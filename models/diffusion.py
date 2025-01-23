import torch
import math
import torch.nn as nn
import torch.nn.functional as F


class TimeEmbedding(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, t):
        half_dim = self.embed_dim // 2
        emb = torch.exp(
            -math.log(10000) * torch.arange(half_dim, device=t.device) / half_dim
        )
        emb = t[:, None] * emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


class DiffusionProcess:
    """
    Class to perform forward and reverse diffusion.
    """

    def __init__(self, timesteps, beta_bounds=(0.0001, 0.02), device=None):
        self._validate_timesteps_and_beta_bounds(timesteps, beta_bounds)
        self.device = device

        betas = torch.linspace(beta_bounds[0], beta_bounds[1], timesteps)
        alphas = 1.0 - betas
        alpha_hat = torch.cumprod(alphas, dim=0)

        self.betas = betas.to(device)
        self.alphas = alphas.to(device)
        self.alpha_hat = alpha_hat.to(device)

        self.sqrt_alpha_hat = torch.sqrt(self.alpha_hat)
        self.sqrt_one_minus_alpha_hat = torch.sqrt(1.0 - self.alpha_hat)

    def forward(self, z, t):
        """
        Forward diffusion: adds noise to `z` for a given time step `t`.

        Args:
            z (torch.Tensor): Input tensor.
            t (torch.Tensor): Time step indices (1D tensor of integers).

        Returns:
            (noisy_z, noise): The noisy version of `z` and the generated noise.
        """
        t = t.to(self.device).long()
        noise = torch.randn_like(z)

        noisy_z = (
            self.sqrt_alpha_hat[t].unsqueeze(-1) * z
            + self.sqrt_one_minus_alpha_hat[t].unsqueeze(-1) * noise
        )
        return noisy_z, noise

    def reverse(
        self, model, num_samples, in_dim, device, denormalize=True, decode=True
    ):
        """
        Reverse diffusion: generates samples from noise by iteratively denoising.

        Args:
            model (nn.Module): Trained model that predicts noise.
            num_samples (int): Number of samples to generate.
            in_dim (int): Dimension of the input data.
            device (torch.device): Device for computations.
            denormalize (bool): Whether to denormalize the generated samples.
            decode (bool): Whether to decode the generated samples.

        Returns:
            torch.Tensor: Generated samples (optionally denormalized or decoded).
        """
        if decode and not denormalize:
            raise ValueError("Cannot decode without denormalizing.")

        model = model.to(device)
        model.eval()

        with torch.no_grad():
            # Start with Gaussian noise
            z = torch.randn(num_samples, in_dim, device=device)
            timesteps = len(self.betas)

            for t in reversed(range(timesteps)):
                t_tensor = torch.full((z.size(0),), t, device=device, dtype=torch.long)
                t_normalized = t_tensor / (timesteps - 1)

                # Predict noise
                predicted_noise = model(z, t_normalized)

                alpha_t = self.alphas[t]
                alpha_hat_t = self.alpha_hat[t]
                beta_t = self.betas[t]

                # Reverse diffusion step
                z = (1 / torch.sqrt(alpha_t)) * (
                    z - beta_t / torch.sqrt(1 - alpha_hat_t + 1e-8) * predicted_noise
                )

                # Add noise if not the last step
                if t > 0:
                    noise = torch.randn_like(z)
                    z += torch.sqrt(beta_t) * noise

            # Optionally denormalize and/or decode
            if denormalize:
                z = model.denormalize(z)
            if decode:
                z = model.decode(z)

            return z

    def _validate_timesteps_and_beta_bounds(self, timesteps: int, beta_bounds: tuple):
        # Check if the number of timesteps is valid
        if timesteps < 1:
            raise ValueError("The number of timesteps must be at least 1.")
        if not isinstance(timesteps, int):
            raise TypeError("The number of timesteps must be an integer.")

        # Check if diffusion parameters are valid
        if not isinstance(beta_bounds, tuple):
            raise TypeError("The beta bounds must be a tuple.")
        if len(beta_bounds) != 2:
            raise ValueError("The beta bounds must contain exactly two elements.")
        if beta_bounds[0] >= beta_bounds[1]:
            raise ValueError("The lower bound must be less than the upper bound.")
        if beta_bounds[0] <= 0:
            raise ValueError("The lower bound must be greater than 0.")


class LatentDiffusionModel(nn.Module):
    def __init__(
        self,
        input_dim,
        time_embed_dim,
        hidden_units,
        dropout,
        depth,
        use_residual: bool = False,
        mean=None,
        std=None,
        decoder: nn.Module = None,
    ):
        """
        Latent Diffusion Model with optional residual connections.

        Args:
            input_dim (int): Dimensionality of the input data.
            time_embed_dim (int): Dimensionality of the time embedding.
            hidden_units (int): Number of hidden units in each layer.
            dropout (float): Dropout rate.
            depth (int): Number of layers.
            use_residual (bool): Whether to use residual connections.
            mean (list or torch.Tensor): Mean for input normalization.
            std (list or torch.Tensor): Standard deviation for input normalization.
            decoder (nn.Module): Decoder model to map the latent space to the original data space.
        """
        super().__init__()
        self.time_embedding = TimeEmbedding(embed_dim=time_embed_dim)
        self.use_residual = use_residual

        # Store normalization parameters as buffers
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(
                mean if mean is not None else [0.0] * input_dim, dtype=torch.float32
            )
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(
                std if std is not None else [1.0] * input_dim, dtype=torch.float32
            )

        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

        # Decoder model is fixed - freeze parameters
        self.decoder = decoder
        if self.decoder is not None:
            for param in self.decoder.parameters():
                param.requires_grad = False

        # Input layer
        self.input_layer = nn.Linear(input_dim + time_embed_dim, hidden_units)
        self.input_bn = nn.BatchNorm1d(hidden_units)
        self.input_activation = nn.SiLU()
        self.input_dropout = nn.Dropout(dropout)

        # Hidden layers
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_units, hidden_units) for _ in range(depth - 1)]
        )
        self.hidden_bns = nn.ModuleList(
            [nn.BatchNorm1d(hidden_units) for _ in range(depth - 1)]
        )
        self.hidden_dropouts = nn.ModuleList(
            [nn.Dropout(dropout) for _ in range(depth - 1)]
        )

        # Output layer
        self.output_layer = nn.Linear(hidden_units, input_dim)

    def forward(self, x, t):
        # Time embedding
        t_emb = self.time_embedding(t)
        t_emb = t_emb.expand(x.size(0), -1)

        # Concatenate input and time embedding
        x_t = torch.cat([x, t_emb], dim=-1)

        # Input layer
        x = self.input_layer(x_t)
        x = self.input_bn(x)
        x = self.input_activation(x)
        x = self.input_dropout(x)

        # Hidden layers with optional residual connections
        for layer, bn, dropout in zip(
            self.hidden_layers, self.hidden_bns, self.hidden_dropouts
        ):
            residual = x
            x = layer(x)
            x = bn(x)
            x = F.silu(x)
            x = dropout(x)
            if self.use_residual:
                x = x + residual

        # Output layer
        x = self.output_layer(x)
        return x

    def denormalize(self, x):
        """
        Denormalize the model output to match the scale of the original data.
        """
        return x * self.std + self.mean

    def decode(self, x_batched):
        """
        Decode the latent representation to the original data space.

        Args:
            x_batched (torch.Tensor): Input tensor of shape (n_batch, d_latent).
        """
        if self.decoder is None:
            raise ValueError("No decoder model provided.")
        return self.decoder(x_batched)
