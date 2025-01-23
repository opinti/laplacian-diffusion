import torch
import torch.nn as nn


class LatentEmbedding(nn.Module):
    """
    A model that:
      1) Applies a linear embedding to an input vector,
      2) Reshapes to (B, C, H, W),
      3) Refines the image with a CNN or U-Net.

    Args:
        data_shape (tuple): (C, H, W) describing how to reshape the linear output before refinement.
        linear_embedding (nn.Module): Model backbone, asumed to be a LinearEmbedding instance.
        head (nn.Module): Model last layers. Either CNNRefine or UNetRefine or another CNN-based refiner.
    """

    def __init__(
        self,
        data_shape: tuple,
        backbone: nn.Module,
        head: nn.Module,
    ):
        super().__init__()

        self.data_shape = data_shape
        self.backbone = backbone
        self.head = head

        # Freeze the backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x):
        """
        x: (B, input_dim) - a flattened input vector, or any d_in dimension.
        Returns: (B, C, H, W) refined output.
        """
        B = x.size(0)

        z = self.backbone(x)  # shape (B, d_out), e.g. d_out = C*H*W
        z_unflatten = z.view(B, *self.data_shape)  # (B, C, H, W)
        z_residual = self.head(z_unflatten)  # (B, C, H, W)
        return z + z_residual.view(B, -1)  # (B, d_out)


class LinearEmbedding(nn.Module):
    def __init__(self, mapping_matrix: torch.Tensor):
        """
        A simple module to apply a linear transformation.

        Args:
            mapping_matrix (torch.Tensor): A tensor of shape (d_out, d_in) defining the linear transformation.
        """
        super(LinearEmbedding, self).__init__()
        self.register_buffer("mapping_matrix", mapping_matrix)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the linear transformation.

        Args:
            x (torch.Tensor): Input tensor of shape (n_batch, d_in).

        Returns:
            torch.Tensor: Transformed tensor of shape (n_batch, d_out).
        """
        return x @ self.mapping_matrix.T


class CNNHead(nn.Module):
    """
    A simple, configurable CNN.
    """

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        num_features=32,
        num_layers=3,
        kernel_size=3,
        padding=1,
        stride=1,
    ):
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            num_features (int): Number of intermediate feature maps.
            num_layers (int): Total number of Conv layers.
            kernel_size (int): Kernel size for each Conv layer.
            padding (int): Padding for each Conv layer.
            stride (int): Stride for each Conv layer.
        """
        super().__init__()

        layers = []
        for i in range(num_layers):

            in_ch = in_channels if i == 0 else num_features
            out_ch = out_channels if i == (num_layers - 1) else num_features

            layers.append(
                nn.Conv2d(
                    in_ch,
                    out_ch,
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                )
            )

        self.conv_layers = nn.ModuleList(layers)
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): (B, in_channels, H, W)
        Returns:
            torch.Tensor: (B, out_channels, H, W)
        """
        out = x
        for i, conv in enumerate(self.conv_layers):
            out = conv(out)
            if i < len(self.conv_layers) - 1:
                out = self.activation(out)
        return out


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNetHead(nn.Module):
    """
    A small U-Net variant for refining  a pre-decoded image.
    For images with shape (B, in_channels, H, W).
    """

    def __init__(self, in_channels=1, out_channels=1, base_features=32):
        super().__init__()

        self.enc1 = DoubleConv(in_channels, base_features)
        self.enc2 = DoubleConv(base_features, base_features * 2)
        self.enc3 = DoubleConv(base_features * 2, base_features * 4)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = DoubleConv(base_features * 4, base_features * 8)

        self.up3 = nn.ConvTranspose2d(
            base_features * 8, base_features * 4, kernel_size=2, stride=2
        )
        self.dec3 = DoubleConv(base_features * 8, base_features * 4)

        self.up2 = nn.ConvTranspose2d(
            base_features * 4, base_features * 2, kernel_size=2, stride=2
        )
        self.dec2 = DoubleConv(base_features * 4, base_features * 2)

        self.up1 = nn.ConvTranspose2d(
            base_features * 2, base_features, kernel_size=2, stride=2
        )
        self.dec1 = DoubleConv(base_features * 2, base_features)

        self.final_conv = nn.Conv2d(base_features, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        b = self.bottleneck(self.pool(e3))

        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.final_conv(d1)
