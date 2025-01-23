from prettytable import PrettyTable
import os
import random
from datetime import datetime
import torch
import torch.optim as optim
import yaml
from models.diffusion import DiffusionProcess, LatentDiffusionModel
from models.latent import LatentEmbedding, LinearEmbedding, CNNHead, UNetHead
from data.datasets import get_data, flatten_dataset


def save_experiment(
    model, optimizer, loss_history, config, experiment_name, save_dir="saved_models"
):
    experiment_path = os.path.join(save_dir, experiment_name)
    os.makedirs(experiment_path, exist_ok=True)

    model_path = os.path.join(experiment_path, "model.pth")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss_history": loss_history,
            "config": config,
        },
        model_path,
    )

    config_path = os.path.join(experiment_path, "config.yml")
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    print(f"Experiment saved in: {experiment_path}")


def load_experiment(experiment_name, save_dir="saved_models"):
    """
    Load experiment configurations, models, and optimizer.

    Args:
        experiment_name (str): Name of the experiment folder.
        save_dir (str): Directory where experiments are saved. Default is "saved_models".

    Returns:
        tuple: Loaded model, diffusion process, optimizer, and config.
    """

    # Paths
    experiment_path = os.path.join(save_dir, experiment_name)
    model_path = os.path.join(experiment_path, "model.pth")
    config_path = os.path.join(experiment_path, "config.yml")

    # Load configuration and checkpoint
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    checkpoint = torch.load(model_path, weights_only=True)

    # Configuration
    data_config = config["data"]
    latent_dim = data_config["latent_dim"]
    encdec_model_config = config["models"]["encoder-decoder"]
    encdec_head = encdec_model_config["head"]
    encdec_config = encdec_model_config["model_config"]
    diffusion_model_config = config["models"]["diffusion"]
    diffusion_training_config = config["training"]["diffusion"]

    # Load data to infer input shape
    X = get_data(data_name=data_config["data_name"], n_samples=1)
    _, dim1, dim2 = flatten_dataset(X, return_shape=True)[1]

    # Extract decoder components
    def extract_checkpoint(prefix):
        return {
            key.replace(f"{prefix}.", ""): value
            for key, value in checkpoint["model_state_dict"].items()
            if key.startswith(prefix)
        }

    decoder_backbone_checkpoint = extract_checkpoint("decoder.backbone")
    decoder_head_checkpoint = extract_checkpoint("decoder.head")

    # Initialize decoder components
    linear_decoder = LinearEmbedding(decoder_backbone_checkpoint["mapping_matrix"])
    head_dec = (
        CNNHead(**encdec_config) if encdec_head == "CNN" else UNetHead(**encdec_config)
    )
    head_dec.load_state_dict(decoder_head_checkpoint)

    decoder = LatentEmbedding(
        data_shape=(1, dim1, dim2),
        backbone=linear_decoder,
        head=head_dec,
    )

    # Extract diffusion model weights
    diffusion_model_checkpoint = {
        key: value
        for key, value in checkpoint["model_state_dict"].items()
        if not key.startswith("decoder")
    }

    # Initialize diffusion components
    diffusion = DiffusionProcess(
        timesteps=diffusion_training_config["timesteps"],
        beta_bounds=tuple(diffusion_training_config["beta_bounds"]),
    )

    model = LatentDiffusionModel(
        input_dim=latent_dim,
        time_embed_dim=diffusion_model_config["time_embed_dim"],
        hidden_units=diffusion_model_config["hidden_units"],
        dropout=diffusion_model_config["dropout"],
        depth=diffusion_model_config["depth"],
        use_residual=diffusion_model_config["use_residual"],
    )

    model.load_state_dict(diffusion_model_checkpoint)
    model.decoder = decoder
    model.eval()

    # Initialize optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=diffusion_training_config["learning_rate"],
        weight_decay=diffusion_training_config["weight_decay"],
    )
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return model, diffusion, optimizer, config


def count_parameters(model):
    """
    Count and display the trainable parameters in the model.

    Args:
        model (torch.nn.Module): The model whose parameters need to be counted.

    Returns:
        None
    """
    table = PrettyTable(["Modules", "Shape", "Parameters"])
    total_params = 0

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, list(parameter.shape), params])
        total_params += params

    print(table)
    print(f"Total Trainable Parameters: {total_params}")
    return


def update_experiment_log(
    experiment_name, config, log_path="saved_models/experiments_log.yml"
):
    """
    Updates a YAML log file with a new experiment's configuration.

    Args:
        experiment_name (str): Name of the experiment.
        config (dict): Experiment configuration dictionary.
        log_path (str): Path to the YAML log file.
    """
    log = {}

    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            log = yaml.safe_load(f) or {}

    log[experiment_name] = config

    with open(log_path, "w") as f:
        yaml.dump(log, f, default_flow_style=False)

    print(f"Experiment '{experiment_name}' added to the log.")


def generate_funny_name():
    adjectives = [
        "dancing",
        "sleepy",
        "hungry",
        "purple",
        "jolly",
        "fluffy",
        "grumpy",
        "spicy",
        "sparkly",
        "noisy",
        "lazy",
        "quirky",
        "silly",
        "bouncy",
        "tasty",
        "fuzzy",
    ]
    nouns = [
        "banana",
        "unicorn",
        "sloth",
        "panda",
        "narwhal",
        "dinosaur",
        "octopus",
        "platypus",
        "mango",
        "waffle",
        "penguin",
        "donut",
        "koala",
        "cactus",
        "ninja",
        "robot",
    ]

    adjective = random.choice(adjectives)
    noun = random.choice(nouns)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    return f"{adjective}-{noun}-{timestamp}"
