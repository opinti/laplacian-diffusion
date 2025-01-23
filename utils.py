from prettytable import PrettyTable
import os
import random
from datetime import datetime
import torch
import torch.optim as optim
import yaml
from models.diffusion import LatentDiffusionModel


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

    experiment_path = os.path.join(save_dir, experiment_name)
    model_path = os.path.join(experiment_path, "model.pth")
    config_path = os.path.join(experiment_path, "config.yml")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    checkpoint = torch.load(model_path)

    model = LatentDiffusionModel(
        input_dim=config["data"]["latent_dim"],
        time_embed_dim=config["model"]["time_embed_dim"],
        hidden_units=config["model"]["hidden_units"],
        dropout=config["model"]["dropout"],
        depth=config["model"]["depth"],
        use_residual=config["model"]["use_residual"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    optimizer = optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return model, optimizer, config


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
