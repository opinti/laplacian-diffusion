import torch
from torchvision import datasets, transforms
from specmf.data import load_data
import numpy as np


torch.set_default_dtype(torch.float32)


# Define the folder for storing data
DATA_DIR = "/Users/orazio/codes/laplacian-diffusion/data"


def get_data(data_name: str, **kwargs):
    """
    Load specified dataset with optional preprocessing.

    Args:
        data_name (str): Name of the dataset ('mnist', 'cifar10', 'darcy').
        data_dir (str): Directory to store datasets.
        **kwargs: Additional parameters for dataset-specific functions.

    Returns:
        Tensor: Processed dataset.
    """
    if data_name == "mnist":
        return get_mnist(**kwargs)
    elif data_name == "cifar10":
        return get_cifar10(**kwargs)
    elif data_name == "darcy":
        return get_darcy_flow(**kwargs)
    else:
        raise ValueError(f"Unknown data_name: {data_name}")


def get_mnist(data_dir: str = DATA_DIR, n_samples: int = 4000, classes: list = None):
    """
    Load and preprocess MNIST dataset, filtering for specific classes.

    Args:
    - data_dir (str): Path to store/load the MNIST dataset.
    - n_samples (int): Number of samples to return.
    - classes (list): List of class labels to filter (e.g., [3, 4]).

    Returns:
    - torch.Tensor: Images of the specified classes.
    - torch.Tensor: Corresponding labels of the specified classes.
    """
    print("Loading MNIST dataset...")
    mnist_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )

    if classes is not None:
        print("Filtering for specific classes...")
        # Filter the dataset for the specified classes
        mnist_dataset = [
            (img, label) for img, label in mnist_dataset if label in classes
        ]

    # Shuffle and sample the filtered data
    print(f"Sampling {n_samples} images...")
    mnist_loader = torch.utils.data.DataLoader(
        mnist_dataset, batch_size=n_samples, shuffle=True
    )
    images, _ = next(iter(mnist_loader))
    return images.squeeze(1)


def get_cifar10(data_dir: str = DATA_DIR, n_samples: int = 2000, class_index: int = 3):
    """
    Load and preprocess CIFAR-10 dataset.
    """
    print("Loading CIFAR-10 dataset...")
    cifar10_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=transforms.Compose([transforms.Grayscale(), transforms.ToTensor()]),
    )

    print(f"Filtering class {class_index}...")
    filtered_data = [img for img, label in cifar10_dataset if label == class_index]
    images_tensor = torch.cat(filtered_data)
    return images_tensor[:n_samples]


def get_darcy_flow(
    data_dir: str = "/Users/orazio/codes/spectral-multifidelity/data",
    n_samples: int = 3000,
):
    """
    Load Darcy flow dataset.
    """
    dataset_name = "darcy-flow"
    X, _ = load_data(
        dataset_name,
        preprocess=True,
        normalize=True,
        flatten=False,
        return_mask=False,
        return_normalization_vars=False,
        data_path=data_dir,
    )
    X = np.transpose(X, axes=(2, 0, 1))
    return X[:n_samples]


def flatten_dataset(data, return_shape=False):
    """
    Flatten tensor from (n, dim1, dim2) to (n, dim1 * dim2).
    """
    if not torch.is_tensor(data):
        data = torch.tensor(data).float()

    if data.dim() != 3:
        raise ValueError(f"Invalid data shape: {data.size()}")

    if return_shape:
        return data.view(data.size(0), -1), data.size()

    return data.view(data.size(0), -1)
