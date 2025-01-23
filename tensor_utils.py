import numpy as np
import torch


def normalize_tensor(X, dim=0):
    """
    Normalize the rows or columns of a numpy array or torch tensor to have zero mean and unit variance.

    Args:
        X (torch.tensor or np.ndarray): The input tensor of shape (n_samples, n_features).
        dim (int): The dimension to normalize along.
    """
    if isinstance(X, np.ndarray):
        X_mean = X.mean(axis=dim)
        X_std = X.std(axis=dim)
    elif isinstance(X, torch.Tensor):
        X_mean = X.mean(dim=dim)
        X_std = X.std(dim=dim)
    return (X - X_mean) / X_std, X_mean, X_std


def compute_linear_transformation(X, Z):
    """
    Compute the linear transformation matrix that maps the latent space Z to the original space X
    such that X = B @ Z.

    Args:
    - X: Data matrix in the original space. Shape (D, N).
    - Z: Data matrix in the latent space. Shape (K, N).
    """
    C_X = X @ X.T  # Shape (D, D)
    C_ZX = Z @ X.T  # Shape (K, D)
    return C_ZX @ torch.linalg.pinv(C_X)
