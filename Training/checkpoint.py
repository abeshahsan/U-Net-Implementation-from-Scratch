"""To save and load checkpoints during training."""

import torch
from Data.transforms import get_train_transforms

def save_checkpoint(state, filename):
    """
    Save model state and optimizer state to a file.

    Args:
    ----------
    state: dict
        A dictionary containing model state, optimizer state, and epoch.
    filename: str
        The name of the file to save the checkpoint.

    Returns:
    ----------
    None
    """

    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(ckpt_file, model, optimizer=None):
    """
    Load model state and optimizer state from a file.

    Args:
    ----------
    ckpt_file: str
        The name of the file to load the checkpoint.
    model: torch.nn.Module
        The model to load the state_dict.
    optimizer: torch.optim.Optimizer, optional
        The optimizer to load the state_dict. Default is None.

    Returns:
    ----------
    int
        The epoch number of the checkpoint.

    """
    checkpoint = torch.load(ckpt_file)
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["epoch"]
