"""Device management utilities for GPU training"""

import torch


def get_device(prefer_gpu=True):
    """
    Automatically detect and return the best available device.

    Args:
        prefer_gpu: If True, prefer GPU when available

    Returns:
        torch.device: The detected device
    """
    if prefer_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        )
    else:
        device = torch.device("cpu")
        print("💻 Using CPU")

    return device


def to_device(data, device):
    """
    Move data to device recursively.

    Args:
        data: Tensor, dict of tensors, or HeteroData
        device: Target device

    Returns:
        Data moved to device
    """
    if hasattr(data, "to"):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return type(data)(to_device(item, device) for item in data)
    else:
        return data


def get_device_info():
    """Print detailed GPU/CPU information."""
    if torch.cuda.is_available():
        print("\n" + "=" * 60)
        print("GPU Information")
        print("=" * 60)
        print(f"Device: {torch.cuda.get_device_name(0)}")
        print(f"Compute Capability: {torch.cuda.get_device_capability(0)}")
        print(
            f"Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        )
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"PyTorch CUDA Enabled: {torch.cuda.is_available()}")
        print("=" * 60 + "\n")
    else:
        print("\n⚠️  CUDA not available. Training will run on CPU.")
