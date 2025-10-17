"""Overall setup for pytorch + cuda"""

import torch


def setup_torch_device() -> torch.device:
    """Select device to run torch calculations on, give priority to CUDA gpus
    then Metal (Apple Silicon) and fallback to CPU if none is available.
    """
    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
