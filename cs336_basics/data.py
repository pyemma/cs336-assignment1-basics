import numpy as np
import torch


def get_batch(x: np.array, batch_size: int, context_length: int, device: str = "mps") -> tuple[torch.Tensor, torch.Tensor]:
    """
        Sample from the input x, return (batch_size, context_length) tensors
    """
    offsets = np.random.randint(0, x.shape[0] - context_length, batch_size)
    offsets = offsets[:, np.newaxis]  # expand for one dimension
    lengths = np.arange(context_length)
    lengths = lengths[np.newaxis, :]
    # use the advanced indexing and broadcasting
    inp_indices = offsets + lengths
    inp = x[inp_indices]
    tgt_indices = inp_indices + 1  # offset by one
    tgt = x[tgt_indices]
    return torch.from_numpy(inp).to(device), torch.from_numpy(tgt).to(device)
