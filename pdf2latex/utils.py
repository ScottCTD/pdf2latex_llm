import torch

def length_until_last(x, value):
    assert x.ndim == 2, "Tensor must be 2D"
    mask = (x == value)
    rev_mask = torch.flip(mask, dims=[1])
    last_rev_idx = rev_mask.int().argmax(dim=1)
    
    has_value = mask.any(dim=1)
    last_idx = x.size(1) - 1 - last_rev_idx
    size_until = torch.where(has_value, last_idx + 1, torch.zeros_like(last_idx))
    return size_until

def index(x: torch.Tensor, value) -> torch.Tensor:
    assert x.ndim == 2, "Tensor must be 2D"

    mask = (x == value)
    has_value = mask.any(dim=1)
    assert torch.all(has_value), "Value not found in at least one row"

    # argmax gives the index of the first True per row
    first_indices = mask.float().argmax(dim=1)
    return first_indices