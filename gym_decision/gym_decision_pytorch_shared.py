"""Shared PyTorch utilities for gym-decision models."""

import random
import torch


def seed_everything(seed: int) -> random.Random:
    """Seed Python and PyTorch RNGs from one integer seed.

    Returns a dedicated random.Random stream initialized from the same seed
    for deterministic data generation.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return random.Random(seed)


def seed_torch_from_rng(rng: random.Random) -> int:
    """Set PyTorch global RNG state using a seed sampled from rng.

    This keeps RNG ownership in the shared Python random.Random streams while
    still making torch parameter initialization reproducible.
    """
    torch_seed = rng.randrange(0, 2**63 - 1)
    torch.manual_seed(torch_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(torch_seed)
    return torch_seed


def configure_torch_determinism(enabled: bool) -> None:
    """Configure PyTorch deterministic behavior.

    When enabled, PyTorch is asked to use deterministic kernels where
    available. This improves reproducibility, especially on GPU.
    """
    torch.use_deterministic_algorithms(enabled)
    torch.backends.cudnn.deterministic = enabled
    torch.backends.cudnn.benchmark = not enabled


def instances_to_tensors(instances) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert a list of InputInstance objects into feature and label tensors."""
    X = torch.tensor([[inst.v, inst.p, inst.e] for inst in instances],
                     dtype=torch.float32)  # shape (num_instances, 3)
    y = torch.tensor([inst.calc_target_value() for inst in instances],
                     # shape (num_instances, 1)
                     dtype=torch.float32).unsqueeze(1)
    return X, y
