"""Centralised seeding so that one master seed deterministically drives:
  - Python random
  - NumPy
  - PyTorch (CPU + CUDA)
  - DataLoader workers

For the source-decomposition experiment, seeds for different stochastic sources can be
controlled independently. Note: PyTorch dropout shares the global CUDA RNG with
weight initialization, so the initialization-varying condition also varies dropout.
"""
from __future__ import annotations
import os
import random
import numpy as np
import torch


def seed_everything(seed: int, deterministic_cudnn: bool = False) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic_cudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id: int) -> None:
    """For DataLoader: each worker gets seed = base_seed + worker_id (set via initial_seed)."""
    base = torch.initial_seed() % 2**32
    np.random.seed(base + worker_id)
    random.seed(base + worker_id)


def make_aug_worker_init_fn(aug_seed: int):
    """For Exp 2: bind augmentation RNGs to a chosen aug_seed independent of
    DataLoader's internal seeding."""
    def _init(worker_id: int) -> None:
        s = (aug_seed + worker_id) & 0xFFFFFFFF
        np.random.seed(s)
        random.seed(s)
        torch.manual_seed(s)
    return _init
