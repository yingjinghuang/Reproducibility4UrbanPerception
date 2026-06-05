"""PyTorch datasets for PP2.0.

Two training targets supported (D1 decision pending):

  1. QScoreDataset
     One image -> 6-vector of TrueSkill mu (with NaN mask for unrated dimensions).
     Loss uses masked MSE so 6-task multi-head training works on sparse labels.

  2. PairwiseDataset
     (left_img, right_img, dim) -> {0, 1} (1 if left wins).
     'equal' votes: option to drop or treat as 0.5 soft label (default: drop for
     binary BCE; flip on `keep_equal=True` to use ternary).

Both datasets attach metadata (city_proxy, global_south, continent, …) for
analysis. Image transforms follow ImageNet defaults (224x224, mean/std normalize).
"""
from __future__ import annotations
from pathlib import Path
from typing import Literal
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

from .paths import (
    DIMENSIONS,
    IMAGES,
    IMAGE_GEO,
    QSCORES,
    SPLITS,
    VOTES_CSV,
)


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def default_train_transform(img_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.15)),
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def default_eval_transform(img_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def tta_transform(img_size: int = 224, seed: int | None = None) -> transforms.Compose:
    """Stochastic transform for test-time augmentation. Reuse train-style aug.

    For deterministic TTA replication, set torch global RNG before each call.
    """
    return default_train_transform(img_size)


def _load_qscore_table() -> pd.DataFrame:
    """Pivot qscores long → image_id × dimension wide on `mu`."""
    qs = pd.read_parquet(QSCORES)
    wide_mu = qs.pivot(index="image_id", columns="dimension", values="mu")
    wide_n = qs.pivot(index="image_id", columns="dimension", values="n_votes").fillna(0).astype(int)
    wide_mu.columns = [f"mu_{c}" for c in wide_mu.columns]
    wide_n.columns = [f"n_{c}" for c in wide_n.columns]
    return wide_mu.join(wide_n).reset_index()


def build_metadata_table() -> pd.DataFrame:
    """Join geo + qscores + splits into a single image-level metadata table.

    Filters out images whose JPEG file is missing on disk.
    """
    geo = pd.read_parquet(IMAGE_GEO)
    qs_wide = _load_qscore_table()
    splits = pd.read_parquet(SPLITS)
    df = geo.merge(qs_wide, on="image_id", how="inner").merge(splits, on="image_id", how="inner")

    # filter on file existence (PP2.0 zip is missing ~600 images)
    df["img_path"] = df["image_id"].apply(lambda i: str(IMAGES / f"{i}.jpg"))
    df["exists"] = df["img_path"].apply(lambda p: Path(p).is_file())
    n_missing = int((~df["exists"]).sum())
    if n_missing:
        df = df[df["exists"]].reset_index(drop=True)
    return df.drop(columns=["exists"])


class QScoreDataset(Dataset):
    """Per-image regression target: 6-vector of TrueSkill mu, with NaN mask."""

    def __init__(
        self,
        split: Literal["train", "val", "test"],
        transform=None,
        meta_df: pd.DataFrame | None = None,
        normalize_targets: bool = True,
    ) -> None:
        if meta_df is None:
            meta_df = build_metadata_table()
        self.meta = meta_df[meta_df["split"] == split].reset_index(drop=True)
        self.transform = transform or (
            default_train_transform() if split == "train" else default_eval_transform()
        )
        self.dimensions = list(DIMENSIONS)
        self.target_cols = [f"mu_{d}" for d in self.dimensions]

        if normalize_targets:
            train_meta = meta_df[meta_df["split"] == "train"]
            self.target_mean = train_meta[self.target_cols].mean().to_numpy().copy()
            self.target_std = train_meta[self.target_cols].std().to_numpy().copy()
            self.target_std[self.target_std < 1e-6] = 1.0
        else:
            self.target_mean = np.zeros(len(self.target_cols))
            self.target_std = np.ones(len(self.target_cols))

    def __len__(self) -> int:
        return len(self.meta)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        row = self.meta.iloc[idx]
        img = Image.open(row["img_path"]).convert("RGB")
        img = self.transform(img)

        raw = row[self.target_cols].to_numpy(dtype=np.float64)
        mask = ~np.isnan(raw)
        target = (raw - self.target_mean) / self.target_std
        target = np.where(mask, target, 0.0)

        return {
            "image": img,
            "target": torch.from_numpy(target).float(),
            "mask": torch.from_numpy(mask).float(),
            "image_id": row["image_id"],
        }


class PairwiseDataset(Dataset):
    """Pairwise vote dataset: (left, right, dim) → {0,1} for binary BCE.

    Equal votes are dropped by default (`keep_equal=False`), matching the
    convention of the original PP2.0 paper.
    """

    def __init__(
        self,
        split: Literal["train", "val", "test"],
        transform=None,
        meta_df: pd.DataFrame | None = None,
        keep_equal: bool = False,
    ) -> None:
        if meta_df is None:
            meta_df = build_metadata_table()
        self.meta = meta_df.set_index("image_id")
        self.transform = transform or (
            default_train_transform() if split == "train" else default_eval_transform()
        )

        # Load votes and filter to those whose BOTH endpoints are in this split
        votes = pd.read_csv(VOTES_CSV)
        if not keep_equal:
            votes = votes[votes["winner"].isin({"left", "right"})]
        keep_ids = set(meta_df[meta_df["split"] == split]["image_id"])
        votes = votes[votes["left_id"].isin(keep_ids) & votes["right_id"].isin(keep_ids)]
        # Also require image files exist (already filtered in meta_df)
        votes = votes[votes["left_id"].isin(self.meta.index) & votes["right_id"].isin(self.meta.index)]
        votes = votes.reset_index(drop=True)
        self.votes = votes

        self.dim2idx = {d: i for i, d in enumerate(DIMENSIONS)}

    def __len__(self) -> int:
        return len(self.votes)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        row = self.votes.iloc[idx]
        l_path = self.meta.loc[row["left_id"], "img_path"]
        r_path = self.meta.loc[row["right_id"], "img_path"]
        l_img = self.transform(Image.open(l_path).convert("RGB"))
        r_img = self.transform(Image.open(r_path).convert("RGB"))
        if row["winner"] == "left":
            label = 1.0
        elif row["winner"] == "right":
            label = 0.0
        else:  # equal
            label = 0.5
        return {
            "left": l_img,
            "right": r_img,
            "dim_idx": self.dim2idx[row["category"]],
            "label": torch.tensor(label, dtype=torch.float32),
            "left_id": row["left_id"],
            "right_id": row["right_id"],
        }


def masked_mse_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """MSE only on positions where mask==1; safe for all-zero mask rows."""
    se = (pred - target) ** 2
    masked = se * mask
    denom = mask.sum().clamp(min=1.0)
    return masked.sum() / denom
