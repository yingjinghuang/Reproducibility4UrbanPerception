"""Compute three image complexity proxies for all images (D9).

Proxies:
  1. shannon_entropy   — Shannon entropy of grayscale histogram (256 bins).
                         Cheap; captures texture / variance.
  2. dino_feat_norm    — L2 norm of DINOv2 ViT-B/14 [CLS] embedding (frozen).
                         Captures "scene richness" in the foundation-model embedding space.
  3. seg_diversity     — Number of distinct semantic classes & Gini diversity from
                         SegFormer-B0 finetuned on ADE20K (150 classes).
                         Captures scene-content heterogeneity.

Output: data_processed/image_complexity.parquet

Runs DINO + SegFormer on GPU 0. ~5-10 minutes for 110K images at batch 64 on RTX 6000.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from skimage.measure import shannon_entropy

ImageFile.LOAD_TRUNCATED_IMAGES = True
from transformers import AutoImageProcessor, AutoModel, SegformerForSemanticSegmentation

from polygeo.paths import IMAGE_GEO, IMAGES, DATA

OUT = DATA / "image_complexity.parquet"

DINO_MODEL = "facebook/dinov2-base"
SEG_MODEL = "nvidia/segformer-b0-finetuned-ade-512-512"
BATCH = 64
NUM_WORKERS = 8
DEVICE = "cuda:0"


class _ImageBatchDataset(Dataset):
    def __init__(self, paths: list[str], tfm) -> None:
        self.paths = paths
        self.tfm = tfm

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.tfm(img), idx


def main() -> None:
    geo = pd.read_parquet(IMAGE_GEO)
    geo["img_path"] = geo["image_id"].apply(lambda i: str(IMAGES / f"{i}.jpg"))
    geo["exists"] = geo["img_path"].apply(lambda p: Path(p).is_file())
    geo = geo[geo["exists"]].reset_index(drop=True)
    print(f"computing complexity on {len(geo):,} images")

    # ---- Pass 1: shannon entropy (CPU, parallel via DataLoader) ----
    print("\n=== shannon entropy ===")
    t0 = time.time()
    ent_tfm = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ])

    def _load_gray(path: str) -> np.ndarray | None:
        try:
            img = Image.open(path).convert("L")
            img = ent_tfm(img)
            return np.asarray(img, dtype=np.uint8)
        except (OSError, Image.DecompressionBombError) as e:
            return None

    entropies = np.full(len(geo), np.nan, dtype=np.float32)
    n_bad = 0
    for i, p in enumerate(geo["img_path"]):
        if i % 20000 == 0:
            print(f"  {i:,} / {len(geo):,}  ({time.time()-t0:.1f}s)  bad={n_bad}")
        arr = _load_gray(p)
        if arr is None:
            n_bad += 1
            continue
        entropies[i] = shannon_entropy(arr)
    print(f"  bad images: {n_bad}")
    print(f"  done in {time.time()-t0:.1f}s; mean={entropies.mean():.3f} std={entropies.std():.3f}")

    # ---- Pass 2: DINOv2 feature norm ----
    print("\n=== DINOv2 feature norm ===")
    t0 = time.time()
    dino_proc = AutoImageProcessor.from_pretrained(DINO_MODEL, use_fast=True)
    dino = AutoModel.from_pretrained(DINO_MODEL).to(DEVICE).eval()
    dino_tfm = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(dino_proc.image_mean, dino_proc.image_std),
    ])

    ds = _ImageBatchDataset(geo["img_path"].tolist(), dino_tfm)
    loader = DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=NUM_WORKERS)
    dino_norms = np.empty(len(geo), dtype=np.float32)
    with torch.no_grad():
        for bi, (imgs, idxs) in enumerate(loader):
            if bi % 100 == 0:
                print(f"  batch {bi}/{len(loader)}  ({time.time()-t0:.1f}s)")
            out = dino(pixel_values=imgs.to(DEVICE, non_blocking=True))
            cls = out.last_hidden_state[:, 0]  # [B, hidden]
            norms = cls.norm(p=2, dim=1).float().cpu().numpy()
            dino_norms[idxs.numpy()] = norms
    del dino
    torch.cuda.empty_cache()
    print(f"  done in {time.time()-t0:.1f}s; mean={dino_norms.mean():.3f} std={dino_norms.std():.3f}")

    # ---- Pass 3: SegFormer diversity ----
    print("\n=== SegFormer-B0 ADE20K diversity ===")
    t0 = time.time()
    seg_proc = AutoImageProcessor.from_pretrained(SEG_MODEL, use_fast=True)
    seg = SegformerForSemanticSegmentation.from_pretrained(SEG_MODEL).to(DEVICE).eval()
    seg_tfm = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(seg_proc.image_mean, seg_proc.image_std),
    ])

    ds2 = _ImageBatchDataset(geo["img_path"].tolist(), seg_tfm)
    loader2 = DataLoader(ds2, batch_size=32, shuffle=False, num_workers=NUM_WORKERS)
    n_classes = seg.config.num_labels  # 150 for ADE20K
    seg_n_classes = np.empty(len(geo), dtype=np.int32)
    seg_gini = np.empty(len(geo), dtype=np.float32)
    seg_entropy = np.empty(len(geo), dtype=np.float32)
    with torch.no_grad():
        for bi, (imgs, idxs) in enumerate(loader2):
            if bi % 100 == 0:
                print(f"  batch {bi}/{len(loader2)}  ({time.time()-t0:.1f}s)")
            logits = seg(pixel_values=imgs.to(DEVICE, non_blocking=True)).logits  # [B, C, h, w]
            preds = logits.argmax(dim=1)  # [B, h, w]
            B = preds.shape[0]
            preds_flat = preds.view(B, -1)
            for j in range(B):
                vals, cnts = torch.unique(preds_flat[j], return_counts=True)
                p = (cnts.float() / cnts.sum()).cpu().numpy()
                gid = idxs[j].item()
                seg_n_classes[gid] = len(vals)
                seg_gini[gid] = float(1.0 - (p ** 2).sum())
                seg_entropy[gid] = float(-(p * np.log2(p + 1e-12)).sum())
    del seg
    torch.cuda.empty_cache()
    print(f"  done in {time.time()-t0:.1f}s; mean #classes={seg_n_classes.mean():.2f}, gini={seg_gini.mean():.3f}, entropy={seg_entropy.mean():.3f}")

    # ---- Output ----
    out = pd.DataFrame({
        "image_id": geo["image_id"].to_numpy(),
        "complexity_shannon": entropies,
        "complexity_dino_norm": dino_norms,
        "complexity_seg_n_classes": seg_n_classes,
        "complexity_seg_gini": seg_gini,
        "complexity_seg_entropy": seg_entropy,
    })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    print(f"\nwrote {OUT}  ({OUT.stat().st_size/1e6:.1f} MB)")

    # Cross-correlations among proxies (sanity for D9)
    cor = out[[
        "complexity_shannon",
        "complexity_dino_norm",
        "complexity_seg_n_classes",
        "complexity_seg_gini",
        "complexity_seg_entropy",
    ]].corr()
    print("\ncross-correlation (Pearson):")
    print(cor.round(3).to_string())


if __name__ == "__main__":
    main()
