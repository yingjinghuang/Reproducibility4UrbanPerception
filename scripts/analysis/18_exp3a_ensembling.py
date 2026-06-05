"""Exp 3a — Ensembling mitigation analysis (compute-only, no new training).

Question: as we ensemble more seeds, does the reproducibility gap shrink across
all three architectures, and does it shrink at the same rate?

Method: bootstrap ensembles of size N from the per-architecture seed pools:
  - ViT-B/16     : 80 seeds → N ∈ {1, 2, 4, 8, 16, 32}
  - ResNet-50    : 40 seeds → N ∈ {1, 2, 4, 8, 16}
  - DINOv2 frozen: 40 seeds → N ∈ {1, 2, 4, 8, 16}
We cap N at pool/2 to avoid finite-pool bias in the bootstrap variance: when
N > S/2 the random subsets overlap so heavily that cross-ensemble std is
artificially compressed by a factor (S-N)/(S-1).

For each (arch, ensemble_size):
  - draw B=200 random seed subsets
  - for each subset, predict by mean of subset predictions
  - measure cross-ensemble per-image rel_std (variance across the B ensembles)

Outputs (rows now carry an `arch` column):
  data_processed/exp3a_ensemble_curve.parquet
  data_processed/exp3a_summary.json
"""
from __future__ import annotations
import sys
import json
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from polygeo.paths import ROOT, DATA, IMAGE_GEO, PREDICTIONS, DIMENSIONS

if os.environ.get("POLYGEO_FONT_DIR"):
    for _f in ("arial.ttf", "arialbd.ttf", "ariali.ttf"):
        _p = Path(os.environ["POLYGEO_FONT_DIR"]) / _f
        if _p.exists():
            fm.fontManager.addfont(str(_p))
mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titlesize": 17,
    "axes.labelsize": 17,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 14,
})

OUT_PARQ = DATA / "exp3a_ensemble_curve.parquet"
OUT_JSON = DATA / "exp3a_summary.json"
FIGS = ROOT / "figures"

ARCHS = [
    ("vit_b16",       "exp1_vit_b16_imagenet21k_seed{s:03d}_test_preds.npz",       80, [1, 2, 4, 8, 16, 32]),
    ("resnet50",      "exp1_resnet50_imagenet21k_seed{s:03d}_test_preds.npz",      40, [1, 2, 4, 8, 16]),
    ("dinov2_frozen", "exp1_dinov2_frozen_imagenet21k_seed{s:03d}_test_preds.npz", 40, [1, 2, 4, 8, 16]),
]
N_BOOTSTRAP = 200  # number of random subsets per ensemble size


def compute_one_arch(arch: str, pattern: str, pool: int,
                     ensemble_sizes: list[int], rng: np.random.Generator) -> pd.DataFrame:
    print(f"\n=== {arch} (pool={pool}) ===")
    arrs = []
    image_ids = None
    for s in range(pool):
        npz = np.load(PREDICTIONS / pattern.format(s=s), allow_pickle=True)
        if image_ids is None:
            image_ids = list(npz["image_ids"])
        arrs.append(npz["preds"].astype(np.float32))
    preds = np.stack(arrs, axis=-1)  # [n_img, n_dim, pool]
    n_img, n_dim, _ = preds.shape
    print(f"  loaded {arch} preds: {preds.shape}")

    mean_per_img_full = preds.mean(axis=2)
    dim_scale = mean_per_img_full.std(axis=0, ddof=0)  # [n_dim]

    chunks = []
    for N in ensemble_sizes:
        if N == 1:
            ens_std = preds.std(axis=2, ddof=0)
        else:
            ensembles = np.empty((n_img, n_dim, N_BOOTSTRAP), dtype=np.float32)
            for b in range(N_BOOTSTRAP):
                idx = rng.choice(pool, size=N, replace=False)
                ensembles[:, :, b] = preds[:, :, idx].mean(axis=2)
            ens_std = ensembles.std(axis=2, ddof=0)
        rel = ens_std / np.maximum(dim_scale[None, :], 1e-6)
        for d, dim in enumerate(DIMENSIONS):
            chunks.append(pd.DataFrame({
                "arch": arch,
                "image_id": image_ids,
                "dimension": dim,
                "ensemble_size": N,
                "ens_std": ens_std[:, d],
                "rel_std": rel[:, d],
            }))
        print(
            f"  N={N:3d}: median rel_std = {np.median(rel):.4f}  "
            f"frac>10% = {(rel > 0.10).mean():.3f}  frac>5% = {(rel > 0.05).mean():.3f}"
        )
    return pd.concat(chunks, ignore_index=True)


def main() -> None:
    rng = np.random.default_rng(20260505)
    parts = [compute_one_arch(arch, pat, pool, ens, rng) for arch, pat, pool, ens in ARCHS]
    long = pd.concat(parts, ignore_index=True)

    geo = pd.read_parquet(IMAGE_GEO)[["image_id", "global_south", "city_proxy", "continent"]]
    long = long.merge(geo, on="image_id", how="left")

    OUT_PARQ.parent.mkdir(parents=True, exist_ok=True)
    long.to_parquet(OUT_PARQ, index=False)
    print(f"\nwrote {OUT_PARQ}  ({OUT_PARQ.stat().st_size/1e6:.2f} MB)")

    # ---- Per-arch summary: median rel_std vs N, threshold-crossing fractions ----
    print("\n=== per-arch ensemble curve: median rel_std vs N ===")
    summary = {"n_bootstrap": N_BOOTSTRAP, "per_arch": {}}
    for arch, _, pool, ensemble_sizes in ARCHS:
        sub = long[long["arch"] == arch]
        rows = []
        for N in ensemble_sizes:
            v = sub[sub["ensemble_size"] == N]["rel_std"]
            rows.append({
                "ensemble_size": N,
                "median_rel_std": float(v.median()),
                "frac_above_5pct": float((v > 0.05).mean()),
                "frac_above_10pct": float((v > 0.10).mean()),
                "frac_above_15pct": float((v > 0.15).mean()),
            })
        summary["per_arch"][arch] = {"pool": pool, "curve": rows}
        print(f"  {arch:15s} (pool={pool}):")
        for r in rows:
            print(f"    N={r['ensemble_size']:3d}: med={r['median_rel_std']:.4f}  "
                  f">5%={r['frac_above_5pct']:.3f}  >10%={r['frac_above_10pct']:.3f}  "
                  f">15%={r['frac_above_15pct']:.3f}")
    OUT_JSON.write_text(json.dumps(summary, indent=2))
    print(f"wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
