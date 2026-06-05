"""Tier 1 — per-image stability across all Exp 1 seeds.

For each of the three architectures (ResNet-50 ×40, ViT-B/16 ×80, DINOv2 ×40):
  - load all per-seed test predictions
  - compute per-image, per-dimension:
      mean, std, range, rel_std (std / dim_scale)
      per-image rank stability via inter-seed Spearman
      disagreement rate (sign of (pred − city_mean) flips across seeds)
  - aggregate to architecture-level summary

Outputs:
  data_processed/tier1_per_image.parquet      long: image_id × dim × arch + stats
  data_processed/tier1_summary.json           per-architecture summary
"""
from __future__ import annotations
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from polygeo.paths import PREDICTIONS, DATA, DIMENSIONS

OUT_PARQ = DATA / "tier1_per_image.parquet"
OUT_JSON = DATA / "tier1_summary.json"

ARCH_PLAN = [
    ("resnet50", "imagenet21k", 40),
    ("vit_b16", "imagenet21k", 80),
    ("dinov2_frozen", "imagenet21k", 40),
]


def load_arch_preds(arch: str, init: str, n_seeds: int) -> tuple[np.ndarray, list[str]]:
    """Returns predictions tensor [n_img, n_dim, n_seeds] and image_ids list."""
    arrays = []
    image_ids = None
    for s in range(n_seeds):
        npz = np.load(PREDICTIONS / f"exp1_{arch}_{init}_seed{s:03d}_test_preds.npz", allow_pickle=True)
        ids = list(npz["image_ids"])
        if image_ids is None:
            image_ids = ids
        else:
            assert ids == image_ids, f"image_ids differ at seed {s}"
        arrays.append(npz["preds"])
    return np.stack(arrays, axis=-1), image_ids


def analyze_arch(arch: str, init: str, n_seeds: int) -> tuple[pd.DataFrame, dict]:
    print(f"\n=== Tier 1: {arch} / {init} (n={n_seeds}) ===")
    preds, image_ids = load_arch_preds(arch, init, n_seeds)
    n_img, n_dim, n = preds.shape
    print(f"  preds shape: {preds.shape}")

    mean_pi = preds.mean(axis=2)
    std_pi = preds.std(axis=2, ddof=0)
    range_pi = preds.max(axis=2) - preds.min(axis=2)
    dim_scale = mean_pi.std(axis=0, ddof=0)            # [n_dim]  baseline scale
    rel_std = std_pi / np.maximum(dim_scale[None, :], 1e-6)

    # disagreement rate: fraction of seed-pairs whose prediction sits on different
    # sides of the global per-dimension median
    medians = np.median(mean_pi, axis=0)               # [n_dim]
    above = (preds > medians[None, :, None]).astype(np.int8)  # [n_img, n_dim, n]
    # for each (img, dim), fraction of seed pairs that disagree on side-of-median
    p_above = above.mean(axis=2)                       # [n_img, n_dim]
    disagree = 2 * p_above * (1 - p_above)             # 0 if all same, 0.5 if 50-50

    # per-image rank stability across seed pairs (Spearman) - sample 10 random pairs
    # to keep cost bounded
    rng = np.random.default_rng(7)
    n_pair_sample = min(10, n * (n - 1) // 2)
    pair_idx = []
    used = set()
    while len(pair_idx) < n_pair_sample:
        i, j = sorted(rng.integers(0, n, 2).tolist())
        if i == j or (i, j) in used:
            continue
        used.add((i, j))
        pair_idx.append((i, j))

    rank_stab_per_dim = []
    for d in range(n_dim):
        rs = []
        for i, j in pair_idx:
            # over all images
            rho, _ = spearmanr(preds[:, d, i], preds[:, d, j])
            rs.append(float(rho))
        rank_stab_per_dim.append((float(np.mean(rs)), float(np.min(rs))))

    # ---- build long-form image-level table ----
    rows = []
    for d, dim in enumerate(DIMENSIONS):
        rows.append(
            pd.DataFrame({
                "image_id": image_ids,
                "arch": arch,
                "dimension": dim,
                "mean_pred": mean_pi[:, d],
                "std_pred": std_pi[:, d],
                "range_pred": range_pi[:, d],
                "rel_std": rel_std[:, d],
                "disagreement_rate": disagree[:, d],
            })
        )
    img_df = pd.concat(rows, ignore_index=True)

    summary = {
        "arch": arch,
        "init": init,
        "n_seeds": int(n),
        "n_images": int(n_img),
        "dim_scale": dict(zip(DIMENSIONS, dim_scale.tolist())),
        "median_std": dict(zip(DIMENSIONS, np.median(std_pi, axis=0).tolist())),
        "median_rel_std": dict(zip(DIMENSIONS, np.median(rel_std, axis=0).tolist())),
        "p95_rel_std": dict(zip(DIMENSIONS, np.percentile(rel_std, 95, axis=0).tolist())),
        "frac_rel_std_gt_5pct": dict(zip(DIMENSIONS, (rel_std > 0.05).mean(axis=0).tolist())),
        "frac_rel_std_gt_10pct": dict(zip(DIMENSIONS, (rel_std > 0.10).mean(axis=0).tolist())),
        "frac_rel_std_gt_25pct": dict(zip(DIMENSIONS, (rel_std > 0.25).mean(axis=0).tolist())),
        "frac_rel_std_gt_50pct": dict(zip(DIMENSIONS, (rel_std > 0.50).mean(axis=0).tolist())),
        "interseed_spearman_mean": dict(
            zip(DIMENSIONS, [t[0] for t in rank_stab_per_dim])
        ),
        "interseed_spearman_min_pair": dict(
            zip(DIMENSIONS, [t[1] for t in rank_stab_per_dim])
        ),
        "median_disagreement_rate": dict(zip(DIMENSIONS, np.median(disagree, axis=0).tolist())),
    }
    print(f"  median rel_std: {[round(v, 3) for v in summary['median_rel_std'].values()]}")
    print(f"  P95 rel_std:    {[round(v, 3) for v in summary['p95_rel_std'].values()]}")
    print(f"  frac>10%:       {[round(v, 3) for v in summary['frac_rel_std_gt_10pct'].values()]}")
    print(f"  frac>25%:       {[round(v, 3) for v in summary['frac_rel_std_gt_25pct'].values()]}")
    print(f"  inter-seed Spearman (mean): {[round(v, 3) for v in summary['interseed_spearman_mean'].values()]}")
    print(f"  inter-seed Spearman (min):  {[round(v, 3) for v in summary['interseed_spearman_min_pair'].values()]}")
    return img_df, summary


def main() -> None:
    summaries = {}
    img_dfs = []
    for arch, init, n_seeds in ARCH_PLAN:
        df, summ = analyze_arch(arch, init, n_seeds)
        summaries[f"{arch}_{init}"] = summ
        img_dfs.append(df)

    out_df = pd.concat(img_dfs, ignore_index=True)
    OUT_PARQ.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(OUT_PARQ, index=False)
    OUT_JSON.write_text(json.dumps(summaries, indent=2))
    print(f"\nwrote {OUT_PARQ} ({OUT_PARQ.stat().st_size/1e6:.2f} MB)")
    print(f"wrote {OUT_JSON}")

    print("\n=== Per-architecture threshold summary (rel_std > 10% on >= 20% of images) ===")
    for key, summ in summaries.items():
        worst_dim, worst_frac = max(summ["frac_rel_std_gt_10pct"].items(), key=lambda kv: kv[1])
        best_dim, best_frac = min(summ["frac_rel_std_gt_10pct"].items(), key=lambda kv: kv[1])
        passes_in_n_dims = sum(v >= 0.20 for v in summ["frac_rel_std_gt_10pct"].values())
        print(
            f"  {key}:  threshold exceeded in {passes_in_n_dims}/6 dims  "
            f"worst-dim={worst_dim} {worst_frac:.1%}  best-dim={best_dim} {best_frac:.1%}"
        )

    # ---- per-arch ranking of 'most unstable dim' ----
    print("\n=== Most unstable dim per arch (frac > 25%) ===")
    for key, summ in summaries.items():
        ranked = sorted(summ["frac_rel_std_gt_25pct"].items(), key=lambda kv: -kv[1])
        print(f"  {key}: {ranked}")


if __name__ == "__main__":
    main()
