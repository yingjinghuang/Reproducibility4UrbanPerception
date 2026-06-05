"""Multi-split robustness analysis.

Consumes the 48 per-seed test predictions from the multi-split sweep
(scripts/28_multisplit_sweep.py) plus the canonical Exp 1 predictions
to answer three concrete questions for the appendix:

  Q1. Does the order-of-magnitude per-image vs aggregate relstd gap survive
      a change in train/test split?
      - For each split (canonical + alt0/1/2), compute per-image median relstd
        and aggregate relstd, and check that the ratio remains > 5x.

  Q2. Does the ResNet vs ViT ranking reversal survive a change in split?
      - Per split, check whether ViT < ResNet at the aggregate scale and
        ViT > ResNet at the per-image scale.

  Q3. How big is split-induced variance relative to seed-induced variance?
      - Restrict to images that appear in the test set of all 3 alt splits.
      - For each such image: compute (a) within-split-across-seeds std and
        (b) across-split-between-split-means std. Compare distributions.

Outputs:
  data_processed/multisplit_robustness.json   headline numbers for the paper
  data_processed/multisplit_per_image.parquet long table of per-image stats
  figures/figA_multisplit_robustness.pdf      appendix figure
"""
from __future__ import annotations
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from polygeo.paths import PREDICTIONS, DATA, DIMENSIONS, ROOT

OUT_JSON = DATA / "multisplit_robustness.json"
OUT_PARQ = DATA / "multisplit_per_image.parquet"
OUT_FIG = ROOT / "figures/figA_multisplit_robustness.pdf"

N_SEEDS_PER_SPLIT = 8
ARCHS = ["resnet50", "vit_b16"]
INIT = "imagenet21k"


def load_multisplit_preds(split_idx: int, arch: str) -> tuple[np.ndarray, list[str]]:
    """Stack 8 seed predictions for (split_idx, arch) into [n_img, n_dim, n_seed]."""
    arrays, image_ids = [], None
    for s in range(N_SEEDS_PER_SPLIT):
        npz_path = PREDICTIONS / f"multisplit_split{split_idx}_{arch}_seed{s:03d}_test_preds.npz"
        if not npz_path.exists():
            raise FileNotFoundError(npz_path)
        npz = np.load(npz_path, allow_pickle=True)
        ids = list(npz["image_ids"])
        if image_ids is None:
            image_ids = ids
        else:
            assert ids == image_ids, f"image_ids drift at split {split_idx} {arch} seed {s}"
        arrays.append(npz["preds"])
    return np.stack(arrays, axis=-1), image_ids


def load_canonical_preds(arch: str, n_seeds: int) -> tuple[np.ndarray, list[str]]:
    arrays, image_ids = [], None
    for s in range(n_seeds):
        npz = np.load(PREDICTIONS / f"exp1_{arch}_{INIT}_seed{s:03d}_test_preds.npz",
                      allow_pickle=True)
        ids = list(npz["image_ids"])
        if image_ids is None:
            image_ids = ids
        else:
            assert ids == image_ids
        arrays.append(npz["preds"])
    return np.stack(arrays, axis=-1), image_ids


def compute_relstd_stats(preds: np.ndarray) -> dict:
    """Return per-image and aggregate relstd statistics for a [n_img, n_dim, n_seed] tensor."""
    mean_pi = preds.mean(axis=2)                                # [n_img, n_dim]
    std_pi = preds.std(axis=2, ddof=0)                          # [n_img, n_dim]
    dim_scale = mean_pi.std(axis=0, ddof=0)                     # [n_dim]
    relstd_pi = std_pi / np.maximum(dim_scale[None, :], 1e-9)   # [n_img, n_dim]
    relstd_per_image = np.median(relstd_pi, axis=1)             # median over dims [n_img]

    # aggregate-scale relstd: per-dim test-set-mean across seeds, then relstd
    test_mean_per_seed = preds.mean(axis=0)                     # [n_dim, n_seed]
    agg_std = test_mean_per_seed.std(axis=1, ddof=0)            # [n_dim]
    agg_relstd = agg_std / np.maximum(dim_scale, 1e-9)          # [n_dim]

    return {
        "median_relstd_per_image": float(np.median(relstd_per_image)),
        "p95_relstd_per_image": float(np.percentile(relstd_per_image, 95)),
        "median_relstd_aggregate": float(np.median(agg_relstd)),
        "mean_relstd_aggregate": float(np.mean(agg_relstd)),
        "scale_gap_ratio": float(np.median(relstd_per_image) / max(np.median(agg_relstd), 1e-9)),
        "dim_scale": dim_scale.tolist(),
        "agg_relstd_per_dim": agg_relstd.tolist(),
    }


def main() -> None:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    # ---- 0. Load geo for populated-place-level rollup ----
    geo = pd.read_parquet(DATA / "image_geo.parquet")[["image_id", "city_proxy", "cc", "continent"]]
    geo_by_id = geo.set_index("image_id")

    # ---- 1. Per-split per-arch headline relstd ----
    per_split_stats: dict[str, dict] = {}
    per_image_records: list[dict] = []

    # Canonical: 40 seeds for ResNet, 80 for ViT (use first 8 to make apples-to-apples,
    # plus the full pool for reference)
    canonical_n_seeds = {"resnet50": 40, "vit_b16": 80}
    print("\n=== Q1/Q2: per-split per-arch relstd ===")
    for arch in ARCHS:
        # canonical (all seeds)
        full_preds, full_ids = load_canonical_preds(arch, canonical_n_seeds[arch])
        full_stats = compute_relstd_stats(full_preds)
        full_stats["arch"] = arch
        full_stats["split_label"] = "canonical_full"
        full_stats["n_seeds"] = canonical_n_seeds[arch]
        full_stats["n_images"] = full_preds.shape[0]
        per_split_stats[f"{arch}_canonical_full"] = full_stats

        # canonical, only first 8 seeds (apples-to-apples comparison vs alt splits)
        sub_preds = full_preds[:, :, :N_SEEDS_PER_SPLIT]
        sub_stats = compute_relstd_stats(sub_preds)
        sub_stats["arch"] = arch
        sub_stats["split_label"] = "canonical_8seeds"
        sub_stats["n_seeds"] = N_SEEDS_PER_SPLIT
        sub_stats["n_images"] = sub_preds.shape[0]
        per_split_stats[f"{arch}_canonical_8seeds"] = sub_stats

        # alt splits (8 seeds each)
        for k in range(3):
            preds, ids = load_multisplit_preds(k, arch)
            st = compute_relstd_stats(preds)
            st["arch"] = arch
            st["split_label"] = f"alt{k}"
            st["n_seeds"] = N_SEEDS_PER_SPLIT
            st["n_images"] = preds.shape[0]
            per_split_stats[f"{arch}_alt{k}"] = st

            # save per-image relstd long for the appendix table
            mean_pi = preds.mean(axis=2)
            std_pi = preds.std(axis=2, ddof=0)
            dim_scale = mean_pi.std(axis=0, ddof=0)
            relstd_pi = std_pi / np.maximum(dim_scale[None, :], 1e-9)
            relstd_per_image = np.median(relstd_pi, axis=1)
            for img_id, r in zip(ids, relstd_per_image):
                per_image_records.append({
                    "image_id": img_id,
                    "arch": arch,
                    "split_label": f"alt{k}",
                    "relstd_per_image": float(r),
                })

        # print headline numbers
        for label in ["canonical_full", "canonical_8seeds", "alt0", "alt1", "alt2"]:
            st = per_split_stats[f"{arch}_{label}"]
            print(f"  {arch:8s} {label:18s}  "
                  f"per-img median={st['median_relstd_per_image']:.3f}  "
                  f"agg median={st['median_relstd_aggregate']:.3f}  "
                  f"ratio={st['scale_gap_ratio']:.1f}x  "
                  f"n_seeds={st['n_seeds']}")

    # ---- 2. Ranking reversal check ----
    print("\n=== Q2: ResNet vs ViT ranking across splits ===")
    ranking_records = []
    for label in ["canonical_8seeds", "alt0", "alt1", "alt2"]:
        rn_agg = per_split_stats[f"resnet50_{label}"]["median_relstd_aggregate"]
        vt_agg = per_split_stats[f"vit_b16_{label}"]["median_relstd_aggregate"]
        rn_img = per_split_stats[f"resnet50_{label}"]["median_relstd_per_image"]
        vt_img = per_split_stats[f"vit_b16_{label}"]["median_relstd_per_image"]
        agg_ranking = "ViT<ResNet" if vt_agg < rn_agg else "ViT>ResNet"
        img_ranking = "ViT<ResNet" if vt_img < rn_img else "ViT>ResNet"
        reversal = (vt_agg < rn_agg) and (vt_img > rn_img)
        ranking_records.append({
            "split_label": label,
            "rn_agg": rn_agg, "vt_agg": vt_agg, "agg_ranking": agg_ranking,
            "rn_img": rn_img, "vt_img": vt_img, "img_ranking": img_ranking,
            "ranking_reversal_holds": reversal,
        })
        print(f"  {label:18s}  agg: {agg_ranking}  ({rn_agg:.3f} vs {vt_agg:.3f})   "
              f"per-img: {img_ranking}  ({rn_img:.3f} vs {vt_img:.3f})   "
              f"reversal={'YES' if reversal else 'no'}")

    # ---- 3. Split-induced variance vs seed-induced variance ----
    print("\n=== Q3: split-induced vs seed-induced variance ===")
    q3_records = []
    for arch in ARCHS:
        all_split_preds: list[np.ndarray] = []
        all_split_ids: list[list[str]] = []
        for k in range(3):
            preds, ids = load_multisplit_preds(k, arch)
            all_split_preds.append(preds)
            all_split_ids.append(ids)

        # intersection of test sets across the 3 alt splits
        common = set(all_split_ids[0]) & set(all_split_ids[1]) & set(all_split_ids[2])
        print(f"  {arch}: {len(common)} images shared across all 3 alt test sets")
        if len(common) < 30:
            print(f"  WARNING: too few shared images for {arch} Q3 analysis")
            continue

        # for each shared image, collect all 3*8=24 predictions
        # within-split std (pooled over 3 splits) vs between-split-mean std
        within_stds, between_stds = [], []
        for img_id in common:
            preds_per_split = []
            for k in range(3):
                idx = all_split_ids[k].index(img_id)
                preds_per_split.append(all_split_preds[k][idx])  # [n_dim, 8]
            preds_per_split = np.stack(preds_per_split, axis=-1)  # [n_dim, 8, 3]

            # within-split std: pool variance across 3 splits, each across 8 seeds
            split_vars = preds_per_split.var(axis=1, ddof=0)  # [n_dim, 3]
            within_var = split_vars.mean(axis=1)              # [n_dim]
            within_std = np.sqrt(within_var)                  # [n_dim]

            # between-split std: std of split means across 3 splits
            split_means = preds_per_split.mean(axis=1)        # [n_dim, 3]
            between_std = split_means.std(axis=1, ddof=0)     # [n_dim]

            within_stds.append(within_std)
            between_stds.append(between_std)

        within_stds = np.array(within_stds)   # [n_common, n_dim]
        between_stds = np.array(between_stds) # [n_common, n_dim]

        # normalize by per-dim scale (use the canonical full-pool scale for stability)
        full_preds, _ = load_canonical_preds(arch, canonical_n_seeds[arch])
        mean_pi = full_preds.mean(axis=2)
        dim_scale = mean_pi.std(axis=0, ddof=0)               # [n_dim]
        within_rel = within_stds / np.maximum(dim_scale[None, :], 1e-9)
        between_rel = between_stds / np.maximum(dim_scale[None, :], 1e-9)

        # collapse over dims by median
        within_per_image = np.median(within_rel, axis=1)
        between_per_image = np.median(between_rel, axis=1)

        rec = {
            "arch": arch,
            "n_shared_images": int(len(common)),
            "within_split_relstd_median": float(np.median(within_per_image)),
            "within_split_relstd_p95": float(np.percentile(within_per_image, 95)),
            "between_split_relstd_median": float(np.median(between_per_image)),
            "between_split_relstd_p95": float(np.percentile(between_per_image, 95)),
            "ratio_between_over_within_median": float(
                np.median(between_per_image) / max(np.median(within_per_image), 1e-9)
            ),
        }
        q3_records.append(rec)
        print(f"  {arch}: within-split median={rec['within_split_relstd_median']:.3f}  "
              f"between-split median={rec['between_split_relstd_median']:.3f}  "
              f"ratio={rec['ratio_between_over_within_median']:.2f}")

    # ---- 4. Save outputs ----
    out = {
        "per_split_stats": per_split_stats,
        "ranking_records": ranking_records,
        "q3_split_vs_seed_variance": q3_records,
        "notes": {
            "Q1": "Compare scale_gap_ratio across split_labels — should remain >5x on all splits.",
            "Q2": "ranking_reversal_holds should be True for all alt splits if finding is robust.",
            "Q3": "ratio_between_over_within < 1 means split-induced variance is smaller than "
                  "seed-induced variance on shared test images.",
        },
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUT_JSON}")

    pd.DataFrame(per_image_records).to_parquet(OUT_PARQ, index=False)
    print(f"wrote {OUT_PARQ}")

    # ---- 5. Appendix figure ----
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    arch_color = {"resnet50": "#4477AA", "vit_b16": "#EE6677"}
    arch_label = {"resnet50": "ResNet-50", "vit_b16": "ViT-B/16"}

    # (a) scale-gap by split: bar chart of per-image vs aggregate median relstd
    splits_to_plot = ["canonical_8seeds", "alt0", "alt1", "alt2"]
    x = np.arange(len(splits_to_plot))
    w = 0.2
    ax = axes[0]
    for i, arch in enumerate(ARCHS):
        per_img = [per_split_stats[f"{arch}_{s}"]["median_relstd_per_image"] for s in splits_to_plot]
        agg = [per_split_stats[f"{arch}_{s}"]["median_relstd_aggregate"] for s in splits_to_plot]
        ax.bar(x + (2*i - 1) * w, per_img, w,
               label=f"{arch_label[arch]} per-image",
               color=arch_color[arch], alpha=0.95)
        ax.bar(x + (2*i + 1) * w, agg, w,
               label=f"{arch_label[arch]} aggregate",
               color=arch_color[arch], alpha=0.35, hatch="///",
               edgecolor=arch_color[arch])
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("canonical_8seeds", "canonical") for s in splits_to_plot])
    ax.set_ylabel(r"median $\sigma_{\mathrm{rel}}$")
    ax.set_title("(a) Scale gap across alternative splits", loc="left",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=8.5, ncol=2, loc="upper right", framealpha=0.95)
    ax.axhline(0.10, color="grey", lw=0.6, ls=":")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # (b) Q3 boxplot: within vs between split relstd
    ax = axes[1]
    if q3_records:
        data = []
        labels = []
        for rec in q3_records:
            arch = rec["arch"]
            # reload the per-image arrays for the box
            all_split_preds = [load_multisplit_preds(k, arch)[0] for k in range(3)]
            all_split_ids = [load_multisplit_preds(k, arch)[1] for k in range(3)]
            common = sorted(set(all_split_ids[0]) & set(all_split_ids[1]) & set(all_split_ids[2]))
            within, between = [], []
            full_preds, _ = load_canonical_preds(arch, canonical_n_seeds[arch])
            mean_pi = full_preds.mean(axis=2)
            dim_scale = mean_pi.std(axis=0, ddof=0)
            for img_id in common:
                pp = [all_split_preds[k][all_split_ids[k].index(img_id)] for k in range(3)]
                pp = np.stack(pp, axis=-1)  # [n_dim, 8, 3]
                within_std = np.sqrt(pp.var(axis=1, ddof=0).mean(axis=1))
                between_std = pp.mean(axis=1).std(axis=1, ddof=0)
                within.append(np.median(within_std / np.maximum(dim_scale, 1e-9)))
                between.append(np.median(between_std / np.maximum(dim_scale, 1e-9)))
            data.append(within); labels.append(f"{arch_label[arch]}\nwithin-split")
            data.append(between); labels.append(f"{arch_label[arch]}\nbetween-split")
        bp = ax.boxplot(data, tick_labels=labels, showfliers=False, patch_artist=True,
                        widths=0.55, medianprops=dict(color="black", lw=1.4))
        # color boxes by arch
        for box_idx, patch in enumerate(bp["boxes"]):
            arch = "resnet50" if box_idx < 2 else "vit_b16"
            shade = 0.95 if box_idx % 2 == 0 else 0.4
            patch.set_facecolor(arch_color[arch])
            patch.set_alpha(shade)
        ax.set_ylabel(r"per-image $\sigma_{\mathrm{rel}}$ (shared images)")
        ax.set_title("(b) Split-induced vs seed-induced variance", loc="left",
                     fontsize=12, fontweight="bold")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    plt.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_FIG, dpi=200, bbox_inches="tight")
    plt.savefig(OUT_FIG.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"wrote {OUT_FIG}")


if __name__ == "__main__":
    main()
