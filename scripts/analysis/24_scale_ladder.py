"""Scale ladder of seed-induced reproducibility.

For each arch and each aggregation scale (image / city / country / continent /
aggregate), compute the seed-induced relative std at that scale. The same
*image-level* dim_scale is used as the normaliser across all scales, so the
ladder values are directly comparable: as we aggregate over more images, the
seed-induced std should shrink (roughly ∝ 1/√N if seed effects are independent
within each spatial unit; slower if correlated).

At each scale S:
    seed_std(u, d)   = std_s [ mean_{i in u} prediction(i, d, s) ]
    rel_std(u, d)    = seed_std(u, d) / dim_scale(d)
    headline(S)      = median over (u, d) of rel_std(u, d)

where dim_scale(d) = std over images of (mean over seeds prediction).

Inputs:
    predictions/exp1_<arch>_imagenet21k_seed*_test_preds.npz
    data_processed/image_geo.parquet
Output:
    data_processed/scale_ladder.json
"""
from __future__ import annotations
import sys
import json
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import pandas as pd

from polygeo.paths import DATA, PREDICTIONS, IMAGE_GEO

OUT_JSON = DATA / "scale_ladder.json"

ARCH_FILES = {
    "vit_b16": "exp1_vit_b16_imagenet21k_seed*_test_preds.npz",
    "resnet50": "exp1_resnet50_imagenet21k_seed*_test_preds.npz",
    "dinov2_frozen": "exp1_dinov2_frozen_imagenet21k_seed*_test_preds.npz",
}


def seed_num(p: Path) -> int:
    m = re.search(r"_seed(\d+)_", p.name)
    return int(m.group(1))


def load_stack(arch: str):
    files = sorted(PREDICTIONS.glob(ARCH_FILES[arch]), key=seed_num)
    arrs, image_ids, dims = [], None, None
    for p in files:
        z = np.load(p, allow_pickle=True)
        if image_ids is None:
            image_ids = list(z["image_ids"])
            dims = list(z["dimensions"])
        arrs.append(z["preds"].astype(np.float32))
    return np.stack(arrs, axis=-1), image_ids, dims  # [n_img, n_dim, n_seeds]


def aggregate_to_unit(
    preds: np.ndarray,
    unit_idx: np.ndarray,
    n_units: int,
) -> np.ndarray:
    """
    Given preds[n_img, n_dim, n_seeds] and unit_idx[n_img] of integers,
    return agg[n_units, n_dim, n_seeds] = mean of preds over images in each unit.
    """
    n_img, n_dim, n_seeds = preds.shape
    out = np.zeros((n_units, n_dim, n_seeds), dtype=np.float32)
    counts = np.bincount(unit_idx, minlength=n_units).astype(np.float32)
    # Vectorised: scatter-add per dim, per seed
    for d in range(n_dim):
        for s in range(n_seeds):
            np.add.at(out[:, d, s], unit_idx, preds[:, d, s])
    out /= np.maximum(counts[:, None, None], 1.0)
    return out, counts


def scale_rel_std(
    preds: np.ndarray,
    unit_idx: np.ndarray | None,
    n_units: int | None,
    dim_scale: np.ndarray,
    min_count: int = 1,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute rel_std per (unit, dim).
    Returns: median, full_flat_array (filtered by min_count), per_unit_rel, counts.
    """
    if unit_idx is None:
        agg = preds.mean(axis=0)  # [n_dim, n_seeds]
        seed_std = agg.std(axis=1, ddof=0)
        rel = seed_std / np.maximum(dim_scale, 1e-9)  # [n_dim]
        return float(np.median(rel)), rel.copy(), rel, np.array([preds.shape[0]])
    else:
        agg, counts = aggregate_to_unit(preds, unit_idx, n_units)
        seed_std = agg.std(axis=2, ddof=0)
        rel = seed_std / np.maximum(dim_scale[None, :], 1e-9)
        mask = counts >= min_count
        flat = rel[mask].reshape(-1)
        return float(np.median(flat)), flat, rel, counts


def per_image_rel_std(preds: np.ndarray, dim_scale: np.ndarray) -> np.ndarray:
    """[n_img, n_dim] image-level rel_std (same as Tier 1)."""
    seed_std = preds.std(axis=2, ddof=0)
    return seed_std / np.maximum(dim_scale[None, :], 1e-9)


def main() -> None:
    geo = pd.read_parquet(IMAGE_GEO)[
        ["image_id", "city_proxy", "cc", "continent"]
    ]

    all_results = {}
    # Per-arch raw distributions for boxplot rendering
    distributions: dict[str, dict[str, np.ndarray]] = {}

    for arch in ARCH_FILES:
        print(f"\n=== {arch} ===")
        preds, image_ids, dims = load_stack(arch)
        n_img, n_dim, n_seeds = preds.shape
        print(f"  preds shape: {preds.shape}")

        gdf = pd.DataFrame({"image_id": image_ids}).merge(geo, on="image_id", how="left")
        missing = gdf[["city_proxy", "cc", "continent"]].isna().any(axis=1)
        if missing.any():
            print(f"  dropping {missing.sum():,} images with missing geo")
        keep_mask = ~missing.to_numpy()
        preds = preds[keep_mask]
        gdf = gdf[keep_mask].reset_index(drop=True)

        mean_per_img = preds.mean(axis=2)
        dim_scale = mean_per_img.std(axis=0, ddof=0)

        img_rel = per_image_rel_std(preds, dim_scale)
        median_img = float(np.median(img_rel))

        city_codes, city_names = pd.factorize(gdf["city_proxy"])
        median_city, city_flat, _, city_counts = scale_rel_std(
            preds, city_codes, len(city_names), dim_scale, min_count=3
        )
        cc_codes, cc_names = pd.factorize(gdf["cc"])
        median_country, country_flat, _, cc_counts = scale_rel_std(
            preds, cc_codes, len(cc_names), dim_scale, min_count=10
        )
        cont_codes, cont_names = pd.factorize(gdf["continent"])
        median_continent, continent_flat, _, cont_counts = scale_rel_std(
            preds, cont_codes, len(cont_names), dim_scale, min_count=10
        )
        median_agg, agg_flat, _, _ = scale_rel_std(preds, None, None, dim_scale)

        # Per-dim breakdown at image scale (for the histogram panel)
        per_dim_image_rel_std = {dims[d]: img_rel[:, d].tolist() for d in range(n_dim)}

        # Percentile summaries for compact JSON
        def pctiles(a):
            a = np.asarray(a).reshape(-1)
            return {
                "p5": float(np.percentile(a, 5)),
                "p25": float(np.percentile(a, 25)),
                "p50": float(np.percentile(a, 50)),
                "p75": float(np.percentile(a, 75)),
                "p95": float(np.percentile(a, 95)),
                "n": int(a.size),
            }

        arch_out = {
            "n_images": int(preds.shape[0]),
            "n_seeds": int(n_seeds),
            "n_dims": n_dim,
            "ladder": {
                "image":      {"median_rel_std": float(median_img),       "n_units": int(preds.shape[0]), "pct": pctiles(img_rel)},
                "city":       {"median_rel_std": float(median_city),      "n_units": int((city_counts >= 3).sum()), "pct": pctiles(city_flat)},
                "country":    {"median_rel_std": float(median_country),   "n_units": int((cc_counts >= 10).sum()), "pct": pctiles(country_flat)},
                "continent":  {"median_rel_std": float(median_continent), "n_units": int((cont_counts >= 10).sum()), "pct": pctiles(continent_flat)},
                "aggregate":  {"median_rel_std": float(median_agg),       "n_units": 1, "pct": pctiles(agg_flat)},
            },
            "dim_scale": {dims[d]: float(dim_scale[d]) for d in range(n_dim)},
        }
        all_results[arch] = arch_out

        # Save raw distributions for boxplot
        distributions[arch] = {
            "image": img_rel.reshape(-1),
            "city": city_flat,
            "country": country_flat,
            "continent": continent_flat,
            "aggregate": agg_flat,
        }

        print(f"  ladder (median rel_std / dim_scale):")
        for scale_name in ["aggregate", "continent", "country", "city", "image"]:
            v = arch_out["ladder"][scale_name]
            print(f"    {scale_name:10s}  {v['median_rel_std']:.4f}   (n_units = {v['n_units']})")

    OUT_JSON.write_text(json.dumps(all_results, indent=2))
    print(f"\nwrote {OUT_JSON}")

    # Save raw distributions as npz for boxplot rendering
    OUT_NPZ = DATA / "scale_ladder_distributions.npz"
    npz_payload = {}
    for arch, scales in distributions.items():
        for scale_name, arr in scales.items():
            npz_payload[f"{arch}__{scale_name}"] = arr.astype(np.float32)
    np.savez_compressed(OUT_NPZ, **npz_payload)
    print(f"wrote {OUT_NPZ}  ({OUT_NPZ.stat().st_size/1e6:.2f} MB)")

    # Cross-arch summary
    print(f"\n{'='*60}\nCross-arch scale ladder (median rel_std)\n{'='*60}")
    print(f"{'arch':18s} {'agg':>8s} {'cont':>8s} {'country':>8s} {'city':>8s} {'image':>8s}")
    for arch in ARCH_FILES:
        L = all_results[arch]["ladder"]
        print(f"{arch:18s} {L['aggregate']['median_rel_std']:>8.4f} "
              f"{L['continent']['median_rel_std']:>8.4f} {L['country']['median_rel_std']:>8.4f} "
              f"{L['city']['median_rel_std']:>8.4f} {L['image']['median_rel_std']:>8.4f}")


if __name__ == "__main__":
    main()
