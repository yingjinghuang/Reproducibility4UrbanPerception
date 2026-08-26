"""Exp. 3a -- corrected ensemble-size analysis (no model training).

The primary analysis treats the available independently seeded runs as an
empirical distribution of training outcomes.  For an ensemble size N, it draws
B bootstrap resamples of N runs *with replacement* and measures the population
standard deviation (ddof=0) of the B ensemble means.  This avoids the finite-
population compression caused by sampling subsets without replacement.

The previous without-replacement calculation is retained only as a diagnostic.
Both methods use the same sigma_rel denominator as the main Tier-1 analysis:
the population SD across test images of the full-pool seed-mean prediction, per
perceptual dimension.

No checkpoints are loaded and no networks are trained.  Inputs are the saved
per-seed test-prediction NPZ files from the main analysis.

Primary outputs
---------------
data_processed/exp3a_ensemble_curve.parquet
    Corrected per-image/per-dimension values used by Figure 4.
data_processed/exp3a_bootstrap_results.csv
    Complete corrected architecture x ensemble-size summary.
data_processed/exp3a_summary.json
    JSON summary retained for compatibility with existing analysis tooling.
data_processed/exp3a_corrected_summary.md
    Human-readable results and diagnostics.

Diagnostic outputs
------------------
data_processed/exp3a_old_vs_new.csv
data_processed/exp3a_sqrt_n_validation.csv
data_processed/exp3a_fpc_validation.csv
data_processed/exp3a_mc_stability.csv
data_processed/exp3a_mc_headline_thresholds.csv
data_processed/exp3a_mc_variation.csv
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import pandas as pd

from polygeo.paths import DATA, DIMENSIONS, IMAGE_GEO, PREDICTIONS


OUT_CURVE = DATA / "exp3a_ensemble_curve.parquet"
OUT_RESULTS = DATA / "exp3a_bootstrap_results.csv"
OUT_JSON = DATA / "exp3a_summary.json"
OUT_MARKDOWN = DATA / "exp3a_corrected_summary.md"
OUT_OLD_NEW = DATA / "exp3a_old_vs_new.csv"
OUT_SQRT = DATA / "exp3a_sqrt_n_validation.csv"
OUT_FPC = DATA / "exp3a_fpc_validation.csv"
OUT_MC = DATA / "exp3a_mc_stability.csv"
OUT_MC_HEADLINES = DATA / "exp3a_mc_headline_thresholds.csv"
OUT_MC_VARIATION = DATA / "exp3a_mc_variation.csv"

DEFAULT_B = 5_000
DEFAULT_BOOTSTRAP_SEED = 20_260_817
DEFAULT_STABILITY_SEEDS = [20_260_817, 20_260_818, 20_260_819, 20_260_820, 20_260_821]
CORE_MIN_IMAGES = 30
EXPECTED_CORE_PLACES = 70
THRESHOLDS = (0.15, 0.10, 0.05)

PRIMARY_METHOD = "bootstrap_with_replacement"
OLD_METHOD = "monte_carlo_without_replacement_diagnostic"


@dataclass(frozen=True)
class Architecture:
    key: str
    label: str
    pattern: str
    pool_size: int
    ensemble_sizes: tuple[int, ...]
    rng_code: int


ARCHITECTURES = (
    Architecture(
        "vit_b16",
        "ViT-B/16",
        "exp1_vit_b16_imagenet21k_seed{s:03d}_test_preds.npz",
        80,
        (1, 2, 4, 8, 16, 32),
        1,
    ),
    Architecture(
        "resnet50",
        "ResNet-50",
        "exp1_resnet50_imagenet21k_seed{s:03d}_test_preds.npz",
        40,
        (1, 2, 4, 8, 16),
        2,
    ),
    Architecture(
        "dinov2_frozen",
        "DINOv2 (frozen)",
        "exp1_dinov2_frozen_imagenet21k_seed{s:03d}_test_preds.npz",
        40,
        (1, 2, 4, 8, 16),
        3,
    ),
)
ARCH_ORDER = {a.key: i for i, a in enumerate(ARCHITECTURES)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bootstrap-replicates",
        "-B",
        type=int,
        default=DEFAULT_B,
        help=f"number of Monte Carlo/bootstrap replicates (default: {DEFAULT_B})",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=DEFAULT_BOOTSTRAP_SEED,
        help=f"fixed RNG seed for final outputs (default: {DEFAULT_BOOTSTRAP_SEED})",
    )
    parser.add_argument(
        "--stability-seeds",
        type=int,
        nargs="*",
        default=DEFAULT_STABILITY_SEEDS,
        help="RNG seeds used for the Monte Carlo stability check",
    )
    parser.add_argument(
        "--skip-long-output",
        action="store_true",
        help="skip the Figure-5 long-form parquet (useful for quick diagnostics)",
    )
    args = parser.parse_args()
    if args.bootstrap_replicates < 2:
        parser.error("--bootstrap-replicates must be at least 2")
    seeds = list(dict.fromkeys([args.bootstrap_seed, *args.stability_seeds]))
    args.stability_seeds = seeds
    return args


def load_arch_predictions(spec: Architecture) -> tuple[np.ndarray, np.ndarray]:
    """Load [image, dimension, seed] predictions and validate all metadata."""
    arrays: list[np.ndarray] = []
    image_ids: np.ndarray | None = None
    dimensions: np.ndarray | None = None
    for seed in range(spec.pool_size):
        path = PREDICTIONS / spec.pattern.format(s=seed)
        if not path.exists():
            raise FileNotFoundError(f"missing prediction file: {path}")
        with np.load(path, allow_pickle=True) as saved:
            ids = np.asarray(saved["image_ids"])
            dims = np.asarray(saved["dimensions"])
            values = np.asarray(saved["preds"], dtype=np.float64)
        if image_ids is None:
            image_ids = ids.copy()
            dimensions = dims.copy()
        else:
            if not np.array_equal(ids, image_ids):
                raise ValueError(f"image order differs for {spec.key}, seed {seed}")
            if not np.array_equal(dims, dimensions):
                raise ValueError(f"dimension order differs for {spec.key}, seed {seed}")
        if values.shape != (len(ids), len(DIMENSIONS)):
            raise ValueError(f"unexpected prediction shape in {path}: {values.shape}")
        if not np.isfinite(values).all():
            raise ValueError(f"non-finite predictions in {path}")
        arrays.append(values)

    assert image_ids is not None and dimensions is not None
    if dimensions.tolist() != list(DIMENSIONS):
        raise ValueError(
            f"{spec.key} dimension order {dimensions.tolist()} != canonical {list(DIMENSIONS)}"
        )
    predictions = np.stack(arrays, axis=-1)
    expected = (len(image_ids), len(DIMENSIONS), spec.pool_size)
    if predictions.shape != expected:
        raise ValueError(f"{spec.key} tensor {predictions.shape} != expected {expected}")
    return predictions, image_ids


def load_geography(image_ids: np.ndarray) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Load the canonical test geography and reproduce the existing 70-place set."""
    columns = ["image_id", "global_south", "city_proxy", "continent"]
    geo = pd.read_parquet(IMAGE_GEO)[columns]
    if not geo["image_id"].is_unique:
        raise ValueError("image_geo.parquet has duplicate image_id rows")
    test_geo = pd.DataFrame({"image_id": image_ids}).merge(
        geo, on="image_id", how="left", validate="one_to_one"
    )
    if test_geo[columns[1:]].isna().any().any():
        missing = test_geo[test_geo["city_proxy"].isna()]["image_id"].tolist()[:5]
        raise ValueError(f"missing geography for test images, examples: {missing}")

    place_counts = test_geo.groupby("city_proxy")["image_id"].nunique()
    core_places = tuple(sorted(place_counts[place_counts >= CORE_MIN_IMAGES].index.tolist()))
    if len(core_places) != EXPECTED_CORE_PLACES:
        raise ValueError(
            f"core-place definition produced {len(core_places)} places, "
            f"expected {EXPECTED_CORE_PLACES}"
        )
    return test_geo, core_places


def sample_weights(
    pool_size: int,
    ensemble_size: int,
    replicates: int,
    rng_seed: int,
    arch_code: int,
    replace: bool,
) -> np.ndarray:
    """Return a [B, S] matrix of equal-weight ensemble membership counts."""
    if ensemble_size > pool_size and not replace:
        raise ValueError("without-replacement ensemble cannot exceed the seed pool")

    # Depend on (reported seed, architecture, N), not loop order.  Reusing this
    # seed sequence across methods also makes N=1 exactly comparable.
    rng = np.random.default_rng(np.random.SeedSequence([rng_seed, arch_code, ensemble_size]))
    if ensemble_size == 1:
        indices = rng.integers(pool_size, size=(replicates, 1))
    elif replace:
        indices = rng.choice(pool_size, size=(replicates, ensemble_size), replace=True)
    else:
        # rng.choice(..., replace=False) is called independently for each
        # replicate, matching the legacy procedure.
        indices = np.empty((replicates, ensemble_size), dtype=np.int16)
        for b in range(replicates):
            indices[b] = rng.choice(pool_size, size=ensemble_size, replace=False)

    weights = np.zeros((replicates, pool_size), dtype=np.float64)
    rows = np.repeat(np.arange(replicates), ensemble_size)
    np.add.at(weights, (rows, indices.reshape(-1)), 1.0 / ensemble_size)
    return weights


def resampled_rel_std(
    predictions: np.ndarray,
    dim_scale: np.ndarray,
    ensemble_size: int,
    replicates: int,
    rng_seed: int,
    arch_code: int,
    replace: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute cross-resample SD and sigma_rel without materializing B tensors.

    If W is the B x S matrix of resample weights and x is one pair's S-vector
    of seed predictions, the B ensemble predictions are W x.  Their population
    variance is x' Cov_B(W) x.  Computing the B-sample weight covariance first
    is algebraically identical to constructing all B ensemble-prediction arrays,
    but uses substantially less memory.
    """
    weights = sample_weights(
        predictions.shape[2],
        ensemble_size,
        replicates,
        rng_seed,
        arch_code,
        replace,
    )
    centered_weights = weights - weights.mean(axis=0, keepdims=True)
    weight_cov = centered_weights.T @ centered_weights / replicates

    flat = predictions.reshape(-1, predictions.shape[2])
    variance = np.sum((flat @ weight_cov) * flat, axis=1)
    ensemble_sd = np.sqrt(np.maximum(variance, 0.0)).reshape(predictions.shape[:2])
    rel_std = ensemble_sd / np.maximum(dim_scale[None, :], 1e-12)
    return ensemble_sd, rel_std


def summarize_rel_std(
    rel_std: np.ndarray,
    spec: Architecture,
    ensemble_size: int,
    replicates: int,
    rng_seed: int,
    sampling_method: str,
    city_per_image: np.ndarray,
    core_places: tuple[str, ...],
) -> dict[str, object]:
    """Apply the existing Figure-5 image and local-place summaries unchanged."""
    pair_places = np.repeat(city_per_image, rel_std.shape[1])
    place_frame = pd.DataFrame(
        {"city_proxy": pair_places, "rel_std": rel_std.reshape(-1)}
    )
    place_profile = (
        place_frame[place_frame["city_proxy"].isin(core_places)]
        .groupby("city_proxy", sort=True)["rel_std"]
        .median()
        .reindex(core_places)
    )
    if len(place_profile) != EXPECTED_CORE_PLACES or place_profile.isna().any():
        raise ValueError("local instability profiles do not cover all 70 core places")

    profile = place_profile.to_numpy(dtype=np.float64)
    below_015 = profile < 0.15
    below_010 = profile < 0.10
    below_005 = profile < 0.05
    return {
        "architecture": spec.key,
        "architecture_label": spec.label,
        "seed_pool_size": spec.pool_size,
        "ensemble_size": ensemble_size,
        "B": replicates,
        "bootstrap_seed": rng_seed,
        "sampling_method": sampling_method,
        # This preserves the existing analysis: the median is over all retained
        # image-dimension pairs, not a label-filtered accuracy subset.
        "per_image_median_sigma_rel": float(np.median(rel_std)),
        "fraction_core_places_below_0.15": float(below_015.mean()),
        "fraction_core_places_below_0.10": float(below_010.mean()),
        "fraction_core_places_below_0.05": float(below_005.mean()),
        "number_core_places_below_0.10": int(below_010.sum()),
        "number_core_places": int(len(profile)),
        "maximum_place_m_u": float(profile.max()),
        "median_place_m_u": float(np.median(profile)),
        "minimum_place_m_u": float(profile.min()),
    }


def make_long_frame(
    ensemble_sd: np.ndarray,
    rel_std: np.ndarray,
    image_ids: np.ndarray,
    spec: Architecture,
    ensemble_size: int,
    replicates: int,
    rng_seed: int,
) -> pd.DataFrame:
    chunks = []
    for d, dimension in enumerate(DIMENSIONS):
        chunks.append(
            pd.DataFrame(
                {
                    "arch": spec.key,
                    "image_id": image_ids,
                    "dimension": dimension,
                    "ensemble_size": ensemble_size,
                    "ens_std": ensemble_sd[:, d],
                    "rel_std": rel_std[:, d],
                    "seed_pool_size": spec.pool_size,
                    "B": replicates,
                    "bootstrap_seed": rng_seed,
                    "sampling_method": PRIMARY_METHOD,
                }
            )
        )
    return pd.concat(chunks, ignore_index=True)


def threshold_crossing(values: pd.DataFrame, column: str, threshold: float) -> str:
    reached = values.loc[values[column] < threshold, "ensemble_size"]
    return str(int(reached.min())) if len(reached) else "not reached"


def all_places_crossing(values: pd.DataFrame) -> str:
    reached = values.loc[
        values["number_core_places_below_0.10"] == EXPECTED_CORE_PLACES,
        "ensemble_size",
    ]
    return str(int(reached.min())) if len(reached) else "not reached"


def headline_rows(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for spec in ARCHITECTURES:
        sub = results[results["architecture"] == spec.key].sort_values("ensemble_size")
        rows.append(
            {
                "architecture": spec.key,
                "architecture_label": spec.label,
                "per_image_below_0.10_N": threshold_crossing(
                    sub, "per_image_median_sigma_rel", 0.10
                ),
                "per_image_below_0.05_N": threshold_crossing(
                    sub, "per_image_median_sigma_rel", 0.05
                ),
                "all_70_places_m_u_below_0.10_N": all_places_crossing(sub),
            }
        )
    return pd.DataFrame(rows)


def sort_results(frame: pd.DataFrame) -> pd.DataFrame:
    order = frame["architecture"].map(ARCH_ORDER)
    return (
        frame.assign(_arch_order=order)
        .sort_values(["_arch_order", "bootstrap_seed", "ensemble_size"])
        .drop(columns="_arch_order")
        .reset_index(drop=True)
    )


def markdown_table(frame: pd.DataFrame, columns: list[str], digits: int = 5) -> str:
    """Small dependency-free Markdown table formatter."""
    shown = frame[columns].copy()
    for column in columns:
        if pd.api.types.is_float_dtype(shown[column]):
            shown[column] = shown[column].map(lambda value: f"{value:.{digits}f}")
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in shown.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def write_json_summary(primary: pd.DataFrame, replicates: int, rng_seed: int) -> None:
    payload: dict[str, object] = {
        "n_bootstrap": replicates,
        "bootstrap_seed": rng_seed,
        "sampling_method": PRIMARY_METHOD,
        "core_place_definition": f">= {CORE_MIN_IMAGES} test images",
        "n_core_places": EXPECTED_CORE_PLACES,
        "per_arch": {},
    }
    for spec in ARCHITECTURES:
        sub = primary[primary["architecture"] == spec.key].sort_values("ensemble_size")
        curve = []
        for row in sub.to_dict(orient="records"):
            curve.append(
                {
                    "ensemble_size": int(row["ensemble_size"]),
                    "median_rel_std": float(row["per_image_median_sigma_rel"]),
                    "fraction_core_places_below_0.15": float(
                        row["fraction_core_places_below_0.15"]
                    ),
                    "fraction_core_places_below_0.10": float(
                        row["fraction_core_places_below_0.10"]
                    ),
                    "fraction_core_places_below_0.05": float(
                        row["fraction_core_places_below_0.05"]
                    ),
                    "number_core_places_below_0.10": int(
                        row["number_core_places_below_0.10"]
                    ),
                    "maximum_place_m_u": float(row["maximum_place_m_u"]),
                    "median_place_m_u": float(row["median_place_m_u"]),
                    "minimum_place_m_u": float(row["minimum_place_m_u"]),
                }
            )
        payload["per_arch"][spec.key] = {"pool": spec.pool_size, "curve": curve}
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    replicates = args.bootstrap_replicates
    primary_seed = args.bootstrap_seed
    stability_seeds: list[int] = args.stability_seeds

    print("Corrected Exp. 3a ensemble analysis")
    print(f"  B={replicates:,}")
    print(f"  final bootstrap seed={primary_seed}")
    print(f"  stability seeds={stability_seeds}")
    print(f"  primary sampling={PRIMARY_METHOD}")

    DATA.mkdir(parents=True, exist_ok=True)
    predictions_by_arch: dict[str, np.ndarray] = {}
    ids_by_arch: dict[str, np.ndarray] = {}
    reference_ids: np.ndarray | None = None
    for spec in ARCHITECTURES:
        predictions, image_ids = load_arch_predictions(spec)
        predictions_by_arch[spec.key] = predictions
        ids_by_arch[spec.key] = image_ids
        if reference_ids is None:
            reference_ids = image_ids
        elif not np.array_equal(reference_ids, image_ids):
            raise ValueError(f"test image order differs across architectures at {spec.key}")
        print(f"  loaded {spec.key}: {predictions.shape}")
    assert reference_ids is not None

    test_geo, core_places = load_geography(reference_ids)
    city_per_image = test_geo["city_proxy"].to_numpy()
    print(
        f"  canonical core places={len(core_places)} "
        f"(>= {CORE_MIN_IMAGES} test images; unchanged definition)"
    )

    primary_rows: list[dict[str, object]] = []
    old_rows: list[dict[str, object]] = []
    stability_rows: list[dict[str, object]] = []
    long_frames: list[pd.DataFrame] = []

    for spec in ARCHITECTURES:
        predictions = predictions_by_arch[spec.key]
        image_ids = ids_by_arch[spec.key]
        full_pool_mean = predictions.mean(axis=2)
        dim_scale = full_pool_mean.std(axis=0, ddof=0)
        print(f"\n=== {spec.label}: S={spec.pool_size}, tensor={predictions.shape} ===")
        print("  denominator SD by dimension:", np.round(dim_scale, 6).tolist())

        for rng_seed in stability_seeds:
            for ensemble_size in spec.ensemble_sizes:
                ensemble_sd, rel_std = resampled_rel_std(
                    predictions,
                    dim_scale,
                    ensemble_size,
                    replicates,
                    rng_seed,
                    spec.rng_code,
                    replace=True,
                )
                row = summarize_rel_std(
                    rel_std,
                    spec,
                    ensemble_size,
                    replicates,
                    rng_seed,
                    PRIMARY_METHOD,
                    city_per_image,
                    core_places,
                )
                stability_rows.append(row)
                if rng_seed == primary_seed:
                    primary_rows.append(row)
                    if not args.skip_long_output:
                        long_frames.append(
                            make_long_frame(
                                ensemble_sd,
                                rel_std,
                                image_ids,
                                spec,
                                ensemble_size,
                                replicates,
                                rng_seed,
                            )
                        )
                    print(
                        f"  corrected N={ensemble_size:2d}: "
                        f"median={row['per_image_median_sigma_rel']:.6f}, "
                        f"core<10%={row['number_core_places_below_0.10']:2d}/70, "
                        f"max m_u={row['maximum_place_m_u']:.6f}"
                    )

        # Recompute the legacy sampling design with the same B and final seed.
        for ensemble_size in spec.ensemble_sizes:
            _, old_rel_std = resampled_rel_std(
                predictions,
                dim_scale,
                ensemble_size,
                replicates,
                primary_seed,
                spec.rng_code,
                replace=False,
            )
            old_rows.append(
                summarize_rel_std(
                    old_rel_std,
                    spec,
                    ensemble_size,
                    replicates,
                    primary_seed,
                    OLD_METHOD,
                    city_per_image,
                    core_places,
                )
            )

    primary = sort_results(pd.DataFrame(primary_rows))
    old = sort_results(pd.DataFrame(old_rows))
    stability = sort_results(pd.DataFrame(stability_rows))

    primary.to_csv(OUT_RESULTS, index=False)
    stability.to_csv(OUT_MC, index=False)

    if not args.skip_long_output:
        long = pd.concat(long_frames, ignore_index=True)
        long = long.merge(
            test_geo, on="image_id", how="left", validate="many_to_one"
        )
        long["is_core_place"] = long["city_proxy"].isin(core_places)
        expected_rows = sum(len(spec.ensemble_sizes) for spec in ARCHITECTURES)
        expected_rows *= len(reference_ids) * len(DIMENSIONS)
        if len(long) != expected_rows:
            raise ValueError(f"long output has {len(long)} rows; expected {expected_rows}")
        long.to_parquet(OUT_CURVE, index=False)
        print(f"\nwrote {OUT_CURVE} ({len(long):,} rows)")

    # 1/sqrt(N) validation for corrected bootstrap results.
    sqrt_rows = []
    for spec in ARCHITECTURES:
        sub = primary[primary["architecture"] == spec.key]
        baseline = float(
            sub.loc[sub["ensemble_size"] == 1, "per_image_median_sigma_rel"].iloc[0]
        )
        for row in sub.to_dict(orient="records"):
            observed = float(row["per_image_median_sigma_rel"]) / baseline
            theoretical = 1.0 / math.sqrt(int(row["ensemble_size"]))
            sqrt_rows.append(
                {
                    "architecture": spec.key,
                    "architecture_label": spec.label,
                    "B": replicates,
                    "bootstrap_seed": primary_seed,
                    "N": int(row["ensemble_size"]),
                    "median_sigma_rel": float(row["per_image_median_sigma_rel"]),
                    "observed_ratio": observed,
                    "one_over_sqrt_N": theoretical,
                    "relative_difference_percent": 100.0
                    * (observed / theoretical - 1.0),
                }
            )
    sqrt_validation = sort_results(
        pd.DataFrame(sqrt_rows).rename(columns={"N": "ensemble_size"})
    ).rename(columns={"ensemble_size": "N"})
    sqrt_validation.to_csv(OUT_SQRT, index=False)

    # Old/new and finite-population-correction validation.
    old_new = old.merge(
        primary,
        on=[
            "architecture",
            "architecture_label",
            "seed_pool_size",
            "ensemble_size",
            "B",
            "bootstrap_seed",
            "number_core_places",
        ],
        suffixes=("_old", "_new"),
        validate="one_to_one",
    )
    old_new_rows = []
    for row in old_new.to_dict(orient="records"):
        pool = int(row["seed_pool_size"])
        ensemble_size = int(row["ensemble_size"])
        old_median = float(row["per_image_median_sigma_rel_old"])
        new_median = float(row["per_image_median_sigma_rel_new"])
        empirical = old_median / new_median
        theoretical = math.sqrt((pool - ensemble_size) / (pool - 1))
        old_new_rows.append(
            {
                "architecture": row["architecture"],
                "architecture_label": row["architecture_label"],
                "S": pool,
                "N": ensemble_size,
                "B": int(row["B"]),
                "bootstrap_seed": int(row["bootstrap_seed"]),
                "old_median_sigma_rel": old_median,
                "new_median_sigma_rel": new_median,
                "empirical_old_new_ratio": empirical,
                "theoretical_FPC_ratio": theoretical,
                "ratio_difference": empirical - theoretical,
                "relative_difference_percent": 100.0
                * (empirical / theoretical - 1.0),
                "old_number_core_places_below_0.10": int(
                    row["number_core_places_below_0.10_old"]
                ),
                "new_number_core_places_below_0.10": int(
                    row["number_core_places_below_0.10_new"]
                ),
                "old_maximum_place_m_u": float(row["maximum_place_m_u_old"]),
                "new_maximum_place_m_u": float(row["maximum_place_m_u_new"]),
            }
        )
    old_new_summary = pd.DataFrame(old_new_rows)
    old_new_summary["_arch_order"] = old_new_summary["architecture"].map(ARCH_ORDER)
    old_new_summary = (
        old_new_summary.sort_values(["_arch_order", "N"])
        .drop(columns="_arch_order")
        .reset_index(drop=True)
    )
    old_new_summary.to_csv(OUT_OLD_NEW, index=False)
    old_new_summary[
        [
            "architecture",
            "architecture_label",
            "S",
            "N",
            "old_median_sigma_rel",
            "new_median_sigma_rel",
            "empirical_old_new_ratio",
            "theoretical_FPC_ratio",
            "ratio_difference",
            "relative_difference_percent",
        ]
    ].to_csv(OUT_FPC, index=False)

    # Monte Carlo threshold invariance and continuous-metric ranges.
    headline_frames = []
    for rng_seed in stability_seeds:
        seed_results = stability[stability["bootstrap_seed"] == rng_seed]
        seed_headlines = headline_rows(seed_results)
        seed_headlines.insert(0, "bootstrap_seed", rng_seed)
        seed_headlines.insert(1, "B", replicates)
        headline_frames.append(seed_headlines)
    mc_headlines = pd.concat(headline_frames, ignore_index=True)
    mc_headlines.to_csv(OUT_MC_HEADLINES, index=False)

    variation = (
        stability.groupby(
            ["architecture", "architecture_label", "seed_pool_size", "ensemble_size"],
            sort=False,
        )
        .agg(
            per_image_median_min=("per_image_median_sigma_rel", "min"),
            per_image_median_max=("per_image_median_sigma_rel", "max"),
            core_fraction_below_010_min=("fraction_core_places_below_0.10", "min"),
            core_fraction_below_010_max=("fraction_core_places_below_0.10", "max"),
            maximum_place_m_u_min=("maximum_place_m_u", "min"),
            maximum_place_m_u_max=("maximum_place_m_u", "max"),
        )
        .reset_index()
    )
    variation["per_image_median_range"] = (
        variation["per_image_median_max"] - variation["per_image_median_min"]
    )
    variation["core_fraction_below_010_range"] = (
        variation["core_fraction_below_010_max"]
        - variation["core_fraction_below_010_min"]
    )
    variation["maximum_place_m_u_range"] = (
        variation["maximum_place_m_u_max"] - variation["maximum_place_m_u_min"]
    )
    variation["_arch_order"] = variation["architecture"].map(ARCH_ORDER)
    variation = variation.sort_values(["_arch_order", "ensemble_size"]).drop(
        columns="_arch_order"
    )
    variation.to_csv(OUT_MC_VARIATION, index=False)

    final_headlines = headline_rows(primary)
    headline_invariant = (
        mc_headlines.groupby("architecture")[[
            "per_image_below_0.10_N",
            "per_image_below_0.05_N",
            "all_70_places_m_u_below_0.10_N",
        ]]
        .nunique()
        .max(axis=1)
        .eq(1)
    )

    write_json_summary(primary, replicates, primary_seed)

    markdown_sections = [
        "# Corrected ensemble-size analysis",
        "",
        f"- Primary method: `{PRIMARY_METHOD}`",
        f"- Bootstrap replicates: B={replicates:,}",
        f"- Final RNG seed: {primary_seed}",
        f"- Stability RNG seeds: {', '.join(map(str, stability_seeds))}",
        f"- Core places: {EXPECTED_CORE_PLACES} places with at least {CORE_MIN_IMAGES} test images",
        "",
        "## Final headline thresholds",
        "",
        markdown_table(
            final_headlines,
            [
                "architecture_label",
                "per_image_below_0.10_N",
                "per_image_below_0.05_N",
                "all_70_places_m_u_below_0.10_N",
            ],
        ),
        "",
        "## Corrected primary results",
        "",
        markdown_table(
            primary,
            [
                "architecture_label",
                "ensemble_size",
                "per_image_median_sigma_rel",
                "number_core_places_below_0.10",
                "fraction_core_places_below_0.10",
                "maximum_place_m_u",
                "median_place_m_u",
                "minimum_place_m_u",
            ],
            digits=6,
        ),
        "",
        "## 1/sqrt(N) validation",
        "",
        markdown_table(
            sqrt_validation,
            [
                "architecture_label",
                "N",
                "median_sigma_rel",
                "observed_ratio",
                "one_over_sqrt_N",
                "relative_difference_percent",
            ],
            digits=6,
        ),
        "",
        "## Without-replacement finite-population diagnostic",
        "",
        markdown_table(
            old_new_summary,
            [
                "architecture_label",
                "S",
                "N",
                "old_median_sigma_rel",
                "new_median_sigma_rel",
                "empirical_old_new_ratio",
                "theoretical_FPC_ratio",
                "relative_difference_percent",
            ],
            digits=6,
        ),
        "",
        "## Monte Carlo stability",
        "",
        markdown_table(
            mc_headlines,
            [
                "bootstrap_seed",
                "architecture_label",
                "per_image_below_0.10_N",
                "per_image_below_0.05_N",
                "all_70_places_m_u_below_0.10_N",
            ],
        ),
        "",
        "Headline thresholds invariant for all architectures: "
        + ("yes" if bool(headline_invariant.all()) else "no"),
        "",
        "Maximum continuous-metric ranges across the five RNG seeds:",
        "",
        f"- per-image median sigma_rel: {variation['per_image_median_range'].max():.8f}",
        f"- fraction of core places below 10%: {variation['core_fraction_below_010_range'].max():.8f}",
        f"- maximum place m_u: {variation['maximum_place_m_u_range'].max():.8f}",
        "",
    ]
    OUT_MARKDOWN.write_text("\n".join(markdown_sections))

    print("\n=== final headline thresholds ===")
    print(final_headlines.to_string(index=False))
    print("\n=== validation maxima ===")
    print(
        "  max |bootstrap ratio / (1/sqrt(N)) - 1| = "
        f"{sqrt_validation['relative_difference_percent'].abs().max():.4f}%"
    )
    print(
        "  max |empirical FPC / theoretical FPC - 1| = "
        f"{old_new_summary['relative_difference_percent'].abs().max():.4f}%"
    )
    print(f"  headline thresholds invariant: {bool(headline_invariant.all())}")
    print(
        "  max MC range, per-image median sigma_rel = "
        f"{variation['per_image_median_range'].max():.8f}"
    )
    print(
        "  max MC range, core fraction below 10% = "
        f"{variation['core_fraction_below_010_range'].max():.8f}"
    )
    print(
        "  max MC range, maximum place m_u = "
        f"{variation['maximum_place_m_u_range'].max():.8f}"
    )
    print("\nwrote:")
    for path in (
        OUT_RESULTS,
        OUT_JSON,
        OUT_MARKDOWN,
        OUT_OLD_NEW,
        OUT_SQRT,
        OUT_FPC,
        OUT_MC,
        OUT_MC_HEADLINES,
        OUT_MC_VARIATION,
    ):
        print(f"  {path}")


if __name__ == "__main__":
    main()
