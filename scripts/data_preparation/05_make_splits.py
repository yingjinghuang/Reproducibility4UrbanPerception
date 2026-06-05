"""Build canonical train/val/test split.

Stratified by city_proxy (so all cities appear in test). Held identical across
all seed runs of the project — split itself is generated once with a fixed
seed and never re-shuffled.

Output:
  data_processed/splits.parquet   columns: image_id, split ∈ {train, val, test}

Strategy:
  - For each city_proxy, take ⌈max(1, 0.10·N)⌉ images for test, same for val,
    rest for train. Cities with N<3 → all images go to test (no train signal anyway).
  - Use a fixed numpy RandomState(seed=20260503) for image ordering within city.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
GEO = ROOT / "data_processed/image_geo.parquet"
QSCORES = ROOT / "data_processed/qscores.parquet"
OUT = ROOT / "data_processed/splits.parquet"

SPLIT_SEED = 20260503
TEST_FRAC = 0.10
VAL_FRAC = 0.10


def main() -> None:
    geo = pd.read_parquet(GEO)
    qs = pd.read_parquet(QSCORES)

    # Only keep images that have at least one TrueSkill rating in any dimension
    rated = qs["image_id"].unique()
    geo = geo[geo["image_id"].isin(rated)].copy()
    print(f"images with ≥1 TrueSkill rating: {len(geo):,}")

    rng = np.random.RandomState(SPLIT_SEED)
    out_rows = []
    for city, grp in geo.groupby("city_proxy", sort=False):
        ids = grp["image_id"].to_numpy()
        rng.shuffle(ids)
        n = len(ids)
        if n < 3:
            for i in ids:
                out_rows.append((i, "test"))
            continue
        n_test = max(1, int(np.ceil(TEST_FRAC * n)))
        n_val = max(1, int(np.ceil(VAL_FRAC * n)))
        test_ids = ids[:n_test]
        val_ids = ids[n_test : n_test + n_val]
        train_ids = ids[n_test + n_val :]
        for i in test_ids:
            out_rows.append((i, "test"))
        for i in val_ids:
            out_rows.append((i, "val"))
        for i in train_ids:
            out_rows.append((i, "train"))

    splits = pd.DataFrame(out_rows, columns=["image_id", "split"])
    print("split counts:")
    print(splits["split"].value_counts().to_string())
    print()

    # Sanity: continent / global_south distribution per split
    merged = splits.merge(geo, on="image_id", how="left")
    print("per-continent counts by split:")
    print(merged.groupby(["continent", "split"]).size().unstack(fill_value=0).to_string())
    print()
    print("global_south split:")
    print(merged.groupby(["global_south", "split"]).size().unstack(fill_value=0).to_string())

    OUT.parent.mkdir(parents=True, exist_ok=True)
    splits.to_parquet(OUT, index=False)
    print(f"\nwrote {OUT}  ({OUT.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
