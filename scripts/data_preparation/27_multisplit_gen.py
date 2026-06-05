"""Generate K alternative train/val/test splits for the multi-split robustness audit.

Each alternative split reuses the same per-city-stratified, 80/10/10 logic as
scripts/05_make_splits.py, varying only the SPLIT_SEED so that the
random shuffle within each city changes. The canonical split
(data_processed/splits.parquet, SPLIT_SEED=20260503) is left untouched.

Outputs:
  data_processed/splits_alt0.parquet  (SPLIT_SEED = 20260601)
  data_processed/splits_alt1.parquet  (SPLIT_SEED = 20260602)
  data_processed/splits_alt2.parquet  (SPLIT_SEED = 20260603)
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
GEO = ROOT / "data_processed/image_geo.parquet"
QSCORES = ROOT / "data_processed/qscores.parquet"
OUT_DIR = ROOT / "data_processed"

ALT_SEEDS = [20260601, 20260602, 20260603]
TEST_FRAC = 0.10
VAL_FRAC = 0.10


def build_split(split_seed: int) -> pd.DataFrame:
    geo = pd.read_parquet(GEO)
    qs = pd.read_parquet(QSCORES)
    rated = qs["image_id"].unique()
    geo = geo[geo["image_id"].isin(rated)].copy()

    rng = np.random.RandomState(split_seed)
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

    return pd.DataFrame(out_rows, columns=["image_id", "split"])


def main() -> None:
    for k, sd in enumerate(ALT_SEEDS):
        df = build_split(sd)
        out_path = OUT_DIR / f"splits_alt{k}.parquet"
        df.to_parquet(out_path, index=False)
        counts = df["split"].value_counts().to_dict()
        print(f"split_alt{k} (seed={sd}) → {out_path.name}  counts={counts}")

    # Quick sanity: alt0 test set should differ substantially from canonical
    canonical = pd.read_parquet(OUT_DIR / "splits.parquet")
    alt0 = pd.read_parquet(OUT_DIR / "splits_alt0.parquet")
    can_test = set(canonical[canonical["split"] == "test"]["image_id"])
    alt_test = set(alt0[alt0["split"] == "test"]["image_id"])
    overlap = len(can_test & alt_test)
    print(f"\noverlap test∩test between canonical and alt0: "
          f"{overlap}/{len(can_test)} canonical test images = {100*overlap/len(can_test):.1f}%")
    print(f"  (a fully independent reshuffle should give ~10% overlap)")


if __name__ == "__main__":
    main()
