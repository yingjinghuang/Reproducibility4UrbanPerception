"""Build per-image table from raw votes.csv.

Each image appears as either left_id or right_id across many votes. We aggregate:
  - canonical lat/lon (mean across appearances; should be near-identical per image)
  - n_votes per dimension
  - n_votes total

Output: data_processed/image_table.parquet
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
VOTES = ROOT / "placePulse/_unpack/rawData/votes.csv"
OUT = ROOT / "data_processed/image_table.parquet"

CATEGORIES = ["safety", "lively", "beautiful", "wealthy", "depressing", "boring"]


def main() -> None:
    print(f"reading {VOTES} ...")
    df = pd.read_csv(VOTES)
    print(f"  {len(df):,} votes, {df['category'].nunique()} categories")

    left = df[["left_id", "left_lat", "left_long", "category"]].rename(
        columns={"left_id": "image_id", "left_lat": "lat", "left_long": "lon"}
    )
    right = df[["right_id", "right_lat", "right_long", "category"]].rename(
        columns={"right_id": "image_id", "right_lat": "lat", "right_long": "lon"}
    )
    long = pd.concat([left, right], ignore_index=True)

    coords = long.groupby("image_id")[["lat", "lon"]].mean()
    n_votes_total = long.groupby("image_id").size().rename("n_votes_total")
    n_votes_by_cat = (
        long.groupby(["image_id", "category"]).size().unstack(fill_value=0)
    )
    n_votes_by_cat.columns = [f"n_votes_{c}" for c in n_votes_by_cat.columns]
    for c in CATEGORIES:
        col = f"n_votes_{c}"
        if col not in n_votes_by_cat.columns:
            n_votes_by_cat[col] = 0
    n_votes_by_cat = n_votes_by_cat[[f"n_votes_{c}" for c in CATEGORIES]]

    table = coords.join(n_votes_total).join(n_votes_by_cat).reset_index()

    print(f"  unique images: {len(table):,}")
    print(f"  vote count summary:")
    print(table[["n_votes_total"] + [f"n_votes_{c}" for c in CATEGORIES]].describe().round(1).to_string())

    # sanity: lat/lon dispersion per image (should be ~0, since same image_id always has same coords)
    coord_check = (
        long.groupby("image_id")[["lat", "lon"]]
        .agg(lambda s: s.max() - s.min())
        .max()
    )
    print(f"  per-image lat range max: {coord_check['lat']:.6f}")
    print(f"  per-image lon range max: {coord_check['lon']:.6f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(OUT, index=False)
    print(f"wrote {OUT}  ({OUT.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
