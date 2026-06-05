"""Compute label dispersion proxies (Tier 0 baseline) since votes.csv has no worker_id.

Two proxies:

  P1. Pair-repeat dispersion
     For each (left_id, right_id, category) tuple, count how many times it appears.
     If ≥ 2 → measure disagreement among repeated votes.
     If pairs with ≥ 2 votes are < 5% of all pairs, P1 is judged unusable.

  P2. BTL violation rate (a.k.a. transitivity violation simplified)
     Using the fitted TrueSkill mu, predict the winner of each vote (mu_left vs mu_right).
     Violation = 1{majority observed winner ≠ TrueSkill prediction}, computed per (i,j,d).
     Per-image violation rate = mean over all pairs containing i (weighted by n_votes_for_pair).
     Per-city aggregation: weighted mean across images in that city.

Output:
  data_processed/label_dispersion_image.parquet  -- per (image, dimension)
  data_processed/label_dispersion_city.parquet   -- per (city_proxy, dimension)
  Also prints whether P1 is usable.
"""
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
VOTES = ROOT / "placePulse/_unpack/rawData/votes.csv"
QSCORES = ROOT / "data_processed/qscores.parquet"
GEO = ROOT / "data_processed/image_geo.parquet"
OUT_IMG = ROOT / "data_processed/label_dispersion_image.parquet"
OUT_CITY = ROOT / "data_processed/label_dispersion_city.parquet"


def check_pair_repeat(df: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    """Return (fraction_of_pairs_with_repeat, table of pair-level dispersion)."""
    # Order-insensitive pair key
    df = df.copy()
    df["pair_key"] = df.apply(
        lambda r: tuple(sorted([r["left_id"], r["right_id"]])) + (r["category"],),
        axis=1,
    )
    sizes = df.groupby("pair_key").size()
    frac_rep = float((sizes >= 2).mean())
    return frac_rep, sizes


def btl_violation(votes: pd.DataFrame, qscores: pd.DataFrame) -> pd.DataFrame:
    """Per-(image, dimension) BTL violation rate.

    For each vote with category=d:
      pred = sign(mu_left - mu_right)  (TrueSkill prediction)
      obs  = +1 if winner==left, -1 if winner==right, 0 if equal
      violation_per_vote = 1 if pred and obs disagree (excluding equals which we drop)
    Per image: mean over all votes containing this image in dimension d.
    """
    print("  joining votes with TrueSkill mu ...")
    qs = qscores[["image_id", "dimension", "mu"]]

    # Drop 'equal' votes for violation rate (they're informative for dispersion but
    # tabulated separately as 'equal_rate').
    df = votes[votes["winner"].isin({"left", "right"})].copy()
    print(f"  {len(df):,} non-equal votes")

    df = df.merge(
        qs.rename(columns={"image_id": "left_id", "mu": "mu_left", "dimension": "category"}),
        on=["left_id", "category"], how="left",
    )
    df = df.merge(
        qs.rename(columns={"image_id": "right_id", "mu": "mu_right", "dimension": "category"}),
        on=["right_id", "category"], how="left",
    )
    df["pred_left"] = (df["mu_left"] > df["mu_right"]).astype(int)
    df["obs_left"] = (df["winner"] == "left").astype(int)
    df["violation"] = (df["pred_left"] != df["obs_left"]).astype(int)

    # Aggregate to image-level per dimension
    left_v = df[["left_id", "category", "violation"]].rename(columns={"left_id": "image_id"})
    right_v = df[["right_id", "category", "violation"]].rename(columns={"right_id": "image_id"})
    long_v = pd.concat([left_v, right_v], ignore_index=True)
    img_disp = (
        long_v.groupby(["image_id", "category"])["violation"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "btl_violation_rate", "count": "n_votes_used", "category": "dimension"})
    )
    return img_disp


def equal_vote_rate(votes: pd.DataFrame) -> pd.DataFrame:
    """Per (image, dimension) fraction of votes that resulted in 'equal'."""
    df = votes.copy()
    df["is_equal"] = (df["winner"] == "equal").astype(int)
    left = df[["left_id", "category", "is_equal"]].rename(columns={"left_id": "image_id"})
    right = df[["right_id", "category", "is_equal"]].rename(columns={"right_id": "image_id"})
    long = pd.concat([left, right], ignore_index=True)
    return (
        long.groupby(["image_id", "category"])["is_equal"]
        .mean()
        .reset_index()
        .rename(columns={"is_equal": "equal_rate", "category": "dimension"})
    )


def main() -> None:
    print(f"reading votes ...")
    votes = pd.read_csv(VOTES)
    print(f"  {len(votes):,} votes")

    print(f"reading qscores ...")
    qs = pd.read_parquet(QSCORES)

    print(f"reading geo ...")
    geo = pd.read_parquet(GEO)[["image_id", "city_proxy", "cc", "global_south", "income_class", "continent"]]

    # ---- P1 check ----
    print("\n=== P1: pair-repeat dispersion ===")
    frac_rep, sizes = check_pair_repeat(votes)
    n_pairs = len(sizes)
    n_repeat = int((sizes >= 2).sum())
    print(f"  unique pairs: {n_pairs:,}")
    print(f"  pairs with ≥ 2 votes: {n_repeat:,}  ({frac_rep:.1%})")
    print(f"  P1 threshold (5%): {'USABLE' if frac_rep >= 0.05 else 'NOT USABLE'}")
    if frac_rep < 0.05:
        print("  → falling back to P2 (BTL violation) only")

    # ---- P2 BTL violation ----
    print("\n=== P2: BTL violation rate ===")
    img_disp = btl_violation(votes, qs)
    print(f"  per-(image, dim) rows: {len(img_disp):,}")
    print(img_disp[["btl_violation_rate", "n_votes_used"]].describe().round(3).to_string())

    # also equal_rate
    eq = equal_vote_rate(votes)
    img_disp = img_disp.merge(eq, on=["image_id", "dimension"], how="left")

    # ---- per-city aggregation ----
    print("\n=== city-level aggregation ===")
    img_with_geo = img_disp.merge(geo, on="image_id", how="left")
    city_disp = (
        img_with_geo.groupby(["city_proxy", "dimension"])
        .apply(
            lambda g: pd.Series(
                {
                    "btl_violation_rate_city": np.average(
                        g["btl_violation_rate"], weights=g["n_votes_used"].clip(lower=1)
                    ),
                    "equal_rate_city": np.average(g["equal_rate"], weights=g["n_votes_used"].clip(lower=1)),
                    "n_images": len(g),
                    "n_votes_used_total": int(g["n_votes_used"].sum()),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    print(f"  per-(city_proxy, dim) rows: {len(city_disp):,}")
    print(city_disp.describe().round(3).to_string())

    OUT_IMG.parent.mkdir(parents=True, exist_ok=True)
    img_disp.to_parquet(OUT_IMG, index=False)
    city_disp.to_parquet(OUT_CITY, index=False)
    print(f"\nwrote {OUT_IMG}  ({OUT_IMG.stat().st_size/1e6:.1f} MB)")
    print(f"wrote {OUT_CITY}  ({OUT_CITY.stat().st_size/1e6:.1f} MB)")

    # ---- top / bottom city sanity ----
    print("\n=== sanity: city-level BTL violation rate by continent (mean across dims) ===")
    cc_disp = (
        img_with_geo.groupby(["continent", "dimension"])
        .apply(
            lambda g: np.average(g["btl_violation_rate"], weights=g["n_votes_used"].clip(lower=1)),
            include_groups=False,
        )
        .unstack()
    )
    print(cc_disp.round(3).to_string())

    print("\n=== global_south split, mean BTL violation rate ===")
    gs_disp = (
        img_with_geo.groupby(["global_south", "dimension"])
        .apply(
            lambda g: np.average(g["btl_violation_rate"], weights=g["n_votes_used"].clip(lower=1)),
            include_groups=False,
        )
        .unstack()
    )
    print(gs_disp.round(3).to_string())


if __name__ == "__main__":
    main()
