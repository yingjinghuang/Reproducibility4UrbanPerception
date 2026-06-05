"""Fit TrueSkill ratings per (image, dimension) from raw pairwise votes.

For each of the 6 PP2.0 dimensions independently:
  - process votes in arrival order
  - 'left' / 'right' winner -> rate1.update_from_match
  - 'equal' -> draw

Output: data_processed/qscores.parquet
  columns: image_id, dimension, mu, sigma, conservative_score, n_votes
"""
from pathlib import Path
import time
import pandas as pd
import trueskill as ts

ROOT = Path(__file__).resolve().parents[2]
VOTES = ROOT / "placePulse/_unpack/rawData/votes.csv"
OUT = ROOT / "data_processed/qscores.parquet"

CATEGORIES = ["safety", "lively", "beautiful", "wealthy", "depressing", "boring"]


def fit_one_dimension(df_dim: pd.DataFrame, env: ts.TrueSkill) -> dict[str, ts.Rating]:
    """Run TrueSkill over all votes of one dimension. Returns {image_id: Rating}."""
    ratings: dict[str, ts.Rating] = {}
    for left_id, right_id, winner in zip(df_dim["left_id"], df_dim["right_id"], df_dim["winner"]):
        r_l = ratings.get(left_id) or env.create_rating()
        r_r = ratings.get(right_id) or env.create_rating()
        if winner == "left":
            r_l_new, r_r_new = env.rate_1vs1(r_l, r_r, drawn=False)
        elif winner == "right":
            r_r_new, r_l_new = env.rate_1vs1(r_r, r_l, drawn=False)
        else:  # 'equal' -> draw
            r_l_new, r_r_new = env.rate_1vs1(r_l, r_r, drawn=True)
        ratings[left_id] = r_l_new
        ratings[right_id] = r_r_new
    return ratings


def main() -> None:
    print(f"reading {VOTES} ...")
    df = pd.read_csv(VOTES)
    print(f"  {len(df):,} votes")

    env = ts.TrueSkill(mu=25.0, sigma=25.0 / 3.0, draw_probability=0.10)
    print(f"  TrueSkill env: mu0={env.mu}, sigma0={env.sigma:.4f}, beta={env.beta:.4f}, draw_prob={env.draw_probability}")

    rows = []
    for cat in CATEGORIES:
        df_dim = df[df["category"] == cat].reset_index(drop=True)
        print(f"  fitting '{cat}': {len(df_dim):,} votes ...", flush=True)
        t0 = time.time()
        ratings = fit_one_dimension(df_dim, env)
        elapsed = time.time() - t0
        print(f"    done in {elapsed:.1f}s, {len(ratings):,} unique images")

        # vote counts per image in this dimension
        vc_left = df_dim["left_id"].value_counts()
        vc_right = df_dim["right_id"].value_counts()
        vc = vc_left.add(vc_right, fill_value=0).astype(int)

        for img, r in ratings.items():
            rows.append(
                {
                    "image_id": img,
                    "dimension": cat,
                    "mu": float(r.mu),
                    "sigma": float(r.sigma),
                    "conservative_score": float(r.mu - 3.0 * r.sigma),
                    "n_votes": int(vc.get(img, 0)),
                }
            )

    out = pd.DataFrame(rows)
    print(f"\nfinal table: {len(out):,} (image,dimension) rows")
    print(out.groupby("dimension")[["mu", "sigma", "n_votes"]].describe().round(3))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    print(f"\nwrote {OUT}  ({OUT.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
