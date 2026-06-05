"""Multi-scale spatial decomposition of per-image rel_std.

Three analyses, all on per-image rel_std (Tier 1):

1. Nested R² across spatial scales
   Fit cumulative OLS with dim FE → + continent FE → + country FE → + city FE.
   Each step's R² gain quantifies the variance share that lives *at that scale
   (above what's already explained by coarser scales)*. Image-level residual
   variance = 1 − R²_full.

2. β_global_south robustness to continent control
   Re-fit Tier 3 M5 with continent fixed effects added, then with continent +
   country FE. If β_GS survives, the GS gap is *within-continent* — not just
   "Africa & South Asia are worse on average."

3. Per-continent and country watchlist
   Mean rel_std per continent; top-10 / bottom-10 countries (N ≥ 500 images).

Focus arch: ViT-B/16 (where geo signal is strongest).

Outputs:
  data_processed/multiscale_decomposition.json
"""
from __future__ import annotations
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from polygeo.paths import DATA, IMAGE_GEO, QSCORES, DIMENSIONS

OUT_JSON = DATA / "multiscale_decomposition.json"
ARCHS = ["vit_b16", "resnet50", "dinov2_frozen"]


def fit_ols_r2(y: np.ndarray, X: np.ndarray) -> tuple[float, float]:
    """Return (R², adjusted R²). X must include intercept column."""
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    n, k = X.shape
    rss = float(resid @ resid)
    tss = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - rss / tss
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / (n - k)
    return r2, adj_r2


def fit_ols_full(y: np.ndarray, X: np.ndarray, names: list[str]) -> dict:
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    n, k = X.shape
    XtUUX = (X * resid[:, None]).T @ (X * resid[:, None])
    cov_hc0 = XtX_inv @ XtUUX @ XtX_inv
    se = np.sqrt(np.maximum(np.diag(cov_hc0), 0.0))
    tss = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float(resid @ resid) / tss
    out = {"_meta": {"n": int(n), "k": int(k), "r2": r2}}
    for i, nm in enumerate(names):
        out[nm] = {
            "coef": float(beta[i]),
            "se_hc0": float(se[i]),
            "z": float(beta[i] / max(se[i], 1e-12)),
            "ci95_lo": float(beta[i] - 1.96 * se[i]),
            "ci95_hi": float(beta[i] + 1.96 * se[i]),
        }
    return out


def assemble_df(arch: str) -> pd.DataFrame:
    t1 = pd.read_parquet(DATA / "tier1_per_image.parquet")
    t1 = t1[t1["arch"] == arch].copy()

    geo = pd.read_parquet(IMAGE_GEO)[
        ["image_id", "city_proxy", "cc", "continent", "income_class", "global_south"]
    ]
    df = t1.merge(geo, on="image_id", how="inner")
    df = df.dropna(subset=["rel_std", "continent", "cc", "city_proxy"]).copy()
    df["log_rel_std"] = np.log(df["rel_std"].clip(1e-4))
    print(f"  [{arch}] rows: {len(df):,}  images: {df['image_id'].nunique():,}  "
          f"continents: {df['continent'].nunique()}  countries: {df['cc'].nunique()}  "
          f"cities: {df['city_proxy'].nunique()}")
    return df


def build_fe(df: pd.DataFrame, levels: list[str]) -> tuple[np.ndarray, list[str]]:
    """One-hot encode a list of categorical columns, drop first level of each."""
    parts = [np.ones((len(df), 1))]
    names = ["intercept"]
    for lvl in levels:
        d = pd.get_dummies(df[lvl], prefix=lvl, drop_first=True).astype(float).to_numpy()
        parts.append(d)
        names.extend([f"{lvl}_{i}" for i in range(d.shape[1])])
    return np.concatenate(parts, axis=1), names


def analysis_nested_r2(df: pd.DataFrame) -> dict:
    """Cumulative R² across nested spatial scales."""
    print("\n=== 1. Nested R² across spatial scales ===")
    y = df["log_rel_std"].to_numpy()
    var_total = float(np.var(y, ddof=0))

    # Within-image variance (residual after all fixed effects up to city)
    # — note tier 1 already collapsed seeds per (image,dim), so the "image"
    # level here is (image,dim); residual = within-(image,dim).
    levels_cumulative = [
        ("dim only", ["dimension"]),
        ("+ continent", ["dimension", "continent"]),
        ("+ country", ["dimension", "continent", "cc"]),
        ("+ city", ["dimension", "continent", "cc", "city_proxy"]),
    ]

    results = {"var_total_log_rel_std": var_total, "n_obs": int(len(df)), "models": []}
    prev_r2 = 0.0
    for label, levels in levels_cumulative:
        X, _ = build_fe(df, levels)
        r2, adj_r2 = fit_ols_r2(y, X)
        delta = r2 - prev_r2
        print(f"  {label:18s}  k={X.shape[1]:5d}  R²={r2:.4f}  adjR²={adj_r2:.4f}  ΔR²={delta:+.4f}")
        results["models"].append({
            "label": label, "levels": levels, "k": int(X.shape[1]),
            "r2": float(r2), "adj_r2": float(adj_r2), "delta_r2": float(delta),
        })
        prev_r2 = r2

    # Variance shares: ΔR² is the share of log(rel_std) variance newly
    # explained at this scale, *above* coarser scales.
    print("\n  Variance share at each scale (above coarser scales):")
    shares = {}
    for i, m in enumerate(results["models"]):
        if i == 0:
            shares["dimension"] = m["r2"]
        else:
            new_scale = m["label"].replace("+ ", "")
            shares[new_scale] = m["delta_r2"]
    shares["within_city_residual"] = 1.0 - results["models"][-1]["r2"]
    for k, v in shares.items():
        print(f"    {k:22s}  {v:+.4f}  ({v * 100:+.2f}%)")
    results["variance_shares"] = shares
    return results


def analysis_beta_gs_robustness(df: pd.DataFrame) -> dict:
    """β_GS with progressively coarser-to-finer geo controls."""
    print("\n=== 2. β_global_south robustness to continent / country control ===")
    # Bring in covariates so we can replicate M5 + add geo FE
    cmplx = pd.read_parquet(DATA / "image_complexity.parquet")
    label_disp = pd.read_parquet(DATA / "label_dispersion_image.parquet")[
        ["image_id", "dimension", "btl_violation_rate"]
    ].rename(columns={"btl_violation_rate": "label_dispersion"})
    qs = pd.read_parquet(QSCORES)[["image_id", "dimension", "n_votes"]].rename(
        columns={"n_votes": "n_votes_dim"}
    )
    splits = pd.read_parquet(DATA / "splits.parquet")
    train_imgs = splits[splits["split"] == "train"]["image_id"]
    geo_all = pd.read_parquet(IMAGE_GEO)
    train_density = (
        geo_all[geo_all["image_id"].isin(train_imgs)]
        .groupby("city_proxy").size().rename("train_density_city").reset_index()
    )
    sub = (
        df
        .merge(cmplx, on="image_id", how="left")
        .merge(label_disp, on=["image_id", "dimension"], how="left")
        .merge(qs, on=["image_id", "dimension"], how="left")
        .merge(train_density, on="city_proxy", how="left")
    )
    sub = sub.dropna(subset=[
        "rel_std", "complexity_shannon", "complexity_dino_norm", "complexity_seg_entropy",
        "label_dispersion", "n_votes_dim", "train_density_city", "global_south",
    ]).copy()
    sub["log_train_density_city"] = np.log1p(sub["train_density_city"])
    sub["log_n_votes_dim"] = np.log1p(sub["n_votes_dim"])
    print(f"  rows for M5 + geo FE: {len(sub):,}")

    cont_cols = [
        "complexity_shannon", "complexity_dino_norm", "complexity_seg_entropy",
        "log_n_votes_dim", "label_dispersion", "log_train_density_city",
        "global_south",
    ]
    cont_X = StandardScaler().fit_transform(sub[cont_cols].to_numpy())
    dim_X = pd.get_dummies(sub["dimension"], prefix="dim", drop_first=True).astype(float).to_numpy()

    specs = [
        ("M5 baseline", []),
        ("M5 + continent FE", ["continent"]),
        ("M5 + country FE", ["cc"]),
        ("M5 + city FE", ["city_proxy"]),
    ]

    y = np.log(sub["rel_std"].clip(1e-4)).to_numpy()
    out = {}
    for label, fe_levels in specs:
        parts = [np.ones((len(sub), 1)), dim_X, cont_X]
        names = ["intercept"] + [f"dim_{i}" for i in range(dim_X.shape[1])] + cont_cols
        for lvl in fe_levels:
            d = pd.get_dummies(sub[lvl], prefix=lvl, drop_first=True).astype(float).to_numpy()
            parts.append(d)
            names.extend([f"{lvl}_{i}" for i in range(d.shape[1])])
        X = np.concatenate(parts, axis=1)
        res = fit_ols_full(y, X, names)
        gs = res["global_south"]
        print(f"  {label:22s}  k={X.shape[1]:5d}  R²={res['_meta']['r2']:.4f}  "
              f"β_GS={gs['coef']:+.4f}  z={gs['z']:+5.2f}  CI=[{gs['ci95_lo']:+.4f},{gs['ci95_hi']:+.4f}]")
        out[label] = {
            "k": int(X.shape[1]), "n": int(len(sub)), "r2": res["_meta"]["r2"],
            "beta_gs": gs,
        }
    return out


def analysis_continent_country(df: pd.DataFrame) -> dict:
    """Per-continent means and country watchlist."""
    print("\n=== 3. Per-continent means + country watchlist ===")
    # Per-continent (pooled across dims)
    cont = df.groupby("continent")["rel_std"].agg(["mean", "median", "count"]).sort_values("mean", ascending=False)
    print("\n  Continent (pooled dims):")
    print(cont.round(4).to_string())

    # Country watchlist — filter to ≥ 500 images
    by_cc = df.groupby("cc").agg(
        n_obs=("rel_std", "size"),
        mean_rel_std=("rel_std", "mean"),
        median_rel_std=("rel_std", "median"),
        gs_flag=("global_south", "first"),
        continent=("continent", "first"),
    )
    big = by_cc[by_cc["n_obs"] >= 500].sort_values("mean_rel_std", ascending=False)
    top10 = big.head(10)
    bot10 = big.tail(10).sort_values("mean_rel_std")
    print(f"\n  Top-10 most unstable countries (N ≥ 500):")
    print(top10.round(4).to_string())
    print(f"\n  Bottom-10 most stable countries (N ≥ 500):")
    print(bot10.round(4).to_string())

    # Per-continent GS gap *within-continent* (does the GS effect live inside continents?)
    print("\n  Within-continent N vs S mean rel_std:")
    rows = []
    for c, g in df.groupby("continent"):
        n = g[g["global_south"] == 0]["rel_std"]
        s = g[g["global_south"] == 1]["rel_std"]
        if len(n) > 50 and len(s) > 50:
            print(f"    {c:18s}  N={n.mean():.4f} (n={len(n):,})  S={s.mean():.4f} (n={len(s):,})  "
                  f"ratio={s.mean()/n.mean():.3f}")
            rows.append({
                "continent": c,
                "mean_north": float(n.mean()), "n_north": int(len(n)),
                "mean_south": float(s.mean()), "n_south": int(len(s)),
                "ratio": float(s.mean() / n.mean()),
            })
        else:
            print(f"    {c:18s}  (skipped: too few in one side)")
    return {
        "continent_summary": cont.reset_index().to_dict("records"),
        "country_top10": top10.reset_index().to_dict("records"),
        "country_bot10": bot10.reset_index().to_dict("records"),
        "within_continent_gs": rows,
    }


def main() -> None:
    print("loading per-image rel_std (Tier 1) for all archs ...")
    per_arch = {}
    for arch in ARCHS:
        print(f"\n{'='*60}\n  ARCH = {arch}\n{'='*60}")
        df = assemble_df(arch)
        per_arch[arch] = {
            "nested_r2": analysis_nested_r2(df),
            "beta_gs_robustness": analysis_beta_gs_robustness(df),
            "geo_summary": analysis_continent_country(df),
        }

    # Cross-arch comparison table
    print(f"\n\n{'='*60}\n  Cross-arch comparison: variance shares\n{'='*60}")
    print(f"{'arch':18s} {'dim':>8s} {'cont':>8s} {'country':>8s} {'city':>8s} {'within':>8s}")
    for arch in ARCHS:
        s = per_arch[arch]["nested_r2"]["variance_shares"]
        print(f"{arch:18s} {s['dimension']:>8.3f} {s['continent']:>8.3f} {s['country']:>8.3f} "
              f"{s['city']:>8.3f} {s['within_city_residual']:>8.3f}")

    print(f"\n{'='*60}\n  Cross-arch: β_GS (M5 baseline → M5+continent FE)\n{'='*60}")
    print(f"{'arch':18s} {'M5 β':>10s} {'M5 z':>8s}  {'+cont β':>10s} {'+cont z':>10s}")
    for arch in ARCHS:
        r = per_arch[arch]["beta_gs_robustness"]
        m5 = r["M5 baseline"]["beta_gs"]
        cf = r["M5 + continent FE"]["beta_gs"]
        print(f"{arch:18s} {m5['coef']:>+10.4f} {m5['z']:>+8.2f}  {cf['coef']:>+10.4f} {cf['z']:>+10.2f}")

    OUT_JSON.write_text(json.dumps({"per_arch": per_arch, "archs": ARCHS}, indent=2, default=str))
    print(f"\nwrote {OUT_JSON}")


if __name__ == "__main__":
    main()
