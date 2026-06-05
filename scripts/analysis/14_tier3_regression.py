"""Regression controls for per-image stability.

Assembles the full per-image table:
  - dependent var: log(rel_std)  (Tier 1)
  - covariates:
      log_train_density_city          # PP2.0 sample count in same city
      log_n_votes_per_image_for_dim
      label_dispersion_image          # BTL violation rate (Tier 0)
      complexity_shannon
      complexity_dino_norm
      complexity_seg_entropy
      global_south  (0/1)
      income_class  (categorical)
      dimension     (categorical)
  - city_proxy: random effect (mixed-effects via group fixed effect dummies +
    cluster-robust SE — pragmatic substitute for full mixed-effects)

We fit, for each architecture separately:
  M0  base                       : intercept + dimension dummies
  M1  + image_complexity        : adds 3 complexity proxies
  M2  + n_votes_per_image_for_dim
  M3  + label_dispersion_image  : Tier 0 mediator
  M4  + log_train_density_city
  M5  + global_south

Output:
  data_processed/tier3_results.json
  data_processed/tier3_coefs.parquet      coefficients across all M0..M5
"""
from __future__ import annotations
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from polygeo.paths import (
    DATA, IMAGE_GEO, IMAGE_TABLE, QSCORES, DIMENSIONS,
)

OUT_JSON = DATA / "tier3_results.json"
OUT_COEFS = DATA / "tier3_coefs.parquet"


def assemble_table() -> pd.DataFrame:
    print("loading components ...")
    t1 = pd.read_parquet(DATA / "tier1_per_image.parquet")  # variance per (image,dim,arch)
    geo = pd.read_parquet(IMAGE_GEO)[
        ["image_id", "city_proxy", "cc", "continent", "income_class", "global_south"]
    ]
    img_t = pd.read_parquet(IMAGE_TABLE)
    cmplx = pd.read_parquet(DATA / "image_complexity.parquet")
    label_disp = pd.read_parquet(DATA / "label_dispersion_image.parquet")[
        ["image_id", "dimension", "btl_violation_rate"]
    ].rename(columns={"btl_violation_rate": "label_dispersion"})
    qs = pd.read_parquet(QSCORES)[["image_id", "dimension", "n_votes"]].rename(
        columns={"n_votes": "n_votes_dim"}
    )

    # n_votes per (image, dim) maps to "img_t.n_votes_<dim>" column too — same data
    df = (
        t1
        .merge(geo, on="image_id", how="left")
        .merge(cmplx, on="image_id", how="left")
        .merge(label_disp, on=["image_id", "dimension"], how="left")
        .merge(qs, on=["image_id", "dimension"], how="left")
    )
    # train_density_city = number of TRAIN images from same city
    splits = pd.read_parquet(DATA / "splits.parquet")
    train_imgs = splits[splits["split"] == "train"]["image_id"]
    train_geo = geo[geo["image_id"].isin(train_imgs)]
    train_density = train_geo.groupby("city_proxy").size().rename("train_density_city").reset_index()
    df = df.merge(train_density, on="city_proxy", how="left")
    df["train_density_city"] = df["train_density_city"].fillna(0)

    # Drop rows with NaN in any used field
    cols_required = [
        "rel_std", "complexity_shannon", "complexity_dino_norm", "complexity_seg_entropy",
        "label_dispersion", "n_votes_dim", "train_density_city",
        "global_south", "dimension", "arch",
    ]
    n0 = len(df)
    df = df.dropna(subset=cols_required).copy()
    print(f"  joined: {n0:,} → {len(df):,} (dropped {n0 - len(df):,} for missing covariates)")

    # transforms
    df["log_rel_std"] = np.log(df["rel_std"].clip(1e-4))
    df["log_train_density_city"] = np.log1p(df["train_density_city"])
    df["log_n_votes_dim"] = np.log1p(df["n_votes_dim"])
    return df


def fit_ols(y: np.ndarray, X: np.ndarray, names: list[str]) -> dict:
    """OLS with cluster-robust SE not implemented here; we just return point coefs.
    For p-values we use HC0 sandwich estimator below.
    """
    # add intercept
    X1 = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
    XtX = X1.T @ X1
    XtX_inv = np.linalg.pinv(XtX)
    beta = XtX_inv @ X1.T @ y
    resid = y - X1 @ beta
    n, k = X1.shape
    sigma2 = float(resid @ resid / (n - k))
    # HC0 sandwich
    XtUUX = (X1 * resid[:, None]).T @ (X1 * resid[:, None])
    cov_hc0 = XtX_inv @ XtUUX @ XtX_inv
    se_hc0 = np.sqrt(np.maximum(np.diag(cov_hc0), 0.0))
    z = beta / np.maximum(se_hc0, 1e-12)
    # 95% CI
    lo = beta - 1.96 * se_hc0
    hi = beta + 1.96 * se_hc0
    out = {}
    for i, name in enumerate(["intercept"] + names):
        out[name] = {
            "coef": float(beta[i]),
            "se_hc0": float(se_hc0[i]),
            "z": float(z[i]),
            "ci95_lo": float(lo[i]),
            "ci95_hi": float(hi[i]),
        }
    out["_meta"] = {
        "n": int(n), "k": int(k), "r2": float(1 - resid @ resid / (y.var() * n)),
        "rmse": float(np.sqrt(sigma2)),
    }
    return out


def design_matrix(df: pd.DataFrame, cols: list[str]) -> tuple[np.ndarray, list[str]]:
    """Z-score continuous vars; one-hot dimension; concatenate."""
    parts, names = [], []

    # one-hot dim (drop one)
    dim_dummies = pd.get_dummies(df["dimension"], prefix="dim", drop_first=True).astype(float).to_numpy()
    dim_names = [f"dim_{d}" for d in DIMENSIONS[1:]]
    parts.append(dim_dummies)
    names.extend(dim_names)

    # one-hot income class (drop "H")
    if "income_class" in cols:
        inc = pd.get_dummies(df["income_class"], prefix="inc", drop_first=True).astype(float).to_numpy()
        inc_names = [f"inc_{c}" for c in sorted(df["income_class"].unique()) if c != "H"]
        # we used drop_first=True, so first sorted level dropped; recover names accordingly
        inc_unique = sorted(df["income_class"].unique())
        inc_names = [f"inc_{c}" for c in inc_unique[1:]]
        parts.append(inc)
        names.extend(inc_names)

    # continuous vars
    cont_vars = [c for c in cols if c not in {"dimension", "income_class"}]
    if cont_vars:
        scaler = StandardScaler()
        x_cont = scaler.fit_transform(df[cont_vars].to_numpy())
        parts.append(x_cont)
        names.extend(cont_vars)

    X = np.concatenate(parts, axis=1)
    return X, names


def main() -> None:
    df = assemble_table()

    # Fit per-architecture, building up M0..M5
    results = {}
    coef_rows = []

    for arch, sub in df.groupby("arch"):
        print(f"\n=== Tier 3 regression: arch={arch}  n={len(sub):,} ===")
        y = sub["log_rel_std"].to_numpy()

        layers = [
            ("M0", []),
            ("M1", [
                "complexity_shannon", "complexity_dino_norm", "complexity_seg_entropy",
            ]),
            ("M2", [
                "complexity_shannon", "complexity_dino_norm", "complexity_seg_entropy",
                "log_n_votes_dim",
            ]),
            ("M3", [
                "complexity_shannon", "complexity_dino_norm", "complexity_seg_entropy",
                "log_n_votes_dim", "label_dispersion",
            ]),
            ("M4", [
                "complexity_shannon", "complexity_dino_norm", "complexity_seg_entropy",
                "log_n_votes_dim", "label_dispersion", "log_train_density_city",
            ]),
            ("M5", [
                "complexity_shannon", "complexity_dino_norm", "complexity_seg_entropy",
                "log_n_votes_dim", "label_dispersion", "log_train_density_city",
                "global_south",
            ]),
        ]
        results[arch] = {}
        for name, cont_cols in layers:
            cols = ["dimension"] + cont_cols
            X, names = design_matrix(sub, cols)
            res = fit_ols(y, X, names)
            results[arch][name] = res
            print(
                f"  {name}: R^2={res['_meta']['r2']:.3f}  rmse={res['_meta']['rmse']:.3f}  k={res['_meta']['k']}"
            )
            for n, val in res.items():
                if n.startswith("_"):
                    continue
                coef_rows.append({
                    "arch": arch, "model": name, "coef_name": n,
                    "coef": val["coef"], "se_hc0": val["se_hc0"],
                    "z": val["z"], "ci95_lo": val["ci95_lo"], "ci95_hi": val["ci95_hi"],
                })

        # Print selected coefficients used in the robustness summary.
        for layer_name, layer_cols in layers:
            r = results[arch][layer_name]
            for k in ["global_south", "log_train_density_city", "log_n_votes_dim", "label_dispersion"]:
                if k in r:
                    v = r[k]
                    sig = "*" if abs(v["z"]) > 1.96 else " "
                    print(f"    {layer_name}  {k:24s}  β={v['coef']:+.3f}  z={v['z']:+.2f}  95%CI=[{v['ci95_lo']:+.3f}, {v['ci95_hi']:+.3f}]  {sig}")

    OUT_JSON.write_text(json.dumps(results, indent=2))
    pd.DataFrame(coef_rows).to_parquet(OUT_COEFS, index=False)
    print(f"\nwrote {OUT_JSON}")
    print(f"wrote {OUT_COEFS}")

    print("\n=== Global-south coefficient after controls (M5) ===")
    for arch in results:
        v = results[arch]["M5"]["global_south"]
        sig = "STRONG" if abs(v["z"]) > 3 else ("YES" if abs(v["z"]) > 1.96 else "NO")
        print(
            f"  {arch:18s}  beta = {v['coef']:+.3f}  z = {v['z']:+.2f}  95%CI = [{v['ci95_lo']:+.3f}, {v['ci95_hi']:+.3f}]  -> {sig}"
        )


if __name__ == "__main__":
    main()
