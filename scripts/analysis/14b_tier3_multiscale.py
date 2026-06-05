"""Multi-scale fixed-effect robustness analysis on log(rel_std).

For each architecture, we ask whether each spatial scale's incremental R^2
survives the addition of per-image confound controls.

  M0   : dim_FE only
  M4   : dim_FE + 3 image complexity proxies + log(n_votes_dim)
         + label_dispersion + log(train_density_city)

For each scale in {continent, country, city}, we then fit:
  M0 + scale_FE      (naïve scale R^2 lift)
  M4 + scale_FE      (controlled scale R^2 lift — does it survive M4?)

Reported metric per (arch, scale):
  ΔR²_naive(scale)      = R²(M0 + scale_FE) - R²(M0)
  ΔR²_controlled(scale) = R²(M4 + scale_FE) - R²(M4)
  survival              = ΔR²_controlled / ΔR²_naive

A survival ratio near 1.0 means the spatial scale's contribution is
independent of per-image confounds (geographic structure is real). A ratio
well below 1.0 means most of the apparent geographic share is explained
away by the per-image controls.

Outputs:
  data_processed/tier3_multiscale_results.json
"""
from __future__ import annotations
import sys
import json
import importlib.util
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from polygeo.paths import DATA, DIMENSIONS

# load assemble_table() from sibling script (digit-prefixed filename → importlib)
_spec = importlib.util.spec_from_file_location(
    "tier3_base", Path(__file__).resolve().parent / "14_tier3_regression.py"
)
_t3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_t3)
assemble_table = _t3.assemble_table

OUT_JSON = DATA / "tier3_multiscale_results.json"

SCALES = ["continent", "cc", "city_proxy"]
SCALE_LABEL = {"continent": "continent", "cc": "country", "city_proxy": "city"}

M4_COVARS = [
    "complexity_shannon", "complexity_dino_norm", "complexity_seg_entropy",
    "log_n_votes_dim", "label_dispersion", "log_train_density_city",
]


def build_dense_X(df: pd.DataFrame, cont_cols: list[str]) -> tuple[np.ndarray, list[str]]:
    """Dense dim-FE one-hot + (optional) z-scored continuous covariates."""
    parts, names = [], []

    # dim FE: 5 dummies (drop first)
    dim_dummies = pd.get_dummies(df["dimension"], prefix="dim", drop_first=True)
    parts.append(dim_dummies.to_numpy(dtype=np.float64))
    names.extend(dim_dummies.columns.tolist())

    # continuous covariates: z-score
    if cont_cols:
        scaler = StandardScaler()
        x_cont = scaler.fit_transform(df[cont_cols].to_numpy(dtype=np.float64))
        parts.append(x_cont)
        names.extend(cont_cols)

    X = np.concatenate(parts, axis=1) if len(parts) > 1 else parts[0]
    return X, names


def build_scale_dummies(df: pd.DataFrame, scale_col: str) -> sparse.csr_matrix:
    """Sparse one-hot dummies for scale FE, drop one to avoid collinearity."""
    cats = pd.Categorical(df[scale_col])
    codes = cats.codes
    k = len(cats.categories)
    rows = np.arange(len(df))[codes >= 0]
    cols = codes[codes >= 0]
    # drop the first category to avoid the dummy variable trap (intercept handled by sklearn)
    keep = cols > 0
    rows = rows[keep]
    cols = cols[keep] - 1
    data = np.ones(len(rows), dtype=np.float64)
    return sparse.csr_matrix((data, (rows, cols)), shape=(len(df), max(0, k - 1)))


def r2(X, y) -> float:
    """Fit OLS (sparse or dense X), return in-sample R²."""
    model = LinearRegression(fit_intercept=True).fit(X, y)
    return float(model.score(X, y))


def run_arch(df_arch: pd.DataFrame) -> dict:
    y = df_arch["log_rel_std"].to_numpy(dtype=np.float64)

    # M0: dim FE only
    X_m0, _ = build_dense_X(df_arch, [])
    r2_m0 = r2(X_m0, y)

    # M4: dim FE + 6 per-image continuous controls
    X_m4, _ = build_dense_X(df_arch, M4_COVARS)
    r2_m4 = r2(X_m4, y)

    out = {
        "n_rows": int(len(df_arch)),
        "r2": {"M0": r2_m0, "M4": r2_m4},
        "scales": {},
    }

    # For each scale, fit M0+scale_FE and M4+scale_FE via sparse hstack
    for scale_col in SCALES:
        D = build_scale_dummies(df_arch, scale_col)
        if D.shape[1] == 0:
            print(f"  scale={scale_col} has only one level, skipping")
            continue
        # combine
        X_m0_sp = sparse.hstack([sparse.csr_matrix(X_m0), D], format="csr")
        X_m4_sp = sparse.hstack([sparse.csr_matrix(X_m4), D], format="csr")
        r2_m0_s = r2(X_m0_sp, y)
        r2_m4_s = r2(X_m4_sp, y)
        d_naive = r2_m0_s - r2_m0
        d_ctrl = r2_m4_s - r2_m4
        survival = (d_ctrl / d_naive) if d_naive > 1e-8 else float("nan")
        out["scales"][SCALE_LABEL[scale_col]] = {
            "n_levels": int(D.shape[1] + 1),
            "r2_M0_plus_scale": r2_m0_s,
            "r2_M4_plus_scale": r2_m4_s,
            "dR2_naive": d_naive,
            "dR2_controlled": d_ctrl,
            "survival_ratio": survival,
        }
        print(
            f"  scale={SCALE_LABEL[scale_col]:9s}  levels={D.shape[1]+1:5d}  "
            f"ΔR²_naive={d_naive*100:5.2f}%  ΔR²_ctrl={d_ctrl*100:5.2f}%  "
            f"survival={survival:.3f}"
        )
    return out


def main() -> None:
    df = assemble_table()

    results = {"per_arch": {}}
    for arch, sub in df.groupby("arch"):
        print(f"\n=== {arch}  n={len(sub):,} ===")
        print(f"  baseline R²: ...")
        res = run_arch(sub)
        results["per_arch"][arch] = res
        print(f"  R²(M0)={res['r2']['M0']:.4f}   R²(M4)={res['r2']['M4']:.4f}")

    OUT_JSON.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {OUT_JSON}")

    # final summary table
    print("\n=== Survival ratio summary (controlled ΔR² / naïve ΔR²) ===")
    print(f"{'arch':18s}  {'continent':>12s}  {'country':>12s}  {'city':>12s}")
    for arch, res in results["per_arch"].items():
        line = f"{arch:18s}"
        for scale in ("continent", "country", "city"):
            v = res["scales"].get(scale)
            if v is None or np.isnan(v["survival_ratio"]):
                line += f"  {'-':>12s}"
            else:
                line += f"  {v['survival_ratio']*100:>10.1f}%"
        print(line)


if __name__ == "__main__":
    main()
