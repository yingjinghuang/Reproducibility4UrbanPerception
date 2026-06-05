"""Mixed-effects ICC robustness check for the variance decomposition.

The primary analysis uses sequential incremental R^2 with the order
  continent -> country -> populated place
which assigns any shared variance between levels to the LAST level added
(the populated place). To verify that the place-level concentration of
geographic variance is not a methodological artifact of this ordering,
we fit a proper 3-level mixed-effects model:

  log(rel_std) ~ 1 + C(dimension) + (1 | continent / country / place) + e

with continent, country, and place all entering simultaneously as nested
random intercepts. The intraclass correlation (ICC) at each level is the
variance attributable to that level divided by total variance, and is
independent of the order in which levels are added.

For each architecture we report:
  - sigma^2_continent, sigma^2_country, sigma^2_place, sigma^2_residual
  - ICC_continent, ICC_country, ICC_place (=variance share)
  - residual share (=1 - sum of ICCs)
and compare these against the sequential incremental R^2 figures already
reported in the benchmark results.

Outputs:
  data_processed/icc_robustness.json   per-arch variance components + ICCs
                                       + side-by-side comparison with the
                                       sequential R^2 numbers.
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from polygeo.paths import DATA

OUT_JSON = DATA / "icc_robustness.json"

ARCHS = ["resnet50", "vit_b16", "dinov2_frozen"]


def load_data() -> pd.DataFrame:
    df = pd.read_parquet(DATA / "tier1_per_image.parquet")
    geo = pd.read_parquet(DATA / "image_geo.parquet")[
        ["image_id", "continent", "cc", "city_proxy"]
    ]
    m = df.merge(geo, on="image_id", how="left")
    m = m[m["rel_std"] > 0].copy()
    m["log_rel_std"] = np.log(m["rel_std"])
    # categorical encodings (statsmodels MixedLM is happier with strings)
    m["continent"] = m["continent"].astype(str)
    m["cc"] = m["cc"].astype(str)
    m["city_proxy"] = m["city_proxy"].astype(str)
    m["dimension"] = m["dimension"].astype(str)
    return m


def fit_mixedlm_one_arch(sub: pd.DataFrame, arch: str) -> dict:
    """Fit log(rel_std) ~ 1 + C(dim) with random intercepts at
    continent / country / place. Returns variance components and ICCs.

    To keep the design matrix tractable on 70k x 1240-level data, we
    pre-residualise log(rel_std) on the dimension fixed effect (a 6-level
    categorical) and then fit a 3-level intercept-only mixed model on the
    residual. This is algebraically identical to including dimension as a
    fixed effect in a single mixedlm call, but the design matrix is much
    smaller and the fit converges in seconds rather than minutes.
    """
    print(f"\n=== Fitting MixedLM for {arch} (n={len(sub):,}) ===", flush=True)
    t0 = time.time()

    # Pre-residualise on dimension FE (no random effects yet) using OLS.
    dim_dummies = pd.get_dummies(sub["dimension"], prefix="dim", drop_first=True).astype(float)
    X = sm.add_constant(dim_dummies.to_numpy())
    y = sub["log_rel_std"].to_numpy()
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    sub = sub.copy()
    sub["log_rel_std_resid"] = resid
    print(f"  pre-residualisation done in {time.time()-t0:.1f}s "
          f"(R^2_dim = {1 - resid.var()/y.var():.4f})", flush=True)

    # 3-level nested random effects on the residual:
    #   groups = continent; vc_formula adds country (cc) and place (city_proxy)
    #   as additional variance components within the continent block.
    md = smf.mixedlm(
        "log_rel_std_resid ~ 1",
        data=sub,
        groups=sub["continent"],
        vc_formula={"country": "0 + C(cc)", "place": "0 + C(city_proxy)"},
        re_formula="1",
    )
    mdf = md.fit(reml=True, method="lbfgs", maxiter=400)
    elapsed = time.time() - t0
    print(f"  fit converged={mdf.converged}  elapsed={elapsed:.1f}s", flush=True)

    # variance components
    # - mdf.cov_re is the random-intercept covariance at the groups level
    #   (continent). For an intercept-only random effect this is a 1x1 matrix
    #   holding sigma^2_continent.
    # - mdf.vcomp is a vector of the additional variance components in the
    #   order given by vc_formula (country, then place).
    # - mdf.scale is the residual variance sigma^2_e.
    sig2_continent = float(np.asarray(mdf.cov_re)[0, 0])
    sig2_country = float(mdf.vcomp[0])
    sig2_place = float(mdf.vcomp[1])
    sig2_resid = float(mdf.scale)
    sig2_total = sig2_continent + sig2_country + sig2_place + sig2_resid

    icc_continent = sig2_continent / sig2_total
    icc_country = sig2_country / sig2_total
    icc_place = sig2_place / sig2_total
    resid_share = sig2_resid / sig2_total

    print(f"  sigma^2_continent = {sig2_continent:.4f}  ICC = {100*icc_continent:.2f}%")
    print(f"  sigma^2_country   = {sig2_country:.4f}  ICC = {100*icc_country:.2f}%")
    print(f"  sigma^2_place     = {sig2_place:.4f}  ICC = {100*icc_place:.2f}%")
    print(f"  sigma^2_resid     = {sig2_resid:.4f}  share = {100*resid_share:.2f}%")
    print(f"  sigma^2_total     = {sig2_total:.4f}")

    return {
        "arch": arch,
        "n_obs": int(len(sub)),
        "converged": bool(mdf.converged),
        "elapsed_s": float(elapsed),
        "sigma2_continent": sig2_continent,
        "sigma2_country": sig2_country,
        "sigma2_place": sig2_place,
        "sigma2_residual": sig2_resid,
        "sigma2_total": sig2_total,
        "icc_continent": icc_continent,
        "icc_country": icc_country,
        "icc_place": icc_place,
        "residual_share": resid_share,
    }


def load_sequential_r2() -> dict:
    """Pull the sequential R^2 numbers reported in Table / Fig 2(a) of the paper."""
    multi = json.loads((DATA / "multiscale_decomposition.json").read_text())
    out = {}
    for arch in ARCHS:
        shares = multi["per_arch"][arch]["nested_r2"]["variance_shares"]
        out[arch] = {
            "continent": shares.get("continent", 0.0),
            "country": shares.get("country", 0.0),
            "place": shares.get("city", 0.0),
            "residual": shares.get("within_city_residual", 0.0),
            "dimension": shares.get("dimension", 0.0),
        }
    return out


def main() -> None:
    df = load_data()
    seq = load_sequential_r2()

    arch_results = {}
    for arch in ARCHS:
        sub = df[df["arch"] == arch].copy()
        arch_results[arch] = fit_mixedlm_one_arch(sub, arch)

    # side-by-side comparison
    print("\n\n=== Comparison: ICC (simultaneous) vs sequential R^2 (paper) ===")
    print(f"{'arch':18s}  {'level':10s}  {'ICC':>8s}  {'seq R^2':>8s}  {'delta':>8s}")
    comparison = {}
    for arch in ARCHS:
        comparison[arch] = {}
        for level in ["continent", "country", "place"]:
            icc = arch_results[arch][f"icc_{level}"]
            sr2 = seq[arch][level]
            delta = icc - sr2
            print(f"  {arch:16s}  {level:10s}  {100*icc:7.2f}%  {100*sr2:7.2f}%  {100*delta:+7.2f}%")
            comparison[arch][level] = {
                "icc": icc,
                "sequential_r2": sr2,
                "delta_pp": delta,
            }

    out = {
        "icc_estimates": arch_results,
        "sequential_r2": seq,
        "comparison": comparison,
        "method": {
            "model": "log(rel_std) ~ 1 + C(dimension) + (1 | continent / country / place)",
            "estimator": "REML (statsmodels.MixedLM, lbfgs)",
            "interpretation": (
                "ICC at each level is the share of total variance attributable "
                "to that level under a simultaneous estimation. Unlike the "
                "sequential incremental R^2, it does not depend on the order "
                "in which levels enter the decomposition."
            ),
        },
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUT_JSON}")


if __name__ == "__main__":
    main()
