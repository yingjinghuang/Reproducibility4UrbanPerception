"""Tier 2 — geographic stratification of per-image variance.

Joins Tier 1 per-image variance with image_geo metadata, then computes:
  - per-city, per-country, per-continent, per-(global_south × dim) summaries
  - simple Bayesian hierarchical pooling for per-city posterior means
    (image -> city -> country -> region partial pooling, numpyro NUTS)

Outputs:
  data_processed/tier2_per_city.parquet              per-city stability metrics
  data_processed/tier2_global_south_split.json       Global N vs S headline
  data_processed/tier2_continent_split.json          per-continent rollup
  data_processed/tier2_hierarchical_posteriors.parquet  posterior means + 95% CIs

Note: a true hierarchical model with 7000+ city_proxy levels can be slow with NUTS.
We restrict the hierarchical model to architectures = ViT-B/16 and dimension = boring
(the most unstable dim) as the primary case, then run a faster frequentist
mixed-effects-style partial pooling for the remaining dimensions.
"""
from __future__ import annotations
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import pandas as pd

from polygeo.paths import DATA, IMAGE_GEO, DIMENSIONS

OUT_CITY = DATA / "tier2_per_city.parquet"
OUT_GS = DATA / "tier2_global_south_split.json"
OUT_CONT = DATA / "tier2_continent_split.json"
OUT_POST = DATA / "tier2_hierarchical_posteriors.parquet"


def main() -> None:
    print("loading Tier 1 per-image variance ...")
    t1 = pd.read_parquet(DATA / "tier1_per_image.parquet")
    print(f"  rows: {len(t1):,}  archs: {t1['arch'].unique().tolist()}")

    print("loading image_geo ...")
    geo = pd.read_parquet(IMAGE_GEO)[
        ["image_id", "city_proxy", "cc", "continent", "income_class", "global_south"]
    ]

    df = t1.merge(geo, on="image_id", how="left")
    if df["city_proxy"].isna().any():
        n_drop = int(df["city_proxy"].isna().sum())
        print(f"  dropping {n_drop} rows with missing city_proxy")
        df = df.dropna(subset=["city_proxy"])

    # ---------------------- Per-(arch × dim × city) ----------------------
    print("\n=== per-city aggregation ===")
    by_city = (
        df.groupby(["arch", "dimension", "city_proxy"])
        .agg(
            n_images=("image_id", "size"),
            mean_std=("std_pred", "mean"),
            mean_rel_std=("rel_std", "mean"),
            median_rel_std=("rel_std", "median"),
            frac_rel_std_gt_10pct=("rel_std", lambda s: float((s > 0.10).mean())),
            frac_rel_std_gt_25pct=("rel_std", lambda s: float((s > 0.25).mean())),
        )
        .reset_index()
    )
    # join continent / cc / income / global_south at the city level
    city_meta = geo.drop_duplicates("city_proxy")[
        ["city_proxy", "cc", "continent", "income_class", "global_south"]
    ]
    by_city = by_city.merge(city_meta, on="city_proxy", how="left")
    by_city.to_parquet(OUT_CITY, index=False)
    print(f"  wrote {OUT_CITY}  ({OUT_CITY.stat().st_size/1e6:.2f} MB)")
    print(f"  per-city rows: {len(by_city):,}")
    print(f"  cities per arch×dim (median): {by_city.groupby(['arch','dimension']).size().median():.0f}")

    # ---------------------- Global N vs S, weighted by n_images ----------------------
    print("\n=== Global North vs South (per arch × dim, weighted by n_images) ===")
    gs_split = {}
    for (arch, dim), grp in df.groupby(["arch", "dimension"]):
        n_g = grp.groupby("global_south")["rel_std"].agg(["mean", "median", "size"]).to_dict("index")
        gs_split[f"{arch}__{dim}"] = {
            "north_mean_rel_std": float(n_g.get(0, {}).get("mean", float("nan"))),
            "south_mean_rel_std": float(n_g.get(1, {}).get("mean", float("nan"))),
            "north_n_images": int(n_g.get(0, {}).get("size", 0)),
            "south_n_images": int(n_g.get(1, {}).get("size", 0)),
            "diff_mean": float(n_g.get(1, {}).get("mean", np.nan) - n_g.get(0, {}).get("mean", np.nan)),
            "ratio_mean": float(n_g.get(1, {}).get("mean", np.nan) / max(n_g.get(0, {}).get("mean", np.nan), 1e-9)),
        }
        print(
            f"  {arch:18s}  {dim:11s}  N={n_g.get(0,{}).get('mean',np.nan):.3f}  "
            f"S={n_g.get(1,{}).get('mean',np.nan):.3f}  diff={gs_split[f'{arch}__{dim}']['diff_mean']:+.3f}  "
            f"ratio={gs_split[f'{arch}__{dim}']['ratio_mean']:.3f}"
        )
    OUT_GS.write_text(json.dumps(gs_split, indent=2))
    print(f"  wrote {OUT_GS}")

    # ---------------------- Continent split ----------------------
    print("\n=== Continent (per arch × dim, mean rel_std) ===")
    cont_rows = []
    for (arch, dim), grp in df.groupby(["arch", "dimension"]):
        for cont, sub in grp.groupby("continent"):
            cont_rows.append({
                "arch": arch, "dimension": dim, "continent": cont,
                "n_images": int(len(sub)),
                "mean_rel_std": float(sub["rel_std"].mean()),
                "median_rel_std": float(sub["rel_std"].median()),
                "frac_rel_std_gt_25pct": float((sub["rel_std"] > 0.25).mean()),
            })
    cont_df = pd.DataFrame(cont_rows)
    OUT_CONT.write_text(cont_df.to_json(orient="records", indent=2))
    print(f"  wrote {OUT_CONT}")
    pivot = cont_df.pivot_table(index=["arch", "continent"], columns="dimension", values="mean_rel_std").round(3)
    print(pivot.to_string())

    # ---------------------- Hierarchical Bayesian pooling (numpyro) ----------------------
    print("\n=== Hierarchical posterior (numpyro NUTS, ViT-B/16 only) ===")
    try:
        import jax.numpy as jnp
        import jax.random as jrand
        import numpyro
        import numpyro.distributions as dist
        from numpyro.infer import MCMC, NUTS

        # Restrict to ViT-B/16 (headline) for the Bayesian model; we run all 6 dims.
        all_post = []
        for dim in DIMENSIONS:
            sub = df[(df["arch"] == "vit_b16") & (df["dimension"] == dim)].copy()
            sub["log_rel_std"] = np.log(sub["rel_std"].clip(1e-4))

            # encode city, country
            city_codes, city_uniques = pd.factorize(sub["city_proxy"], sort=True)
            country_codes, country_uniques = pd.factorize(sub["cc"], sort=True)
            cont_codes, cont_uniques = pd.factorize(sub["continent"], sort=True)

            # Build country -> cont and city -> country lookups for nested hierarchy
            city_to_country_lut = sub.drop_duplicates("city_proxy").set_index("city_proxy").loc[city_uniques, "cc"].map(
                {c: i for i, c in enumerate(country_uniques)}
            ).to_numpy()
            country_to_cont_lut = sub.drop_duplicates("cc").set_index("cc").loc[country_uniques, "continent"].map(
                {c: i for i, c in enumerate(cont_uniques)}
            ).to_numpy()

            y = sub["log_rel_std"].to_numpy()
            x_city = city_codes
            n_city = len(city_uniques)
            n_country = len(country_uniques)
            n_cont = len(cont_uniques)

            def model(y, x_city, city_to_country, country_to_cont, n_city, n_country, n_cont):
                mu_global = numpyro.sample("mu_global", dist.Normal(0.0, 2.0))
                sigma_obs = numpyro.sample("sigma_obs", dist.HalfNormal(1.0))
                sigma_cont = numpyro.sample("sigma_cont", dist.HalfNormal(1.0))
                sigma_country = numpyro.sample("sigma_country", dist.HalfNormal(1.0))
                sigma_city = numpyro.sample("sigma_city", dist.HalfNormal(1.0))
                a_cont = numpyro.sample(
                    "a_cont", dist.Normal(jnp.zeros(n_cont), sigma_cont)
                )
                a_country = numpyro.sample(
                    "a_country",
                    dist.Normal(a_cont[country_to_cont], sigma_country),
                )
                a_city = numpyro.sample(
                    "a_city",
                    dist.Normal(a_country[city_to_country], sigma_city),
                )
                eta = mu_global + a_city[x_city]
                numpyro.sample("y", dist.Normal(eta, sigma_obs), obs=y)

            kernel = NUTS(model, target_accept_prob=0.85)
            mcmc = MCMC(kernel, num_warmup=400, num_samples=600, num_chains=2, progress_bar=False)
            mcmc.run(
                jrand.PRNGKey(20260505 + DIMENSIONS.index(dim)),
                y=jnp.asarray(y, dtype=jnp.float32),
                x_city=jnp.asarray(x_city, dtype=jnp.int32),
                city_to_country=jnp.asarray(city_to_country_lut, dtype=jnp.int32),
                country_to_cont=jnp.asarray(country_to_cont_lut, dtype=jnp.int32),
                n_city=int(n_city), n_country=int(n_country), n_cont=int(n_cont),
            )
            samp = mcmc.get_samples()
            # posterior per-city: log_rel_std mean = mu_global + a_city
            mu_global = np.asarray(samp["mu_global"])  # [S]
            a_city = np.asarray(samp["a_city"])        # [S, n_city]
            sigma_cont_v = np.asarray(samp["sigma_cont"])
            sigma_country_v = np.asarray(samp["sigma_country"])
            sigma_city_v = np.asarray(samp["sigma_city"])
            sigma_obs_v = np.asarray(samp["sigma_obs"])

            mean_log_rel = a_city.mean(axis=0) + mu_global.mean()
            lo_log_rel = np.percentile(a_city, 2.5, axis=0) + np.percentile(mu_global, 2.5)
            hi_log_rel = np.percentile(a_city, 97.5, axis=0) + np.percentile(mu_global, 97.5)

            post = pd.DataFrame({
                "dimension": dim,
                "city_proxy": city_uniques,
                "post_mean_log_rel_std": mean_log_rel,
                "post_lo_log_rel_std": lo_log_rel,
                "post_hi_log_rel_std": hi_log_rel,
                "post_mean_rel_std": np.exp(mean_log_rel),
                "post_lo_rel_std": np.exp(lo_log_rel),
                "post_hi_rel_std": np.exp(hi_log_rel),
            })
            all_post.append(post)
            print(
                f"  {dim:11s}: σ_cont={sigma_cont_v.mean():.3f} σ_country={sigma_country_v.mean():.3f} "
                f"σ_city={sigma_city_v.mean():.3f} σ_obs={sigma_obs_v.mean():.3f} "
                f"(n_city={n_city}, n_country={n_country}, n_cont={n_cont})"
            )

        post_df = pd.concat(all_post, ignore_index=True)
        post_df.to_parquet(OUT_POST, index=False)
        print(f"  wrote {OUT_POST}  ({OUT_POST.stat().st_size/1e6:.2f} MB)")

    except Exception as e:
        print(f"  hierarchical model FAILED: {e!r}")
        print("  proceeding without it; only frequentist per-city estimates are saved.")


if __name__ == "__main__":
    main()
