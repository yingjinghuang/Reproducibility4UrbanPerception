"""Exp 2 source decomposition analysis.

For each of the 3 source groups (init+dropout, data, aug) on ViT-B/16:
  - Compute per-image variance across the 20 group-runs.
  - Per-arch ratio: how much of Exp1's all-source per-image variance does this
    source explain on average?
  - Per-(image, dim) variance partition: image-level proportions.
  - Re-fit Tier 3 regression per source group to test if β_global_south
    survives in each — if it does, GS gap is robust across sources.
  - Per-source rel_std maps (light-weight, no hierarchical model).

Outputs:
  data_processed/exp2_per_image_variance.parquet      per (img, dim, source) std
  data_processed/exp2_variance_partition.json         per-dim aggregate variance share
  data_processed/exp2_tier3_per_source.json           Tier 3 β_GS per source group
  data_processed/exp2_global_south_split.json         per-source N/S ratio
  figures/fig3_source_decomposition.{pdf,png}
"""
from __future__ import annotations
import sys
import json
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from polygeo.paths import ROOT, DATA, IMAGE_GEO, PREDICTIONS, DIMENSIONS, IMAGE_TABLE, QSCORES

if os.environ.get("POLYGEO_FONT_DIR"):
    for _f in ("arial.ttf", "arialbd.ttf", "ariali.ttf"):
        _p = Path(os.environ["POLYGEO_FONT_DIR"]) / _f
        if _p.exists():
            fm.fontManager.addfont(str(_p))
mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titlesize": 21,
    "axes.labelsize": 21,
    "xtick.labelsize": 19,
    "ytick.labelsize": 19,
    "legend.fontsize": 17,
})

OUT_PARQ = DATA / "exp2_per_image_variance.parquet"
OUT_PART = DATA / "exp2_variance_partition.json"
OUT_T3 = DATA / "exp2_tier3_per_source.json"
OUT_GS = DATA / "exp2_global_south_split.json"
FIGS = ROOT / "figures"
FIGS.mkdir(exist_ok=True)


def load_source_preds(source: str, arch: str, n: int = 20):
    """Stack 20 predictions for a given source group: returns [n_img, n_dim, n_seeds] + image_ids."""
    arrays = []
    image_ids = None
    for s in range(n):
        if source == "init_dropout":
            run = f"exp2_init_dropout_{arch}_i{s:03d}_d000_a000"
        elif source == "data":
            run = f"exp2_data_{arch}_i000_d{s:03d}_a000"
        elif source == "aug":
            run = f"exp2_aug_{arch}_i000_d000_a{s:03d}"
        else:
            raise ValueError(source)
        npz = np.load(PREDICTIONS / f"{run}_test_preds.npz", allow_pickle=True)
        ids = list(npz["image_ids"])
        if image_ids is None:
            image_ids = ids
        else:
            assert ids == image_ids
        arrays.append(npz["preds"])
    return np.stack(arrays, axis=-1), image_ids


def per_image_long(source: str, arch: str, preds: np.ndarray, image_ids: list[str]) -> pd.DataFrame:
    n_img, n_dim, _ = preds.shape
    mean_pi = preds.mean(axis=2)
    std_pi = preds.std(axis=2, ddof=0)
    dim_scale = mean_pi.std(axis=0, ddof=0)
    rel_std = std_pi / np.maximum(dim_scale[None, :], 1e-6)
    rows = []
    for d, dim in enumerate(DIMENSIONS):
        rows.append(pd.DataFrame({
            "image_id": image_ids,
            "source": source,
            "arch": arch,
            "dimension": dim,
            "mean_pred": mean_pi[:, d],
            "std_pred": std_pi[:, d],
            "rel_std": rel_std[:, d],
        }))
    return pd.concat(rows, ignore_index=True)


# ---- minimal OLS w/ HC0 (copied from script 14) ----
def fit_ols(y, X, names):
    X1 = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
    XtX = X1.T @ X1
    XtX_inv = np.linalg.pinv(XtX)
    beta = XtX_inv @ X1.T @ y
    resid = y - X1 @ beta
    n, k = X1.shape
    sigma2 = float(resid @ resid / (n - k))
    XtUUX = (X1 * resid[:, None]).T @ (X1 * resid[:, None])
    cov_hc0 = XtX_inv @ XtUUX @ XtX_inv
    se_hc0 = np.sqrt(np.maximum(np.diag(cov_hc0), 0.0))
    z = beta / np.maximum(se_hc0, 1e-12)
    out = {}
    for i, name in enumerate(["intercept"] + names):
        out[name] = {
            "coef": float(beta[i]),
            "se_hc0": float(se_hc0[i]),
            "z": float(z[i]),
            "ci95_lo": float(beta[i] - 1.96 * se_hc0[i]),
            "ci95_hi": float(beta[i] + 1.96 * se_hc0[i]),
        }
    out["_meta"] = {"n": int(n), "k": int(k), "r2": float(1 - resid @ resid / (y.var() * n))}
    return out


def design_matrix_simple(df, cols):
    from sklearn.preprocessing import StandardScaler
    parts, names = [], []
    dim_dummies = pd.get_dummies(df["dimension"], prefix="dim", drop_first=True).astype(float).to_numpy()
    parts.append(dim_dummies); names.extend([f"dim_{d}" for d in DIMENSIONS[1:]])
    cont_vars = cols
    if cont_vars:
        x_cont = StandardScaler().fit_transform(df[cont_vars].to_numpy())
        parts.append(x_cont); names.extend(cont_vars)
    return np.concatenate(parts, axis=1), names


ARCHES = ["vit_b16", "resnet50"]
ARCH_LABEL = {"vit_b16": "ViT-B/16", "resnet50": "ResNet-50"}


def compute_arch(arch: str, sources: list[str]):
    """Run per-arch pipeline: per-image long df, variance partition, source_stats."""
    print(f"=== loading per-source predictions for {arch} ===")
    long_dfs, source_stats = [], {}
    for src in sources:
        preds, ids = load_source_preds(src, arch)
        df = per_image_long(src, arch, preds, ids)
        long_dfs.append(df)
        source_stats[src] = {
            "median_rel_std_per_dim": dict(zip(DIMENSIONS, df.groupby("dimension")["rel_std"].median().tolist())),
            "frac_rel_std_gt_10pct_per_dim": dict(zip(DIMENSIONS, df.groupby("dimension")["rel_std"].apply(lambda s: float((s > 0.10).mean())).tolist())),
            "frac_rel_std_gt_25pct_per_dim": dict(zip(DIMENSIONS, df.groupby("dimension")["rel_std"].apply(lambda s: float((s > 0.25).mean())).tolist())),
        }
        print(f"  {src}: median rel_std per dim = {[round(v,3) for v in source_stats[src]['median_rel_std_per_dim'].values()]}")
    arch_df = pd.concat(long_dfs, ignore_index=True)

    print(f"\n=== variance partition ({arch}) ===")
    pivot = arch_df.pivot_table(index=["image_id", "dimension"], columns="source", values="std_pred")
    pivot["var_init_dropout"] = pivot["init_dropout"] ** 2
    pivot["var_data"] = pivot["data"] ** 2
    pivot["var_aug"] = pivot["aug"] ** 2
    pivot["var_sum"] = pivot["var_init_dropout"] + pivot["var_data"] + pivot["var_aug"]
    parts = {}
    for dim in DIMENSIONS:
        sub = pivot.xs(dim, level="dimension")
        parts[dim] = {
            "share_init_dropout": float((sub["var_init_dropout"] / sub["var_sum"]).mean()),
            "share_data": float((sub["var_data"] / sub["var_sum"]).mean()),
            "share_aug": float((sub["var_aug"] / sub["var_sum"]).mean()),
            "mean_var_sum": float(sub["var_sum"].mean()),
        }
        print(f"  {dim:11s}: init+drop {parts[dim]['share_init_dropout']:.1%}  data {parts[dim]['share_data']:.1%}  aug {parts[dim]['share_aug']:.1%}")
    return arch_df, parts, source_stats


def compute_tier3_per_source(arch_df: pd.DataFrame, sources: list[str], geo, cmplx, label_disp, qs, train_density):
    res_per_src = {}
    for src in sources:
        sub = (
            arch_df[arch_df["source"] == src]
            .merge(geo, on="image_id", how="left")
            .merge(cmplx, on="image_id", how="left")
            .merge(label_disp, on=["image_id", "dimension"], how="left")
            .merge(qs, on=["image_id", "dimension"], how="left")
            .merge(train_density, on="city_proxy", how="left")
        )
        sub = sub.dropna(subset=[
            "rel_std", "complexity_shannon", "complexity_dino_norm", "complexity_seg_entropy",
            "label_dispersion", "n_votes_dim", "train_density_city", "global_south",
        ])
        sub["log_rel_std"] = np.log(sub["rel_std"].clip(1e-4))
        sub["log_train_density_city"] = np.log1p(sub["train_density_city"])
        sub["log_n_votes_dim"] = np.log1p(sub["n_votes_dim"])
        cont_cols = [
            "complexity_shannon", "complexity_dino_norm", "complexity_seg_entropy",
            "log_n_votes_dim", "label_dispersion", "log_train_density_city",
            "global_south",
        ]
        X, names = design_matrix_simple(sub, cont_cols)
        res = fit_ols(sub["log_rel_std"].to_numpy(), X, names)
        res_per_src[src] = res
        gs = res["global_south"]
        print(f"  {src:14s}: β_GS = {gs['coef']:+.4f}  z = {gs['z']:+5.2f}  95%CI = [{gs['ci95_lo']:+.4f}, {gs['ci95_hi']:+.4f}]")
    return res_per_src


def city_spearman(arch_df: pd.DataFrame, sources: list[str], geo_city):
    merged_city = arch_df.merge(geo_city, on="image_id", how="left")
    city_med = (merged_city.groupby(["source", "city_proxy"])["rel_std"].median().reset_index())
    city_n = merged_city.groupby(["source", "city_proxy"]).size().reset_index(name="n")
    city_med = city_med.merge(city_n, on=["source", "city_proxy"])
    pivot = city_med.pivot_table(index="city_proxy", columns="source", values="rel_std").dropna()
    pivot_n = city_med.pivot_table(index="city_proxy", columns="source", values="n").dropna()
    keep = (pivot_n >= 180).all(axis=1)
    pivot = pivot[keep]
    rho = pivot.corr(method="spearman").reindex(sources, axis=0).reindex(sources, axis=1)
    return rho, len(pivot)


def main() -> None:
    sources = ["init_dropout", "data", "aug"]

    # ---- 1-2. per-arch per-source loading + variance partition ----
    per_arch_df = {}
    per_arch_parts = {}
    per_arch_source_stats = {}
    for arch in ARCHES:
        arch_df, parts, source_stats = compute_arch(arch, sources)
        per_arch_df[arch] = arch_df
        per_arch_parts[arch] = parts
        per_arch_source_stats[arch] = source_stats

    out_df = pd.concat(list(per_arch_df.values()), ignore_index=True)
    OUT_PARQ.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(OUT_PARQ, index=False)
    print(f"\nwrote {OUT_PARQ}  ({OUT_PARQ.stat().st_size/1e6:.2f} MB)")

    OUT_PART.write_text(json.dumps({
        "per_arch": {
            arch: {"per_dim": per_arch_parts[arch], "per_source_overall": per_arch_source_stats[arch]}
            for arch in ARCHES
        }
    }, indent=2))
    print(f"wrote {OUT_PART}")

    # ---- 3. Tier 3 regression per (arch, source) ----
    print("\n=== Tier 3 (M5) per (arch, source) ===")
    geo = pd.read_parquet(IMAGE_GEO)[["image_id", "city_proxy", "global_south", "income_class"]]
    cmplx = pd.read_parquet(DATA / "image_complexity.parquet")
    label_disp = pd.read_parquet(DATA / "label_dispersion_image.parquet")[
        ["image_id", "dimension", "btl_violation_rate"]
    ].rename(columns={"btl_violation_rate": "label_dispersion"})
    qs = pd.read_parquet(QSCORES)[["image_id", "dimension", "n_votes"]].rename(columns={"n_votes": "n_votes_dim"})
    splits = pd.read_parquet(DATA / "splits.parquet")
    train_imgs = splits[splits["split"] == "train"]["image_id"]
    train_density = (
        pd.read_parquet(IMAGE_GEO)[pd.read_parquet(IMAGE_GEO)["image_id"].isin(train_imgs)]
        .groupby("city_proxy").size().rename("train_density_city").reset_index()
    )

    t3_results = {}
    for arch in ARCHES:
        print(f"--- {arch} ---")
        t3_results[arch] = compute_tier3_per_source(
            per_arch_df[arch], sources, geo, cmplx, label_disp, qs, train_density
        )
    OUT_T3.write_text(json.dumps(t3_results, indent=2))
    print(f"wrote {OUT_T3}")

    # ---- 4. per-source N vs S split per arch ----
    print("\n=== Global South vs North per (arch, source) ===")
    gs_split = {}
    for arch in ARCHES:
        for src in sources:
            sub = per_arch_df[arch][per_arch_df[arch]["source"] == src].merge(
                geo[["image_id", "global_south"]], on="image_id", how="left"
            )
            for dim in DIMENSIONS:
                grp = sub[sub["dimension"] == dim]
                n_val = grp[grp["global_south"] == 0]["rel_std"].mean()
                s_val = grp[grp["global_south"] == 1]["rel_std"].mean()
                gs_split[f"{arch}__{src}__{dim}"] = {
                    "north": float(n_val), "south": float(s_val),
                    "ratio": float(s_val / max(n_val, 1e-9)),
                }
    OUT_GS.write_text(json.dumps(gs_split, indent=2))
    print(f"wrote {OUT_GS}")

    # ---- 5. Fig 3: source decomposition figure (2 rows × 3 cols: ViT / ResNet-50) ----
    print("\n=== rendering Fig 3 (2×3, ViT row + ResNet-50 row) ===")
    SOURCE_COLORS = {"init_dropout": "#9467bd", "data": "#1f77b4", "aug": "#2ca02c"}
    SOURCE_LABEL = {"init_dropout": "Init", "data": "Data", "aug": "Aug"}

    geo_city = pd.read_parquet(IMAGE_GEO)[["image_id", "city_proxy"]]
    rho_per_arch = {}
    n_cities_per_arch = {}
    for arch in ARCHES:
        rho, n_c = city_spearman(per_arch_df[arch], sources, geo_city)
        rho_per_arch[arch] = rho
        n_cities_per_arch[arch] = n_c

    fig = plt.figure(figsize=(15.0, 9.2))
    gs = fig.add_gridspec(2, 3, wspace=0.32, hspace=0.55)

    panel_letters = [["(a)", "(b)", "(c)"], ["(d)", "(e)", "(f)"]]
    for row, arch in enumerate(ARCHES):
        parts = per_arch_parts[arch]
        arch_df = per_arch_df[arch]

        # Panel: variance shares per dim (stacked bars)
        ax = fig.add_subplot(gs[row, 0])
        dims = list(DIMENSIONS)
        shares_init = [parts[d]["share_init_dropout"] for d in dims]
        shares_data = [parts[d]["share_data"] for d in dims]
        shares_aug = [parts[d]["share_aug"] for d in dims]
        x = np.arange(len(dims))
        ax.bar(x, shares_init, color=SOURCE_COLORS["init_dropout"], label=SOURCE_LABEL["init_dropout"])
        ax.bar(x, shares_data, bottom=shares_init, color=SOURCE_COLORS["data"], label=SOURCE_LABEL["data"])
        ax.bar(x, shares_aug, bottom=np.array(shares_init) + np.array(shares_data),
               color=SOURCE_COLORS["aug"], label=SOURCE_LABEL["aug"])
        ax.set_xticks(x); ax.set_xticklabels(dims, rotation=20, ha="right")
        ax.set_ylabel(f"{ARCH_LABEL[arch]}\nPer-image variance share")
        ax.set_ylim(0, 1)
        ax.set_title(panel_letters[row][0], loc="left", fontsize=19, fontweight="bold")
        if row == 1:
            ax.legend(loc="lower center", fontsize=20, ncol=3, bbox_to_anchor=(0.5, -0.55),
                      frameon=False, handlelength=1.2, handletextpad=0.4, columnspacing=1.5)
        ax.grid(axis="y", alpha=0.3)

        # Panel: per-image rel_std distribution per source (pooled across dims)
        ax = fig.add_subplot(gs[row, 1])
        bins = np.linspace(0, 0.5, 50)
        for src in sources:
            vals = arch_df[arch_df["source"] == src]["rel_std"].values
            ax.hist(vals, bins=bins, density=True, histtype="step", linewidth=2.0,
                    color=SOURCE_COLORS[src], label=SOURCE_LABEL[src])
        ax.set_xlabel(r"Per-image $\sigma_{\mathrm{rel}}$")
        ax.set_ylabel("Density")
        ax.set_title(panel_letters[row][1], loc="left", fontsize=19, fontweight="bold")
        if row == 1:
            ax.legend(loc="lower center", fontsize=20, ncol=3, bbox_to_anchor=(0.5, -0.55),
                      frameon=False, handlelength=1.5, handletextpad=0.4, columnspacing=1.5)
        ax.grid(alpha=0.3)

        # Panel: cross-source Spearman ρ at city level (lower triangular only)
        ax = fig.add_subplot(gs[row, 2])
        rho = rho_per_arch[arch]
        n_c = n_cities_per_arch[arch]
        n = len(sources)
        mask = np.zeros_like(rho.values, dtype=bool)
        mask[np.triu_indices(n, k=1)] = True
        masked = np.ma.masked_where(mask, rho.values)
        cmap = plt.cm.viridis.copy()
        cmap.set_bad("white")
        im = ax.imshow(masked, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")
        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        src_short = [SOURCE_LABEL[s] for s in sources]
        ax.set_xticklabels(src_short)
        ax.set_yticklabels(src_short)
        for i in range(n):
            for j in range(n):
                if j > i:
                    continue
                v = rho.values[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v < 0.6 else "black", fontsize=17)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.set_title(panel_letters[row][2], loc="left", fontsize=19, fontweight="bold")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Spearman ρ")
        print(f"  {arch}: city-Spearman from n={n_c}")

    fig.savefig(FIGS / "fig3_source_decomposition.pdf", bbox_inches="tight")
    fig.savefig(FIGS / "fig3_source_decomposition.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("saved figures/fig3_source_decomposition.{pdf,png}")


if __name__ == "__main__":
    main()
