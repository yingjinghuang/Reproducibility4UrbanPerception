"""Nature-style redraws of the four data-driven main figures.

This script is deliberately plotting-only: it reads the existing analysis
artifacts in ``data_processed/`` and never overwrites the original figure
files. New files are written to ``figures/nature_v1/``.

Examples
--------
Render all four figures::

    python scripts/figures/31_nature_style_figures.py

Render selected figures::

    python scripts/figures/31_nature_style_figures.py --fig 1 4

Use a different output directory::

    python scripts/figures/31_nature_style_figures.py --out figures/nature_v2

Design goals
------------
* journal-like compact typography and thin axes
* color-blind-friendly, muted palette
* minimal grid / chart junk
* direct visual hierarchy rather than saturated decoration
* vector PDF + publication-resolution PNG
* scientific content and data selection kept consistent with the original
  figure scripts
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

from polygeo.paths import DATA, DIMENSIONS, IMAGE_GEO, ROOT


# -----------------------------------------------------------------------------
# Shared visual system
# -----------------------------------------------------------------------------
MM_TO_IN = 1 / 25.4
DOUBLE_COLUMN = 183 * MM_TO_IN  # ~7.20 in, full-width journal figure
SINGLE_COLUMN = 89 * MM_TO_IN   # ~3.50 in

ARCHS = ["resnet50", "vit_b16", "dinov2_frozen"]
ARCH_LABEL = {
    "resnet50": "ResNet-50",
    "vit_b16": "ViT-B/16",
    "dinov2_frozen": "DINOv2 (frozen)",
}
ARCH_SHORT = {
    "resnet50": "ResNet",
    "vit_b16": "ViT",
    "dinov2_frozen": "DINOv2",
}
# Okabe-Ito-inspired, robust in print and for most color-vision deficiencies.
ARCH_COLOR = {
    "resnet50": "#0072B2",
    "vit_b16": "#D55E00",
    "dinov2_frozen": "#009E73",
}
ARCH_MARKER = {"resnet50": "s", "vit_b16": "o", "dinov2_frozen": "^"}

INK = "#222222"
MID = "#6E6E6E"
LIGHT = "#D9D9D9"
VERY_LIGHT = "#EFEFEF"

SCALE_ORDER = ["dimension", "continent", "country", "city", "within_city_residual"]
SCALE_LABEL = {
    "dimension": "Dimension",
    "continent": "Continent",
    "country": "Country",
    "city": "Place",
    "within_city_residual": "Per-image",
}
SCALE_COLOR = {
    "dimension": "#C7C7C7",
    "continent": "#4C78A8",
    "country": "#A8C5E5",
    "city": "#59A14F",
    "within_city_residual": "#E15759",
}

SOURCE_ORDER = ["init_dropout", "data", "aug"]
SOURCE_LABEL = {
    "init_dropout": "Initialization + dropout",
    "data": "Data order",
    "aug": "Augmentation",
}
SOURCE_SHORT = {"init_dropout": "Init. + dropout", "data": "Data", "aug": "Aug."}
SOURCE_COLOR = {"init_dropout": "#7F3C8D", "data": "#2F6B9A", "aug": "#11A579"}

CONTINENT_COLOR = {
    "North America": "#7A7A7A",
    "South America": "#D55E00",
    "Europe": "#0072B2",
    "Asia": "#8A5FB5",
    "Africa": "#009E73",
    "Oceania": "#CC79A7",
}
CC_DISPLAY = {"TW": "CN-TW"}


def _register_font() -> None:
    """Prefer Arial/Helvetica-like sans fonts without requiring bundled fonts."""
    font_dir = os.environ.get("POLYGEO_FONT_DIR")
    if font_dir:
        for name in ("arial.ttf", "arialbd.ttf", "ariali.ttf"):
            p = Path(font_dir) / name
            if p.exists():
                fm.fontManager.addfont(str(p))

    available = {f.name for f in fm.fontManager.ttflist}
    for candidate in ("Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"):
        if candidate in available:
            mpl.rcParams["font.family"] = candidate
            return
    mpl.rcParams["font.family"] = "sans-serif"


def set_nature_style() -> None:
    _register_font()
    mpl.rcParams.update({
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.transparent": False,
        "savefig.facecolor": "white",
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "axes.edgecolor": INK,
        "axes.linewidth": 0.65,
        "axes.labelcolor": INK,
        "axes.labelsize": 8.3,
        "axes.titlesize": 8.3,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.labelsize": 7.3,
        "ytick.labelsize": 7.3,
        "xtick.major.width": 0.65,
        "ytick.major.width": 0.65,
        "xtick.minor.width": 0.45,
        "ytick.minor.width": 0.45,
        "xtick.major.size": 3.2,
        "ytick.major.size": 3.2,
        "xtick.minor.size": 1.8,
        "ytick.minor.size": 1.8,
        "legend.fontsize": 7.1,
        "legend.frameon": False,
        "lines.linewidth": 1.25,
        "lines.markersize": 4.6,
        "patch.linewidth": 0.6,
        "text.color": INK,
        "axes.titlepad": 5,
    })


def clean_axis(ax: plt.Axes, *, left: bool = True, bottom: bool = True) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(left)
    ax.spines["bottom"].set_visible(bottom)
    ax.tick_params(direction="out", top=False, right=False)


def panel_label(ax: plt.Axes, label: str, x: float = -0.12, y: float = 1.04) -> None:
    ax.text(x, y, label, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=10.2, fontweight="bold", color=INK)


def pct_log_formatter(x: float, _pos: int) -> str:
    pct = 100 * x
    if pct >= 10:
        return f"{pct:.0f}%"
    if pct >= 1:
        return f"{pct:g}%"
    return f"{pct:.1f}%"


def require(paths: Iterable[Path], hint: str = "") -> None:
    missing = [p for p in paths if not p.exists()]
    if not missing:
        return
    msg = "Missing required plotting inputs:\n" + "\n".join(f"  - {p}" for p in missing)
    if hint:
        msg += f"\n\n{hint}"
    raise FileNotFoundError(msg)


def save_pair(fig: plt.Figure, out_dir: Path, stem: str, dpi: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / f"{stem}.pdf"
    png = out_dir / f"{stem}.png"
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(png, bbox_inches="tight", pad_inches=0.04, dpi=dpi)
    plt.close(fig)
    print(f"saved {pdf.relative_to(ROOT)}")
    print(f"saved {png.relative_to(ROOT)}")


# -----------------------------------------------------------------------------
# Figure 1 — scale ladder
# -----------------------------------------------------------------------------
def draw_fig1(out_dir: Path, dpi: int) -> None:
    json_path = DATA / "scale_ladder.json"
    dist_path = DATA / "scale_ladder_distributions.npz"
    require([json_path, dist_path], "Run scripts/analysis/24_scale_ladder.py first if these files are absent.")

    ladder = json.loads(json_path.read_text())
    dist = np.load(dist_path)
    scale_order = ["aggregate", "continent", "country", "city", "image"]
    scale_labels = ["Aggregate", "Continent", "Country", "Place", "Image"]
    xs = np.arange(len(scale_order), dtype=float)
    offsets = {"resnet50": -0.18, "vit_b16": 0.0, "dinov2_frozen": 0.18}

    fig, ax = plt.subplots(figsize=(DOUBLE_COLUMN, 3.15))
    medians: dict[str, list[float]] = {a: [] for a in ARCHS}

    for si, scale in enumerate(scale_order):
        for arch in ARCHS:
            values = np.asarray(dist[f"{arch}__{scale}"], dtype=float)
            x = xs[si] + offsets[arch]
            color = ARCH_COLOR[arch]
            med = float(np.median(values))
            medians[arch].append(med)
            bp = ax.boxplot(
                values,
                positions=[x],
                widths=0.145,
                patch_artist=True,
                showfliers=False,
                whis=(5, 95),
                manage_ticks=False,
                zorder=2,
                medianprops={"color": color, "linewidth": 1.05},
                boxprops={
                    "facecolor": mpl.colors.to_rgba(color, 0.12),
                    "edgecolor": color,
                    "linewidth": 0.75,
                },
                whiskerprops={"color": color, "linewidth": 0.7},
                capprops={"color": color, "linewidth": 0.7},
            )
            for patch in bp["boxes"]:
                patch.set_zorder(2)

    for arch in ARCHS:
        xline = xs + offsets[arch]
        ax.plot(
            xline,
            medians[arch],
            color=ARCH_COLOR[arch],
            marker=ARCH_MARKER[arch],
            markerfacecolor="white",
            markeredgecolor=ARCH_COLOR[arch],
            markeredgewidth=0.85,
            linewidth=1.35,
            markersize=4.2,
            label=ARCH_LABEL[arch],
            zorder=4,
        )

    total_images = ladder["vit_b16"]["n_images"]
    n_per_unit = {
        scale: total_images / max(ladder["vit_b16"]["ladder"][scale]["n_units"], 1)
        for scale in scale_order
    }
    vit_image = ladder["vit_b16"]["ladder"]["image"]["median_rel_std"]
    reference = np.array([vit_image / np.sqrt(n_per_unit[s]) for s in scale_order], dtype=float)
    ax.plot(xs, reference, color=MID, lw=0.85, ls=(0, (1.4, 2.0)), zorder=1)
    if reference[-2] > 0 and reference[-1] > 0:
        ax.text(xs[-1] - 0.02, reference[-1] * 0.82, r"$1/\sqrt{n}$ reference",
                fontsize=6.7, color=MID, ha="right", va="top")

    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=5))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(FuncFormatter(pct_log_formatter))
    ax.set_ylabel(r"Seed-induced relative variability, $\sigma_{\mathrm{rel}}$")
    ax.set_xticks(xs)
    ax.set_xticklabels(scale_labels)
    ax.set_xlim(-0.48, 4.48)
    ax.margins(y=0.12)
    ax.grid(axis="y", which="major", color=VERY_LIGHT, linewidth=0.55, zorder=0)
    clean_axis(ax)
    ax.legend(loc="upper left", ncol=3, columnspacing=1.4, handlelength=1.8,
              bbox_to_anchor=(0.0, 1.01), borderaxespad=0)

    fig.subplots_adjust(left=0.11, right=0.985, bottom=0.17, top=0.90)
    save_pair(fig, out_dir, "fig1_scale_ladder_nature_v1", dpi)


# -----------------------------------------------------------------------------
# Figure 2 — geographic variance structure + place ranks
# -----------------------------------------------------------------------------
def _short_place(city_proxy: str, cc: str) -> str:
    return f"{city_proxy.split(',')[0].strip()} ({CC_DISPLAY.get(cc, cc)})"


def _stagger_labels(items: list[tuple], min_spacing: float, low: float, high: float) -> list[tuple]:
    """Greedy vertical label placement while staying inside plotting limits."""
    if not items:
        return []
    items = sorted(items, key=lambda x: x[1])
    placed = []
    y = low
    for item in items:
        target = max(float(item[1]), y)
        placed.append((*item, target))
        y = target + min_spacing
    overflow = placed[-1][-1] - high
    if overflow > 0:
        placed = [(*p[:-1], p[-1] - overflow) for p in placed]
    for i in range(len(placed) - 2, -1, -1):
        max_y = placed[i + 1][-1] - min_spacing
        if placed[i][-1] > max_y:
            placed[i] = (*placed[i][:-1], max_y)
    return placed


def draw_fig2(out_dir: Path, dpi: int) -> None:
    dec_path = DATA / "multiscale_decomposition.json"
    city_path = DATA / "tier2_per_city.parquet"
    require([dec_path, city_path], "Run scripts/analysis/22_multiscale_geo.py and 13_tier2_geo.py first if needed.")

    per_arch = json.loads(dec_path.read_text())["per_arch"]

    fig = plt.figure(figsize=(DOUBLE_COLUMN, 4.55))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.78, 1.65], wspace=0.29)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    x = np.arange(len(ARCHS))
    bottom = np.zeros(len(ARCHS), dtype=float)
    for key in SCALE_ORDER:
        vals = np.array([per_arch[a]["nested_r2"]["variance_shares"][key] for a in ARCHS])
        ax_a.bar(x, vals, bottom=bottom, width=0.62, color=SCALE_COLOR[key],
                 edgecolor="white", linewidth=0.65)
        for xi, btm, val in zip(x, bottom, vals):
            if val >= 0.075:
                center = btm + val / 2
                txt_color = "white" if key in {"continent", "city", "within_city_residual"} else INK
                ax_a.text(xi, center, f"{val:.0%}", ha="center", va="center",
                          fontsize=6.8, fontweight="bold", color=txt_color)
        bottom += vals

    ax_a.set_xticks(x)
    ax_a.set_xticklabels([ARCH_SHORT[a] for a in ARCHS])
    ax_a.set_ylim(0, 1)
    ax_a.set_yticks(np.linspace(0, 1, 6))
    ax_a.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:.0%}"))
    ax_a.set_ylabel(r"Share of variance in $\log\,\sigma_{\mathrm{rel}}$")
    ax_a.grid(axis="y", color=VERY_LIGHT, lw=0.5, zorder=0)
    clean_axis(ax_a)
    panel_label(ax_a, "a", x=-0.22, y=1.02)

    scale_handles = [Patch(facecolor=SCALE_COLOR[k], edgecolor="none", label=SCALE_LABEL[k]) for k in SCALE_ORDER]
    ax_a.legend(handles=scale_handles, loc="upper left", bbox_to_anchor=(-0.02, -0.13),
                ncol=2, columnspacing=1.0, handlelength=1.0, handletextpad=0.45)

    df = pd.read_parquet(city_path)
    agg = (df.groupby(["arch", "city_proxy", "cc", "continent"])
             .agg(rel_std=("median_rel_std", "mean"), n_images=("n_images", "min"))
             .reset_index())
    agg = agg[agg["n_images"] >= 30]
    piv = agg.pivot_table(index=["city_proxy", "cc", "continent"], columns="arch", values="rel_std")
    piv = piv.dropna(subset=ARCHS)
    ranks = pd.DataFrame({a: piv[a].rank(ascending=False, method="min") for a in ARCHS}, index=piv.index)
    n_places = len(ranks)
    if n_places == 0:
        raise ValueError("No common places available for Fig. 2 after filtering.")

    top_k, bot_k = 5, 3
    top_set: set = set()
    bot_set: set = set()
    for arch in ARCHS:
        top_set |= set(ranks[ranks[arch] <= top_k].index)
        bot_set |= set(ranks[ranks[arch] >= n_places - (bot_k - 1)].index)
    highlighted = top_set | bot_set
    xs = np.arange(len(ARCHS))

    for idx in ranks.index:
        if idx in highlighted:
            continue
        ys = [ranks.loc[idx, a] for a in ARCHS]
        ax_b.plot(xs, ys, color="#D2D2D2", lw=0.5, alpha=0.55, zorder=1)

    for idx in highlighted:
        color = CONTINENT_COLOR.get(idx[2], MID)
        ys = np.array([ranks.loc[idx, a] for a in ARCHS], dtype=float)
        is_top = idx in top_set
        ax_b.plot(xs, ys, color=color, lw=1.25 if is_top else 0.9,
                  alpha=0.98 if is_top else 0.82, zorder=3)
        ax_b.scatter(xs, ys, s=17 if is_top else 12, facecolor=color,
                     edgecolor="white", linewidth=0.45, zorder=4)

    raw_items = [(idx, float(ranks.loc[idx, ARCHS[-1]])) for idx in highlighted]
    placed = _stagger_labels(raw_items, min_spacing=2.15, low=1.0, high=float(n_places))
    x_anchor = xs[-1]
    x_label = xs[-1] + 0.30
    for idx, y_anchor, y_label in placed:
        color = CONTINENT_COLOR.get(idx[2], MID)
        name = _short_place(idx[0], idx[1])
        ax_b.plot([x_anchor + 0.03, x_label - 0.04], [y_anchor, y_label],
                  color=color, lw=0.55, alpha=0.68, zorder=2)
        ax_b.text(x_label, y_label, name, ha="left", va="center", fontsize=6.4,
                  color=color, fontweight="medium" if idx in top_set else "normal")

    ax_b.set_xticks(xs)
    ax_b.set_xticklabels([ARCH_SHORT[a] for a in ARCHS])
    ax_b.set_xlim(-0.18, xs[-1] + 1.46)
    ax_b.set_ylim(n_places + 2.5, -1.5)
    ax_b.set_yticks([])
    ax_b.axhline(top_k + 0.5, color=LIGHT, lw=0.5, ls=(0, (2, 2)), zorder=0)
    ax_b.axhline(n_places - bot_k + 0.5, color=LIGHT, lw=0.5, ls=(0, (2, 2)), zorder=0)
    ax_b.text(-0.035, 0.99, "more\nunstable", transform=ax_b.transAxes,
              ha="right", va="top", fontsize=6.7, color=MID, fontstyle="italic")
    ax_b.text(-0.035, 0.01, "more\nstable", transform=ax_b.transAxes,
              ha="right", va="bottom", fontsize=6.7, color=MID, fontstyle="italic")
    clean_axis(ax_b, left=False)
    panel_label(ax_b, "b", x=-0.08, y=1.02)

    present = [c for c in CONTINENT_COLOR if c in {idx[2] for idx in ranks.index}]
    continent_handles = [Line2D([0], [0], color=CONTINENT_COLOR[c], lw=1.7, label=c) for c in present]
    ax_b.legend(handles=continent_handles, loc="upper left", bbox_to_anchor=(0.0, -0.13),
                ncol=3, columnspacing=1.1, handlelength=1.5, handletextpad=0.45)

    fig.subplots_adjust(left=0.08, right=0.94, top=0.96, bottom=0.20)
    save_pair(fig, out_dir, "fig2_multiscale_decomposition_nature_v1", dpi)


# -----------------------------------------------------------------------------
# Figure 3 — source decomposition
# -----------------------------------------------------------------------------
def _city_spearman(exp2: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], dict[str, int]]:
    geo_city = pd.read_parquet(IMAGE_GEO)[["image_id", "city_proxy"]]
    rho_by_arch: dict[str, pd.DataFrame] = {}
    n_by_arch: dict[str, int] = {}
    for arch in ("vit_b16", "resnet50"):
        merged = exp2[exp2["arch"] == arch].merge(geo_city, on="image_id", how="left")
        city_med = merged.groupby(["source", "city_proxy"])["rel_std"].median().reset_index()
        city_n = merged.groupby(["source", "city_proxy"]).size().reset_index(name="n")
        city_med = city_med.merge(city_n, on=["source", "city_proxy"])
        pivot = city_med.pivot_table(index="city_proxy", columns="source", values="rel_std").dropna()
        pivot_n = city_med.pivot_table(index="city_proxy", columns="source", values="n").dropna()
        keep = (pivot_n >= 180).all(axis=1)
        pivot = pivot.loc[keep]
        rho_by_arch[arch] = (
            pivot.corr(method="spearman")
            .reindex(SOURCE_ORDER, axis=0)
            .reindex(SOURCE_ORDER, axis=1)
        )
        n_by_arch[arch] = len(pivot)
    return rho_by_arch, n_by_arch


def draw_fig3(out_dir: Path, dpi: int) -> None:
    parq_path = DATA / "exp2_per_image_variance.parquet"
    part_path = DATA / "exp2_variance_partition.json"
    require(
        [parq_path, part_path, IMAGE_GEO],
        "Run scripts/analysis/17_exp2_analysis.py first if the Exp. 2 derived tables are absent. "
        "The Nature-style script itself does not need the raw prediction files.",
    )

    exp2 = pd.read_parquet(parq_path)
    parts_json = json.loads(part_path.read_text())["per_arch"]
    rho_by_arch, n_cities = _city_spearman(exp2)
    arch_rows = ["vit_b16", "resnet50"]

    fig = plt.figure(figsize=(DOUBLE_COLUMN, 5.55))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.18, 1.0, 0.92], wspace=0.34, hspace=0.42)
    axes = [[fig.add_subplot(gs[r, c]) for c in range(3)] for r in range(2)]
    letters = [["a", "b", "c"], ["d", "e", "f"]]

    rho_cmap = LinearSegmentedColormap.from_list(
        "rho_blue", ["#F7FBFF", "#6BAED6", "#08306B"]
    )
    im_for_cb = None

    for r, arch in enumerate(arch_rows):
        arch_df = exp2[exp2["arch"] == arch]
        per_dim = parts_json[arch]["per_dim"]

        ax = axes[r][0]
        dims = list(DIMENSIONS)
        y = np.arange(len(dims))
        left = np.zeros(len(dims), dtype=float)
        for src in SOURCE_ORDER:
            key = {
                "init_dropout": "share_init_dropout",
                "data": "share_data",
                "aug": "share_aug",
            }[src]
            vals = np.array([per_dim[d][key] for d in dims], dtype=float)
            ax.barh(y, vals, left=left, height=0.63, color=SOURCE_COLOR[src],
                    edgecolor="white", linewidth=0.5)
            for yi, lft, val in zip(y, left, vals):
                if val >= 0.17:
                    ax.text(lft + val / 2, yi, f"{val:.0%}", ha="center", va="center",
                            fontsize=6.0, color="white", fontweight="bold")
            left += vals
        ax.set_yticks(y)
        ax.set_yticklabels([d.capitalize() for d in dims])
        ax.set_xlim(0, 1)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:.0%}"))
        ax.set_xlabel("Per-image variance share")
        ax.invert_yaxis()
        ax.grid(axis="x", color=VERY_LIGHT, lw=0.45, zorder=0)
        clean_axis(ax)
        panel_label(ax, letters[r][0], x=-0.18, y=1.02)

        ax = axes[r][1]
        bins = np.linspace(0, 0.5, 56)
        for src in SOURCE_ORDER:
            vals = arch_df.loc[arch_df["source"] == src, "rel_std"].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            ax.hist(vals, bins=bins, density=True, histtype="step", linewidth=1.25,
                    color=SOURCE_COLOR[src], label=SOURCE_SHORT[src])
        ax.set_xlim(0, 0.5)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{100*v:.0f}%"))
        ax.set_xlabel(r"Per-image $\sigma_{\mathrm{rel}}$")
        ax.set_ylabel("Density")
        ax.grid(axis="y", color=VERY_LIGHT, lw=0.45, zorder=0)
        clean_axis(ax)
        panel_label(ax, letters[r][1], x=-0.18, y=1.02)

        ax = axes[r][2]
        rho = rho_by_arch[arch].to_numpy(dtype=float)
        mask = np.triu(np.ones_like(rho, dtype=bool), k=1)
        masked = np.ma.masked_where(mask, rho)
        im = ax.imshow(masked, cmap=rho_cmap, vmin=0, vmax=1,
                       interpolation="nearest", aspect="equal")
        im_for_cb = im
        labels = ["Init.", "Data", "Aug."]
        ax.set_xticks(np.arange(3)); ax.set_xticklabels(labels)
        ax.set_yticks(np.arange(3)); ax.set_yticklabels(labels)
        for i in range(3):
            for j in range(i + 1):
                v = rho[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.8,
                            color="white" if v >= 0.58 else INK, fontweight="medium")
        ax.text(0.98, -0.17, f"n = {n_cities[arch]} places", transform=ax.transAxes,
                ha="right", va="top", fontsize=6.2, color=MID)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        panel_label(ax, letters[r][2], x=-0.18, y=1.02)

        axes[r][0].text(-0.52, 0.5, ARCH_LABEL[arch], transform=axes[r][0].transAxes,
                        rotation=90, ha="center", va="center", fontsize=7.6,
                        fontweight="bold")

    source_handles = [
        Line2D([0], [0], color=SOURCE_COLOR[s], lw=2.0, label=SOURCE_LABEL[s])
        for s in SOURCE_ORDER
    ]
    fig.legend(handles=source_handles, loc="upper center", bbox_to_anchor=(0.48, 1.005),
               ncol=3, columnspacing=1.4, handlelength=1.8, handletextpad=0.5)
    if im_for_cb is not None:
        cbar = fig.colorbar(
            im_for_cb,
            ax=[axes[0][2], axes[1][2]],
            fraction=0.035,
            pad=0.03,
            ticks=[0, 0.5, 1.0],
        )
        cbar.set_label("Spearman rank correlation, ρ", fontsize=7.2)
        cbar.ax.tick_params(labelsize=6.7, width=0.5, length=2)
        cbar.outline.set_linewidth(0.5)

    fig.subplots_adjust(left=0.105, right=0.91, top=0.91, bottom=0.09)
    save_pair(fig, out_dir, "fig3_source_decomposition_nature_v1", dpi)


# -----------------------------------------------------------------------------
# Figure 4 — ensemble mitigation
# -----------------------------------------------------------------------------
def _first_n_at_or_above(ns: list[int], ys: list[float], target: float) -> int | None:
    for n, y in zip(ns, ys):
        if y >= target:
            return n
    return None


def draw_fig4(out_dir: Path, dpi: int) -> None:
    curve_path = DATA / "exp3a_ensemble_curve.parquet"
    require([curve_path], "Run scripts/analysis/18_exp3a_ensembling.py first if this file is absent.")

    long = pd.read_parquet(curve_path)
    required_columns = {
        "sampling_method", "B", "bootstrap_seed", "is_core_place", "arch",
        "ensemble_size", "rel_std", "city_proxy",
    }
    missing = required_columns.difference(long.columns)
    if missing:
        raise ValueError(f"ensemble curve lacks required columns: {sorted(missing)}")
    if set(long["sampling_method"].unique()) != {"bootstrap_with_replacement"}:
        raise ValueError("Figure 4 expects bootstrap_with_replacement results only.")
    if set(long["B"].unique()) != {5000}:
        raise ValueError("Figure 4 expects B=5,000 camera-ready results.")
    if set(long["bootstrap_seed"].unique()) != {20260817}:
        raise ValueError("Figure 4 expects bootstrap seed 20260817.")

    all_ns = [1, 2, 4, 8, 16, 32]
    arch_ns = {
        "vit_b16": [1, 2, 4, 8, 16, 32],
        "resnet50": [1, 2, 4, 8, 16],
        "dinov2_frozen": [1, 2, 4, 8, 16],
    }
    long = long[long["ensemble_size"].isin(all_ns)].copy()

    med_by_arch: dict[str, list[float]] = {}
    for arch in ARCHS:
        sub = long[long["arch"] == arch]
        med_by_arch[arch] = [
            float(sub[sub["ensemble_size"] == n]["rel_std"].median())
            for n in arch_ns[arch]
        ]

    core_places = sorted(long.loc[long["is_core_place"], "city_proxy"].unique())
    if len(core_places) != 70:
        raise ValueError(f"expected 70 core places, found {len(core_places)}")

    coverage: dict[str, list[float]] = {}
    for arch in ARCHS:
        sub = long[(long["arch"] == arch) & (long["city_proxy"].isin(core_places))]
        per_city = sub.groupby(["ensemble_size", "city_proxy"])["rel_std"].median().reset_index()
        coverage[arch] = [
            float((per_city.loc[per_city["ensemble_size"] == n, "rel_std"].to_numpy() < 0.10).mean() * 100)
            for n in arch_ns[arch]
        ]

    fig, axes = plt.subplots(
        1, 2, figsize=(DOUBLE_COLUMN, 3.25), gridspec_kw={"wspace": 0.30}
    )

    ax = axes[0]
    for thr in (0.05, 0.10, 0.15):
        ax.axhline(thr, color="#B7B7B7", lw=0.65, ls=(0, (3, 2)), zorder=0)
        ax.text(34.0, thr, f"{thr:.0%}", color=MID, fontsize=6.2,
                ha="left", va="center")
    for arch in ARCHS:
        ns = arch_ns[arch]
        ys = med_by_arch[arch]
        ax.plot(ns, ys, color=ARCH_COLOR[arch], marker=ARCH_MARKER[arch],
                markerfacecolor="white", markeredgewidth=0.8, markersize=4.5,
                lw=1.35, label=ARCH_LABEL[arch], zorder=3)
        ax.text(ns[-1] * 1.07, ys[-1], f"{ys[-1]:.1%}", color=ARCH_COLOR[arch],
                fontsize=6.2, ha="left", va="center")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(all_ns)
    ax.set_xticklabels([str(n) for n in all_ns])
    ax.yaxis.set_major_formatter(FuncFormatter(pct_log_formatter))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(0.85, 47)
    ax.set_xlabel("Ensemble size, $N$")
    ax.set_ylabel(r"Median per-image $\sigma_{\mathrm{rel}}$")
    ax.grid(axis="y", which="major", color=VERY_LIGHT, lw=0.5, zorder=0)
    clean_axis(ax)
    panel_label(ax, "a", x=-0.16, y=1.03)

    ax = axes[1]
    for arch in ARCHS:
        ns = arch_ns[arch]
        ys = coverage[arch]
        ax.plot(ns, ys, color=ARCH_COLOR[arch], marker=ARCH_MARKER[arch],
                markerfacecolor="white", markeredgewidth=0.8, markersize=4.5,
                lw=1.35, zorder=3)
        first_full = _first_n_at_or_above(ns, ys, 99.999)
        if first_full is not None:
            ax.scatter([first_full], [100], s=28, facecolor=ARCH_COLOR[arch],
                       edgecolor="white", linewidth=0.6, zorder=4)
            ax.text(first_full, 96.0, f"N={first_full}", color=ARCH_COLOR[arch],
                    fontsize=6.1, ha="center", va="top")
    ax.set_xscale("log", base=2)
    ax.set_xticks(all_ns)
    ax.set_xticklabels([str(n) for n in all_ns])
    ax.set_xlim(0.85, 38)
    ax.set_ylim(-2, 104)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:.0f}%"))
    ax.set_xlabel("Ensemble size, $N$")
    ax.set_ylabel("Core places below 10% tolerance")
    ax.grid(axis="y", color=VERY_LIGHT, lw=0.5, zorder=0)
    clean_axis(ax)
    panel_label(ax, "b", x=-0.16, y=1.03)

    handles = [
        Line2D([0], [0], color=ARCH_COLOR[a], marker=ARCH_MARKER[a],
               markerfacecolor="white", markeredgecolor=ARCH_COLOR[a],
               lw=1.3, markersize=4.2, label=ARCH_LABEL[a])
        for a in ARCHS
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.51, 1.01),
               ncol=3, columnspacing=1.35, handlelength=1.9, handletextpad=0.5)

    fig.subplots_adjust(left=0.095, right=0.965, bottom=0.18, top=0.86)
    save_pair(fig, out_dir, "fig4_ensembling_nature_v1", dpi)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate Nature-style versions of the four data-driven main figures."
    )
    p.add_argument(
        "--fig",
        nargs="+",
        choices=["1", "2", "3", "4", "all"],
        default=["all"],
        help="Figures to render (default: all). Example: --fig 1 4",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "figures" / "nature_v1",
        help="Output directory (default: figures/nature_v1)",
    )
    p.add_argument("--dpi", type=int, default=300, help="PNG output dpi (default: 300)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_nature_style()
    out_dir = args.out if args.out.is_absolute() else ROOT / args.out
    requested = {"1", "2", "3", "4"} if "all" in args.fig else set(args.fig)

    print(f"Nature-style output directory: {out_dir}")
    if "1" in requested:
        draw_fig1(out_dir, args.dpi)
    if "2" in requested:
        draw_fig2(out_dir, args.dpi)
    if "3" in requested:
        draw_fig3(out_dir, args.dpi)
    if "4" in requested:
        draw_fig4(out_dir, args.dpi)
    print("done")


if __name__ == "__main__":
    main()
