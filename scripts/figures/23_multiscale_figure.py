"""Fig 11: multi-scale variance decomposition + per-city slope chart (combined).

Two stacked panels:
  (a) Cumulative R² across nested spatial scales as 3 stacked horizontal bars
      (one per architecture). Each segment is one scale's variance share of
      log(rel_std). Shows per-image dominance across archs and architecturally-
      modulated continent / city structure.
  (b) Per-city reproducibility slope chart: each of 70 cities common to all 3
      archs is a polyline connecting its rank-by-rel_std across ResNet → ViT →
      DINOv2. Top-5 unstable + bottom-3 stable in any arch highlighted and
      labeled, colored by continent. Background lines are the rest.

Inputs:
  data_processed/multiscale_decomposition.json
  data_processed/tier2_per_city.parquet
Output:
  figures/fig2_multiscale_decomposition.{pdf,png}
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

from polygeo.paths import ROOT, DATA

FIGS = ROOT / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

if os.environ.get("POLYGEO_FONT_DIR"):
    for _f in ("arial.ttf", "arialbd.ttf", "ariali.ttf"):
        _p = Path(os.environ["POLYGEO_FONT_DIR"]) / _f
        if _p.exists():
            fm.fontManager.addfont(str(_p))
mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titlesize": 21,
    "axes.labelsize": 20,
    "xtick.labelsize": 19,
    "ytick.labelsize": 17,
    "legend.fontsize": 16,
    "figure.dpi": 130,
})


# ---- panel (a) variance decomposition ----
SCALE_ORDER = ["dimension", "continent", "country", "city", "within_city_residual"]
SCALE_LABEL = {
    "dimension": "Dimension",
    "continent": "Continent",
    "country": "Country",
    "city": "Place",
    "within_city_residual": "Per-image",
}
SCALE_COLOR = {
    "dimension": "#bdbdbd",
    "continent": "#1f77b4",
    "country": "#9ecae1",
    "city": "#2ca02c",
    "within_city_residual": "#d62728",
}
ARCH_DISPLAY = {
    "vit_b16": "ViT",
    "resnet50": "ResNet",
    "dinov2_frozen": "DINOv2",
}
ARCHS = ["resnet50", "vit_b16", "dinov2_frozen"]


def panel_variance_share(ax, per_arch: dict) -> None:
    """Vertical stacked bars: x = arch, y = cumulative share 0-100%.
    Stack order from bottom to top: dimension → continent → country → city → within-city."""
    x_positions = np.arange(len(ARCHS))
    bar_w = 0.55
    for xi, arch in zip(x_positions, ARCHS):
        shares = per_arch[arch]["nested_r2"]["variance_shares"]
        bottom = 0.0
        for key in SCALE_ORDER:
            v = shares[key]
            ax.bar(xi, v, bottom=bottom, width=bar_w, color=SCALE_COLOR[key],
                   edgecolor="white", linewidth=1.0)
            if v >= 0.05:
                txt_color = "white" if key in {"continent", "city", "within_city_residual"} else "black"
                ax.text(xi, bottom + v / 2, f"{v * 100:.0f}%",
                        ha="center", va="center", fontsize=15, color=txt_color, fontweight="bold")
            bottom += v

    ax.set_xticks(x_positions)
    ax.set_xticklabels([ARCH_DISPLAY[a] for a in ARCHS], rotation=20, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_xlim(-0.55, len(ARCHS) - 0.45)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_yticklabels([f"{y:.0%}" for y in np.linspace(0, 1, 6)])
    ax.set_ylabel(r"Share of $\log \sigma_{\mathrm{rel}}$ variance")
    ax.set_title("(a)", loc="left", fontsize=22, fontweight="bold")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    handles = [plt.Rectangle((0, 0), 1, 1, color=SCALE_COLOR[k]) for k in SCALE_ORDER]
    labels = [SCALE_LABEL[k] for k in SCALE_ORDER]
    ax.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.02),
              ncol=3, frameon=False, fontsize=16, handlelength=0.9,
              columnspacing=0.7, handletextpad=0.35)


# ---- panel (b) city slope chart ----
ARCH_LABELS = {"resnet50": "ResNet", "vit_b16": "ViT", "dinov2_frozen": "DINOv2"}

CONTINENT_COLOR_ALL = {
    "North America": "#9aa0a6",
    "Asia": "#7a3f99",
    "South America": "#c0392b",
    "Europe": "#1f77b4",
    "Africa": "#2ca02c",
    "Oceania": "#e67e22",
}


CC_DISPLAY = {"TW": "CN-TW"}


def short_name(city_proxy: str, cc: str) -> str:
    return f"{city_proxy.split(',')[0].strip()} ({CC_DISPLAY.get(cc, cc)})"


def panel_city_slope(ax) -> None:
    df = pd.read_parquet(DATA / "tier2_per_city.parquet")
    agg = (df.groupby(["arch", "city_proxy", "cc", "continent"])
             .agg(rel_std=("median_rel_std", "mean"), n_images=("n_images", "min"))
             .reset_index())
    agg = agg[agg["n_images"] >= 30]

    piv = agg.pivot_table(index=["city_proxy", "cc", "continent"],
                          columns="arch", values="rel_std").dropna(subset=ARCHS)
    ranks = pd.DataFrame({a: piv[a].rank(ascending=False, method="min") for a in ARCHS},
                         index=piv.index)
    N = len(ranks)

    TOP_K = 5
    BOT_K = 3
    top_set, bot_set = set(), set()
    for a in ARCHS:
        top_set |= set(ranks[ranks[a] <= TOP_K].index)
        bot_set |= set(ranks[ranks[a] >= N - (BOT_K - 1)].index)

    xs = np.arange(len(ARCHS))

    # background lines: continent-colored, thin & faint
    for idx in ranks.index:
        if idx in top_set or idx in bot_set:
            continue
        cont = idx[2]
        color = CONTINENT_COLOR_ALL.get(cont, "#444444")
        ys = [ranks.loc[idx, a] for a in ARCHS]
        ax.plot(xs, ys, color=color, lw=0.7, alpha=0.35, zorder=1)

    def draw(group: set, lw: float, dot_size: float, alpha: float) -> None:
        for idx in group:
            cont = idx[2]
            color = CONTINENT_COLOR_ALL.get(cont, "#444444")
            ys = [ranks.loc[idx, a] for a in ARCHS]
            ax.plot(xs, ys, color=color, lw=lw, alpha=alpha, zorder=3,
                    solid_capstyle="round")
            ax.scatter(xs, ys, color=color, s=dot_size, edgecolor="white",
                       lw=0.9, zorder=4)

    draw(top_set, lw=2.4, dot_size=55, alpha=0.95)
    draw(bot_set, lw=1.8, dot_size=35, alpha=0.85)

    def collect(group: set, is_top: bool):
        # anchor every label at the rightmost arch (DINOv2 frozen): label y =
        # DINOv2 rank, leader line is short and never crosses the panel.
        items = []
        for idx in group:
            anchor_ai = len(ARCHS) - 1
            anchor_y = ranks.loc[idx, ARCHS[anchor_ai]]
            items.append((idx, anchor_y, anchor_ai))
        return items

    def stagger(items, min_spacing: float):
        items = sorted(items, key=lambda x: x[1])
        out, last_y = [], -1e9
        for idx, y_anchor, peak_ai in items:
            ny = max(y_anchor, last_y + min_spacing)
            out.append((idx, y_anchor, ny, peak_ai))
            last_y = ny
        return out

    def emit(placed, fontsize, label_x_offset):
        for idx, y_anchor, y_label, peak_ai in placed:
            color = CONTINENT_COLOR_ALL.get(idx[2], "#444444")
            name = short_name(idx[0], idx[1])
            x_dot = xs[peak_ai]
            x_label = xs[-1] + label_x_offset
            ax.plot([x_label - 0.04, x_dot], [y_label, y_anchor],
                    color=color, lw=0.6, alpha=0.55, zorder=2)
            ax.text(x_label, y_label, name, ha="left", va="center",
                    fontsize=fontsize, color=color)

    # since labels all anchor at DINOv2 rank, top + bot must be staggered together
    raw = collect(top_set, is_top=True) + collect(bot_set - top_set, is_top=False)
    placed = stagger(raw, min_spacing=2.2)
    top_idx_set = set(top_set)
    sized = [(idx, y_anchor, y_label, peak_ai,
              15 if idx in top_idx_set else 14)
             for idx, y_anchor, y_label, peak_ai in placed]
    for idx, y_anchor, y_label, peak_ai, fs in sized:
        color = CONTINENT_COLOR_ALL.get(idx[2], "#444444")
        name = short_name(idx[0], idx[1])
        x_dot = xs[peak_ai]
        x_label = xs[-1] + 0.18
        ax.plot([x_label - 0.04, x_dot], [y_label, y_anchor],
                color=color, lw=0.6, alpha=0.55, zorder=2)
        ax.text(x_label, y_label, name, ha="left", va="center",
                fontsize=fs, color=color)

    ax.set_xticks(xs)
    ax.set_xticklabels([ARCH_LABELS[a] for a in ARCHS])
    ax.set_xlim(-0.35, len(ARCHS) - 1 + 1.75)
    ax.invert_yaxis()
    ax.set_ylim(N + 6, -5)
    ax.set_yticks([])
    ax.set_ylabel("")
    ax.set_title("(b)", loc="left", fontsize=22, fontweight="bold")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.axhline(TOP_K + 0.5, color="#bbbbbb", lw=0.6, ls=":", zorder=0)
    ax.axhline(N - BOT_K + 0.5, color="#bbbbbb", lw=0.6, ls=":", zorder=0)

    # direction hints replacing y-axis ticks
    ax.text(-0.02, 0.99, "more\nunstable", transform=ax.transAxes,
            ha="right", va="top", fontsize=16, color="#666", fontstyle="italic")
    ax.text(-0.02, 0.01, "more\nstable", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=16, color="#666", fontstyle="italic")

    # legend: every continent present in the 70 cities (line color matches the figure)
    present = {idx[2] for idx in ranks.index}
    leg_order = [c for c in CONTINENT_COLOR_ALL if c in present]
    handles = [plt.Line2D([0], [0], color=CONTINENT_COLOR_ALL[c], lw=2.5) for c in leg_order]
    ax.legend(handles, leg_order, loc="lower center", bbox_to_anchor=(0.5, 1.02),
              ncol=3, frameon=False, fontsize=17,
              handlelength=1.5, handletextpad=0.4, columnspacing=1.5)


def main() -> None:
    data = json.loads((DATA / "multiscale_decomposition.json").read_text())
    per_arch = data["per_arch"]

    fig = plt.figure(figsize=(12.5, 9.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 2.0], wspace=0.25)
    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])
    panel_variance_share(ax_a, per_arch)
    panel_city_slope(ax_b)

    fig.savefig(FIGS / "fig2_multiscale_decomposition.pdf", bbox_inches="tight")
    fig.savefig(FIGS / "fig2_multiscale_decomposition.png", bbox_inches="tight", dpi=160)
    plt.close(fig)
    print("saved figures/fig2_multiscale_decomposition.{pdf,png}")


if __name__ == "__main__":
    main()
