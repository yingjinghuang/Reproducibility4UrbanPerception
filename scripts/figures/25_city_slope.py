"""Per-city reproducibility ranking slope chart across architectures.

Each city is a polyline connecting its rank-by-rel_std across ResNet-50, ViT-B/16,
DINOv2-frozen. Top-5 unstable + bottom-3 stable in any arch are highlighted and
labeled; colored by continent. Background lines are the rest of the 70 cities
common to all 3 archs.

Inputs:
  data_processed/tier2_per_city.parquet
Output:
  figures/fig12_city_slope.{pdf,png}
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

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
    "axes.titlesize": 16,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.dpi": 130,
})

ARCHS = ["resnet50", "vit_b16", "dinov2_frozen"]
ARCH_LABELS = {"resnet50": "ResNet", "vit_b16": "ViT", "dinov2_frozen": "DINOv2"}

CONTINENT_COLOR = {
    "North America": "#9aa0a6",
    "Asia": "#7a3f99",
    "South America": "#c0392b",
    "Europe": "#1f77b4",
    "Africa": "#2ca02c",
    "Oceania": "#e67e22",
}


def short_name(city_proxy: str, cc: str) -> str:
    return f"{city_proxy.split(',')[0].strip()} ({cc})"


def main() -> None:
    df = pd.read_parquet(DATA / "tier2_per_city.parquet")

    agg = (df.groupby(["arch", "city_proxy", "cc", "continent"])
             .agg(rel_std=("median_rel_std", "mean"), n_images=("n_images", "min"))
             .reset_index())
    agg = agg[agg["n_images"] >= 30]

    piv = agg.pivot_table(index=["city_proxy", "cc", "continent"],
                          columns="arch", values="rel_std")
    piv = piv.dropna(subset=ARCHS)

    ranks = pd.DataFrame({a: piv[a].rank(ascending=False, method="min") for a in ARCHS},
                         index=piv.index)
    N = len(ranks)
    print(f"N cities (>=30 imgs, common to 3 archs): {N}")

    # highlight: top-5 in any arch OR bottom-3 in any arch
    TOP_K = 5
    BOT_K = 3
    top_set = set()
    bot_set = set()
    for a in ARCHS:
        top_set |= set(ranks[ranks[a] <= TOP_K].index)
        bot_set |= set(ranks[ranks[a] >= N - (BOT_K - 1)].index)

    fig, ax = plt.subplots(figsize=(13.0, 9.0))
    xs = np.arange(len(ARCHS))

    # background lines
    for idx in ranks.index:
        if idx in top_set or idx in bot_set:
            continue
        ys = [ranks.loc[idx, a] for a in ARCHS]
        ax.plot(xs, ys, color="#dddddd", lw=0.6, alpha=0.7, zorder=1)

    # draw highlighted lines + dots
    def draw_lines(group: set, lw: float, dot_size: float, alpha: float) -> None:
        for idx in group:
            cont = idx[2]
            color = CONTINENT_COLOR.get(cont, "#444444")
            ys = [ranks.loc[idx, a] for a in ARCHS]
            ax.plot(xs, ys, color=color, lw=lw, alpha=alpha, zorder=3,
                    solid_capstyle="round")
            ax.scatter(xs, ys, color=color, s=dot_size, edgecolor="white",
                       lw=0.9, zorder=4)

    draw_lines(top_set, lw=2.2, dot_size=55, alpha=0.92)
    draw_lines(bot_set, lw=1.5, dot_size=35, alpha=0.65)

    # ---- LABEL PLACEMENT (no collisions) ----
    # For each highlighted city, decide side:
    #   peaks at ResNet (i=0) OR ViT (i=1) → LEFT
    #   peaks at DINOv2 only (i=2)         → RIGHT
    # Anchor at the arch where the city is at its extreme rank.
    # Stagger vertical positions within each side to avoid overlap.

    def classify_and_place(group: set, is_top: bool):
        left, right = [], []
        for idx in group:
            ys = [ranks.loc[idx, a] for a in ARCHS]
            extreme_ai = int(np.argmin(ys)) if is_top else int(np.argmax(ys))
            y_anchor = ys[extreme_ai]
            if extreme_ai == 2:
                right.append((idx, y_anchor, extreme_ai))
            else:
                left.append((idx, y_anchor, extreme_ai))
        return left, right

    def stagger(items, min_spacing: float):
        # items: list of (idx, y_anchor, peak_ai), sort by y, force vertical spacing
        items_sorted = sorted(items, key=lambda x: x[1])
        out = []
        last_y = -1e9
        for idx, y_anchor, peak_ai in items_sorted:
            ny = max(y_anchor, last_y + min_spacing)
            out.append((idx, y_anchor, ny, peak_ai))
            last_y = ny
        return out

    def emit_labels(placed, side: str, fontsize: float, label_x_offset: float):
        for idx, y_anchor, y_label, peak_ai in placed:
            cont = idx[2]
            color = CONTINENT_COLOR.get(cont, "#444444")
            name = short_name(idx[0], idx[1])

            x_dot = xs[peak_ai]
            if side == "left":
                x_label = xs[0] - label_x_offset
                ha = "right"
            else:
                x_label = xs[-1] + label_x_offset
                ha = "left"

            # leader line from label to its dot
            ax.plot([x_label + (0.04 if side == "left" else -0.04), x_dot],
                    [y_label, y_anchor],
                    color=color, lw=0.6, alpha=0.55, zorder=2)
            ax.text(x_label, y_label, name, ha=ha, va="center",
                    fontsize=fontsize, color=color)

    top_left, top_right = classify_and_place(top_set, is_top=True)
    bot_left, bot_right = classify_and_place(bot_set, is_top=False)

    top_left = stagger(top_left, min_spacing=2.0)
    top_right = stagger(top_right, min_spacing=2.0)
    bot_left = stagger(bot_left, min_spacing=2.0)
    bot_right = stagger(bot_right, min_spacing=2.0)

    emit_labels(top_left, "left", fontsize=10.5, label_x_offset=0.18)
    emit_labels(top_right, "right", fontsize=10.5, label_x_offset=0.18)
    emit_labels(bot_left, "left", fontsize=9.5, label_x_offset=0.18)
    emit_labels(bot_right, "right", fontsize=9.5, label_x_offset=0.18)

    ax.set_xticks(xs)
    ax.set_xticklabels([ARCH_LABELS[a] for a in ARCHS])
    ax.set_xlim(-1.55, len(ARCHS) - 1 + 1.55)
    ax.invert_yaxis()
    ax.set_ylim(N + 4, -4)
    ax.set_ylabel("City rank by reproducibility\n(1 = most unstable, larger = more stable)")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    ax.axhline(TOP_K + 0.5, color="#bbbbbb", lw=0.6, ls=":", zorder=0)
    ax.axhline(N - BOT_K + 0.5, color="#bbbbbb", lw=0.6, ls=":", zorder=0)

    handles = [plt.Line2D([0], [0], color=c, lw=2.5)
               for c in CONTINENT_COLOR.values()]
    labels = list(CONTINENT_COLOR.keys())
    ax.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.01),
              ncol=6, frameon=False, fontsize=12,
              handlelength=1.5, handletextpad=0.4, columnspacing=1.5)

    # cross-arch Spearman ρ inset
    spearman = piv[ARCHS].rank(ascending=False).corr(method="spearman")
    ρ_text = (
        "Cross-arch Spearman ρ (n=70):\n"
        f"  ResNet · ViT   = {spearman.loc['resnet50','vit_b16']:.2f}\n"
        f"  ResNet · DINOv2 = {spearman.loc['resnet50','dinov2_frozen']:.2f}\n"
        f"  ViT · DINOv2   = {spearman.loc['vit_b16','dinov2_frozen']:.2f}"
    )
    ax.text(0.98, 0.34, ρ_text, transform=ax.transAxes,
            ha="right", va="top", fontsize=10.5, color="#444",
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                      edgecolor="#cccccc", lw=0.5))

    fig.tight_layout()
    fig.savefig(FIGS / "fig12_city_slope.pdf", bbox_inches="tight")
    fig.savefig(FIGS / "fig12_city_slope.png", bbox_inches="tight", dpi=160)
    plt.close(fig)
    print("saved figures/fig12_city_slope.{pdf,png}")


if __name__ == "__main__":
    main()
