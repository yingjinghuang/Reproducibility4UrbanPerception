"""Generate the ensembling mitigation figure."""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from polygeo.paths import ROOT, DATA


FIGS = ROOT / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

if os.environ.get("POLYGEO_FONT_DIR"):
    for font_file in ("arial.ttf", "arialbd.ttf", "ariali.ttf"):
        font_path = Path(os.environ["POLYGEO_FONT_DIR"]) / font_file
        if font_path.exists():
            fm.fontManager.addfont(str(font_path))

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 15,
    "figure.dpi": 130,
})

MIN_IMG_PER_PLACE = 30
ALL_NS = [1, 2, 4, 8, 16, 32]
ARCH_NS = {
    "vit_b16": [1, 2, 4, 8, 16, 32],
    "resnet50": [1, 2, 4, 8, 16],
    "dinov2_frozen": [1, 2, 4, 8, 16],
}
ARCH_LABELS = {
    "vit_b16": "ViT-B/16",
    "resnet50": "ResNet-50",
    "dinov2_frozen": "DINOv2 frozen",
}
ARCH_COLORS = {
    "vit_b16": "#5b3a98",
    "resnet50": "#1f77b4",
    "dinov2_frozen": "#2ca02c",
}
ARCH_MARKERS = {
    "vit_b16": "o",
    "resnet50": "s",
    "dinov2_frozen": "^",
}
ARCH_ORDER = ["vit_b16", "resnet50", "dinov2_frozen"]

THRESHOLDS = [0.05, 0.10, 0.15]
THRESHOLD_COLORS = {0.05: "#c0392b", 0.10: "#e8a444", 0.15: "#888888"}
COVERAGE_THRESHOLD = 0.10


def main() -> None:
    long = pd.read_parquet(DATA / "exp3a_ensemble_curve.parquet")
    long = long[long["ensemble_size"].isin(ALL_NS)].copy()

    med_by_arch = {}
    for arch in ARCH_ORDER:
        sub = long[long["arch"] == arch]
        med_by_arch[arch] = [
            float(sub[sub["ensemble_size"] == n]["rel_std"].median())
            for n in ARCH_NS[arch]
        ]

    n_images = (
        long[(long["arch"] == "vit_b16") & (long["ensemble_size"] == 1)]
        .groupby("city_proxy")["image_id"]
        .nunique()
    )
    places = n_images[n_images >= MIN_IMG_PER_PLACE].index.tolist()

    coverage_by_arch = {}
    for arch in ARCH_ORDER:
        per_place = (
            long[(long["arch"] == arch) & (long["city_proxy"].isin(places))]
            .groupby(["ensemble_size", "city_proxy"])["rel_std"]
            .median()
            .reset_index()
        )
        coverage_by_arch[arch] = [
            float((per_place[per_place["ensemble_size"] == n]["rel_std"] < COVERAGE_THRESHOLD).mean() * 100)
            for n in ARCH_NS[arch]
        ]

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8))

    ax = axes[0]
    for threshold in THRESHOLDS:
        ax.axhline(
            threshold,
            color=THRESHOLD_COLORS[threshold],
            linewidth=1.0,
            linestyle="--",
            alpha=0.7,
            zorder=2,
        )
        ax.text(
            ALL_NS[-1] * 1.08,
            threshold,
            f"{int(threshold * 100)}%",
            color=THRESHOLD_COLORS[threshold],
            fontsize=13,
            fontweight="bold",
            va="center",
            ha="left",
        )

    for arch in ARCH_ORDER:
        ns = ARCH_NS[arch]
        ys = med_by_arch[arch]
        ax.plot(
            ns,
            ys,
            marker=ARCH_MARKERS[arch],
            color=ARCH_COLORS[arch],
            linewidth=2.2,
            markersize=8,
            zorder=4,
            label=ARCH_LABELS[arch],
        )
        ax.annotate(
            f"{ys[-1]:.3f}",
            (ns[-1], ys[-1]),
            xytext=(8, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=11,
            color=ARCH_COLORS[arch],
        )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Ensemble size $N$")
    ax.set_ylabel(r"Per-image median $\sigma_{\mathrm{rel}}$")
    ax.set_title("(a)", loc="left", fontsize=19, fontweight="bold", pad=14)
    ax.set_xticks(ALL_NS)
    ax.set_xticklabels([f"$2^{{{int(np.log2(n))}}}$" for n in ALL_NS])
    ax.set_xlim(ALL_NS[0] * 0.85, ALL_NS[-1] * 1.55)
    ax.legend(loc="lower left", fontsize=14, frameon=False, handletextpad=0.4)
    ax.grid(alpha=0.3, which="both")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    ax = axes[1]
    for arch in ARCH_ORDER:
        ns = ARCH_NS[arch]
        coverage = coverage_by_arch[arch]
        ax.plot(
            ns,
            coverage,
            marker=ARCH_MARKERS[arch],
            color=ARCH_COLORS[arch],
            linewidth=2.2,
            markersize=8,
            label=ARCH_LABELS[arch],
        )
        for n, y in zip(ns, coverage):
            if 0.5 < y < 99.5:
                ax.annotate(
                    f"{y:.0f}",
                    (n, y),
                    xytext=(0, 9),
                    textcoords="offset points",
                    ha="center",
                    fontsize=11,
                    color=ARCH_COLORS[arch],
                )

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Ensemble size $N$")
    ax.set_ylabel(
        f"% of places with median $\\sigma_{{\\mathrm{{rel}}}} < {int(COVERAGE_THRESHOLD * 100)}\\%$"
    )
    ax.set_title("(b)", loc="left", fontsize=19, fontweight="bold", pad=14)
    ax.set_xticks(ALL_NS)
    ax.set_xticklabels([f"$2^{{{int(np.log2(n))}}}$" for n in ALL_NS])
    ax.set_xlim(ALL_NS[0] * 0.85, ALL_NS[-1] * 1.15)
    ax.set_ylim(-4, 106)
    ax.legend(loc="lower right", fontsize=14, frameon=False, handletextpad=0.4)
    ax.grid(alpha=0.3, which="both")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    fig.savefig(FIGS / "fig4_ensembling.pdf", bbox_inches="tight")
    fig.savefig(FIGS / "fig4_ensembling.png", bbox_inches="tight", dpi=160)
    plt.close(fig)
    print("saved figures/fig4_ensembling.{pdf,png}")


if __name__ == "__main__":
    main()
