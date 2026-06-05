"""Generate the aggregate-vs-local stability figure."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

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
    "axes.titlesize": 21,
    "axes.labelsize": 21,
    "xtick.labelsize": 19,
    "ytick.labelsize": 19,
    "legend.fontsize": 17,
    "figure.dpi": 130,
})

ARCH_COLORS = {
    "resnet50": "#1f77b4",
    "vit_b16": "#d62728",
    "dinov2_frozen": "#2ca02c",
}
ARCH_LABELS = {
    "resnet50": "ResNet-50",
    "vit_b16": "ViT-B/16",
    "dinov2_frozen": "DINOv2 frozen",
}


def main() -> None:
    ladder = json.loads((DATA / "scale_ladder.json").read_text())
    dist = np.load(DATA / "scale_ladder_distributions.npz")

    scale_order = ["aggregate", "continent", "country", "city", "image"]
    scale_labels = ["Aggregate", "Continent", "Country", "Place", "Image"]
    archs = ["resnet50", "vit_b16", "dinov2_frozen"]
    box_width = 0.22
    x_scale = np.arange(len(scale_order))

    fig, ax = plt.subplots(1, 1, figsize=(8.5, 4.6))
    medians_by_arch = {arch: [] for arch in archs}

    for scale_idx, scale in enumerate(scale_order):
        for arch_idx, arch in enumerate(archs):
            values = dist[f"{arch}__{scale}"]
            xpos = x_scale[scale_idx] + (arch_idx - 1) * box_width
            ax.boxplot(
                values,
                positions=[xpos],
                widths=box_width * 0.85,
                patch_artist=True,
                showfliers=False,
                whis=(5, 95),
                medianprops={"color": "black", "linewidth": 1.4},
                boxprops={
                    "facecolor": ARCH_COLORS[arch],
                    "alpha": 0.55,
                    "edgecolor": ARCH_COLORS[arch],
                },
                whiskerprops={"color": ARCH_COLORS[arch], "linewidth": 1.2},
                capprops={"color": ARCH_COLORS[arch], "linewidth": 1.2},
            )
            medians_by_arch[arch].append(np.median(values))

    for arch in archs:
        offset = (archs.index(arch) - 1) * box_width
        ax.plot(
            x_scale + offset,
            medians_by_arch[arch],
            marker="o",
            linewidth=1.8,
            markersize=6,
            color=ARCH_COLORS[arch],
            label=ARCH_LABELS[arch],
            zorder=5,
        )

    total_images = ladder["vit_b16"]["n_images"]
    n_per_unit = {
        scale: total_images / max(ladder["vit_b16"]["ladder"][scale]["n_units"], 1)
        for scale in scale_order
    }
    vit_image = ladder["vit_b16"]["ladder"]["image"]["median_rel_std"]
    reference = [vit_image / np.sqrt(n_per_unit[scale]) for scale in scale_order]
    ax.plot(
        x_scale,
        reference,
        "k:",
        linewidth=1.2,
        alpha=0.6,
        label="1/sqrt(N) reference\n(from image scale)",
    )

    ax.set_xticks(x_scale)
    ax.set_xticklabels(scale_labels)
    ax.set_xlim(-0.6, len(scale_order) - 0.4)
    ax.set_yscale("log")
    ax.set_ylabel(r"Seed-induced $\sigma_{\mathrm{rel}}$")
    ax.legend(loc="lower right", fontsize=14, framealpha=0.9)
    ax.grid(alpha=0.3, which="both", axis="y")

    fig.tight_layout()
    fig.savefig(FIGS / "fig1_aggregate_vs_perimage.pdf", bbox_inches="tight")
    fig.savefig(FIGS / "fig1_aggregate_vs_perimage.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("saved figures/fig1_aggregate_vs_perimage.{pdf,png}")


if __name__ == "__main__":
    main()
