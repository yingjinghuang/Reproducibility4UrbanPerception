"""Figure 4: Ensembling across all three architectures.

Two panels, both using the 70 core places from Section 4.2
(>= 30 test images each), with one line per architecture:
  (a) per-image median rel_std vs N, log-log, with horizontal threshold
      lines at rel_std = 5%, 10%, 15%. Read off the minimum N each
      architecture needs to hit any target threshold.
  (b) % of core places whose local instability profile m_u falls below
      the 10% operational tolerance, vs N.

ViT-B/16 has a pool of 80 seeds (N up to 32); ResNet-50 and DINOv2 frozen
have pools of 40 (N up to 16). Values must come from the corrected B=5,000
bootstrap analysis with replacement.

Inputs:
  data_processed/exp3a_ensemble_curve.parquet   (now carries `arch` col)
Output:
  figures/fig4_ensembling.{pdf,png}
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from polygeo.paths import DATA, ROOT

FIGS = ROOT / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
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
    "vit_b16":       [1, 2, 4, 8, 16, 32],
    "resnet50":      [1, 2, 4, 8, 16],
    "dinov2_frozen": [1, 2, 4, 8, 16],
}
ARCH_LABEL = {"vit_b16": "ViT-B/16", "resnet50": "ResNet-50", "dinov2_frozen": "DINOv2 (frozen)"}
ARCH_COLOR = {"vit_b16": "#5b3a98", "resnet50": "#1f77b4", "dinov2_frozen": "#2ca02c"}
ARCH_MARKER = {"vit_b16": "o", "resnet50": "s", "dinov2_frozen": "^"}
ARCHS_ORDER = ["vit_b16", "resnet50", "dinov2_frozen"]

THRESHOLDS_A = [0.05, 0.10, 0.15]
THRESHOLD_COLOR_A = {0.05: "#c0392b", 0.10: "#e8a444", 0.15: "#888888"}

COVERAGE_THR = 0.10  # canonical engineering target for panel (b)


def main() -> None:
    long = pd.read_parquet(DATA / "exp3a_ensemble_curve.parquet")
    required_columns = {"sampling_method", "B", "bootstrap_seed", "is_core_place"}
    missing = required_columns.difference(long.columns)
    if missing:
        raise ValueError(f"ensemble curve lacks corrected-analysis metadata: {sorted(missing)}")
    if set(long["sampling_method"].unique()) != {"bootstrap_with_replacement"}:
        raise ValueError("Figure 4 must use only with-replacement bootstrap results")
    if set(long["B"].unique()) != {5000}:
        raise ValueError("Figure 4 expects the camera-ready B=5,000 results")
    if set(long["bootstrap_seed"].unique()) != {20260817}:
        raise ValueError("Figure 4 expects the documented bootstrap seed 20260817")
    long = long[long["ensemble_size"].isin(ALL_NS)].copy()

    # ---- per-image median rel_std vs N, per arch ----
    med_by_arch = {}
    for arch in ARCHS_ORDER:
        ns = ARCH_NS[arch]
        sub = long[long["arch"] == arch]
        med_by_arch[arch] = [float(sub[sub["ensemble_size"] == N]["rel_std"].median()) for N in ns]
    print("per-image median rel_std vs N:")
    for arch in ARCHS_ORDER:
        ns = ARCH_NS[arch]
        vals = med_by_arch[arch]
        print(f"  {ARCH_LABEL[arch]:18s}: " + "  ".join(f"N={N}:{v:.4f}" for N, v in zip(ns, vals)))

    # ---- canonical 70-core-place coverage at COVERAGE_THR ----
    core_places = sorted(long.loc[long["is_core_place"], "city_proxy"].unique())
    n_places = len(core_places)
    if n_places != 70:
        raise ValueError(f"expected 70 core places, found {n_places}")
    print(f"\n{n_places} core places (>= {MIN_IMG_PER_PLACE} test images)")

    coverage_by_arch = {}
    for arch in ARCHS_ORDER:
        ns = ARCH_NS[arch]
        per_city = (long[(long["arch"] == arch) & (long["city_proxy"].isin(core_places))]
                    .groupby(["ensemble_size", "city_proxy"])["rel_std"]
                    .median().reset_index())
        cov = []
        for N in ns:
            v = per_city[per_city["ensemble_size"] == N]["rel_std"].values
            cov.append(float((v < COVERAGE_THR).mean() * 100))
        coverage_by_arch[arch] = cov

    print(f"\ncore-place coverage (% of {n_places} places with m_u < {COVERAGE_THR:.0%}):")
    for arch in ARCHS_ORDER:
        ns = ARCH_NS[arch]
        print(f"  {ARCH_LABEL[arch]:18s}: " + "  ".join(f"N={N}:{c:5.1f}" for N, c in zip(ns, coverage_by_arch[arch])))

    # ---- render: 2 panels ----
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8))

    # Panel (a): decay curves, one per arch, with threshold reference lines
    ax = axes[0]
    for thr in THRESHOLDS_A:
        ax.axhline(thr, color=THRESHOLD_COLOR_A[thr], lw=1.0, ls="--", alpha=0.7, zorder=2)
        ax.text(ALL_NS[-1] * 1.08, thr,
                f"{int(thr * 100)}%",
                color=THRESHOLD_COLOR_A[thr], fontsize=13, fontweight="bold",
                va="center", ha="left")
    for arch in ARCHS_ORDER:
        ns = ARCH_NS[arch]
        ys = med_by_arch[arch]
        ax.plot(ns, ys, marker=ARCH_MARKER[arch], color=ARCH_COLOR[arch],
                lw=2.2, markersize=8, zorder=4, label=ARCH_LABEL[arch])
        # annotate the endpoint
        endpoint_offset = -11 if arch == "vit_b16" else 0
        ax.annotate(f"{ys[-1]:.3f}", (ns[-1], ys[-1]), xytext=(8, endpoint_offset),
                    textcoords="offset points", ha="left", va="center",
                    fontsize=11, color=ARCH_COLOR[arch])
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Ensemble size $N$")
    ax.set_ylabel(r"Per-image median $\sigma_{\mathrm{rel}}$")
    ax.set_title("(a)", loc="left", fontsize=19, fontweight="bold", pad=14)
    ax.set_xticks(ALL_NS)
    ax.set_xticklabels([f"$2^{{{int(np.log2(N))}}}$" for N in ALL_NS])
    ax.set_xlim(ALL_NS[0] * 0.85, ALL_NS[-1] * 1.55)
    ax.legend(loc="lower left", fontsize=14, frameon=False, handletextpad=0.4)
    ax.grid(alpha=0.3, which="both")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # Panel (b): core-place coverage at COVERAGE_THR, one line per arch
    ax = axes[1]
    for arch in ARCHS_ORDER:
        ns = ARCH_NS[arch]
        cov = coverage_by_arch[arch]
        ax.plot(ns, cov, marker=ARCH_MARKER[arch], color=ARCH_COLOR[arch],
                lw=2.2, markersize=8, label=ARCH_LABEL[arch])
        for N, y in zip(ns, cov):
            if y <= 0.5 or y >= 99.5:
                continue
            ax.annotate(f"{y:.0f}", (N, y), xytext=(0, 9),
                        textcoords="offset points", ha="center",
                        fontsize=11, color=ARCH_COLOR[arch])
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Ensemble size $N$")
    ax.set_ylabel(f"% of core places with $m_u < {int(COVERAGE_THR*100)}\\%$")
    ax.set_title("(b)", loc="left", fontsize=19, fontweight="bold", pad=14)
    ax.set_xticks(ALL_NS)
    ax.set_xticklabels([f"$2^{{{int(np.log2(N))}}}$" for N in ALL_NS])
    ax.set_xlim(ALL_NS[0] * 0.85, ALL_NS[-1] * 1.15)
    ax.set_ylim(-4, 106)
    ax.legend(loc="lower right", fontsize=14, frameon=False, handletextpad=0.4)
    ax.grid(alpha=0.3, which="both")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    fig.tight_layout()
    fig.savefig(FIGS / "fig4_ensembling.pdf", bbox_inches="tight")
    fig.savefig(FIGS / "fig4_ensembling.png", bbox_inches="tight", dpi=160)
    plt.close(fig)
    print("\nsaved corrected figures/fig4_ensembling.{pdf,png}")


if __name__ == "__main__":
    main()
