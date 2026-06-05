# PolyGeoVision Benchmark Artifacts

Benchmark and reproducibility artifact release for the urban visual perception reproducibility audit.

This repository contains the evaluation code for the accompanying SIGSPATIAL study. The per-seed predictions and spatial stability metadata are released as separate artifact archives, because keeping them in the git history would make the repository unnecessarily heavy. The repository does not include manuscript source, raw Place Pulse 2.0 imagery, or model checkpoints.

## Layout

```text
.
├── src/polygeo/                 reusable training and data-loading package
├── scripts/
│   ├── data_preparation/        raw PP2.0 processing, geocoding, splits, covariates
│   ├── training/                main training and robustness sweep launchers
│   ├── analysis/                stability, variance, ensembling, and ICC analyses
│   └── figures/                 figure-generation scripts
├── data_processed/              target directory for derived metadata artifacts
├── predictions/                 target directory for per-seed prediction artifacts
├── figures/                     generated figures, intentionally empty in git by default
├── checkpoints/                 generated model checkpoints, not released
├── runs/                        generated run summaries and logs, not released
├── environment.yml
└── requirements.txt
```

## Artifact Archives

Download or unpack the artifact archives into the repository root before running analyses:

- `polygeovision_predictions_328runs.tar.gz`: 328 per-seed `.npz` prediction files for reported runs.
- `polygeovision_metadata.tar.gz`: derived metadata and analysis outputs.

After unpacking, the `predictions/` directory contains files corresponding to the 328 runs reported in the paper:

- 160 main multi-seed runs: ResNet-50, ViT-B/16, and DINOv2 frozen.
- 120 source-decomposition runs: ResNet-50 and ViT-B/16.
- 48 multi-split robustness runs: ResNet-50 and ViT-B/16 on three alternative splits.

The unpacked `data_processed/` directory contains derived metadata and analysis outputs, including Q-scores, split assignments, reverse-geocoded spatial hierarchy, per-image covariates, per-image stability tables, multi-scale decomposition results, multi-split robustness summaries, and ICC robustness outputs.

Raw street-view imagery is not redistributed. To rerun data preparation or training from scratch, obtain Place Pulse 2.0 from the official source and place it in the expected local layout.

```bash
tar -xzf polygeovision_predictions_328runs.tar.gz
tar -xzf polygeovision_metadata.tar.gz
```

## Setup

```bash
conda env create -f environment.yml
conda activate perception_stable
```

If using pip instead:

```bash
python -m pip install -r requirements.txt
```

All paths are resolved relative to the repository root by default. Set `POLYGEO_ROOT=/path/to/repo` if running scripts from another working directory.

## Reproduction Workflow

Data preparation:

```bash
python scripts/data_preparation/01_build_image_table.py
python scripts/data_preparation/02_reverse_geocode.py
python scripts/data_preparation/03_fit_trueskill.py
python scripts/data_preparation/04_label_dispersion.py
python scripts/data_preparation/05_make_splits.py
python scripts/data_preparation/06_smoke_test_dataloader.py
python scripts/data_preparation/07_image_complexity.py
```

Main training:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/training/11_exp1_main_sweep.py
CUDA_VISIBLE_DEVICES=0 python scripts/training/16_exp2_decompose.py
CUDA_VISIBLE_DEVICES=0 python scripts/training/16b_exp2_decompose_resnet.py
```

Main analyses:

```bash
python scripts/analysis/12_tier1_variance.py
python scripts/analysis/13_tier2_geo.py
python scripts/analysis/14_tier3_regression.py
python scripts/analysis/14b_tier3_multiscale.py
python scripts/analysis/17_exp2_analysis.py
python scripts/analysis/18_exp3a_ensembling.py
python scripts/analysis/22_multiscale_geo.py
python scripts/analysis/24_scale_ladder.py
```

Robustness checks:

```bash
python scripts/data_preparation/27_multisplit_gen.py
CUDA_VISIBLE_DEVICES=0 python scripts/training/28_multisplit_sweep.py
python scripts/analysis/29_multisplit_analysis.py
python scripts/analysis/30_icc_robustness.py
```

Figure generation:

```bash
python scripts/figures/15_figures.py
python scripts/figures/23_multiscale_figure.py
python scripts/figures/26_fig4_ensembling.py
python scripts/figures/25_city_slope.py
```
