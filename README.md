# Investigating the Multi-Scale Reproducibility of GeoAI Models for Urban Perception

This repository provides the code and data-package instructions for the SIGSPATIAL study, "Investigating the Multi-Scale Reproducibility of GeoAI Models for Urban Perception."

The release contains the training pipeline, evaluation scripts, figure-generation code, and instructions for using the accompanying prediction and metadata packages. Raw Place Pulse 2.0 imagery, manuscript source files, and trained model checkpoints are not included.

## Data Packages

The per-seed predictions and derived metadata are distributed separately from the git repository to keep the repository lightweight.

Zenodo DOI: [10.5281/zenodo.20561660](https://doi.org/10.5281/zenodo.20561660)

The release uses two downloadable packages:

- `polygeovision_predictions_328runs.tar.gz`: per-seed test predictions for the 328 reported training runs.
- `polygeovision_metadata.tar.gz`: derived metadata and analysis tables used by the evaluation scripts.

Place both files in the repository root and unpack them:

```bash
tar -xzf polygeovision_predictions_328runs.tar.gz
tar -xzf polygeovision_metadata.tar.gz
```

After unpacking, `predictions/` contains the 328 prediction files used in the study:

- 160 main multi-seed runs: ResNet-50, ViT-B/16, and frozen DINOv2.
- 120 source-decomposition runs: ResNet-50 and ViT-B/16.
- 48 multi-split robustness runs: ResNet-50 and ViT-B/16 on three alternative splits.

After unpacking, `data_processed/` contains the derived inputs needed for analysis: TrueSkill Q-scores, split assignments, reverse-geocoded spatial hierarchy, image-level covariates, per-image and per-place stability summaries, multi-scale decomposition outputs, ensembling summaries, multi-split robustness outputs, and ICC robustness outputs.

Raw Place Pulse 2.0 imagery is not redistributed. To rerun the full pipeline from raw data, obtain Place Pulse 2.0 from the official source and place it in the expected local layout.

## Repository Layout

```text
.
├── src/polygeo/                 reusable package code
├── scripts/
│   ├── data_preparation/        PP2.0 preprocessing, geocoding, splits, covariates
│   ├── training/                training and robustness-sweep launchers
│   ├── analysis/                stability, variance, ensembling, and ICC analyses
│   └── figures/                 figure-generation scripts
├── data_processed/              destination for the derived metadata package
├── predictions/                 destination for the per-seed prediction package
├── figures/                     generated figures
├── checkpoints/                 local output directory for trained model weights
├── runs/                        local output directory for run logs and summaries
├── environment.yml
└── requirements.txt
```

`checkpoints/`, `runs/`, and `figures/` are local output directories. They are ignored by git except for placeholder files.

## Environment

Create the conda environment:

```bash
conda env create -f environment.yml
conda activate perception_stable
```

Alternatively, install from `requirements.txt`:

```bash
python -m pip install -r requirements.txt
```

Paths are resolved relative to the repository root by default. If running scripts from another working directory, set:

```bash
export POLYGEO_ROOT=/path/to/repository
```

## Reproducing the Analyses

If the prediction and metadata packages have been unpacked, the analysis scripts can be run without retraining the models.

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
python scripts/analysis/29_multisplit_analysis.py
python scripts/analysis/30_icc_robustness.py
```

Figure generation:

```bash
python scripts/figures/15_figures.py
python scripts/figures/23_multiscale_figure.py
python scripts/analysis/17_exp2_analysis.py
python scripts/figures/26_fig4_ensembling.py
```

The optional city-slope supporting figure can be regenerated with:

```bash
python scripts/figures/25_city_slope.py
```

## Rerunning Training

Training requires the raw PP2.0 imagery and a CUDA-capable environment.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/training/11_exp1_main_sweep.py
CUDA_VISIBLE_DEVICES=0 python scripts/training/16_exp2_decompose.py
CUDA_VISIBLE_DEVICES=0 python scripts/training/16b_exp2_decompose_resnet.py
CUDA_VISIBLE_DEVICES=0 python scripts/training/28_multisplit_sweep.py
```

The training scripts write run summaries to `runs/`, model weights to `checkpoints/`, and predictions to `predictions/`.

## Rerunning Data Preparation

These scripts require the raw Place Pulse 2.0 files.

```bash
python scripts/data_preparation/01_build_image_table.py
python scripts/data_preparation/02_reverse_geocode.py
python scripts/data_preparation/03_fit_trueskill.py
python scripts/data_preparation/04_label_dispersion.py
python scripts/data_preparation/05_make_splits.py
python scripts/data_preparation/06_smoke_test_dataloader.py
python scripts/data_preparation/07_image_complexity.py
```

## Citation

The manuscript citation will be added after publication.

For now, the prediction and metadata packages can be cited through the Zenodo DOI listed above.
