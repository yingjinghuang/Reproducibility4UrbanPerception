# Investigating the Multi-Scale Reproducibility of GeoAI Models for Urban Perception

> **Accepted as a full paper at ACM SIGSPATIAL 2026.** The final proceedings citation, paper DOI, and page range are forthcoming.

This repository is the public reproducibility companion for the SIGSPATIAL study, "Investigating the Multi-Scale Reproducibility of GeoAI Models for Urban Perception."

Project page: [yingjinghuang.github.io/Reproducibility4UrbanPerception](https://yingjinghuang.github.io/Reproducibility4UrbanPerception/)

It contains the portable training pipeline, analysis and figure scripts, and instructions for the accompanying data packages. Raw Place Pulse 2.0 imagery, manuscript source files, trained model checkpoints, and generated release archives are intentionally kept outside this Git repository.

## Quick Start

Clone the code and create the recorded Python 3.11 environment:

```bash
git clone https://github.com/yingjinghuang/Reproducibility4UrbanPerception.git
cd Reproducibility4UrbanPerception
conda env create -f environment.yml
conda activate perception_stable
```

Download the two data packages from Zenodo into the repository root, then unpack them:

```bash
tar -xzf polygeovision_predictions_328runs.tar.gz
tar -xzf polygeovision_metadata.tar.gz
```

The archives populate `predictions/` and `data_processed/`. The archive files and extracted data are ignored by Git. A first analysis or figure can then be run from the repository root:

```bash
python scripts/analysis/12_tier1_variance.py
python scripts/figures/15_figures.py
```

## Data Packages

The per-seed predictions and derived metadata are distributed separately from the git repository to keep the repository lightweight.

Zenodo DOI: [10.5281/zenodo.20561660](https://doi.org/10.5281/zenodo.20561660)

The release uses two downloadable packages:

| Package | Extracts to | Purpose |
| --- | --- | --- |
| `polygeovision_predictions_328runs.tar.gz` | `predictions/` | Per-seed test predictions for the 328 reported training runs |
| `polygeovision_metadata.tar.gz` | `data_processed/` | Derived metadata and analysis tables used by the evaluation scripts |

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

## Reproduction Workflow

The numbered script names preserve their provenance from the research workflow; the directories group them by task.

| Stage | Code | Required inputs | Main outputs |
| --- | --- | --- | --- |
| Data preparation | `scripts/data_preparation/` | Raw Place Pulse 2.0 data | Tables in `data_processed/` |
| Training | `scripts/training/` and `src/polygeo/` | Prepared tables and raw imagery | Predictions, run summaries, and checkpoints |
| Analysis | `scripts/analysis/` | Released predictions and metadata | Stability and robustness tables |
| Figures | `scripts/figures/` | Released or regenerated analysis tables | PDF and PNG files in `figures/` |

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
├── environment.yml              conda entry point (Python 3.11)
└── requirements.txt             pinned full environment
```

`data_processed/`, `predictions/`, `checkpoints/`, `runs/`, and `figures/` contain downloaded or generated artifacts. They are ignored by Git except for documentation and placeholder files.

## Environment

Create the conda environment:

```bash
conda env create -f environment.yml
conda activate perception_stable
```

Alternatively, install from `requirements.txt`:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

`requirements.txt` records the full CUDA 12.8 training environment and includes the PyTorch CUDA package index. Analysis scripts themselves do not train models, but using the recorded environment keeps the numerical stack consistent with the study.

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

## Repository Check

To verify that all released Python files parse without running the experiments:

```bash
python -m compileall -q src scripts
```

## Citation

The final ACM citation will be added after the proceedings metadata becomes available. Until then, use the following provisional citation:

```bibtex
@inproceedings{huang2026multiscale,
  author    = {Yingjing Huang and Krzysztof Janowicz and Mina Karimi and
               Zilong Liu and Songling Wang and Annika Suess and
               Alexandra Fortacz-Lazan},
  title     = {Investigating the Multi-Scale Reproducibility of {GeoAI}
               Models for Urban Perception},
  booktitle = {Proceedings of the 34th ACM SIGSPATIAL International
               Conference on Advances in Geographic Information Systems},
  year      = {2026},
  note      = {Accepted full paper; DOI forthcoming}
}
```

For now, the prediction and metadata packages can be cited through the Zenodo DOI listed above.
