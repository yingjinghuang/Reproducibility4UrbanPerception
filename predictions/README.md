# Per-Seed Predictions

This directory is the target location for `polygeovision_predictions_328runs.tar.gz`. After unpacking, it contains 328 per-seed test prediction files for the runs reported in the paper:

- `exp1_*_test_preds.npz`: main multi-seed analysis
- `exp2_*_test_preds.npz`: source-decomposition analysis
- `multisplit_*_test_preds.npz`: multi-split robustness analysis

Each file stores model predictions and the corresponding test image IDs. Pilot and exploratory prediction files are not included.

Download DOI: https://doi.org/10.5281/zenodo.20561660
