# Per-Seed Predictions

This directory is populated by unpacking `polygeovision_predictions_328runs.tar.gz` from the repository root. It contains 328 per-seed test prediction files for the runs reported in the paper:

- `exp1_*_test_preds.npz`: main multi-seed analysis
- `exp2_*_test_preds.npz`: source-decomposition analysis
- `multisplit_*_test_preds.npz`: multi-split robustness analysis

Each NumPy archive contains `image_ids`, `preds`, `target_mean`, `target_std`, and `dimensions`. Pilot and exploratory prediction files are not included.

The package is a generated research artifact and is not tracked in Git. Keep this README and `.gitkeep` when replacing or rebuilding the package.

Download DOI: https://doi.org/10.5281/zenodo.20561660
