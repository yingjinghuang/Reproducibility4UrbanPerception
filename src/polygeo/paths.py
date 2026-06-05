"""Canonical paths for the project.

By default paths are resolved relative to the repository root. Set
``POLYGEO_ROOT`` to override this when running scripts from another location.
"""
import os
from pathlib import Path

ROOT = Path(os.environ.get("POLYGEO_ROOT", Path(__file__).resolve().parents[2])).resolve()
DATA = ROOT / "data_processed"
IMAGES = DATA / "googleviews-110802files"

VOTES_CSV = ROOT / "placePulse/_unpack/rawData/votes.csv"
IMAGE_TABLE = DATA / "image_table.parquet"
IMAGE_GEO = DATA / "image_geo.parquet"
QSCORES = DATA / "qscores.parquet"
LABEL_DISP_IMG = DATA / "label_dispersion_image.parquet"
LABEL_DISP_CITY = DATA / "label_dispersion_city.parquet"
SPLITS = DATA / "splits.parquet"

CHECKPOINTS = ROOT / "checkpoints"
PREDICTIONS = ROOT / "predictions"
RUNS = ROOT / "runs"

DIMENSIONS = ("safety", "lively", "beautiful", "wealthy", "depressing", "boring")
