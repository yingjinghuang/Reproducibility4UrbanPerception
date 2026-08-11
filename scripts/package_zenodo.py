"""Build deterministic Zenodo data packages for the reproducibility release.

The script packages only the explicitly enumerated study artifacts. It avoids
accidentally including pilot predictions, caches, or unrelated working files.
It uses only the Python standard library so it can validate and package an
existing result tree before the project environment is installed.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import tarfile
import tempfile
import zipfile
from pathlib import Path


PREDICTION_GROUPS = (
    ("exp1_vit_b16_imagenet21k_seed*_test_preds.npz", 80),
    ("exp1_resnet50_imagenet21k_seed*_test_preds.npz", 40),
    ("exp1_dinov2_frozen_imagenet21k_seed*_test_preds.npz", 40),
    ("exp2_init_dropout_vit_b16_*_test_preds.npz", 20),
    ("exp2_data_vit_b16_*_test_preds.npz", 20),
    ("exp2_aug_vit_b16_*_test_preds.npz", 20),
    ("exp2_init_dropout_resnet50_*_test_preds.npz", 20),
    ("exp2_data_resnet50_*_test_preds.npz", 20),
    ("exp2_aug_resnet50_*_test_preds.npz", 20),
    ("multisplit_split0_vit_b16_seed*_test_preds.npz", 8),
    ("multisplit_split1_vit_b16_seed*_test_preds.npz", 8),
    ("multisplit_split2_vit_b16_seed*_test_preds.npz", 8),
    ("multisplit_split0_resnet50_seed*_test_preds.npz", 8),
    ("multisplit_split1_resnet50_seed*_test_preds.npz", 8),
    ("multisplit_split2_resnet50_seed*_test_preds.npz", 8),
)

METADATA_FILES = (
    "data_processed/exp1_summary.json",
    "data_processed/exp2_global_south_split.json",
    "data_processed/exp2_per_image_variance.parquet",
    "data_processed/exp2_resnet50_summary.json",
    "data_processed/exp2_summary.json",
    "data_processed/exp2_tier3_per_source.json",
    "data_processed/exp2_variance_partition.json",
    "data_processed/exp3a_ensemble_curve.parquet",
    "data_processed/exp3a_summary.json",
    "data_processed/exp3b_compare.json",
    "data_processed/exp3b_per_image_variance.parquet",
    "data_processed/exp3b_summary.json",
    "data_processed/icc_robustness.json",
    "data_processed/image_complexity.parquet",
    "data_processed/image_geo.parquet",
    "data_processed/image_table.parquet",
    "data_processed/label_dispersion_city.parquet",
    "data_processed/label_dispersion_image.parquet",
    "data_processed/multiscale_decomposition.json",
    "data_processed/multisplit_per_image.parquet",
    "data_processed/multisplit_robustness.json",
    "data_processed/multisplit_sweep_summary.json",
    "data_processed/pilot_per_image_variance.parquet",
    "data_processed/pilot_summary.json",
    "data_processed/pilot_variance_summary.json",
    "data_processed/qscores.parquet",
    "data_processed/scale_ladder.json",
    "data_processed/scale_ladder_distributions.npz",
    "data_processed/sigspatial_seed_budget_bootstrap.csv",
    "data_processed/sigspatial_seed_budget_summary.csv",
    "data_processed/splits.parquet",
    "data_processed/splits_alt0.parquet",
    "data_processed/splits_alt1.parquet",
    "data_processed/splits_alt2.parquet",
    "data_processed/tier1_per_image.parquet",
    "data_processed/tier1_summary.json",
    "data_processed/tier2_continent_split.json",
    "data_processed/tier2_global_south_split.json",
    "data_processed/tier2_hierarchical_posteriors.parquet",
    "data_processed/tier2_per_city.parquet",
    "data_processed/tier3_coefs.parquet",
    "data_processed/tier3_multiscale_results.json",
    "data_processed/tier3_results.json",
    "data_processed/tta_per_image.parquet",
    "data_processed/tta_vs_seed_summary.json",
)

EXPECTED_NPZ_MEMBERS = {
    "dimensions.npy",
    "image_ids.npy",
    "preds.npy",
    "target_mean.npy",
    "target_std.npy",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_npz(path: Path, expected_members: set[str] | None = None) -> None:
    try:
        with zipfile.ZipFile(path) as archive:
            bad_member = archive.testzip()
            if bad_member is not None:
                raise ValueError(f"CRC failure in {path}: {bad_member}")
            members = set(archive.namelist())
    except zipfile.BadZipFile as exc:
        raise ValueError(f"invalid NumPy archive: {path}") from exc

    if expected_members is not None and members != expected_members:
        missing = sorted(expected_members - members)
        extra = sorted(members - expected_members)
        raise ValueError(f"unexpected members in {path}: missing={missing}, extra={extra}")


def validate_parquet(path: Path) -> None:
    with path.open("rb") as handle:
        header = handle.read(4)
        handle.seek(-4, os.SEEK_END)
        footer = handle.read(4)
    if header != b"PAR1" or footer != b"PAR1":
        raise ValueError(f"invalid Parquet magic bytes: {path}")


def collect_predictions(source_root: Path) -> tuple[list[Path], list[str]]:
    prediction_dir = source_root / "predictions"
    selected: set[Path] = set()
    for pattern, expected_count in PREDICTION_GROUPS:
        matches = set(prediction_dir.glob(pattern))
        if len(matches) != expected_count:
            raise ValueError(
                f"{pattern}: expected {expected_count} files, found {len(matches)}"
            )
        overlap = selected & matches
        if overlap:
            raise ValueError(f"prediction patterns overlap: {sorted(overlap)}")
        selected.update(matches)

    if len(selected) != 328:
        raise ValueError(f"expected 328 selected predictions, found {len(selected)}")

    all_npz = set(prediction_dir.glob("*.npz"))
    excluded = sorted(path.name for path in all_npz - selected)
    files = sorted(selected)
    for path in files:
        validate_npz(path, EXPECTED_NPZ_MEMBERS)
    return files, excluded


def collect_metadata(source_root: Path) -> list[Path]:
    files = [source_root / relative for relative in METADATA_FILES]
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise ValueError(f"missing metadata files: {missing}")

    for path in files:
        if path.suffix == ".json":
            with path.open(encoding="utf-8") as handle:
                json.load(handle)
        elif path.suffix == ".parquet":
            validate_parquet(path)
        elif path.suffix == ".npz":
            validate_npz(path)
        elif path.stat().st_size == 0:
            raise ValueError(f"empty metadata file: {path}")
    return files


def add_directory(archive: tarfile.TarFile, name: str) -> None:
    info = tarfile.TarInfo(name)
    info.type = tarfile.DIRTYPE
    info.mode = 0o755
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    info.mtime = 0
    archive.addfile(info)


def add_file(archive: tarfile.TarFile, path: Path, arcname: str) -> None:
    info = tarfile.TarInfo(arcname)
    info.size = path.stat().st_size
    info.mode = 0o644
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    info.mtime = 0
    with path.open("rb") as handle:
        archive.addfile(info, handle)


def build_archive(output: Path, root_name: str, files: list[tuple[Path, str]]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as raw:
            with gzip.GzipFile(
                filename="", mode="wb", compresslevel=9, fileobj=raw, mtime=0
            ) as compressed:
                with tarfile.open(
                    fileobj=compressed, mode="w", format=tarfile.USTAR_FORMAT
                ) as archive:
                    add_directory(archive, root_name)
                    for path, arcname in sorted(files, key=lambda item: item[1]):
                        add_file(archive, path, arcname)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def file_record(path: Path, arcname: str) -> dict[str, object]:
    return {
        "path": arcname,
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def parse_args() -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=repository_root,
        help="tree containing predictions/ and data_processed/ (default: repository root)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repository_root / "release_assets",
        help="destination for Zenodo-ready files",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    output_dir = args.output_dir.resolve()

    predictions, excluded = collect_predictions(source_root)
    metadata = collect_metadata(source_root)

    prediction_items = [(path, f"predictions/{path.name}") for path in predictions]
    metadata_items = [
        (path, path.relative_to(source_root).as_posix()) for path in metadata
    ]

    prediction_archive = output_dir / "polygeovision_predictions_328runs.tar.gz"
    metadata_archive = output_dir / "polygeovision_metadata.tar.gz"
    build_archive(prediction_archive, "predictions", prediction_items)
    build_archive(metadata_archive, "data_processed", metadata_items)

    manifest = {
        "schema_version": 1,
        "packages": {
            prediction_archive.name: {
                "file_count": len(prediction_items),
                "size_bytes": prediction_archive.stat().st_size,
                "sha256": sha256(prediction_archive),
                "files": [file_record(path, arcname) for path, arcname in prediction_items],
            },
            metadata_archive.name: {
                "file_count": len(metadata_items),
                "size_bytes": metadata_archive.stat().st_size,
                "sha256": sha256(metadata_archive),
                "files": [file_record(path, arcname) for path, arcname in metadata_items],
            },
        },
        "excluded_prediction_files": excluded,
    }

    manifest_path = output_dir / "zenodo_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    checksum_paths = (prediction_archive, metadata_archive, manifest_path)
    checksum_path = output_dir / "SHA256SUMS"
    checksum_path.write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in checksum_paths),
        encoding="utf-8",
    )

    print(f"selected predictions: {len(predictions)}")
    print(f"excluded prediction artifacts: {len(excluded)}")
    print(f"selected metadata files: {len(metadata)}")
    for path in (*checksum_paths, checksum_path):
        print(f"wrote {path} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
