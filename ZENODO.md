# Zenodo Release Checklist

The Zenodo deposit contains generated research artifacts, while the portable code remains in the GitHub repository.

## Build the Upload Files

From a standalone checkout whose `predictions/` and `data_processed/` directories are populated:

```bash
python scripts/package_zenodo.py
```

For the development layout where this repository is checked out as `release/` inside the research workspace:

```bash
python scripts/package_zenodo.py \
  --source-root .. \
  --output-dir ../release_assets
```

The command validates file counts, NPZ integrity, JSON syntax, and Parquet magic bytes before producing:

- `polygeovision_predictions_328runs.tar.gz` — 328 selected prediction files
- `polygeovision_metadata.tar.gz` — 45 selected metadata and analysis files
- `zenodo_manifest.json` — per-package and per-file sizes and SHA-256 hashes
- `SHA256SUMS` — checksums for the two packages and manifest

The selection intentionally excludes pilot and exploratory prediction files. The package archives are deterministic: identical inputs produce identical archive checksums.

## Verify Before Uploading

```bash
cd release_assets
sha256sum -c SHA256SUMS
tar -tzf polygeovision_predictions_328runs.tar.gz | head
tar -tzf polygeovision_metadata.tar.gz | head
```

Upload all four generated files, not the `release_assets/` directory itself.

## Complete the Zenodo Draft

1. If this dataset already has a published Zenodo record, use **New version** from that record so the new files remain linked to the existing concept DOI.
2. Use the dataset resource type and describe the two packages separately in the record description.
3. Add all creators and affiliations, the release date and version, the intended data license, and the GitHub repository as a related identifier.
4. Upload the four generated files and compare their displayed sizes with `zenodo_manifest.json`.
5. Save the draft, preview the landing page, test the GitHub and DOI links, and publish only after the checks pass.

Zenodo documentation:

- [Create a new upload](https://help.zenodo.org/docs/deposit/create-new-upload/)
- [Manage versions](https://help.zenodo.org/docs/deposit/manage-versions/)
- [Describe records](https://help.zenodo.org/docs/deposit/describe-records/)
