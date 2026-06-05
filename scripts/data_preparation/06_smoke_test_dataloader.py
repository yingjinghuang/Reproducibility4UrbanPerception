"""Smoke test: build datasets, iterate a few batches, dump shape summary."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from torch.utils.data import DataLoader

from polygeo.data import QScoreDataset, PairwiseDataset, masked_mse_loss, build_metadata_table


def main() -> None:
    print("=== building metadata table ===")
    meta = build_metadata_table()
    print(f"  {len(meta):,} images with image file present")
    print(f"  splits: {meta['split'].value_counts().to_dict()}")
    print()

    print("=== QScoreDataset (Q-score regression) ===")
    for split in ["train", "val", "test"]:
        ds = QScoreDataset(split=split, meta_df=meta)
        print(f"  {split}: n={len(ds):,}  target_mean={ds.target_mean.round(3).tolist()}")
    ds = QScoreDataset(split="train", meta_df=meta)
    sample = ds[0]
    print(f"  sample shapes: image={tuple(sample['image'].shape)}  target={tuple(sample['target'].shape)}  mask_sum={sample['mask'].sum().item()}")
    loader = DataLoader(ds, batch_size=8, shuffle=True, num_workers=2)
    batch = next(iter(loader))
    print(f"  batch: image={tuple(batch['image'].shape)}  target={tuple(batch['target'].shape)}  mask={tuple(batch['mask'].shape)}")
    # masked loss smoke
    import torch
    pred = torch.zeros_like(batch["target"])
    loss = masked_mse_loss(pred, batch["target"], batch["mask"])
    print(f"  masked MSE on zero pred: {loss.item():.4f}")
    print()

    print("=== PairwiseDataset ===")
    pds = PairwiseDataset(split="train", meta_df=meta)
    print(f"  train pairs: {len(pds):,}")
    sample = pds[0]
    print(f"  sample: left={tuple(sample['left'].shape)}  dim_idx={sample['dim_idx']}  label={sample['label'].item()}")
    ploader = DataLoader(pds, batch_size=4, shuffle=True, num_workers=2)
    pbatch = next(iter(ploader))
    print(f"  batch: left={tuple(pbatch['left'].shape)}  dim_idx={pbatch['dim_idx'].tolist()}  label={pbatch['label'].tolist()}")


if __name__ == "__main__":
    main()
