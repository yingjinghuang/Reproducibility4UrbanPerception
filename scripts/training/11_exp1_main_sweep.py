"""Main multi-seed sweep: 160 runs total.

Run plan:
  - ResNet-50 (full FT, 6 epochs, batch 128, lr 1e-4):     seeds 0..39  (40)
  - ViT-B/16  (full FT, 6 epochs, batch 64,  lr 3e-5):     seeds 0..79  (80)
  - DINOv2 frozen + MLP head (10 epochs, batch 128, 1e-3): seeds 0..39  (40)

All hyperparameters are fixed to the per-architecture defaults in polygeo.train.
Only the master seed varies between runs; it controls weight initialization,
data shuffling, augmentation randomness, and dropout randomness.

Each run saves:
  - runs/{run_name}/{config.json, train_log.csv, summary.json}
  - checkpoints/{run_name}.pt
  - predictions/{run_name}_test_preds.npz

The launcher is restartable: runs with an existing summary.json are skipped.
"""
from __future__ import annotations
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np

from polygeo.train import TrainConfig, run_one, get_arch_defaults
from polygeo.paths import RUNS, ROOT


SWEEP_PLAN = [
    ("resnet50", "imagenet21k", 40),
    ("vit_b16", "imagenet21k", 80),
    ("dinov2_frozen", "imagenet21k", 40),
]


def main() -> None:
    t_total = time.time()
    summary_rows = []
    n_skipped = 0
    n_done = 0
    n_failed = 0

    for arch, init, n_seeds in SWEEP_PLAN:
        defaults = get_arch_defaults(arch, init)
        for seed in range(n_seeds):
            run_name = f"exp1_{arch}_{init}_seed{seed:03d}"
            run_dir = RUNS / run_name
            done_marker = run_dir / "summary.json"
            if done_marker.exists():
                summary_rows.append({**json.loads(done_marker.read_text()), "run_name": run_name})
                n_skipped += 1
                continue

            cfg = TrainConfig(
                arch=arch,
                init=init,
                seed=seed,
                epochs=defaults["epochs"],
                batch_size=defaults["batch_size"],
                lr=defaults["lr"],
                weight_decay=defaults.get("weight_decay", 0.05),
                run_name=run_name,
                save_checkpoint=True,
                save_predictions=True,
            )
            print(f"\n>>> {run_name}  ({n_done}/{sum(p[2] for p in SWEEP_PLAN)} so far, skipped {n_skipped})")
            t0 = time.time()
            try:
                summ = run_one(cfg)
                summ["run_name"] = run_name
                summary_rows.append(summ)
                n_done += 1
                print(f">>> {run_name} done in {(time.time()-t0)/60:.1f} min")
            except Exception as e:
                print(f">>> {run_name} FAILED: {e!r}")
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / "FAILED.txt").write_text(repr(e))
                n_failed += 1

    print("\n=== Exp 1 sweep summary ===")
    by_arch = {}
    for row in summary_rows:
        key = f"{row['config']['arch']}_{row['config']['init']}"
        by_arch.setdefault(key, []).append(row)

    rolled = {}
    for key, rows in by_arch.items():
        rmse = [r["test_rmse_mean"] for r in rows]
        elapsed = [r["elapsed_h"] for r in rows]
        print(
            f"  {key}: n={len(rows)}  "
            f"test_rmse = {np.mean(rmse):.4f} ± {np.std(rmse):.4f}  "
            f"GPU-h/run = {np.mean(elapsed):.3f}  total = {np.sum(elapsed):.1f} h"
        )
        rolled[key] = {
            "n_seeds": len(rows),
            "test_rmse_mean": float(np.mean(rmse)),
            "test_rmse_std": float(np.std(rmse)),
            "test_rmse_per_seed": rmse,
            "elapsed_h_per_run_mean": float(np.mean(elapsed)),
            "elapsed_h_total": float(np.sum(elapsed)),
        }

    out = {
        "rolled": rolled,
        "n_done": n_done,
        "n_skipped": n_skipped,
        "n_failed": n_failed,
        "total_wallclock_h": (time.time() - t_total) / 3600.0,
    }
    (ROOT / "data_processed/exp1_summary.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {ROOT/'data_processed/exp1_summary.json'}")
    print(f"total wallclock: {out['total_wallclock_h']:.2f} h  done={n_done} skipped={n_skipped} failed={n_failed}")


if __name__ == "__main__":
    main()
