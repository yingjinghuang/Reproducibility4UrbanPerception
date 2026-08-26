"""Exp 2 — Source decomposition on ViT-B/16.

Three groups, 20 runs each (60 total):
  Group A "init+dropout":  init_seed varies (0..19), data_seed=0, aug_seed=0
  Group B "data":           data_seed varies (0..19), init_seed=0, aug_seed=0
  Group C "aug":            aug_seed varies (0..19), init_seed=0, data_seed=0

Run (init_seed=0, data_seed=0, aug_seed=0) is shared across groups as the
reference run and provides a duplicate check for the seeding implementation.
Restartable: skips runs whose summary.json already exists.
"""
from __future__ import annotations
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np

from polygeo.train_decompose import DecomposeConfig, run_decompose
from polygeo.train import get_arch_defaults
from polygeo.paths import RUNS, ROOT


N_PER_GROUP = 20
ARCH = "vit_b16"
INIT = "imagenet21k"


def main() -> None:
    defaults = get_arch_defaults(ARCH, INIT)
    summary_rows = []
    n_done, n_skipped, n_failed = 0, 0, 0
    t_total = time.time()

    plan = []
    for s in range(N_PER_GROUP):
        plan.append(("init_dropout", s, 0, 0))
    for s in range(N_PER_GROUP):
        plan.append(("data", 0, s, 0))
    for s in range(N_PER_GROUP):
        plan.append(("aug", 0, 0, s))

    for group, init_s, data_s, aug_s in plan:
        run_name = f"exp2_{group}_vit_b16_i{init_s:03d}_d{data_s:03d}_a{aug_s:03d}"
        run_dir = RUNS / run_name
        done_marker = run_dir / "summary.json"
        if done_marker.exists():
            summary_rows.append({**json.loads(done_marker.read_text()), "run_name": run_name, "group": group})
            n_skipped += 1
            continue
        cfg = DecomposeConfig(
            arch=ARCH,
            init=INIT,
            init_seed=init_s,
            data_seed=data_s,
            aug_seed=aug_s,
            epochs=defaults["epochs"],
            batch_size=defaults["batch_size"],
            lr=defaults["lr"],
            weight_decay=defaults.get("weight_decay", 0.05),
            run_name=run_name,
            save_checkpoint=False,
            save_predictions=True,
        )
        print(f"\n>>> {run_name} ({n_done}/{len(plan)} done, {n_skipped} skipped)")
        try:
            summ = run_decompose(cfg)
            summ["run_name"] = run_name
            summ["group"] = group
            summary_rows.append(summ)
            n_done += 1
        except Exception as e:
            print(f">>> {run_name} FAILED: {e!r}")
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "FAILED.txt").write_text(repr(e))
            n_failed += 1

    # Roll up per group
    print("\n=== Exp 2 summary ===")
    by_group = {}
    for row in summary_rows:
        by_group.setdefault(row["group"], []).append(row)

    rolled = {}
    for group, rows in by_group.items():
        rmse = [r["test_rmse_mean"] for r in rows]
        elapsed = [r["elapsed_h"] for r in rows]
        rolled[group] = {
            "n_seeds": len(rows),
            "test_rmse_mean": float(np.mean(rmse)),
            "test_rmse_std": float(np.std(rmse)),
            "test_rmse_per_seed": rmse,
            "elapsed_h_per_run_mean": float(np.mean(elapsed)),
            "elapsed_h_total": float(np.sum(elapsed)),
        }
        print(
            f"  {group}: n={len(rows)}  "
            f"test_rmse = {np.mean(rmse):.5f} ± {np.std(rmse):.5f}  "
            f"GPU-h/run = {np.mean(elapsed):.3f}  total = {np.sum(elapsed):.1f} h"
        )

    out = {
        "rolled": rolled,
        "n_done": n_done, "n_skipped": n_skipped, "n_failed": n_failed,
        "total_wallclock_h": (time.time() - t_total) / 3600.0,
    }
    (ROOT / "data_processed/exp2_summary.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {ROOT/'data_processed/exp2_summary.json'}")


if __name__ == "__main__":
    main()
