"""Training variant for the source-decomposition experiment.

Three independent seeds:
  - init_seed   : weight initialization (and global RNG before model creation)
  - data_seed   : DataLoader shuffle order
  - aug_seed    : augmentation worker RNGs

PyTorch dropout uses the same global CUDA RNG as weight initialization, so the
initialization-varying condition also varies dropout. We label the three groups
accordingly:
  - "init+dropout"  : init_seed varies, data_seed and aug_seed fixed at 0
  - "data"          : data_seed varies, others fixed at 0
  - "aug"           : aug_seed varies, others fixed at 0
"""
from __future__ import annotations
import csv
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import QScoreDataset, build_metadata_table, masked_mse_loss
from .models import build
from .paths import CHECKPOINTS, PREDICTIONS, RUNS, DIMENSIONS
from .seeding import seed_everything, make_aug_worker_init_fn


@dataclass
class DecomposeConfig:
    arch: str = "vit_b16"
    init: str = "imagenet21k"
    init_seed: int = 0
    data_seed: int = 0
    aug_seed: int = 0
    epochs: int = 6
    batch_size: int = 64
    lr: float = 3e-5
    weight_decay: float = 0.05
    warmup_frac: float = 0.05
    num_workers: int = 8
    img_size: int = 224
    device: str = "cuda:0"
    amp: bool = True
    save_checkpoint: bool = True
    save_predictions: bool = True
    run_name: str | None = None
    grad_clip: float = 1.0


def cosine_warmup_lr(step: int, total_steps: int, base_lr: float, warmup_frac: float) -> float:
    warmup = max(1, int(total_steps * warmup_frac))
    if step < warmup:
        return base_lr * step / warmup
    progress = (step - warmup) / max(1, total_steps - warmup)
    return base_lr * 0.5 * (1.0 + np.cos(np.pi * progress))


@torch.no_grad()
def evaluate(model, loader, device, target_mean, target_std):
    model.eval()
    sse = np.zeros(len(DIMENSIONS), dtype=np.float64)
    cnt = np.zeros(len(DIMENSIONS), dtype=np.int64)
    preds_all, ids_all = [], []
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["target"].to(device, non_blocking=True)
        m = batch["mask"].to(device, non_blocking=True)
        p = model(x)
        sq = ((p - y) ** 2 * m).cpu().numpy()
        sse += sq.sum(axis=0)
        cnt += m.cpu().numpy().sum(axis=0).astype(np.int64)
        p_denorm = p.cpu().numpy() * target_std + target_mean
        preds_all.append(p_denorm)
        ids_all.extend(batch["image_id"])
    rmse_per_dim = np.sqrt(sse / np.maximum(cnt, 1))
    preds = np.concatenate(preds_all, axis=0)
    return rmse_per_dim, preds, ids_all


def run_decompose(cfg: DecomposeConfig) -> dict:
    if cfg.run_name is None:
        cfg.run_name = f"exp2_{cfg.arch}_init{cfg.init_seed:03d}_data{cfg.data_seed:03d}_aug{cfg.aug_seed:03d}"
    run_dir = RUNS / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))

    print(f"\n=== {cfg.run_name} ===")

    # 1. Set global RNGs to init_seed BEFORE model creation
    seed_everything(cfg.init_seed)

    meta = build_metadata_table()
    train_ds = QScoreDataset("train", meta_df=meta)
    val_ds = QScoreDataset("val", meta_df=meta)
    test_ds = QScoreDataset("test", meta_df=meta)
    target_mean = train_ds.target_mean
    target_std = train_ds.target_std

    # 2. DataLoader uses an explicit Generator seeded by data_seed
    g = torch.Generator()
    g.manual_seed(cfg.data_seed)

    # 3. Worker RNGs follow aug_seed
    aug_init_fn = make_aug_worker_init_fn(cfg.aug_seed)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
        worker_init_fn=aug_init_fn, generator=g,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
    )

    device = torch.device(cfg.device)
    model_kwargs = {"init": cfg.init} if cfg.arch == "vit_b16" else {}
    model = build(cfg.arch, **model_kwargs).to(device)

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr, weight_decay=cfg.weight_decay,
    )
    total_steps = cfg.epochs * len(train_loader)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp)

    log_csv = (run_dir / "train_log.csv").open("w", newline="")
    log_writer = csv.writer(log_csv)
    log_writer.writerow(["epoch", "step", "train_loss", "val_rmse_mean", *[f"val_rmse_{d}" for d in DIMENSIONS]])

    history = []
    step = 0
    t_start = time.time()
    for epoch in range(cfg.epochs):
        model.train()
        running, n = 0.0, 0
        for batch in train_loader:
            x = batch["image"].to(device, non_blocking=True)
            y = batch["target"].to(device, non_blocking=True)
            m = batch["mask"].to(device, non_blocking=True)
            lr = cosine_warmup_lr(step, total_steps, cfg.lr, cfg.warmup_frac)
            for g_ in optim.param_groups:
                g_["lr"] = lr
            optim.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=cfg.amp):
                pred = model(x)
                loss = masked_mse_loss(pred, y, m)
            scaler.scale(loss).backward()
            if cfg.grad_clip:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], cfg.grad_clip
                )
            scaler.step(optim)
            scaler.update()
            running += loss.item() * x.size(0)
            n += x.size(0)
            step += 1

        train_loss = running / max(n, 1)
        rmse_per_dim, _, _ = evaluate(model, val_loader, device, target_mean, target_std)
        val_rmse_mean = float(rmse_per_dim.mean())
        log_writer.writerow([epoch, step, train_loss, val_rmse_mean, *rmse_per_dim.tolist()])
        log_csv.flush()
        history.append({"epoch": epoch, "train_loss": train_loss, "val_rmse_mean": val_rmse_mean})

    log_csv.close()

    test_rmse, test_preds, test_ids = evaluate(model, test_loader, device, target_mean, target_std)
    elapsed = time.time() - t_start

    summary = {
        "config": asdict(cfg),
        "history": history,
        "test_rmse_per_dim": dict(zip(DIMENSIONS, test_rmse.tolist())),
        "test_rmse_mean": float(test_rmse.mean()),
        "elapsed_s": elapsed,
        "elapsed_h": elapsed / 3600.0,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    if cfg.save_predictions:
        PREDICTIONS.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            PREDICTIONS / f"{cfg.run_name}_test_preds.npz",
            image_ids=np.array(test_ids),
            preds=test_preds,
            target_mean=target_mean,
            target_std=target_std,
            dimensions=np.array(DIMENSIONS),
        )

    if cfg.save_checkpoint:
        CHECKPOINTS.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"model": model.state_dict(), "config": asdict(cfg)},
            CHECKPOINTS / f"{cfg.run_name}.pt",
        )

    print(f"  done {cfg.run_name}  test_rmse={float(test_rmse.mean()):.4f}  elapsed={elapsed/60:.1f}m")
    return summary
