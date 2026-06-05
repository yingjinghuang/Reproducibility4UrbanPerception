"""Single Q-score regression training run for the multi-seed audit.

Hyperparameters are fixed per architecture (Pilot will tune them on a held-out seed,
production runs use the fixed config). The seed alone differs across runs.
"""
from __future__ import annotations
import argparse
import csv
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from polygeo.data import QScoreDataset, build_metadata_table, masked_mse_loss
from polygeo.models import build
from polygeo.paths import CHECKPOINTS, PREDICTIONS, RUNS, DIMENSIONS
from polygeo.seeding import seed_everything, worker_init_fn


@dataclass
class TrainConfig:
    arch: str = "vit_b16"
    init: str = "imagenet21k"  # only for vit_b16
    seed: int = 0
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


def get_arch_defaults(arch: str, init: str) -> dict:
    if arch == "resnet50":
        return dict(lr=1e-4, weight_decay=1e-4, epochs=6, batch_size=128)
    if arch == "vit_b16":
        return dict(lr=3e-5, weight_decay=0.05, epochs=6, batch_size=64)
    if arch == "dinov2_frozen":
        return dict(lr=1e-3, weight_decay=1e-4, epochs=10, batch_size=128)
    raise ValueError(arch)


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
        # accumulate per-dimension MSE on normalized scale
        sq = ((p - y) ** 2 * m).cpu().numpy()
        sse += sq.sum(axis=0)
        cnt += m.cpu().numpy().sum(axis=0).astype(np.int64)
        # de-normalize for saving
        p_denorm = p.cpu().numpy() * target_std + target_mean
        preds_all.append(p_denorm)
        ids_all.extend(batch["image_id"])
    rmse_per_dim = np.sqrt(sse / np.maximum(cnt, 1))
    preds = np.concatenate(preds_all, axis=0)
    return rmse_per_dim, preds, ids_all


def run_one(cfg: TrainConfig) -> dict:
    seed_everything(cfg.seed, deterministic_cudnn=False)

    if cfg.run_name is None:
        cfg.run_name = f"{cfg.arch}_{cfg.init}_seed{cfg.seed:03d}"
    run_dir = RUNS / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))

    print(f"\n=== {cfg.run_name} ===")
    print(f"config: {asdict(cfg)}")

    meta = build_metadata_table()
    train_ds = QScoreDataset("train", meta_df=meta)
    val_ds = QScoreDataset("val", meta_df=meta)
    test_ds = QScoreDataset("test", meta_df=meta)
    print(f"datasets — train: {len(train_ds):,}, val: {len(val_ds):,}, test: {len(test_ds):,}")

    g = torch.Generator()
    g.manual_seed(cfg.seed)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
        worker_init_fn=worker_init_fn, generator=g,
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
    model_kwargs = {}
    if cfg.arch == "vit_b16":
        model_kwargs["init"] = cfg.init
    model = build(cfg.arch, **model_kwargs).to(device)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"model: {cfg.arch}  trainable params: {n_train/1e6:.1f}M")

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr, weight_decay=cfg.weight_decay,
    )
    total_steps = cfg.epochs * len(train_loader)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp)

    target_mean = train_ds.target_mean
    target_std = train_ds.target_std

    history = []
    log_csv = (run_dir / "train_log.csv").open("w", newline="")
    log_writer = csv.writer(log_csv)
    log_writer.writerow(["epoch", "step", "train_loss", "val_rmse_mean", *[f"val_rmse_{d}" for d in DIMENSIONS]])

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
            if step % 100 == 0:
                print(f"  e{epoch} s{step}/{total_steps}  lr={lr:.2e}  loss={running/n:.4f}  elapsed={time.time()-t_start:.0f}s")

        train_loss = running / max(n, 1)
        rmse_per_dim, _, _ = evaluate(model, val_loader, device, target_mean, target_std)
        val_rmse_mean = float(rmse_per_dim.mean())
        print(f"  epoch {epoch} done  train_loss={train_loss:.4f}  val_rmse_mean={val_rmse_mean:.4f}  per-dim={rmse_per_dim.round(3).tolist()}")
        log_writer.writerow([epoch, step, train_loss, val_rmse_mean, *rmse_per_dim.tolist()])
        log_csv.flush()
        history.append({"epoch": epoch, "train_loss": train_loss, "val_rmse_mean": val_rmse_mean})

    log_csv.close()

    print("running test ...")
    test_rmse, test_preds, test_ids = evaluate(model, test_loader, device, target_mean, target_std)
    print(f"  test_rmse_mean={float(test_rmse.mean()):.4f}  per-dim={test_rmse.round(3).tolist()}")

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

    return summary


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--arch", default="vit_b16", choices=["resnet50", "vit_b16", "dinov2_frozen"])
    p.add_argument("--init", default="imagenet21k", choices=["imagenet21k", "dinov2"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--no-checkpoint", action="store_true")
    p.add_argument("--run-name", default=None)
    args = p.parse_args()

    defaults = get_arch_defaults(args.arch, args.init)
    cfg = TrainConfig(
        arch=args.arch,
        init=args.init,
        seed=args.seed,
        epochs=args.epochs if args.epochs is not None else defaults["epochs"],
        batch_size=args.batch_size if args.batch_size is not None else defaults["batch_size"],
        lr=args.lr if args.lr is not None else defaults["lr"],
        weight_decay=defaults.get("weight_decay", 0.05),
        num_workers=args.num_workers,
        device=args.device,
        amp=not args.no_amp,
        save_checkpoint=not args.no_checkpoint,
        run_name=args.run_name,
    )
    return cfg


if __name__ == "__main__":
    cfg = parse_args()
    run_one(cfg)
