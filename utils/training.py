"""Utility functions for training/validation with AMP support (PyTorch >=2.1).
Safe to import as `from utils.training import *`.
"""
from __future__ import annotations

import logging
from contextlib import nullcontext
from typing import List

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch import amp  # PyTorch 2.1+ unified AMP API

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# helper ----------------------------------------------------------------------
# -----------------------------------------------------------------------------

def batch_to_device(obj, device: torch.device):
    """Recursively move *obj* to *device* (non‑blocking when possible)."""
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, dict):
        return {k: batch_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(batch_to_device(v, device) for v in obj)
    return obj


def process_batch(batch: dict, device: torch.device):
    """Split a raw batch into (power, imu, target‑dict)."""
    batch = batch_to_device(batch, device)
    pw    = batch["power_window"].float()   # (B, win, 8)
    imu   = batch["imu_window"].float()     # (B, win, 6)
    target = {
        "accel":  batch["accel"].float(),   # (B, 6)
        "thrust": batch["thrust"].float()   # (B, 8)
    }
    return pw, imu, target

# -----------------------------------------------------------------------------
# training loop ---------------------------------------------------------------
# -----------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    scaler: amp.GradScaler,
    epoch: int,
    warmup_steps: int = 0,
    scheduler: Optimizer | None = None,
    step_per_batch: bool = False,
    amp_enabled: bool = True,
    grad_clip: float | None = None,
) -> float:
    """Run one epoch; return average loss."""
    device = next(model.parameters()).device
    model.train()

    running_loss, seen = 0.0, 0
    global_step = (epoch - 1) * len(loader)

    for batch_idx, batch in enumerate(loader, 1):
        pw, imu, tgt = process_batch(batch, device)

        # warm‑up lr ----------------------------------------------------------------
        global_step += 1
        if warmup_steps and global_step <= warmup_steps:
            warm_ratio = global_step / warmup_steps
            for group in optimizer.param_groups:
                group["lr"] = group["initial_lr"] * warm_ratio

        optimizer.zero_grad(set_to_none=True)

        ctx = amp.autocast(device_type="cuda", enabled=amp_enabled) if amp_enabled else nullcontext()
        with ctx:
            out   = model(pw, imu)
            loss  = criterion(out, tgt)

        if not torch.isfinite(loss):
            logger.warning(f"[Skip] epoch {epoch} batch {batch_idx}: NaN/Inf loss")
            continue
        if torch.isnan(loss) or torch.isinf(loss):
            logger.error(f"[NaN] epoch={epoch} batch={batch_idx}  lr={optimizer.param_groups[0]['lr']:.2e}")
            for n, p in model.named_parameters():
                if torch.isnan(p).any() or torch.isinf(p).any():
                    logger.error(f"  param {n} has NaN/Inf")
            raise RuntimeError("NaN detected – aborting to keep checkpoint clean")

        # backward ------------------------------------------------------------------
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        if scheduler and step_per_batch:
            scheduler.step()

        bs = pw.size(0)
        running_loss += loss.item() * bs
        seen += bs

    if scheduler and not step_per_batch:
        scheduler.step()

    return running_loss / max(seen, 1)


def custom_train_model(
    model: nn.Module,
    criterion: nn.Module,
    train_loader: DataLoader,
    epochs: int,
    optimizer: Optimizer,
    scheduler: Optimizer | None = None,
    warmup_steps: int = 0,
    grad_clip: float | None = None,
    amp_enabled: bool = True,
    step_per_batch: bool = False,
) -> List[float]:
    """Full training routine; returns list of epoch losses."""
    scaler = amp.GradScaler(enabled=amp_enabled)
    history: List[float] = []

    for ep in range(1, epochs + 1):
        epoch_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler,
            ep, warmup_steps, scheduler, step_per_batch, amp_enabled, grad_clip
        )
        history.append(epoch_loss)
        lr = optimizer.param_groups[0]["lr"]
        logger.info(f"Epoch[{ep:3d}/{epochs}] train‑loss={epoch_loss:.6f} lr={lr:.2e}")

    return history

# -----------------------------------------------------------------------------
# validation ------------------------------------------------------------------
# -----------------------------------------------------------------------------

@torch.no_grad()
def validate_model(
    model: nn.Module,
    criterion: nn.Module,
    loader: DataLoader,
    amp_enabled: bool = True,
) -> float:
    """Evaluate *model*; return average loss."""
    model.eval()
    device = next(model.parameters()).device
    total, seen = 0.0, 0

    ctx_mgr = amp.autocast(device_type="cuda", enabled=amp_enabled) if amp_enabled else nullcontext()

    for batch in loader:
        pw, imu, tgt = process_batch(batch, device)
        with ctx_mgr:
            loss = criterion(model(pw, imu), tgt)
        bs = pw.size(0)
        total += loss.item() * bs
        seen += bs

    avg_loss = total / max(seen, 1)
    logger.info(f"[Validate] loss={avg_loss:.6f}")
    return avg_loss
