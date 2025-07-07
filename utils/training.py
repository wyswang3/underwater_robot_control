"""Utility functions for training/validation with AMP support (PyTorch >=2.1).
Safe to import as `from utils.training import *`.
Supports different loss signatures for pure lstm/mlp (MSELoss) and physics/hybrid (EnhancedDynamicsLoss).
"""
from __future__ import annotations

import logging
import torch
from torch.cuda.amp import autocast, GradScaler
from typing import Optional

# Setup logger
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------
# helper ---------------------------------------------------------------------
# ----------------------------------------------------------------------------

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
    """Split a raw batch into (power, imu, target_dict)."""
    batch = batch_to_device(batch, device)
    pw    = batch["power_window"].float()
    imu   = batch.get("imu_window")
    if imu is not None:
        imu = imu.float()
    # Build target dict for physics/hybrid loss
    target = {
        "accel":  batch["accel"].float(),
        "thrust": batch.get("thrust_labels", batch.get("thrust")).float()
    }
    return pw, imu, target

# ----------------------------------------------------------------------------
# training loop --------------------------------------------------------------
# ----------------------------------------------------------------------------

def train_one_epoch(
    model,
    criterion,
    dataloader,
    optimizer,
    scheduler,
    warmup_steps: int,
    grad_clip: float,
    amp_enabled: bool,
    step_per_batch: bool
) -> float:
    """Run one epoch of training and return average loss."""
    model.train()
    total_loss = 0.0
    scaler = GradScaler() if amp_enabled else None

    for step, batch in enumerate(dataloader):
        pw, imu, target = process_batch(batch, next(model.parameters()).device)

        optimizer.zero_grad()
        if amp_enabled:
            with autocast():
                out = model(pw, imu)
                # decide loss call signature
                if isinstance(criterion, torch.nn.MSELoss):
                    # pure lstm/mlp: out is tensor or dict with accel_pred
                    pred = out.get('accel_pred', out) if isinstance(out, dict) else out
                    loss = criterion(pred, target['accel'])
                else:
                    # physics/hybrid: criterion expects (out, batch_dict)
                    loss = criterion(out, target)
            scaler.scale(loss).backward()
            if grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(pw, imu)
            if isinstance(criterion, torch.nn.MSELoss):
                pred = out.get('accel_pred', out) if isinstance(out, dict) else out
                loss = criterion(pred, target['accel'])
            else:
                loss = criterion(out, target)
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        if scheduler and step_per_batch:
            scheduler.step()
        total_loss += loss.item()

    if scheduler and not step_per_batch:
        scheduler.step()

    return total_loss / len(dataloader)


def custom_train_model(
    model,
    criterion,
    train_loader,
    val_loader: Optional[torch.utils.data.DataLoader] = None,
    epochs: int = 1,
    optimizer=None,
    scheduler=None,
    warmup_steps: int = 0,
    grad_clip: float = 0.0,
    amp_enabled: bool = False,
    step_per_batch: bool = False
) -> dict:
    """Run full training for multiple epochs, with optional validation.
    Logs epoch, train loss, and learning rate at INFO level.
    Returns history with 'train_loss' and 'val_loss'."""
    history = {'train_loss': [], 'val_loss': []}
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model, criterion, train_loader,
            optimizer, scheduler, warmup_steps,
            grad_clip, amp_enabled, step_per_batch
        )
        history['train_loss'].append(train_loss)

        lr = optimizer.param_groups[0]['lr']
        if val_loader is not None:
            val_loss = validate_model(model, criterion, val_loader, amp_enabled)
            history['val_loss'].append(val_loss)
            logger.info(
                f"Epoch[{epoch:3d}/{epochs}] train-loss={train_loss:.6f} "
                f"val-loss={val_loss:.6f} lr={lr:.2e}"
            )
        else:
            logger.info(
                f"Epoch[{epoch:3d}/{epochs}] train-loss={train_loss:.6f} lr={lr:.2e}"
            )
    return history


def validate_model(
    model,
    criterion,
    loader,
    amp_enabled: bool = False
) -> float:
    """Compute average loss over validation/test loader."""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            pw, imu, target = process_batch(batch, next(model.parameters()).device)
            if amp_enabled:
                with autocast():
                    out = model(pw, imu)
                    if isinstance(criterion, torch.nn.MSELoss):
                        pred = out.get('accel_pred', out) if isinstance(out, dict) else out
                        loss = criterion(pred, target['accel'])
                    else:
                        loss = criterion(out, target)
            else:
                out = model(pw, imu)
                if isinstance(criterion, torch.nn.MSELoss):
                    pred = out.get('accel_pred', out) if isinstance(out, dict) else out
                    loss = criterion(pred, target['accel'])
                else:
                    loss = criterion(out, target)
            total_loss += loss.item()
    return total_loss / len(loader)