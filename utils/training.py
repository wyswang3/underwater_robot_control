import logging, torch, math
import torch.nn as nn
from contextlib import nullcontext
from torch.cuda import amp

logger = logging.getLogger(__name__)

# ╭───────────────────╮
# │ 1.  辅助函数      │
# ╰───────────────────╯
def batch_to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, dict):
        return {k: batch_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(batch_to_device(v, device) for v in obj)
    return obj


def process_batch(batch, device):
    """
    返回:
      power_window: (B, win, 8)
      imu_window  : (B, win, 6)
      target_dict : {accel:(B,6), thrust:(B,8)}
    """
    batch = batch_to_device(batch, device)
    pw = batch["power_window"].float()
    imu = batch["imu_window"].float()
    accel = batch["accel"].float()
    target = {
        "accel": accel,                 # 已经是 6 维
        "thrust": batch["thrust"].float()
    }
    return pw, imu, target

# ╭───────────────────╮
# │ 2.  训练函数      │
# ╰───────────────────╯
def train_one_epoch(
        model, loader, criterion,
        optimizer, scaler,
        epoch, warmup_steps,
        scheduler=None, step_per_batch=False,
        amp_enabled=True, grad_clip=None):

    device = next(model.parameters()).device
    model.train()
    running, n = 0.0, 0
    global_step = (epoch - 1) * len(loader)

    for b, batch in enumerate(loader, 1):
        pw, imu, tgt = process_batch(batch, device)

        # ── Warm‑up ──────────────────
        global_step += 1
        if warmup_steps and global_step <= warmup_steps:
            for g in optimizer.param_groups:
                g["lr"] = g["initial_lr"] * global_step / warmup_steps

        optimizer.zero_grad(set_to_none=True)

        ctx = amp.autocast(enabled=amp_enabled) if amp_enabled else nullcontext()
        with ctx:
            out   = model(pw, imu)
            loss  = criterion(out, tgt)           # ★ 只返回一个标量
        if torch.isfinite(loss):
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            logger.warning(f"[Skip] epoch{epoch} batch{b} NaN/Inf loss")
            continue

        if scheduler and step_per_batch:
            scheduler.step()

        bs = pw.size(0)
        running += loss.item() * bs
        n += bs

        #if b % 100 == 0:
          #  logger.info(f"Epoch{epoch} [{b:4d}/{len(loader)}] "
           #             f"loss {loss.item():.4f} lr {optimizer.param_groups[0]['lr']:.2e}")

    if scheduler and not step_per_batch:
        scheduler.step()

    return running / max(n, 1)


def custom_train_model(
        model, criterion, train_loader,
        epochs, optimizer, scheduler=None,
        warmup_steps=0, grad_clip=None,
        amp_enabled=True, step_per_batch=False):

    device = next(model.parameters()).device
    scaler = amp.GradScaler(enabled=amp_enabled)

    history = []
    for ep in range(1, epochs + 1):
        epoch_loss = train_one_epoch(
            model, train_loader, criterion,
            optimizer, scaler, ep, warmup_steps,
            scheduler, step_per_batch,
            amp_enabled, grad_clip
        )
        history.append(epoch_loss)
        logger.info(f"Epoch[{ep:3d}/{epochs}]  train‑loss {epoch_loss:.6f} "
                    f"lr {optimizer.param_groups[0]['lr']:.2e}")

    return history


# ╭───────────────────╮
# │ 3.  验证函数      │
# ╰───────────────────╯
@torch.no_grad()
def validate_model(model, criterion, loader, amp_enabled=True):
    model.eval()
    device = next(model.parameters()).device
    total, n = 0.0, 0
    ctx_mgr = amp.autocast(enabled=amp_enabled) if amp_enabled else nullcontext()

    for batch in loader:
        pw, imu, tgt = process_batch(batch, device)
        with ctx_mgr:
            loss = criterion(model(pw, imu), tgt)
        total += loss.item() * pw.size(0)
        n += pw.size(0)

    avg = total / max(n, 1)
    logger.info(f"[Validate] loss {avg:.6f}")
    return avg
