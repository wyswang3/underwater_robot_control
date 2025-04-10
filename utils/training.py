import torch
import torch.nn as nn
from contextlib import nullcontext
from torch import amp


# ------------------------------------------------------------------
# 1. 把任意嵌套结构搬到指定 device
# ------------------------------------------------------------------
def batch_to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, dict):
        return {k: batch_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(batch_to_device(v, device) for v in obj)
    return obj


# ------------------------------------------------------------------
# 2. 拆分 batch
# ------------------------------------------------------------------
def process_batch(batch, device):
    batch = batch_to_device(batch, device)
    pw, imu = batch["power_window"], batch["imu_window"]
    # features: 将 power_window 与 imu 拼接成 (B, W*14)
    features = torch.cat([pw, imu], dim=2).flatten(1)

    accel = batch["accel"]
    lin_t, ang_t = accel[:, :3], accel[:, 3:]
    thrust = batch["thrust"]
    return features.float(), lin_t.float(), ang_t.float(), thrust.float()


# ------------------------------------------------------------------
# 3. 训练
# ------------------------------------------------------------------
def custom_train_model(
        model, criterion, train_loader,
        epochs, lr, weight_decay, grad_clip,
        scheduler=None, *, step_per_batch=False
):
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # 是否启用自动混合精度
    amp_enabled = torch.cuda.is_available() and device.type == "cuda"
    scaler = amp.GradScaler(enabled=amp_enabled)

    train_losses = []

    for ep in range(1, epochs + 1):
        model.train()
        running_loss, total_samples = 0.0, 0

        # 循环遍历所有 batch
        for batch_idx, batch in enumerate(train_loader, start=1):
            feat, lin_t, ang_t, tau = process_batch(batch, device)

            optimizer.zero_grad(set_to_none=True)
            # 保持 tau 的数据类型与 feat 一致
            tau = tau.to(feat.dtype)

            ctx = amp.autocast(device_type="cuda") if amp_enabled else nullcontext()
            with ctx:
                out = model(feat, tau)
                # 假设模型输出 (pred_lin, pred_ang, M, D)
                pred_lin, pred_ang, M, D, v_pred = out
                loss = criterion(pred_lin, lin_t, pred_ang, ang_t, tau, M, D, v_pred)

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"[WARN] Epoch {ep} Batch {batch_idx}: NaN/Inf loss — skipping batch")
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

            bs = feat.size(0)
            running_loss += loss.item() * bs
            total_samples += bs

            # 可打印每个 batch 的 loss 信息（可选）
            #if batch_idx % 10 == 0:
             #   print(f"Epoch {ep} Batch {batch_idx}: Loss = {loss.item():.6f}")

            if scheduler and step_per_batch:
                scheduler.step()

        if scheduler and not step_per_batch:
            scheduler.step()

        epoch_loss = running_loss / max(total_samples, 1)
        train_losses.append(epoch_loss)
        print(f"Epoch [{ep:3d}/{epochs}]  Loss: {epoch_loss:.6f}")

    return model, train_losses


# ------------------------------------------------------------------
# 4. 验证
# ------------------------------------------------------------------
def validate_model(model: nn.Module, criterion: nn.Module, val_loader) -> float:
    """
    验证模型：
      - 模型处于 eval 模式；
      - 采用自动混合精度进行计算（如果可用）；
      - 对每个 batch 计算损失并累积；
      - 返回平均验证损失。

    参数：
      model: 模型实例
      criterion: 损失函数
      val_loader: 验证数据加载器

    返回：
      平均验证损失 (float)
    """
    model.eval()
    device = next(model.parameters()).device
    amp_enabled = torch.cuda.is_available() and device.type == "cuda"

    total_loss, total_samples = 0.0, 0
    with torch.no_grad():
        for batch in val_loader:
            # 处理数据：feat, lin_t, ang_t, tau 均已发送到相应 device 上
            feat, lin_t, ang_t, tau = process_batch(batch, device)
            tau = tau.to(feat.dtype)

            # 根据设备情况选择自动混合精度上下文
            ctx = amp.autocast(device_type="cuda") if amp_enabled else nullcontext()
            with ctx:
                # 此处设 return_v_pred=True 保证返回 5 个输出（包括 v_pred）
                pred_lin, pred_ang, M, D, v_pred = model(feat, tau, return_v_pred=True)
                loss = criterion(pred_lin, lin_t, pred_ang, ang_t, tau, M, D, v_pred)

            bs = feat.size(0)
            total_loss += loss.item() * bs
            total_samples += bs

    avg_loss = total_loss / max(total_samples, 1)
    print(f"Validation Loss: {avg_loss:.6f}")
    return avg_loss