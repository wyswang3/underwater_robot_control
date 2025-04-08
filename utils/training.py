import torch
from torch.cuda.amp import autocast, GradScaler
import torch.nn as nn


def batch_to_device(batch, device):
    """
    递归地将 batch 内所有 Tensor（包括嵌套在字典或列表中的）移动到指定 device 上。
    """
    if isinstance(batch, dict):
        return {key: batch_to_device(val, device) for key, val in batch.items()}
    elif isinstance(batch, list):
        return [batch_to_device(item, device) for item in batch]
    elif isinstance(batch, torch.Tensor):
        return batch.to(device)
    else:
        return batch


def process_batch(batch, device):
    """
    从 batch 字典中提取输入数据和标签，转换并移动到指定 device 上。

    假设 batch 包含：
      - 'power_window': Tensor, shape (B, window_size, 8)
      - 'imu_window':   Tensor, shape (B, window_size, 6)
      - 'accel':        Tensor, shape (B, 6)  （前3为线性加速度，后3为角加速度）
      - 'thrust':       Tensor, shape (B, 6)
    返回：
      - features: Tensor, shape (B, window_size*14)
      - true_lin: Tensor, shape (B, 3)
      - true_ang: Tensor, shape (B, 3)
      - thrust:   Tensor, shape (B, 6)
    """
    batch = batch_to_device(batch, device)
    power_window = batch['power_window']  # (B, W, 8)
    imu_window = batch['imu_window']  # (B, W, 6)
    features = torch.cat([power_window, imu_window], dim=2).view(power_window.size(0), -1)
    accel_all = batch['accel']  # (B, 6)
    true_lin = accel_all[:, :3]
    true_ang = accel_all[:, 3:]
    thrust = batch['thrust']  # (B, 6)

    return features.float(), true_lin.float(), true_ang.float(), thrust.float()


def custom_train_model(model, criterion, train_loader, epochs, lr, weight_decay, grad_clip, scheduler=None):
    """
    训练模型，支持混合精度、梯度裁剪和学习率调度。

    参数：
      - model: 待训练模型
      - criterion: 物理感知损失函数（此处要求调用格式为:
                   (pred_lin, true_lin, pred_ang, true_ang, tau, M, D)）
      - train_loader: 训练 DataLoader（返回字典）
      - epochs, lr, weight_decay, grad_clip: 训练超参数
      - scheduler: 学习率调度器（可选）

    返回：
      - model: 训练后的模型
      - train_losses: 每个 epoch 的平均损失列表
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = GradScaler()  # 混合精度训练
    train_losses = []
    model.train()
    device = next(model.parameters()).device

    for epoch in range(epochs):
        running_loss = 0.0
        for batch in train_loader:
            features, true_lin, true_ang, thrust = process_batch(batch, device)
            optimizer.zero_grad()
            with autocast():
                # 现在模型返回 4 个值
                pred_lin, pred_ang, M, D = model(features, thrust)
                # 此处假设我们使用 thrust 作为 tau
                loss = criterion(pred_lin, true_lin, pred_ang, true_ang, thrust, M, D)
            scaler.scale(loss).backward()
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * features.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        print(f"Epoch [{epoch + 1}/{epochs}] - Loss: {epoch_loss:.6f}")
        if scheduler is not None:
            scheduler.step()
    return model, train_losses


def validate_model(model, criterion, val_loader):
    """
    在验证集上评估模型，返回平均 loss。

    参数：
      - model: 待评估模型
      - criterion: 损失函数，要求接受 (pred_lin, true_lin, pred_ang, true_ang, tau, M, D)
      - val_loader: 验证 DataLoader
    返回：
      - avg_loss (float): 验证集上的平均 loss
    """
    model.eval()
    device = next(model.parameters()).device
    total_loss = 0.0
    sample_count = 0
    with torch.no_grad():
        for batch in val_loader:
            features, true_lin, true_ang, thrust = process_batch(batch, device)
            pred_lin, pred_ang, M, D = model(features, thrust)
            loss = criterion(pred_lin, true_lin, pred_ang, true_ang, thrust, M, D)
            batch_size = batch['accel'].size(0)
            total_loss += loss.item() * batch_size
            sample_count += batch_size
    avg_loss = total_loss / sample_count if sample_count > 0 else 0.0
    return avg_loss
