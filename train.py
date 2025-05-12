import os
import torch
import numpy as np
import argparse
import multiprocessing
from tqdm import tqdm  # 用于显示训练进度

# =============== 项目内部模块 ===============
from config import Config
from models.dynamics_net import (
    EnhancedPhysicsNet,
    DirectMappingNet,
    HybridDynamicsModel,
    EnhancedDynamicsLoss
)
from utils.preprocessing import load_thrust_allocation_matrix
from utils.visualization import plot_loss_curve, visualize_global_accuracy,visualize_overall_error
from utils.random_segment_fit_utils import run_random_segment_fit
from utils.dataset import PreprocessedDataset
import evaluate  # 评估脚本


def batch_to_device(batch, device):
    """
    将字典或序列形式的 batch 中的每个 tensor 移到指定 device 上
    """
    if isinstance(batch, dict):
        return {k: batch_to_device(v, device) for k, v in batch.items()}
    elif isinstance(batch, (list, tuple)):
        return type(batch)(batch_to_device(v, device) for v in batch)
    elif torch.is_tensor(batch):
        return batch.to(device, non_blocking=True)
    else:
        return batch


def check_continuous_zeros_in_array(arr, threshold_fraction=0.8, min_continuous=20):
    """
    检查一维数组 arr 中零值所占比例以及连续零值的最大长度。
    返回 (fraction, max_run)
    """
    zeros = (arr == 0)
    fraction = zeros.mean()
    max_run = 0
    current_run = 0
    for z in zeros:
        if z:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 0
    return fraction, max_run


def check_dataset(dataset, threshold_fraction=0.8, min_continuous=20):
    """
    检查数据集中 'power_window' (B,T,8) 与 'imu_window' (B,T,6) 中连续的0值情况，
    打印存在较高比例和长连续0值的样本信息。
    """
    for i in range(len(dataset)):
        sample = dataset[i]
        power_data = sample['power_window']
        imu_data = sample['imu_window']
        if torch.is_tensor(power_data):
            power_data = power_data.numpy()
        if torch.is_tensor(imu_data):
            imu_data = imu_data.numpy()
        for ch in range(power_data.shape[1]):
            frac, max_run = check_continuous_zeros_in_array(power_data[:, ch],
                                                            threshold_fraction,
                                                            min_continuous)
            if frac >= threshold_fraction and max_run >= min_continuous:
                print(f"[WARN] Sample {i}, MotorPower ch={ch}, frac={frac:.2f}, max_run={max_run}")
        for ch in range(imu_data.shape[1]):
            frac, max_run = check_continuous_zeros_in_array(imu_data[:, ch],
                                                            threshold_fraction,
                                                            min_continuous)
            if frac >= threshold_fraction and max_run >= min_continuous:
                print(f"[WARN] Sample {i}, IMU ch={ch}, frac={frac:.2f}, max_run={max_run}")


def custom_train_model(model, loss_fn, train_loader, optimizer, scheduler=None,
                       epochs=100, grad_clip=1.0):
    """
    Custom training function:
      - For each epoch, compute the average loss over all batches.
      - When NaN loss is detected, switch to end-to-end mode and reset optimizer.
      - If using OneCycleLR, call scheduler.step() after each batch;
        otherwise, call scheduler.step(avg_loss) at the end of each epoch.
    """
    switched = False  # Whether already switched to end-to-end mode
    epoch_losses = []
    device = next(model.parameters()).device

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        sample_count = 0

        # 直接遍历 train_loader，不使用 tqdm 进度条包装
        for batch in train_loader:
            batch = batch_to_device(batch, device)
            outputs = model(batch['power_window'], batch['imu_window'])
            loss = loss_fn(outputs, batch)

            if torch.isnan(loss):
                if not switched:
                    print("NaN detected, switching to end-to-end mode and resetting optimizer.")
                    model.switch_to_e2e()  # Ensure the model implements this method.
                    optimizer = torch.optim.AdamW(model.parameters(),
                                                  lr=optimizer.defaults['lr'],
                                                  weight_decay=optimizer.defaults['weight_decay'])
                    switched = True
                continue  # Skip current batch

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            # For OneCycleLR: call scheduler.step() after each optimizer step
            if scheduler and isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                scheduler.step()

            b_size = batch['accel'].size(0)
            total_loss += loss.item() * b_size
            sample_count += b_size

        avg_loss = total_loss / sample_count if sample_count > 0 else 0.0
        epoch_losses.append(avg_loss)
        # For non-OneCycleLR schedulers, call scheduler.step(avg_loss) at epoch end.
        if scheduler and not isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
            scheduler.step(avg_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch + 1}/{epochs}] - Loss: {avg_loss:.6f}, LR: {current_lr:.2e}")

    return model, epoch_losses


def validate_model(model, loss_fn, val_loader):
    """
    验证模型，返回在验证集上的平均 loss
    """
    model.eval()
    device = next(model.parameters()).device
    total_loss = 0.0
    sample_count = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch_to_device(batch, device)
            outputs = model(batch['power_window'], batch['imu_window'])
            loss = loss_fn(outputs, batch)
            b_size = batch['accel'].size(0)
            total_loss += loss.item() * b_size
            sample_count += b_size
    avg_loss = total_loss / sample_count if sample_count > 0 else 0.0
    return avg_loss


def main():
    # Windows 下 DataLoader 多进程需使用 spawn 模式
    multiprocessing.set_start_method('spawn', force=True)

    # 1) 读取配置
    cfg = Config()
    cfg.print_config()

    torch.manual_seed(42)
    np.random.seed(42)

    # 2) 设备设置
    if cfg.device.DEVICE.startswith("cuda") and torch.cuda.is_available():
        device = torch.device(cfg.device.DEVICE)
    else:
        print(f"[WARN] 配置中的设备 {cfg.device.DEVICE} 不可用，切换为 CPU")
        device = torch.device("cpu")
    print("Using device:", device)

    # 3) 加载推力分配矩阵 (6x8)
    thrust_matrix_np = load_thrust_allocation_matrix(cfg.paths.THRUST_MATRIX_FILE)
    thrust_matrix = torch.tensor(thrust_matrix_np, device=device)

    # 4) 初始化网络
    physics_net = EnhancedPhysicsNet(thrust_matrix=thrust_matrix,
                                     window_size=cfg.training.WINDOW_SIZE,
                                     hidden_dim=cfg.training.PHYSICS_HIDDEN_DIM)
    e2e_net = DirectMappingNet(window_size=cfg.training.WINDOW_SIZE,
                               hidden_dim=cfg.training.E2E_HIDDEN_DIM)
    model = HybridDynamicsModel(physics_net, e2e_net).to(device)

    # 5) 定义损失函数
    criterion = EnhancedDynamicsLoss(alpha=cfg.training.ALPHA,
                                     beta=cfg.training.BETA,
                                     gamma=cfg.training.GAMMA)

    # 6) 加载数据集
    full_dataset = PreprocessedDataset(features_file=cfg.paths.TRAIN_FEATURES_FILE,
                                       accel_file=cfg.paths.TRAIN_ACCEL_LABELS_FILE,
                                       angular_accel_file=cfg.paths.TRAIN_ANGULAR_ACCEL_LABELS_FILE,
                                       thrust_file=cfg.paths.TRAIN_THRUST_LABELS_FILE,
                                       window_size=cfg.training.WINDOW_SIZE)
    print("=== 检查数据集零值情况 ===")
    check_dataset(full_dataset, threshold_fraction=0.8, min_continuous=8)
    print("数据集检查结束.")

    # 7) 划分训练/验证集
    split_ratio = 0.8
    train_size = int(len(full_dataset) * split_ratio)
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=cfg.training.BATCH_SIZE,
                                               shuffle=True,
                                               num_workers=cfg.training.NUM_WORKERS)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=cfg.training.BATCH_SIZE,
                                             shuffle=False,
                                             num_workers=cfg.training.NUM_WORKERS)

    # 8) 设置学习率调度器 —— 使用 OneCycleLR
    total_steps = cfg.training.NUM_EPOCHS * len(train_loader)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.LEARNING_RATE,
                                  weight_decay=cfg.training.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg.training.LEARNING_RATE,
        total_steps=total_steps,
        pct_start=0.2,  # 前20%的步数用于 warm-up
        anneal_strategy='linear',
        final_div_factor=cfg.training.LEARNING_RATE / cfg.training.MIN_LR
    )

    # 9) 开始训练
    print("=== 开始训练 ===")
    model, train_losses = custom_train_model(model, criterion, train_loader,
                                             optimizer=optimizer,
                                             scheduler=scheduler,
                                             epochs=cfg.training.NUM_EPOCHS,
                                             grad_clip=cfg.training.CLIP_GRAD_NORM)

    # 10) 保存模型
    checkpoint_path = os.path.join(cfg.paths.MODEL_DIR, "model_checkpoint.pt")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"模型已保存 -> {checkpoint_path}")

    # 11) 绘制训练损失曲线
    loss_plot_path = os.path.join(cfg.paths.SPLITS_DIR, "training_loss.png")
    plot_loss_curve(train_losses, save_path=loss_plot_path)
    print(f"训练损失曲线已保存 -> {loss_plot_path}")

    # 12) 验证集评估
    val_loss = validate_model(model, criterion, val_loader)
    print(f"Validation Loss (final): {val_loss:.6f}")

    # 13) 调用评估脚本
    eval_args = argparse.Namespace(checkpoint=checkpoint_path)
    print("=== 开始评估 ===")
    evaluate.main(eval_args)

    # 14) 预测结果可视化
    pred_plot_path = os.path.join(cfg.paths.SPLITS_DIR, "prediction_comparison.png")
    visualize_global_accuracy(model, val_loader, device, save_path=pred_plot_path)
    print(f"Prediction visualization saved -> {pred_plot_path}")
    overall_error_path = os.path.join(cfg.paths.SPLITS_DIR, "overall_error.png")
    visualize_overall_error(model, val_loader, device, save_path=overall_error_path)
    print(f"Overall error visualization saved -> {overall_error_path}")

    # 15) 随机数据段预测对比
    full_dataset = PreprocessedDataset(features_file=cfg.paths.TRAIN_FEATURES_FILE,
                                       accel_file=cfg.paths.TRAIN_ACCEL_LABELS_FILE,
                                       angular_accel_file=cfg.paths.TRAIN_ANGULAR_ACCEL_LABELS_FILE,
                                       thrust_file=cfg.paths.TRAIN_THRUST_LABELS_FILE,
                                       window_size=cfg.training.WINDOW_SIZE)
    random_seg_plot_path = os.path.join(cfg.paths.SPLITS_DIR, "random_segment_comparison.png")
    run_random_segment_fit(cfg, model, full_dataset, device, dt=0.11, segment_duration=22,
                           save_path=random_seg_plot_path)
    print(f"随机数据段预测对比图已保存 -> {random_seg_plot_path}")


if __name__ == "__main__":
    main()
