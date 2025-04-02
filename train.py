import os
import torch
import numpy as np
import argparse
import multiprocessing

# =============== 项目内部模块 ===============
from config import Config
from models.dynamics_net import (
    EnhancedPhysicsNet,
    # DirectMappingNet 已经从模型中移除
    HybridDynamicsModel,
    EnhancedDynamicsLoss
)
from utils.preprocessing import load_thrust_allocation_matrix
from utils.visualization import plot_loss_curve, visualize_predictions
from utils.random_segment_fit_utils import run_random_segment_fit
from utils.dataset import PreprocessedDataset
import evaluate  # 评估脚本
#激活服务器虚拟环境：conda activate /home/furui/pzy/wys_lstm/wyswang3_env
# 辅助函数：保证所有 tensor 至少有2个维度（batch 维度）
def ensure_batch_dim(batch):
    for k in batch:
        if isinstance(batch[k], torch.Tensor) and batch[k].dim() == 1:
            batch[k] = batch[k].unsqueeze(0)
    return batch

def check_continuous_zeros_in_array(arr, threshold_fraction=0.8, min_continuous=20):
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
    for i in range(len(dataset)):
        sample = dataset[i]
        power_data = sample['power_window']
        imu_data = sample['imu_window']
        if isinstance(power_data, torch.Tensor):
            power_data = power_data.numpy()
        if isinstance(imu_data, torch.Tensor):
            imu_data = imu_data.numpy()
        for ch in range(power_data.shape[1]):  # 8通道
            frac, max_run = check_continuous_zeros_in_array(power_data[:, ch],
                                                            threshold_fraction,
                                                            min_continuous)
            if frac >= threshold_fraction and max_run >= min_continuous:
                print(f"[WARN] Sample {i}, MotorPower ch={ch}, frac={frac:.2f}, max_run={max_run}")
        for ch in range(imu_data.shape[1]):  # 6通道
            frac, max_run = check_continuous_zeros_in_array(imu_data[:, ch],
                                                            threshold_fraction,
                                                            min_continuous)
            if frac >= threshold_fraction and max_run >= min_continuous:
                print(f"[WARN] Sample {i}, IMU ch={ch}, frac={frac:.2f}, max_run={max_run}")

def custom_train_model(model, loss_fn, train_loader,
                       epochs=100, lr=1e-4, weight_decay=1e-5,
                       grad_clip=1.0):
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    switched = False  # 是否切换模式（例如切换到端到端模式）
    epoch_losses = []
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        sample_count = 0
        for batch in train_loader:
            # 保证每个 batch 的 tensor 至少有2个维度
            batch = ensure_batch_dim(batch)
            for k in batch:
                batch[k] = batch[k].to(device)
            outputs = model(batch['power_window'], batch['imu_window'])
            loss = loss_fn(outputs, batch)
            if torch.isnan(loss):
                if not switched:
                    print("检测到 NaN，切换到端到端模式 & 重置优化器")
                    # 如果模型支持模式切换，可调用 model.switch_to_e2e() 之类的接口
                    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
                    switched = True
                continue  # 跳过当前 batch
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            batch_size = batch['accel'].size(0)
            total_loss += loss.item() * batch_size
            sample_count += batch_size
        avg_loss = total_loss / sample_count if sample_count > 0 else 0.0
        epoch_losses.append(avg_loss)
        print(f"Epoch[{epoch + 1}/{epochs}] - Loss: {avg_loss:.6f}")
    return model, epoch_losses

def validate_model(model, loss_fn, val_loader):
    model.eval()
    device = next(model.parameters()).device
    total_loss = 0.0
    sample_count = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = ensure_batch_dim(batch)
            for k in batch:
                batch[k] = batch[k].to(device)
            outputs = model(batch['power_window'], batch['imu_window'])
            loss = loss_fn(outputs, batch)
            batch_size = batch['accel'].size(0)
            total_loss += loss.item() * batch_size
            sample_count += batch_size
    avg_loss = total_loss / sample_count if sample_count > 0 else 0.0
    return avg_loss

def main():
    multiprocessing.set_start_method('spawn', force=True)
    cfg = Config()
    cfg.print_config()
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device(cfg.device.DEVICE)

    # 加载推力分配矩阵 (6x8)
    thrust_matrix_np = load_thrust_allocation_matrix(cfg.paths.THRUST_MATRIX_FILE)
    thrust_matrix = torch.tensor(thrust_matrix_np, device=device)

    # 初始化网络：使用改进后的 EnhancedPhysicsNet（端到端分支已从模型中移除）
    physics_net = EnhancedPhysicsNet(
        thrust_matrix=thrust_matrix,
        window_size=cfg.training.WINDOW_SIZE,
        hidden_dim=cfg.training.PHYSICS_HIDDEN_DIM
    )
    # 此处我们只使用物理约束网络（EnhancedPhysicsNet），若需要融合端到端分支，请自行扩展
    model = physics_net  # 或者 HybridDynamicsModel(physics_net, e2e_net) 如果有端到端分支
    model.to(device)

    # 定义损失函数（使用配置中的权重，注意此处不再传入 delta 参数）
    criterion = EnhancedDynamicsLoss(
        alpha=cfg.training.ALPHA,
        beta=cfg.training.BETA,
        gamma=cfg.training.GAMMA
    )

    # 加载预处理数据集
    full_dataset = PreprocessedDataset(
        features_file=cfg.paths.TRAIN_FEATURES_FILE,
        accel_file=cfg.paths.TRAIN_ACCEL_LABELS_FILE,
        angular_accel_file=cfg.paths.TRAIN_ANGULAR_ACCEL_LABELS_FILE,
        thrust_file=cfg.paths.TRAIN_THRUST_LABELS_FILE,
        window_size=cfg.training.WINDOW_SIZE
    )

    print("=== 检查数据集零值情况 ===")
    check_dataset(full_dataset, threshold_fraction=0.8, min_continuous=8)
    print("数据检查结束.")

    # 划分训练/验证集
    split_ratio = 0.8
    train_size = int(len(full_dataset) * split_ratio)
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.training.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.training.NUM_WORKERS
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.training.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.training.NUM_WORKERS
    )

    print("=== 开始训练 ===")
    model, train_losses = custom_train_model(
        model, criterion, train_loader,
        epochs=cfg.training.NUM_EPOCHS,
        lr=cfg.training.LEARNING_RATE,
        weight_decay=cfg.training.WEIGHT_DECAY,
        grad_clip=cfg.training.CLIP_GRAD_NORM
    )

    checkpoint_path = os.path.join(cfg.paths.MODEL_DIR, "model_checkpoint.pt")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"模型已保存 -> {checkpoint_path}")

    loss_plot_path = os.path.join(cfg.paths.SPLITS_DIR, "training_loss.png")
    plot_loss_curve(train_losses, save_path=loss_plot_path)
    print(f"训练损失曲线已保存 -> {loss_plot_path}")

    val_loss = validate_model(model, criterion, val_loader)
    print(f"Validation Loss (final): {val_loss:.6f}")

    eval_args = argparse.Namespace(checkpoint=checkpoint_path)
    print("=== 开始评估 ===")
    evaluate.main(eval_args)

    pred_plot_path = os.path.join(cfg.paths.SPLITS_DIR, "prediction_comparison.png")
    visualize_predictions(model, val_loader, device, save_path=pred_plot_path)
    print(f"预测可视化已保存 -> {pred_plot_path}")

    random_seg_plot_path = os.path.join(cfg.paths.SPLITS_DIR, "random_segment_comparison.png")
    run_random_segment_fit(cfg, model, full_dataset, device, dt=0.11, segment_duration=20,
                           save_path=random_seg_plot_path)
    print(f"随机数据段预测对比图已保存 -> {random_seg_plot_path}")

if __name__ == "__main__":
    main()
