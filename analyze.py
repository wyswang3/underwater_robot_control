#!/usr/bin/env python3
import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from config import Config
from models.dynamics_net import EnhancedPhysicsNet, DirectMappingNet, HybridDynamicsModel
from utils.preprocessing import load_thrust_allocation_matrix
from utils.dataset import PreprocessedDataset


def main():
    # 1. 加载配置与设备
    cfg = Config()
    device = torch.device(cfg.device.DEVICE)

    # 设置采样间隔 dt（单位秒），假设为0.5秒（如果配置中有该参数，可直接使用）
    dt = 0.5
    # 每个样本覆盖的时间 = WINDOW_SIZE * dt
    sample_time = cfg.training.WINDOW_SIZE * dt
    # 要获得大约15秒的数据段，需要连续采样的个数
    num_samples = int(15 / sample_time)
    if num_samples < 1:
        num_samples = 1
    print(
        f"Each sample covers {sample_time:.2f}s; selecting {num_samples} consecutive samples for a total of ~{num_samples * sample_time:.2f}s.")

    # 2. 加载推力分配矩阵 (6x8)
    thrust_matrix_np = load_thrust_allocation_matrix(cfg.paths.THRUST_MATRIX_FILE)
    thrust_matrix = torch.tensor(thrust_matrix_np, device=device)

    # 3. 构造模型（物理网络和端到端网络构成混合模型）
    physics_net = EnhancedPhysicsNet(
        thrust_matrix,
        window_size=cfg.training.WINDOW_SIZE,
        hidden_dim=cfg.training.PHYSICS_HIDDEN_DIM
    )
    e2e_net = DirectMappingNet(
        window_size=cfg.training.WINDOW_SIZE,
        hidden_dim=cfg.training.E2E_HIDDEN_DIM
    )
    model = HybridDynamicsModel(physics_net, e2e_net)
    model.to(device)
    model.eval()

    # 4. 加载训练好的模型检查点
    checkpoint_path = os.path.join(cfg.paths.MODEL_DIR, "model_checkpoint.pt")
    if not os.path.exists(checkpoint_path):
        print("Checkpoint not found:", checkpoint_path)
        return
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)

    # 5. 加载预处理后的数据集
    dataset = PreprocessedDataset(
        features_file=cfg.paths.TRAIN_FEATURES_FILE,
        accel_file=cfg.paths.TRAIN_ACCEL_LABELS_FILE,
        angular_accel_file=cfg.paths.TRAIN_ANGULAR_ACCEL_LABELS_FILE,
        thrust_file=cfg.paths.TRAIN_THRUST_LABELS_FILE,
        window_size=cfg.training.WINDOW_SIZE
    )
    total_samples = len(dataset)
    if total_samples < num_samples:
        print("Dataset is too small for the requested segment length.")
        return

    # 随机选取一个起始索引，确保可以取出 num_samples 连续样本
    start_idx = random.randint(0, total_samples - num_samples)
    print(f"Selected segment starting at index {start_idx} covering {num_samples} consecutive samples.")

    # 6. 收集连续样本数据
    segment_power = []
    segment_imu = []
    segment_accel_measured = []
    for i in range(start_idx, start_idx + num_samples):
        sample = dataset[i]
        segment_power.append(sample['power_window'])  # shape: (T, 8)
        segment_imu.append(sample['imu_window'])  # shape: (T, 6)
        segment_accel_measured.append(sample['accel'])  # shape: (3,) —— 取最后一帧的加速度作为标签
    # 将列表堆叠为 tensor，形状：(num_samples, T, channels) 或 (num_samples, 3)
    segment_power = torch.stack(segment_power, dim=0).to(device)
    segment_imu = torch.stack(segment_imu, dim=0).to(device)
    segment_accel_measured = torch.stack(segment_accel_measured, dim=0).to(device)  # shape: (num_samples, 3)

    # 7. 利用模型对该段数据进行预测
    with torch.no_grad():
        outputs = model(segment_power, segment_imu)
        # 若输出为字典，则取 'accel_pred'
        if isinstance(outputs, dict):
            segment_accel_pred = outputs.get('accel_pred')
        else:
            segment_accel_pred = outputs

    # 有时模型可能预测6轴数据，而标签只有3轴，这里假设取预测结果的前3轴与测量数据对比
    if segment_accel_pred.shape[1] != segment_accel_measured.shape[1]:
        segment_accel_pred = segment_accel_pred[:, :segment_accel_measured.shape[1]]

    # 转换为 numpy 数组以便绘图
    segment_accel_pred = segment_accel_pred.cpu().numpy()  # shape: (num_samples, 3)
    segment_accel_measured = segment_accel_measured.cpu().numpy()  # shape: (num_samples, 3)

    # 构造时间轴（每个样本覆盖 sample_time 秒）
    time_axis = np.arange(num_samples) * sample_time

    # 8. 绘制预测与测量加速度的对比曲线：对每个轴绘制一张子图
    num_axes = segment_accel_measured.shape[1]
    fig, axs = plt.subplots(nrows=num_axes, ncols=1, figsize=(10, 4 * num_axes))
    if num_axes == 1:
        axs = [axs]
    for ax, axis_idx in zip(axs, range(num_axes)):
        ax.plot(time_axis, segment_accel_measured[:, axis_idx], 'o-', label='Measured')
        ax.plot(time_axis, segment_accel_pred[:, axis_idx], 's--', label='Predicted')
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(f"Acceleration Axis {axis_idx + 1}")
        ax.set_title(f"Acceleration Comparison on Axis {axis_idx + 1}")
        ax.legend()
        ax.grid(True)
    fig.tight_layout()
    save_path_fig = os.path.join(cfg.paths.SPLITS_DIR, "segment_acceleration_comparison.png")
    os.makedirs(os.path.dirname(save_path_fig), exist_ok=True)
    fig.savefig(save_path_fig)
    plt.show()
    plt.close(fig)

    # 9. 计算并输出整体 RMSE（所有轴综合）
    rmse = np.sqrt(np.mean((segment_accel_pred - segment_accel_measured) ** 2))
    print(f"Overall RMSE for the selected {num_samples * sample_time:.1f}s segment: {rmse:.3f}")


if __name__ == "__main__":
    main()
