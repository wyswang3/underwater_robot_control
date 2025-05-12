import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional   # ← 新增这一行


def select_random_segment(dataset, num_samples):
    """
    从数据集中随机选取连续的 num_samples 个样本
    返回：选取的样本列表和起始索引
    """
    total_samples = len(dataset)
    if total_samples < num_samples:
        raise ValueError("数据集样本数量不足，无法选取连续数据段。")
    start_idx = random.randint(0, total_samples - num_samples)
    segment_samples = [dataset[i] for i in range(start_idx, start_idx + num_samples)]
    return segment_samples, start_idx


def extract_segment_data(segment_samples, device):
    """
    从选取的样本列表中提取所需数据：
      - 每个样本包含 'power_window' (T, 8)，'imu_window' (T, 6)，'accel' (3,)
    返回：将各个样本堆叠为 tensor
    """
    segment_power = []
    segment_imu = []
    segment_accel_measured = []
    for sample in segment_samples:
        segment_power.append(sample['power_window'])
        segment_imu.append(sample['imu_window'])
        segment_accel_measured.append(sample['accel'])  # 假设使用最后一帧加速度作为标签
    segment_power = torch.stack(segment_power, dim=0).to(device)
    segment_imu = torch.stack(segment_imu, dim=0).to(device)
    segment_accel_measured = torch.stack(segment_accel_measured, dim=0).to(device)
    return segment_power, segment_imu, segment_accel_measured


def predict_segment(model, segment_power, segment_imu):
    """
    使用模型对选取的数据段进行预测，返回预测加速度。
    如果模型输出为字典，则取 'accel_pred'
    """
    with torch.no_grad():
        outputs = model(segment_power, segment_imu)
        if isinstance(outputs, dict):
            preds = outputs.get('accel_pred', None)
            if preds is None:
                raise ValueError("模型输出中未找到 'accel_pred'")
        else:
            preds = outputs
    return preds


def plot_segment_comparison(time_axis: np.ndarray,
                            gt: np.ndarray,
                            pred: np.ndarray,
                            save_path: Optional[str] = None) -> None:
    # 全局字体设置：Times 系列，字号 9pt
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'font.size': 9
    })

    n_axes = gt.shape[1]
    axis_names = (
        ["Linear Accel X", "Linear Accel Y", "Linear Accel Z",
         "Angular Accel X", "Angular Accel Y", "Angular Accel Z"]
        if n_axes == 6 else
        [f"Axis {i+1}" for i in range(n_axes)]
    )

    # 等宽两列，使用 constrained_layout 自动紧凑
    height = 8 * n_axes / 6
    fig, axes = plt.subplots(
        n_axes, 2,
        figsize=(8, height),
        sharex='col',
        constrained_layout=True
    )
    if n_axes == 1:
        axes = axes.reshape(1, 2)

    for i in range(n_axes):
        # 左：Measured vs Predicted
        ax_l = axes[i, 0]
        ax_l.plot(time_axis, gt[:, i],  'o-', color='red',
                  linewidth=1.5, markersize=3, markerfacecolor='none',
                  label='Measured')
        ax_l.plot(time_axis, pred[:, i], 'o-', color='blue',
                  linewidth=1.5, markersize=3, markerfacecolor='none',
                  label='Predicted')
        ax_l.set_ylabel(axis_names[i], fontsize=10)
        ax_l.tick_params(axis='y', labelsize=10)
        if i == 0:
            ax_l.legend(frameon=False, fontsize=9, loc='upper right')
        if i < n_axes - 1:
            ax_l.tick_params(labelbottom=False)
        else:
            ax_l.set_xlabel("Time (s)", fontsize=10)
            ax_l.tick_params(axis='x', labelsize=10)
        ax_l.grid(False)

        # 右：Residual
        ax_r = axes[i, 1]
        resid = pred[:, i] - gt[:, i]
        ax_r.plot(time_axis, resid, 'o-', color='purple',
                  linewidth=1.5, markersize=2, markerfacecolor='none',
                  label='Residual')
        ax_r.axhline(0, color='red', linestyle='--', linewidth=1.5)
        ax_r.set_ylabel("Residual", fontsize=10)
        ax_r.tick_params(axis='y', labelsize=10)
        if i == 0:
            ax_r.legend(frameon=False, fontsize=9, loc='upper right')
        if i < n_axes - 1:
            ax_r.tick_params(labelbottom=False)
        else:
            ax_r.set_xlabel("Time (s)", fontsize=10)
            ax_r.tick_params(axis='x', labelsize=10)
        ax_r.grid(False)

        # 对齐左列 y‐labels
        fig.align_ylabels(axes[:, 0])
        # 再微调一下边缘留白
        fig.tight_layout(pad=0.5)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)

def run_random_segment_fit(cfg, model, dataset, device, dt=0.5, segment_duration=30, save_path=None):
    """
    综合调用上述函数：
      - 根据 dt 和窗口大小计算每个样本覆盖的时间
      - 计算选取的连续样本数，使总时长约为 segment_duration 秒
      - 从数据集中随机选取连续样本，提取数据、预测、可视化并计算 RMSE
    """
    # 每个样本覆盖的时长
    sample_time = cfg.training.WINDOW_SIZE * dt
    num_samples = int(segment_duration / sample_time)
    if num_samples < 1:
        num_samples = 1
    print(
        f"每个样本覆盖 {sample_time:.2f}s; 将选取 {num_samples} 个连续样本，总时长约 {num_samples * sample_time:.2f}s.")

    # 随机选取连续样本
    segment_samples, start_idx = select_random_segment(dataset, num_samples)
    print(f"选取数据段起始索引：{start_idx}")

    # 提取数据
    segment_power, segment_imu, segment_accel_measured = extract_segment_data(segment_samples, device)
    # 模型预测
    segment_accel_pred = predict_segment(model, segment_power, segment_imu)
    # 如果预测结果维度与标签不一致，则取前几轴
    if segment_accel_pred.shape[1] != segment_accel_measured.shape[1]:
        segment_accel_pred = segment_accel_pred[:, :segment_accel_measured.shape[1]]
    # 转换为 numpy 数组
    segment_accel_pred = segment_accel_pred.cpu().numpy()
    segment_accel_measured = segment_accel_measured.cpu().numpy()
    # 构造时间轴
    time_axis = np.arange(num_samples) * sample_time

    # 如果没有传入保存路径，则默认保存到 SPLITS_DIR 下
    if save_path is None:
        save_path = os.path.join(cfg.paths.SPLITS_DIR, "random_segment_comparison.png")

    # 绘制对比图
    plot_segment_comparison(time_axis, segment_accel_measured, segment_accel_pred, save_path=save_path)

    # 计算整体 RMSE
    rmse = np.sqrt(np.mean((segment_accel_pred - segment_accel_measured) ** 2))
    print(f"随机抽取的 {num_samples * sample_time:.1f}s 数据段整体 RMSE: {rmse:.3f}")
