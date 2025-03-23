import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt


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


def plot_segment_comparison(time_axis, measured, predicted, save_path=None):
    """
    绘制选取数据段中每个加速度轴随时间变化的对比图，包括：
      - 左侧：测量值与预测值的时间序列对比
      - 右侧：预测残差（预测值-测量值）随时间变化
    每个子图标题中显示该轴的 RMSE 值。
    """
    num_axes = measured.shape[1]
    # 创建 num_axes 行，2列的子图
    fig, axs = plt.subplots(nrows=num_axes, ncols=2, figsize=(14, 4 * num_axes))

    for i in range(num_axes):
        # 左侧：时间序列对比
        ax_ts = axs[i, 0] if num_axes > 1 else axs[0]
        ax_ts.plot(time_axis, measured[:, i], 'o-', label='Measured')
        ax_ts.plot(time_axis, predicted[:, i], 's--', label='Predicted')
        ax_ts.set_xlabel("Time (s)")
        ax_ts.set_ylabel(f"Axis {i + 1}")
        # 计算该轴 RMSE
        axis_rmse = np.sqrt(np.mean((predicted[:, i] - measured[:, i]) ** 2))
        ax_ts.set_title(f"Axis {i + 1} Time Series (RMSE: {axis_rmse:.3f})")
        ax_ts.legend()
        ax_ts.grid(True)

        # 右侧：残差（预测误差）对比
        ax_res = axs[i, 1] if num_axes > 1 else axs[1]
        residual = predicted[:, i] - measured[:, i]
        ax_res.plot(time_axis, residual, 'o-', color='purple', label='Residual')
        ax_res.axhline(0, color='red', linestyle='--')
        ax_res.set_xlabel("Time (s)")
        ax_res.set_ylabel("Residual")
        ax_res.set_title(f"Axis {i + 1} Residual")
        ax_res.legend()
        ax_res.grid(True)

    fig.tight_layout()
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path)
    plt.show()
    plt.close(fig)


def run_random_segment_fit(cfg, model, dataset, device, dt=0.5, segment_duration=20):
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
    # 设置保存路径：存放到 cfg.paths.SPLITS_DIR 下
    save_path = os.path.join(cfg.paths.SPLITS_DIR, "random_segment_comparison.png")
    # 绘制对比图
    plot_segment_comparison(time_axis, segment_accel_measured, segment_accel_pred, save_path=save_path)
    # 计算整体 RMSE
    rmse = np.sqrt(np.mean((segment_accel_pred - segment_accel_measured) ** 2))
    print(f"随机抽取的 {num_samples * sample_time:.1f}s 数据段整体 RMSE: {rmse:.3f}")
