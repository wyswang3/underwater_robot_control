import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt


def select_random_segment(dataset, num_samples: int):
    """
    从数据集中随机选取连续的 num_samples 个样本。
    如果数据集样本数量不足，则抛出 ValueError。

    返回：segment_samples (list)，start_idx (int)
      - segment_samples: 连续选取的样本列表
      - start_idx: 选取段在数据集中的起始索引
    """
    total_samples = len(dataset)
    if total_samples < num_samples:
        raise ValueError("数据集样本数量不足，无法选取连续数据段。")
    start_idx = random.randint(0, total_samples - num_samples)
    segment_samples = [dataset[i] for i in range(start_idx, start_idx + num_samples)]
    return segment_samples, start_idx


def extract_segment_data(segment_samples, device):
    """
    从选取的样本列表中提取所需数据并移动到指定 device：
      - 'power_window' (T, 8)
      - 'imu_window'   (T, 6)
      - 'accel'        (6,)
      - 'thrust'       (6,)

    返回四个 Tensor，分别是：
      - segment_power: (N, T, 8)
      - segment_imu:   (N, T, 6)
      - segment_accel_measured: (N, 6)
      - segment_thrust: (N, 6)
      其中 N = num_samples。
    """
    segment_power = []
    segment_imu = []
    segment_accel_measured = []
    segment_thrust = []

    for sample in segment_samples:
        segment_power.append(sample['power_window'])
        segment_imu.append(sample['imu_window'])
        segment_accel_measured.append(sample['accel'])
        segment_thrust.append(sample['thrust'])

    segment_power = torch.stack(segment_power, dim=0).to(device)
    segment_imu = torch.stack(segment_imu, dim=0).to(device)
    segment_accel_measured = torch.stack(segment_accel_measured, dim=0).to(device)
    segment_thrust = torch.stack(segment_thrust, dim=0).to(device)
    return segment_power, segment_imu, segment_accel_measured, segment_thrust


def plot_segment_comparison(time_axis: np.ndarray,
                            measured: np.ndarray,
                            predicted: np.ndarray,
                            save_path: str = None):
    """
    绘制选取数据段中各加速度轴随时间变化的对比图：
      - 左侧显示真实值与预测值的时间序列对比
      - 右侧显示预测残差随时间变化
    若数据有 6 个通道，则假设前 3 个为线性加速度，后 3 个为角加速度，
    标题中显示每个通道的 RMSE。

    参数：
      - time_axis: 一维时间序列 (shape: (N,))
      - measured: 真实加速度 (N, num_axes)
      - predicted: 预测加速度 (N, num_axes)
      - save_path: 若不为 None，则保存图像到此路径
    """
    num_axes = measured.shape[1]
    # 如果数据有 6 维通道，则定义命名；否则自动命名
    if num_axes == 6:
        axis_names = ["Linear Accel X", "Linear Accel Y", "Linear Accel Z",
                      "Angular Accel X", "Angular Accel Y", "Angular Accel Z"]
    else:
        axis_names = [f"Axis {i + 1}" for i in range(num_axes)]

    fig, axs = plt.subplots(nrows=num_axes, ncols=2, figsize=(14, 4 * num_axes))
    if num_axes == 1:
        axs = np.array([axs])  # 保证统一索引

    for i in range(num_axes):
        # 左侧时序对比
        ax_ts = axs[i, 0]
        ax_ts.plot(time_axis, measured[:, i], 'o-', label='Measured')
        ax_ts.plot(time_axis, predicted[:, i], 's--', label='Predicted')
        rmse = np.sqrt(np.mean((predicted[:, i] - measured[:, i]) ** 2))
        ax_ts.set_title(f"{axis_names[i]} Time Series (RMSE: {rmse:.3f})")
        ax_ts.set_xlabel("Time (s)")
        ax_ts.set_ylabel("Value")
        ax_ts.legend()
        ax_ts.grid(True)

        # 右侧残差图
        ax_res = axs[i, 1]
        residual = predicted[:, i] - measured[:, i]
        ax_res.plot(time_axis, residual, 'o-', color='purple', label='Residual')
        ax_res.axhline(0, color='red', linestyle='--')
        ax_res.set_title(f"{axis_names[i]} Residual")
        ax_res.set_xlabel("Time (s)")
        ax_res.set_ylabel("Residual")
        ax_res.legend()
        ax_res.grid(True)

    fig.tight_layout()
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300)
    plt.show()
    plt.close(fig)


def run_random_segment_fit(cfg, model, dataset, device,
                           dt: float = 0.1,
                           segment_duration: float = 15,
                           save_path: str = None):
    """
    综合调用上面函数，执行以下步骤：
      1) 计算每个样本覆盖的时间 = WINDOW_SIZE * dt
      2) 计算需要连续抽取的样本数量 num_samples, 使总时长约为 segment_duration
      3) 随机选取连续样本，提取数据并预测，然后可视化与计算 RMSE

    参数：
      - cfg: 配置对象，包含 training.WINDOW_SIZE
      - model: 已训练的模型
      - dataset: 数据集对象
      - device: 模型所在的 device
      - dt: 每个时间步的采样时间
      - segment_duration: 要选取的时长（秒）
      - save_path: 如果指定，则保存可视化结果到此路径
    """
    # 每个样本覆盖的时长 = window_size * dt
    sample_time = cfg.training.WINDOW_SIZE * dt
    # 连续样本数
    num_samples = int(segment_duration / sample_time)
    num_samples = max(1, num_samples)
    print(
        f"每个样本覆盖 {sample_time:.2f}s; 将选取 {num_samples} 个连续样本，总时长约 {num_samples * sample_time:.2f}s.")

    # 随机选取
    segment_samples, start_idx = select_random_segment(dataset, num_samples)
    print(f"选取数据段起始索引：{start_idx}")

    # 提取数据
    seg_power, seg_imu, seg_accel_measured, seg_thrust = extract_segment_data(segment_samples, device)
    # 拼接 power_window 和 imu_window => (N, T, 14)，再展平 => (N, T*14)
    segment_features = torch.cat([seg_power, seg_imu], dim=2).view(seg_power.size(0), -1)

    # 调用模型预测
    segment_accel_pred = predict_segment(model, segment_features, seg_thrust)

    # 检查维度一致性，不一致则截断
    if segment_accel_pred.shape[1] != seg_accel_measured.shape[1]:
        segment_accel_pred = segment_accel_pred[:, :seg_accel_measured.shape[1]]

    # 转为 numpy 以绘图
    segment_accel_pred = segment_accel_pred.cpu().numpy()
    seg_accel_measured = seg_accel_measured.cpu().numpy()

    # 构造时间轴
    time_axis = np.arange(num_samples) * sample_time

    if save_path is None:
        save_path = os.path.join(cfg.paths.SPLITS_DIR, "random_segment_comparison.png")

    # 绘图对比
    plot_segment_comparison(time_axis, seg_accel_measured, segment_accel_pred, save_path=save_path)

    # 计算整体 RMSE
    rmse = np.sqrt(np.mean((segment_accel_pred - seg_accel_measured) ** 2))
    print(f"随机抽取的 {num_samples * sample_time:.1f}s 数据段整体 RMSE: {rmse:.3f}")


def predict_segment(model, segment_features: torch.Tensor, thrust: torch.Tensor) -> torch.Tensor:
    """
    对选取的连续数据段进行预测。
    假设模型 forward 接口为 model(features, thrust)，返回四个值 (pred_lin, pred_ang, M, D)，
    这里只使用 pred_lin, pred_ang。
    """
    model.eval()
    with torch.no_grad():
        outputs = model(segment_features, thrust)
        if isinstance(outputs, tuple) and len(outputs) >= 2:
            pred_lin, pred_ang = outputs[:2]
        else:
            raise ValueError("Model output does not match expected format (pred_lin, pred_ang, ...).")
        preds = torch.cat([pred_lin, pred_ang], dim=1)  # (N, 6)
    return preds
