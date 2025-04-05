import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import matplotlib
from torch.utils.data import DataLoader
from typing import Optional, Union, List

# 设置支持中文字体，避免中文字符缺失
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['axes.unicode_minus'] = False

def plot_loss_curve(train_losses: Union[List[float], np.ndarray],
                    val_losses: Optional[Union[List[float], np.ndarray]] = None,
                    save_path: Optional[str] = None) -> None:
    """
    绘制训练过程中的损失曲线

    参数:
      - train_losses: 每个 epoch 的训练损失列表或数组
      - val_losses: 每个 epoch 的验证损失（可选）
      - save_path: 如果指定，则将图像保存到该路径
    """
    epochs = np.arange(1, len(train_losses) + 1)
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_losses, label='训练损失', marker='o')
    if val_losses is not None:
        plt.plot(epochs, val_losses, label='验证损失', marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("训练与验证损失曲线")
    plt.legend()
    plt.grid(True)
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.show()
    plt.close()

def visualize_predictions(model: torch.nn.Module,
                          dataloader: DataLoader,
                          device: torch.device,
                          num_samples: int = 6,
                          save_path: Optional[str] = None) -> None:
    """
    可视化模型在一个 batch 上的预测结果，并计算每个样本的 RMSE 误差。
    每个子图对应一个样本，显示真实值与预测值的比较，并在标题中标注 RMSE。

    参数：
      - model: 训练好的模型，接受输入形状 (B, window_size*14) 和 (B, 6)
      - dataloader: DataLoader，用于获取一个 batch 数据
      - device: 模型所在设备
      - num_samples: 从 batch 中取出的样本数量（默认 6）
      - save_path: 如果指定，则将图像保存到该路径
    """
    model.eval()
    with torch.no_grad():
        try:
            batch = next(iter(dataloader))
        except StopIteration:
            print("Visualization failed: DataLoader has no data!")
            return

        # 将 batch 中所有张量移动到指定设备
        for k in batch:
            batch[k] = batch[k].to(device)

        # 将 'power_window' 和 'imu_window' 拼接后展平
        # 假设 'power_window': (B, W, 8) 和 'imu_window': (B, W, 6)
        features = torch.cat([batch['power_window'], batch['imu_window']], dim=2)  # (B, W, 14)
        features = features.view(features.size(0), -1)  # (B, W*14)

        thrust = batch['thrust']  # (B, 6)

        # 模型前向传播
        outputs = model(features, thrust)
        # 这里假设模型返回的是一个元组 (pred_lin, pred_ang)
        pred_lin, pred_ang = outputs
        preds = torch.cat([pred_lin, pred_ang], dim=1)  # (B, 6)
        predictions = preds.cpu().numpy()  # (B, 6)
        targets = batch['accel'].cpu().numpy()  # (B, 6)

    B, dim_pred = predictions.shape
    if dim_pred != targets.shape[1]:
        print(f"Visualization failed: Prediction dimension ({dim_pred}) != Target dimension ({targets.shape[1]})")
        return

    # 只取前 num_samples 个样本进行可视化
    predictions = predictions[:num_samples]
    targets = targets[:num_samples]

    # 创建子图
    fig, axes = plt.subplots(nrows=num_samples, ncols=1, figsize=(12, 4 * num_samples))
    if num_samples == 1:
        axes = [axes]
    x_ticks = np.arange(dim_pred)
    for i, ax in enumerate(axes):
        sample_pred = predictions[i]
        sample_target = targets[i]
        rmse = np.sqrt(np.mean((sample_pred - sample_target) ** 2))
        ax.plot(x_ticks, sample_target, label='真实值', marker='o')
        ax.plot(x_ticks, sample_pred, label='预测值', marker='x')
        ax.set_title(f"样本 {i + 1}: 预测 vs 真实 (RMSE: {rmse:.3f})")
        ax.set_xlabel("维度")
        ax.set_ylabel("数值")
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.show()
    plt.close()
