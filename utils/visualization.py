import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import matplotlib

# 使用支持中文的字体（如 SimHei），避免中文字符缺失警告
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['axes.unicode_minus'] = False


def plot_loss_curve(train_losses, val_losses=None, save_path=None):
    """
    绘制训练过程中的损失曲线

    参数:
      train_losses: list or numpy array，每个 epoch 的训练损失
      val_losses: list or numpy array，每个 epoch 的验证损失（可选）
      save_path: str，如果指定，则保存图像到该路径
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
        # 自动创建保存目录
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.show()
    plt.close()


def visualize_predictions(model, dataloader, device, num_samples=6, save_path=None):
    """
    可视化模型在一个完整 batch 中的预测结果，并计算每个样本的RMSE误差。
    每个子图对应一个样本，x轴表示各维度（例如6维），曲线上分别绘制真实值与预测值，
    标题中标注该样本的RMSE，以更准确评估预测精度。

    参数：
      - model: 训练好的模型（EnhancedPhysicsNet/DirectMappingNet/HybridDynamicsModel）
      - dataloader: 数据加载器 (DataLoader)，用于获取一个完整 batch 的数据
      - device: 模型所在设备 (torch.device)
      - num_samples: 从 batch 中取出的样本数量（默认6）
      - save_path: 如果指定，则将图像保存到该路径
    """
    model.eval()
    with torch.no_grad():
        try:
            batch = next(iter(dataloader))
        except StopIteration:
            print("Visualization failed: DataLoader has no data!")
            return

        # 将batch中所有张量移动到device上
        for k in batch:
            batch[k] = batch[k].to(device)

        outputs = model(batch['power_window'], batch['imu_window'])
        if isinstance(outputs, dict):
            preds = outputs.get('accel_pred', None)
            if preds is None:
                print("Visualization failed: 'accel_pred' not found in model output!")
                return
        else:
            preds = outputs

        predictions = preds.cpu().numpy()  # shape: (B, D)
        targets = batch['accel'].cpu().numpy()  # shape: (B, D)

    B, dim_pred = predictions.shape
    _, dim_tar = targets.shape
    if dim_pred != dim_tar:
        print(f"Visualization failed: Prediction dimension ({dim_pred}) != Target dimension ({dim_tar})")
        return

    # 只取前 num_samples 个样本进行可视化
    predictions = predictions[:num_samples]
    targets = targets[:num_samples]

    # 使用subplots创建多个子图
    fig, axes = plt.subplots(nrows=num_samples, ncols=1, figsize=(12, 4 * num_samples))
    if num_samples == 1:
        axes = [axes]  # 保证axes是列表
    x_ticks = np.arange(dim_pred)  # 对应维度索引，例如 [0,1,...,dim_pred-1]
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
