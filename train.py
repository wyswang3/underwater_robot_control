import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import argparse

# --------------------------
# 项目内部模块导入
# --------------------------
from config import Config
cfg = Config()

from models.dynamics_net import (
    EnhancedPhysicsNet,
    DirectMappingNet,
    HybridDynamicsModel,
    EnhancedDynamicsLoss,
    train_model  # 自定义训练函数（在此版本中为 custom_train_model）
)
from utils.preprocessing import load_thrust_allocation_matrix

# 导入评估模块和可视化函数
import evaluate
from utils.visualization import plot_loss_curve, visualize_predictions
from utils.dataset import PreprocessedDataset
# --------------------------
# 数据集定义（与 evaluate.py 共享）
# --------------------------
# --------------------------
# 自定义训练函数（包含 NaN 检测与网络切换功能，记录每个 epoch 损失）
# --------------------------
def check_continuous_zeros_in_array(arr, threshold_fraction=0.8, min_continuous=2):
    """
    检查一维数组 arr 中零值所占比例以及连续零值的最大长度。
    :param arr: 1D numpy array
    :param threshold_fraction: 如果零值比例超过此阈值，则视为异常
    :param min_continuous: 如果连续零值长度超过此值，则视为异常
    :return: (fraction, max_run)
    """
    zeros = (arr == 0)
    fraction = zeros.mean()
    max_run = 0
    current_run = 0
    for z in zeros:
        if z:
            current_run += 1
            if current_run > max_run:
                max_run = current_run
        else:
            current_run = 0
    return fraction, max_run


def check_dataset(dataset, threshold_fraction=0.8, min_continuous=20):
    """
    检查数据集中每个样本的电机功率数据和 IMU 数据是否存在大量且连续的0值。
    打印出有问题的样本和通道信息。
    :param dataset: 一个实现 __getitem__ 返回字典的 PyTorch 数据集，
                    其中必须包含 'power_window' 和 'imu_window' 键。
    :param threshold_fraction: 零值比例阈值，默认0.8
    :param min_continuous: 连续0值的最小长度阈值，默认2（你可根据实际窗口大小调整）
    """
    for i in range(len(dataset)):
        sample = dataset[i]
        # 检查电机功率数据：假设 shape (window_size, 8)
        power_data = sample['power_window']
        if isinstance(power_data, torch.Tensor):
            power_data = power_data.numpy()
        for ch in range(power_data.shape[1]):
            frac, max_run = check_continuous_zeros_in_array(power_data[:, ch],
                                                            threshold_fraction, min_continuous)
            if frac >= threshold_fraction and max_run >= min_continuous:
                print(
                    f"Warning: Sample {i} - Motor power channel {ch} has {frac:.2f} zeros with a continuous run of {max_run}.")

        # 检查 IMU 数据：假设 shape (window_size, 6)
        imu_data = sample['imu_window']
        if isinstance(imu_data, torch.Tensor):
            imu_data = imu_data.numpy()
        for ch in range(imu_data.shape[1]):
            frac, max_run = check_continuous_zeros_in_array(imu_data[:, ch],
                                                            threshold_fraction, min_continuous)
            if frac >= threshold_fraction and max_run >= min_continuous:
                print(
                    f"Warning: Sample {i} - IMU channel {ch} has {frac:.2f} zeros with a continuous run of {max_run}.")


def custom_train_model(model, loss_fn, train_loader, epochs=100, lr=1e-4):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg.training.WEIGHT_DECAY)
    model.train()
    device = torch.device(cfg.device.DEVICE)
    epoch_losses = []
    switched = False  # 标志是否已经切换到端到端模式

    for epoch in range(epochs):
        total_loss = 0.0
        for batch in train_loader:
            for k in batch:
                batch[k] = batch[k].to(device)
            outputs = model(batch['power_window'], batch['imu_window'])
            loss = loss_fn(outputs, batch)
            if torch.isnan(loss):
                if not switched:
                    print("检测到 NaN 值，切换到端到端网络训练")
                    model.switch_to_e2e()
                    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg.training.WEIGHT_DECAY)
                    switched = True
                continue  # 跳过当前 batch
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.CLIP_GRAD_NORM)
            optimizer.step()
            total_loss += loss.item() * batch['accel'].size(0)
        avg_loss = total_loss / len(train_loader.dataset)
        epoch_losses.append(avg_loss)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.6f}")
    return model, epoch_losses

# --------------------------
# 主函数：数据加载、模型初始化、训练、评估及结果可视化
# --------------------------
def main():
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device(cfg.device.DEVICE)

    # 加载推力分配矩阵 (6x8)
    thrust_matrix_np = load_thrust_allocation_matrix(cfg.paths.THRUST_MATRIX_FILE)
    thrust_matrix = torch.tensor(thrust_matrix_np, device=device)

    # 初始化物理引导网络和端到端网络，并使用混合网络包装
    physics_net = EnhancedPhysicsNet(
        thrust_matrix,
        window_size=cfg.training.WINDOW_SIZE,
        hidden_dim=cfg.training.HIDDEN_DIM
    )
    e2e_net = DirectMappingNet(
        window_size=cfg.training.WINDOW_SIZE,
        hidden_dim=cfg.training.HIDDEN_DIM
    )
    model = HybridDynamicsModel(physics_net, e2e_net)
    model.to(device)

    # 初始化损失函数
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

    # 检查数据集中是否存在大量且连续的0值（电机功率和IMU数据）
    print("开始检查数据集连续0值情况……")
    check_dataset(full_dataset, threshold_fraction=0.8, min_continuous=8)
    print("数据集检查完成。")

    split_ratio = 0.8
    train_size = int(len(full_dataset) * split_ratio)
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=cfg.training.BATCH_SIZE, shuffle=True, num_workers=cfg.training.NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=cfg.training.BATCH_SIZE, shuffle=False, num_workers=cfg.training.NUM_WORKERS)

    print("开始训练……")
    model, train_losses = custom_train_model(model, criterion, train_loader,
                                             epochs=cfg.training.NUM_EPOCHS,
                                             lr=cfg.training.LEARNING_RATE)
    checkpoint_path = os.path.join(cfg.paths.MODEL_DIR, "model_checkpoint.pt")
    torch.save(model.state_dict(), checkpoint_path)
    print("训练结束，模型已保存到", checkpoint_path)

    # 保存训练损失曲线到 SPLITS_DIR
    loss_plot_path = os.path.join(cfg.paths.SPLITS_DIR, "training_loss.png")
    plot_loss_curve(train_losses, save_path=loss_plot_path)
    print("训练损失曲线已保存到", loss_plot_path)

    # 调用评估模块进行评估（使用训练过程中选择的 active_net 模式）
    eval_args = argparse.Namespace(checkpoint=checkpoint_path)
    print("开始评估……")
    evaluate.main(eval_args)

    # 可视化验证集预测对比图，并保存到 SPLITS_DIR
    pred_plot_path = os.path.join(cfg.paths.SPLITS_DIR, "prediction_comparison.png")
    visualize_predictions(model, val_loader, device, save_path=pred_plot_path)
    print("预测对比图已保存到", pred_plot_path)

if __name__ == "__main__":
    main()
