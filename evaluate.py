import os
import torch
from torch.utils.data import DataLoader
import argparse

from config import Config

cfg = Config()

# 更新导入：使用 SimpleHydroNet（而非 DeepHydroNet 或其他）
from models.dynamics_net import SimpleHydroNet
from utils.preprocessing import load_thrust_allocation_matrix
from utils.dataset import PreprocessedDataset


def validate_model(model, val_loader):
    """
    在验证集上计算平均损失。
    计算方式：分别计算线性和角加速度的 MSE，然后取平均。

    参数:
      - model: 待评估模型，输入要求为 (B, window_size*14) 和 (B, 6)
      - val_loader: 验证集 DataLoader
    返回:
      - avg_loss (float): 验证集上的平均 loss
    """
    model.eval()
    device = next(model.parameters()).device
    total_loss = 0.0
    sample_count = 0

    mse = torch.nn.MSELoss()

    with torch.no_grad():
        for batch in val_loader:
            # 将所有张量移动到指定设备
            for key in batch:
                batch[key] = batch[key].to(device)
            # 拼接 'power_window' 与 'imu_window' 为模型输入 (B, window_size*14)
            features = torch.cat([batch['power_window'], batch['imu_window']], dim=2)  # (B, W, 14)
            features = features.view(features.size(0), -1)  # (B, W*14)
            thrust = batch['thrust']  # (B, 6)
            # 前向传播
            pred_lin, pred_ang = model(features, thrust)
            # 真实标签：'accel' (B, 6) 分别拆分为 (B,3)
            true_lin = batch['accel'][:, :3]
            true_ang = batch['accel'][:, 3:]
            # 计算 MSELoss 分别
            loss_lin = mse(pred_lin, true_lin)
            loss_ang = mse(pred_ang, true_ang)
            loss = (loss_lin + loss_ang) / 2.0

            batch_size = batch['accel'].size(0)
            total_loss += loss.item() * batch_size
            sample_count += batch_size

    avg_loss = total_loss / sample_count if sample_count > 0 else 0.0
    return avg_loss


def main(args):
    device = torch.device(cfg.device.DEVICE)

    # 构造评估数据集
    full_dataset = PreprocessedDataset(
        features_file=cfg.paths.TRAIN_FEATURES_FILE,
        accel_file=cfg.paths.TRAIN_ACCEL_LABELS_FILE,
        angular_accel_file=cfg.paths.TRAIN_ANGULAR_ACCEL_LABELS_FILE,
        thrust_file=cfg.paths.TRAIN_THRUST_LABELS_FILE,
        window_size=cfg.training.WINDOW_SIZE
    )
    eval_loader = DataLoader(
        full_dataset,
        batch_size=cfg.training.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.training.NUM_WORKERS
    )

    # 构造新网络模型：使用 SimpleHydroNet
    model = SimpleHydroNet(
        window_size=cfg.training.WINDOW_SIZE,
        input_dim=cfg.training.INPUT_DIM if hasattr(cfg.training, 'INPUT_DIM') else 14,
        hidden_dim=cfg.training.PHYSICS_HIDDEN_DIM
    )
    model.to(device)

    # 加载模型检查点
    checkpoint_path = args.checkpoint
    if not os.path.exists(checkpoint_path):
        print("Checkpoint file not found:", checkpoint_path)
        return
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)

    # 开始评估
    print("开始评估……")
    avg_loss = validate_model(model, eval_loader)
    print("评估完成，平均总损失：{:.6f}".format(avg_loss))

