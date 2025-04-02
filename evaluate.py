import os
import torch
import numpy as np
from torch.utils.data import DataLoader
import argparse

# --------------------------
# 项目内部模块导入
# --------------------------
from config import Config

cfg = Config()

from models.dynamics_net import (
    EnhancedPhysicsNet,
    HybridDynamicsModel,
    EnhancedDynamicsLoss
)
from utils.preprocessing import load_thrust_allocation_matrix
from utils.dataset import PreprocessedDataset


# 占位的端到端网络，输出形状为 (B,6)
class DummyE2ENet(torch.nn.Module):
    def __init__(self, window_size=5, hidden_dim=256):
        super().__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear((8 + 6) * window_size, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 6)
        )

    def forward(self, power_window, imu_window):
        x = torch.cat([power_window, imu_window], dim=-1).flatten(1)
        return self.fc(x)


def validate_model(model, loss_fn, val_loader):
    """
    对模型进行验证，返回在验证集上的平均损失。
    """
    model.eval()
    device = next(model.parameters()).device
    total_loss = 0.0
    sample_count = 0
    with torch.no_grad():
        for batch in val_loader:
            for k in batch:
                batch[k] = batch[k].to(device)
            outputs = model(batch['power_window'], batch['imu_window'])
            loss = loss_fn(outputs, batch)
            batch_size = batch['accel'].size(0)
            total_loss += loss.item() * batch_size
            sample_count += batch_size
    avg_loss = total_loss / sample_count if sample_count > 0 else 0.0
    return avg_loss


def main(args):
    device = torch.device(cfg.device.DEVICE)

    # 构造评估数据集（使用整个预处理数据）
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

    # 加载推力分配矩阵 (6x8)
    thrust_matrix_np = load_thrust_allocation_matrix(cfg.paths.THRUST_MATRIX_FILE)
    thrust_matrix = torch.tensor(thrust_matrix_np, device=device)

    # 初始化物理分支网络
    physics_net = EnhancedPhysicsNet(
        thrust_matrix=thrust_matrix,
        window_size=cfg.training.WINDOW_SIZE,
        hidden_dim=cfg.training.PHYSICS_HIDDEN_DIM
    )
    # 使用 DummyE2ENet 作为端到端分支
    e2e_net = DummyE2ENet(
        window_size=cfg.training.WINDOW_SIZE,
        hidden_dim=cfg.training.E2E_HIDDEN_DIM
    )
    # 构造混合模型
    model = HybridDynamicsModel(physics_net, e2e_net)
    model.to(device)

    # 加载模型检查点
    checkpoint_path = args.checkpoint
    if not os.path.exists(checkpoint_path):
        print("Checkpoint file not found:", checkpoint_path)
        return
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)

    # 初始化损失函数
    # 注意：若 TrainingConfig 中未定义 DELTA，则使用默认值 0.1
    criterion = EnhancedDynamicsLoss(
        alpha=cfg.training.ALPHA,
        beta=cfg.training.BETA,
        gamma=cfg.training.GAMMA,
        delta=cfg.training.DELTA if hasattr(cfg.training, "DELTA") else 0.1
    )

    print("开始评估……")
    avg_loss = validate_model(model, criterion, eval_loader)
    print("评估完成，平均总损失：{:.6f}".format(avg_loss))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="模型评估")
    parser.add_argument("--checkpoint", type=str, default="model_checkpoint.pt", help="模型检查点路径")
    parser.add_argument("--mode", type=str, default="physics", help="评估模式：physics 或 e2e")
    args = parser.parse_args()
    main(args)
