# evaluate.py
import os
import torch
from torch.utils.data import DataLoader
import argparse

from config import Config
cfg = Config()

# 从新网络模型中导入 DynamicsCore 和 PhysicsGuidedLoss
from models.dynamics_net import DynamicsCore, PhysicsGuidedLoss
from utils.preprocessing import load_thrust_allocation_matrix
from utils.dataset import PreprocessedDataset

def validate_model(model, loss_fn, val_loader):
    """
    在验证集上计算平均损失。

    :param model: 待评估模型，输入要求 (B, window_size*14) 和 (B, 6)
    :param loss_fn: 损失函数，要求输入 (pred_acc, true_acc, thrust, pred_ang, true_ang)
    :param val_loader: 验证集 DataLoader
    :return: avg_loss (float)
    """
    model.eval()  # 设置为评估模式
    device = next(model.parameters()).device

    total_loss = 0.0
    sample_count = 0

    with torch.no_grad():
        for batch in val_loader:
            # 将 batch 中所有张量移动到相同的 device 上
            for key in batch:
                batch[key] = batch[key].to(device)

            # 拼接 power_window 和 imu_window 形成模型输入，形状转换为 (B, window_size*14)
            features = torch.cat([batch['power_window'], batch['imu_window']], dim=2)  # (B, W, 14)
            features = features.view(features.size(0), -1)  # (B, W*14)

            thrust = batch['thrust']  # (B, 6)

            # 前向传播
            pred_acc, pred_ang = model(features, thrust)

            # 拆分真实标签：accel 标签 (B,6) 拆分为线性 (B,3) 和角 (B,3)
            true_acc = batch['accel'][:, :3]
            true_ang = batch['accel'][:, 3:]

            loss = loss_fn(pred_acc, true_acc, thrust, pred_ang, true_ang)

            batch_size = batch['accel'].size(0)
            total_loss += loss.item() * batch_size
            sample_count += batch_size

    avg_loss = total_loss / sample_count if sample_count > 0 else 0.0
    return avg_loss

def main(args):
    device = torch.device(cfg.device.DEVICE)

    # 构造评估数据集（这里使用整个预处理数据作为评估集）
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

    # 如果模型需要推力分配矩阵，此处加载（当前 DynamicsCore 不直接用到，可选）
    # thrust_matrix_np = load_thrust_allocation_matrix(cfg.paths.THRUST_MATRIX_FILE)
    # thrust_matrix = torch.tensor(thrust_matrix_np, device=device)

    # 构造新网络模型：使用 DynamicsCore
    model = DynamicsCore(
        input_dim=14,
        hidden_dim=cfg.training.PHYSICS_HIDDEN_DIM,
        window_size=cfg.training.WINDOW_SIZE
    )
    model.to(device)

    # 加载模型检查点
    checkpoint_path = args.checkpoint
    if not os.path.exists(checkpoint_path):
        print("Checkpoint file not found:", checkpoint_path)
        return
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)

    # 初始化损失函数
    criterion = PhysicsGuidedLoss(
        alpha=cfg.training.ALPHA,
        beta=cfg.training.BETA,
        gamma=cfg.training.GAMMA
    )

    # 开始评估
    print("开始评估……")
    avg_loss = validate_model(model, criterion, eval_loader)
    print("评估完成，平均总损失：{:.6f}".format(avg_loss))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help="模型检查点路径")
    # 新网络结构下不需要额外的模式切换参数
    args = parser.parse_args()
    main(args)
