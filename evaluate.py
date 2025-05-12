#!/usr/bin/env python
# evaluate.py
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
    DirectMappingNet,
    HybridDynamicsModel,
    EnhancedDynamicsLoss,
)
from utils.preprocessing import load_thrust_allocation_matrix
from utils.dataset import PreprocessedDataset
from utils.visualization import aggregate_predictions  # 新增聚合函数导入


def validate_model(model, loss_fn, val_loader):
    """
    对模型进行验证，返回在验证集上的平均损失 avg_loss。
    """
    model.eval()
    device = next(model.parameters()).device

    total_loss = 0.0
    sample_count = 0
    with torch.no_grad():
        for batch in val_loader:
            # 移动到设备
            for k in batch:
                batch[k] = batch[k].to(device)
            outputs = model(batch['power_window'], batch['imu_window'])
            loss = loss_fn(outputs, batch)
            bsz = batch['accel'].size(0)
            total_loss += loss.item() * bsz
            sample_count += bsz
    return total_loss / sample_count if sample_count > 0 else 0.0


def main(args):
    device = torch.device(cfg.device.DEVICE)

    # 读取验证集
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

    # 加载推力矩阵
    thrust_matrix_np = load_thrust_allocation_matrix(cfg.paths.THRUST_MATRIX_FILE)
    thrust_matrix = torch.tensor(thrust_matrix_np, device=device)

    # 构建模型
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

    # 加载检查点
    checkpoint_path = args.checkpoint
    if not os.path.exists(checkpoint_path):
        print("Checkpoint file not found:", checkpoint_path)
        return
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)

    # 切换子网络模式（可选）
    if hasattr(model, 'active_net') and args.mode:
        mode = args.mode.lower()
        if mode in ['physics', 'e2e']:
            model.active_net = mode
            print(f"Switched to mode: {mode}")
        else:
            print(f"Invalid mode {args.mode}, using default: {model.active_net}")

    # 定义损失函数
    criterion = EnhancedDynamicsLoss(
        alpha=cfg.training.ALPHA,
        beta=cfg.training.BETA,
        gamma=cfg.training.GAMMA
    )

    # 验证
    print("Starting evaluation...")
    avg_loss = validate_model(model, criterion, eval_loader)
    print(f"Average loss: {avg_loss:.6f}")

    # 计算全数据集6轴RMSE和MAE
    preds, gts = aggregate_predictions(model, eval_loader, device, max_samples=len(full_dataset))
    rmse_per_dim = torch.sqrt(torch.mean((preds - gts) ** 2, dim=0))
    mae_per_dim = torch.mean(torch.abs(preds - gts), dim=0)
    # 计算 RMSE/MAE
    rmse_per_dim = torch.sqrt(torch.mean((preds - gts) ** 2, dim=0)).cpu().numpy()
    mae_per_dim = torch.mean(torch.abs(preds - gts), dim=0).cpu().numpy()

    # 转成 NumPy 以后，直接用 np.round 支持的小数位参数
    rmse_per_dim_4 = np.round(rmse_per_dim, 4)
    mae_per_dim_4 = np.round(mae_per_dim, 4)

    print("RMSE per dimension:", rmse_per_dim_4.tolist())
    print("MAE  per dimension:", mae_per_dim_4.tolist())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate dynamics model")
    parser.add_argument(
        '--checkpoint', type=str,
        default='models/checkpoints/model_checkpoint.pt',
        help='Path to checkpoint file'
    )
    parser.add_argument(
        '--mode', type=str, choices=['physics','e2e'], default='',
        help='Optional: evaluate physics or e2e subnetwork'
    )
    args = parser.parse_args()
    main(args)
