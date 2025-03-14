#evaluate.py
import os
import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
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

# 假设数据集类 PreprocessedDataset 已经在 train.py 中定义，
# 你可以直接从 train.py 导入或单独提取到 utils 模块中
from utils.dataset import PreprocessedDataset


def validate_model(model, loss_fn, val_loader):
    """
    对模型进行验证，返回在验证集上的平均损失 avg_loss。

    :param model: 已训练的模型 (physics/e2e/hybrid)
    :param loss_fn: 损失函数 (如 EnhancedDynamicsLoss)
    :param val_loader: 验证集 DataLoader
    :return: avg_loss (float), 验证集平均 loss
    """
    model.eval()  # 设置为评估模式
    device = next(model.parameters()).device  # 取得模型参数所在设备

    total_loss = 0.0
    sample_count = 0

    with torch.no_grad():
        for batch in val_loader:
            # 将 batch 中所有张量移动到相同的 device 上
            for k in batch:
                batch[k] = batch[k].to(device)

            # 前向传播
            outputs = model(batch['power_window'], batch['imu_window'])

            # 计算损失
            loss = loss_fn(outputs, batch)

            # 累加加权损失
            batch_size = batch['accel'].size(0)
            total_loss += loss.item() * batch_size
            sample_count += batch_size

    avg_loss = total_loss / sample_count if sample_count > 0 else 0.0

    # 如果需要再次切回训练模式，可加:
    # model.train()

    return avg_loss


def main(args):
    device = torch.device(cfg.device.DEVICE)

    # 构造评估数据集（这里使用整个预处理数据作为评估集；如需要可按比例划分）
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

    # 构造混合网络：同时包含物理引导网络和端到端网络
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

    # 加载模型检查点
    checkpoint_path = args.checkpoint
    if not os.path.exists(checkpoint_path):
        print("Checkpoint file not found:", checkpoint_path)
        return
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)

    # 不强制修改 active_net，而是先打印当前网络模式（注意：state_dict 不保存 active_net，
    # 如果没有额外保存该属性，模型会使用 __init__ 中的默认值）
    if hasattr(model, "active_net"):
        print("当前评估模式（训练过程中确定）：", model.active_net)

    # 如果命令行参数中指定了评估模式，则根据参数进行切换；否则保持训练时模式
    if hasattr(args, "mode") and args.mode:
        eval_mode = args.mode.lower()
        if eval_mode in ["physics", "e2e"]:
            model.active_net = eval_mode
            print("根据命令行参数切换评估模式为：", eval_mode)
        else:
            print("未知模式参数，保持当前模式：", model.active_net)

    # 初始化损失函数
    criterion = EnhancedDynamicsLoss(
        alpha=cfg.training.ALPHA,
        beta=cfg.training.BETA,
        gamma=cfg.training.GAMMA
    )

    # 开始评估
    print("开始评估……")
    avg_loss = validate_model(model, criterion, eval_loader)
    print("评估完成，平均总损失：{:.6f}".format(avg_loss))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="模型评估")
    parser.add_argument("--checkpoint", type=str, default="model_checkpoint.pt", help="模型检查点路径")
    parser.add_argument("--mode", type=str, default="e2e", help="评估模式：physics 或 e2e")
    args = parser.parse_args()
    main(args)
