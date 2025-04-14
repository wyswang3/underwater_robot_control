#!/usr/bin/env python
# evaluate.py
import os
import argparse
from typing import Union

from utils.log_helper import setup_logging  # 根据 cfg.DEBUG 设置日志级别
setup_logging()

import logging
import torch
from torch.utils.data import DataLoader

# ── 项目内部模块 ────────────────────────────────────────────
from config import Config
from models.dynamics_net import (
    EnhancedPhysicsNet_LSTM,
    DirectMappingNet_LSTM_Fusion,
    HybridDynamicsModel,
    EnhancedDynamicsLoss
)
from utils.preprocessing import load_thrust_allocation_matrix
from utils.dataset import PreprocessedDataset
from utils.training import validate_model

logger = logging.getLogger(__name__)

# ╭──────────────────────────────╮
# │       评估主流程             │
# ╰──────────────────────────────╯
def evaluate_model(ckpt: Union[str, os.PathLike]) -> None:
    cfg = Config()
    device = torch.device(cfg.device.DEVICE if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # 1) 数据集
    dataset = PreprocessedDataset(
        cfg.paths.TRAIN_FEATURES_FILE,
        cfg.paths.TRAIN_ACCEL_LABELS_FILE,
        cfg.paths.TRAIN_ANGULAR_ACCEL_LABELS_FILE,
        cfg.paths.TRAIN_THRUST_LABELS_FILE,
        window_size=cfg.training.WINDOW_SIZE
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.training.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.training.NUM_WORKERS,
        pin_memory=(device.type == "cuda")
    )
    logger.info(f"Eval samples: {len(dataset)}")

    # 2) 网络结构
    T = torch.tensor(
        load_thrust_allocation_matrix(cfg.paths.THRUST_MATRIX_FILE),
        dtype=torch.float32, device=device
    )

    physics_net = EnhancedPhysicsNet_LSTM(
        thrust_matrix=T,
        lstm_hidden_dim=cfg.training.HIDDEN_DIM,
        num_layers=cfg.training.LSTM_LAYERS,
        debug=cfg.DEBUG
    ).to(device)

    e2e_hidden_factor = getattr(cfg.training, "E2E_HIDDEN_FACTOR", 0.5)
    fusion_hidden_dim = getattr(cfg.training, "FUSION_HIDDEN_DIM", 256)
    e2e_net = DirectMappingNet_LSTM_Fusion(
        hidden_dim=int(cfg.training.HIDDEN_DIM * e2e_hidden_factor),
        num_layers=cfg.training.LSTM_LAYERS,
        fusion_hidden_dim=fusion_hidden_dim,
        debug=cfg.DEBUG
    ).to(device)

    model = HybridDynamicsModel(physics_net, e2e_net, debug=cfg.DEBUG).to(device)
    logger.info("Model graph built.")

    # 3) 加载权重
    ckpt = os.fspath(ckpt)
    if not os.path.isfile(ckpt):
        logger.error(f"Checkpoint not found: {ckpt}")
        return
    model.load_state_dict(torch.load(ckpt, map_location=device), strict=True)
    logger.info(f"Loaded checkpoint: {ckpt}")

    # 4) 损失函数
    criterion = EnhancedDynamicsLoss(alpha=cfg.training.LAMBDA_PHY).to(device)

    # 5) 验证
    avg_loss = validate_model(model, criterion, loader, amp_enabled=(device.type == "cuda"))
    print(f"\nEvaluation complete — average loss: {avg_loss:.6f}")


# ╭──────────────────────────────╮
# │            CLI              │
# ╰──────────────────────────────╯
def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate hybrid dynamics model")
    parser.add_argument(
        "-c", "--checkpoint",
        default="models/checkpoints/model.pt",
        help="Path to checkpoint file"
    )
    args = parser.parse_args()
    evaluate_model(args.checkpoint)


if __name__ == "__main__":
    main()
