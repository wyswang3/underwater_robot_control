#!/usr/bin/env python
"""
evaluate.py

Evaluate script supporting 'pure_lstm', 'direct', and 'hybrid' branches.
Falls back to RMSE if validation loss returns NaN.
"""
import os
import math
import argparse
import logging
from typing import Optional

import torch
from torch.utils.data import DataLoader

from config import Config
from utils.preprocessing import load_thrust_allocation_matrix
from utils.dataset import PreprocessedDataset
from utils.training import validate_model

# Model imports
from models.pure_lstm_model import PureLSTMNet
from models.dynamics_net import (
    DirectMappingNet_LSTM_Fusion,
    EnhancedPhysicsNet_LSTM,
    HybridDynamicsModel,
    EnhancedDynamicsLoss,
)

logger = logging.getLogger(__name__)


def build_model_and_criterion(
    cfg: Config,
    device: torch.device,
    thrust_matrix: Optional[torch.Tensor]
):
    """
    Instantiate model and corresponding loss criterion based on
    cfg.training.MODEL_TYPE.
    """
    mtype = cfg.training.MODEL_TYPE.replace('-', '_').lower()

    if mtype == 'pure_lstm':
        model = PureLSTMNet(
            input_size=cfg.training.INPUT_DIM,
            hidden_size=cfg.training.PURE_LSTM_HIDDEN_DIM,
            num_layers=cfg.training.PURE_LSTM_LAYERS,
            dropout=cfg.training.PURE_LSTM_DROPOUT,
            output_size=cfg.training.PURE_LSTM_OUTPUT_DIM,
        ).to(device)
        criterion = torch.nn.MSELoss().to(device)
        logger.info('Built PureLSTMNet + MSELoss')

    elif mtype == 'direct':
        model = DirectMappingNet_LSTM_Fusion(
            hidden=int(cfg.training.HIDDEN_DIM * cfg.training.E2E_HIDDEN_FACTOR),
            layers=cfg.training.LSTM_LAYERS,
            fusion_hidden=cfg.training.FUSION_HIDDEN_DIM,
            debug=cfg.DEBUG
        ).to(device)
        criterion = EnhancedDynamicsLoss(
            beta=cfg.training.LAMBDA_PHY,
            linear_w=getattr(cfg.training, 'LIN_WEIGHT', 0.7),
            use_reg=False
        ).to(device)
        logger.info('Built DirectMappingNet_LSTM_Fusion + EnhancedDynamicsLoss(use_reg=False)')

    elif mtype == 'hybrid':
        phys = EnhancedPhysicsNet_LSTM(
            T=thrust_matrix,
            hidden=cfg.training.HIDDEN_DIM,
            layers=cfg.training.LSTM_LAYERS,
            debug=cfg.DEBUG
        ).to(device)
        e2e = DirectMappingNet_LSTM_Fusion(
            hidden=int(cfg.training.HIDDEN_DIM * cfg.training.E2E_HIDDEN_FACTOR),
            layers=cfg.training.LSTM_LAYERS,
            fusion_hidden=cfg.training.FUSION_HIDDEN_DIM,
            debug=cfg.DEBUG
        ).to(device)
        model = HybridDynamicsModel(phys, e2e, debug=cfg.DEBUG).to(device)
        criterion = EnhancedDynamicsLoss(
            beta=cfg.training.LAMBDA_PHY,
            linear_w=getattr(cfg.training, 'LIN_WEIGHT', 0.7),
            use_reg=True
        ).to(device)
        logger.info('Built HybridDynamicsModel + EnhancedDynamicsLoss(use_reg=True)')

    else:
        raise ValueError(f"Unknown MODEL_TYPE='{cfg.training.MODEL_TYPE}'")

    return model, criterion


def evaluate_model(ckpt_path: str) -> None:
    """
    Load checkpoint and evaluate on validation set.
    """
    # Load config and normalize model type
    cfg = Config()
    cfg.training.MODEL_TYPE = cfg.training.MODEL_TYPE.replace('-', '_').lower()

    # Device
    device = torch.device(cfg.device.DEVICE if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    # Dataset and loader
    ds = PreprocessedDataset(
        cfg.paths.TRAIN_FEATURES_FILE,
        cfg.paths.TRAIN_ACCEL_LABELS_FILE,
        cfg.paths.TRAIN_ANGULAR_ACCEL_LABELS_FILE,
        cfg.paths.TRAIN_THRUST_LABELS_FILE,
        window_size=cfg.training.WINDOW_SIZE
    )
    loader = DataLoader(
        ds,
        batch_size=cfg.training.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.training.NUM_WORKERS,
        pin_memory=(device.type == 'cuda')
    )
    logger.info(f"Eval samples: {len(ds)}")

    # Thrust matrix for physics-based models
    thrust_matrix: Optional[torch.Tensor] = None
    if cfg.training.MODEL_TYPE in ('direct', 'hybrid'):
        arr = load_thrust_allocation_matrix(cfg.paths.THRUST_MATRIX_FILE)
        thrust_matrix = torch.tensor(arr, dtype=torch.float32, device=device)
        logger.info(f"Loaded thrust matrix shape: {thrust_matrix.shape}")

    # Build model and loss
    model, criterion = build_model_and_criterion(cfg, device, thrust_matrix)

    # ------------------------------------------------------
    # Load checkpoint  (兼容整体模型  &  state_dict 两种格式)
    # ------------------------------------------------------
    if not os.path.isfile(ckpt_path):
        logger.error(f"Checkpoint not found: {ckpt_path}")
        return

    obj = torch.load(ckpt_path, map_location=device)

    # ——— 情况 1：整体 nn.Module 对象 ————————————————
    #   • 直接用加载出的模型替代 build_model_and_criterion 里的空壳
    #   • 仍保留 criterion，用来计算验证损失
    if isinstance(obj, torch.nn.Module):
        model = obj.to(device).eval()
        logger.info(f"Loaded FULL model: {ckpt_path}")

    # ——— 情况 2：state_dict 或包装字典 ——————————————
    #   • obj 是 dict ⇒ 继续走旧逻辑
    else:
        state_dict = None

        # 2-a 纯 state_dict：所有 value 都是张量
        if isinstance(obj, dict) and all(torch.is_tensor(v) for v in obj.values()):
            state_dict = obj

        # 2-b 自定义包装，如 {'model_state': …}
        elif isinstance(obj, dict) and 'model_state' in obj:
            state_dict = obj['model_state']

        else:
            raise TypeError("Unrecognized checkpoint format")

        # NaN / Inf 检查
        if any(torch.isnan(v).any() or torch.isinf(v).any()
               for v in state_dict.values() if torch.is_tensor(v)):
            raise RuntimeError("Checkpoint contains NaN/Inf – abort evaluation.")

        model.load_state_dict(state_dict, strict=True)
        logger.info(f"Loaded state_dict: {ckpt_path}")


    # Validate
    avg_loss = validate_model(model, criterion, loader, amp_enabled=(device.type=='cuda'))
    if math.isnan(avg_loss):
        logger.warning("Validation loss is NaN, falling back to RMSE.")
        from utils.visualization import aggregate_predictions
        preds, tgts = aggregate_predictions(model, loader, device, max_samples=1000)
        rmse = torch.sqrt(torch.mean((preds - tgts) ** 2)).item()
        logger.info(f"RMSE fallback: {rmse:.6f}")
        print(f"\nRMSE fallback: {rmse:.6f}")
    else:
        logger.info(f"Validation complete — average loss: {avg_loss:.6f}")
        print(f"\nValidation complete — average loss: {avg_loss:.6f}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    parser.add_argument(
        '-c', '--checkpoint',
        default='models/checkpoints/model.pt',
        help='Path to checkpoint file'
    )
    args = parser.parse_args()
    evaluate_model(args.checkpoint)


if __name__ == '__main__':
    main()
