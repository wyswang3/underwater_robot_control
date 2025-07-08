#!/usr/bin/env python
"""
evaluate.py

Evaluate script for 'pure_lstm', 'direct', and 'hybrid'.
Falls back to RMSE if validation loss is NaN.
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
from utils.dataset import PreprocessedDataset, split_dataset
from utils.training import validate_model
from utils.log_helper import setup_logging

# Model imports
from models.pure_lstm_model import PureLSTMNet
from models.dynamics_net import (
    DirectMappingNet,
    EnhancedPhysicsNet,
    HybridDynamicsModel,
    EnhancedDynamicsLoss,
)

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


def build_model_and_criterion(
    cfg: Config,
    device: torch.device,
    thrust_matrix: Optional[torch.Tensor]
):
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
        model = DirectMappingNet(
            p_hidden=cfg.training.DIRECT_P_HIDDEN,
            i_hidden=cfg.training.DIRECT_I_HIDDEN,
            fusion_hidden=cfg.training.FUSION_HIDDEN_DIM,
        ).to(device)
        criterion = torch.nn.MSELoss().to(device)
        logger.info('Built DirectMappingNet + MSELoss')
    elif mtype == 'hybrid':
        phys = EnhancedPhysicsNet(
            T=thrust_matrix,
            enc_hidden=cfg.training.PHY_ENCODER_HIDDEN,
            layers=cfg.training.PHY_LAYERS,
            dropout=cfg.training.PHY_DROPOUT,
        ).to(device)
        direct = DirectMappingNet(
            p_hidden=cfg.training.DIRECT_P_HIDDEN,
            i_hidden=cfg.training.DIRECT_I_HIDDEN,
            fusion_hidden=cfg.training.FUSION_HIDDEN_DIM,
        ).to(device)
        model = HybridDynamicsModel(direct, phys).to(device)
        criterion = EnhancedDynamicsLoss(
            beta=cfg.training.REG_WEIGHT,
            linear_w=cfg.training.LINEAR_WEIGHT,
            use_reg=True,
        ).to(device)
        logger.info('Built HybridDynamicsModel + EnhancedDynamicsLoss(use_reg=True)')
    else:
        raise ValueError(f"Unknown MODEL_TYPE='{cfg.training.MODEL_TYPE}'")
    return model, criterion


def evaluate_model(ckpt_path: str) -> None:
    # Load config
    cfg = Config()
    cfg.training.MODEL_TYPE = cfg.training.MODEL_TYPE.replace('-', '_').lower()

    # Device
    device = torch.device(cfg.device.DEVICE if torch.cuda.is_available() else 'cpu')
    logger.info(f"Evaluation on device: {device}")

    # Dataset and loader
    ds = PreprocessedDataset(
        features_file=cfg.paths.TRAIN_FEATURES_FILE,
        accel_file=cfg.paths.TRAIN_ACCEL_LABELS_FILE,
        angular_accel_file=cfg.paths.TRAIN_ANGULAR_ACCEL_LABELS_FILE,
        thrust_file=cfg.paths.TRAIN_THRUST_LABELS_FILE,
        window_size=cfg.training.WINDOW_SIZE,
        clean_data=False,
    )
    loader = DataLoader(
        ds,
        batch_size=cfg.training.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.training.NUM_WORKERS,
        pin_memory=(device.type == 'cuda'),
    )
    logger.info(f"Eval samples: {len(ds)}")

    # Thrust matrix
    thrust_matrix = None
    if cfg.training.MODEL_TYPE in ('direct', 'hybrid'):
        arr = load_thrust_allocation_matrix(cfg.paths.THRUST_MATRIX_FILE)
        thrust_matrix = torch.tensor(arr, dtype=torch.float32, device=device)
        logger.info(f"Loaded thrust matrix shape: {thrust_matrix.shape}")

    # Build model + loss
    model, criterion = build_model_and_criterion(cfg, device, thrust_matrix)

    # Load checkpoint
    if not os.path.isfile(ckpt_path):
        logger.error(f"Checkpoint not found: {ckpt_path}")
        return

    obj = torch.load(ckpt_path, map_location=device)
    if isinstance(obj, torch.nn.Module):
        model = obj.to(device).eval()
        logger.info(f"Loaded full model: {ckpt_path}")
    else:
        state_dict = obj.get('model_state', obj) if isinstance(obj, dict) else obj
        model.load_state_dict(state_dict)
        model.to(device).eval()
        logger.info(f"Loaded state_dict: {ckpt_path}")

    # Validation
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
    parser.add_argument('-c', '--checkpoint',
                        default=str(Config().paths.MODEL_DIR / 'model_hybrid.pt'),
                        help='Path to checkpoint')
    args = parser.parse_args()
    evaluate_model(args.checkpoint)


if __name__ == '__main__':
    main()