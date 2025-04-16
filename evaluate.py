#!/usr/bin/env python
# evaluate.py
"""
This module evaluates the hybrid dynamics model.
If the computed validation loss is NaN, the module falls back to evaluating the model using RMSE.
"""
import os
import argparse
import math
from typing import Union

from utils.log_helper import setup_logging  # Set up logging based on cfg.DEBUG

setup_logging()

import logging
import torch
from torch.utils.data import DataLoader

# ── Project Modules ────────────────────────────────────────────
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


def evaluate_model(ckpt: Union[str, os.PathLike]) -> None:
    """
    Evaluate the hybrid dynamics model with the given checkpoint.

    If the computed validation loss is NaN, an alternative RMSE evaluation is performed.
    """
    cfg = Config()
    device = torch.device(cfg.device.DEVICE if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # 1) Load dataset
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

    # 2) Build model architecture
    T = torch.tensor(
        load_thrust_allocation_matrix(cfg.paths.THRUST_MATRIX_FILE),
        dtype=torch.float32, device=device
    )
    physics_net = EnhancedPhysicsNet_LSTM(
        T=T,  # Use the thrust allocation matrix T
        hidden=cfg.training.HIDDEN_DIM,
        layers=cfg.training.LSTM_LAYERS,
        debug=cfg.DEBUG
    )
    e2e_hidden_factor = getattr(cfg.training, "E2E_HIDDEN_FACTOR", 0.5)
    fusion_hidden_dim = getattr(cfg.training, "FUSION_HIDDEN_DIM", 256)
    e2e_net = DirectMappingNet_LSTM_Fusion(
        hidden=int(cfg.training.HIDDEN_DIM * e2e_hidden_factor),
        layers=cfg.training.LSTM_LAYERS,
        fusion_hidden=fusion_hidden_dim,
        debug=cfg.DEBUG
    ).to(device)

    model = HybridDynamicsModel(physics_net, e2e_net, debug=cfg.DEBUG).to(device)
    logger.info("Model graph built.")

    # 3) Load checkpoint
    ckpt = os.fspath(ckpt)
    if not os.path.isfile(ckpt):
        logger.error(f"Checkpoint not found: {ckpt}")
        return
    state = torch.load(ckpt, map_location=device)
    # Check for NaN or Inf in checkpoint values
    bad = any(torch.isnan(v).any() or torch.isinf(v).any()
              for v in state.values() if torch.is_tensor(v))
    if bad:
        raise RuntimeError("Checkpoint contains NaN/Inf – abort evaluation.")
    model.load_state_dict(state, strict=True)
    logger.info(f"Loaded checkpoint: {ckpt}")

    # 4) Define loss function
    criterion = EnhancedDynamicsLoss(
        beta=cfg.training.LAMBDA_PHY,  # Regularization weight
        linear_w=getattr(cfg.training, "LIN_WEIGHT", 0.7),
        use_reg=False
    ).to(device)

    # 5) Validate the model
    avg_loss = validate_model(model, criterion, loader, amp_enabled=(device.type == "cuda"))
    if math.isnan(avg_loss):
        logger.warning("Validation loss returned NaN. Falling back to RMSE evaluation.")
        # Fallback: aggregate predictions and calculate RMSE as alternative evaluation metric.
        from utils.visualization import aggregate_predictions
        preds, targets = aggregate_predictions(model, loader, device, max_samples=1000)
        alt_rmse = torch.sqrt(torch.mean((preds - targets) ** 2)).item()
        logger.info(f"Alternative evaluation (RMSE): {alt_rmse:.6f}")
        print(f"\nAlternative evaluation (RMSE): {alt_rmse:.6f}")
    else:
        logger.info(f"Evaluation complete — average loss: {avg_loss:.6f}")
        print(f"\nEvaluation complete — average loss: {avg_loss:.6f}")


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
