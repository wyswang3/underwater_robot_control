#!/usr/bin/env python
"""
train.py

Unified training script supporting 'pure_lstm', 'direct', and 'hybrid' branches.
Features:
  - Branch selection via config.training.MODEL_TYPE
  - Optional sliding window override via CLI
  - Automatic data loading, cleaning, and splitting
  - Dynamic optimizer & scheduler setup
  - Structured logging per epoch with LR and losses
  - Post-training: save checkpoint, plot loss curve, validate, evaluate, visualize
"""
import os
import platform
import multiprocessing
import argparse
import logging

import torch
import numpy as np
from torch.optim.lr_scheduler import OneCycleLR, SequentialLR, CosineAnnealingLR, LinearLR, LambdaLR

from config import Config
from utils.preprocessing import load_thrust_allocation_matrix
from utils.dataset import PreprocessedDataset, check_dataset, split_dataset
from utils.visualization import plot_loss_curve, visualize_predictions, visualize_overall_accuracy
from utils.random_segment_fit_utils import run_random_segment_fit
from utils.training import custom_train_model, validate_model
from evaluate import evaluate_model
from utils.log_helper import setup_logging

# Model imports
from models.pure_lstm_model import PureLSTMNet
from models.dynamics_net import (
    DirectMappingNet_LSTM_Fusion,
    EnhancedPhysicsNet_LSTM,
    HybridDynamicsModel,
    EnhancedDynamicsLoss,
)

# ──────────────────────────────────────────────────────────────────────────────
setup_logging()
logger = logging.getLogger(__name__)


def make_scheduler(optimizer, cfg, steps_per_epoch):
    """Create LR scheduler according to config."""
    stype = cfg.training.LR_SCHEDULER_TYPE.lower()
    total_steps = cfg.training.NUM_EPOCHS * steps_per_epoch
    if stype == "onecyclelr":
        sched = OneCycleLR(
            optimizer,
            max_lr=cfg.training.MAX_LR,
            total_steps=total_steps,
            pct_start=cfg.training.LR_SCHEDULER_PCT_START,
            final_div_factor=cfg.training.MAX_LR / cfg.training.MIN_LR,
        )
        return sched, True
    elif stype == "cosine":
        warm = int(0.1 * cfg.training.NUM_EPOCHS)
        sched = SequentialLR(
            optimizer,
            schedulers=[
                LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warm),
                CosineAnnealingLR(optimizer, T_max=cfg.training.NUM_EPOCHS - warm, eta_min=cfg.training.MIN_LR),
            ],
            milestones=[warm],
        )
        return sched, False
    else:
        warm = cfg.training.WARMUP_STEPS or int(0.05 * total_steps)
        def poly(step):
            if step < warm:
                return (step + 1) / warm
            return (1 - (step - warm) / (total_steps - warm)) ** 2
        sched = LambdaLR(optimizer, lr_lambda=poly)
        return sched, True


def build_model_and_loss(cfg, device, thrust_matrix=None):
    """Instantiate model and corresponding loss based on MODEL_TYPE."""
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
            debug=cfg.DEBUG,
        ).to(device)
        criterion = EnhancedDynamicsLoss(
            beta=cfg.training.LAMBDA_PHY,
            linear_w=getattr(cfg.training, 'LIN_WEIGHT', 0.7),
            use_reg=False,
        ).to(device)
        logger.info('Built DirectMappingNet_LSTM_Fusion + EnhancedDynamicsLoss(use_reg=False)')
    elif mtype == 'hybrid':
        phys = EnhancedPhysicsNet_LSTM(
            T=thrust_matrix,
            hidden=cfg.training.HIDDEN_DIM,
            layers=cfg.training.LSTM_LAYERS,
            debug=cfg.DEBUG,
        ).to(device)
        e2e = DirectMappingNet_LSTM_Fusion(
            hidden=int(cfg.training.HIDDEN_DIM * cfg.training.E2E_HIDDEN_FACTOR),
            layers=cfg.training.LSTM_LAYERS,
            fusion_hidden=cfg.training.FUSION_HIDDEN_DIM,
            debug=cfg.DEBUG,
        ).to(device)
        model = HybridDynamicsModel(phys, e2e, debug=cfg.DEBUG).to(device)
        criterion = EnhancedDynamicsLoss(
            beta=cfg.training.LAMBDA_PHY,
            linear_w=getattr(cfg.training, 'LIN_WEIGHT', 0.7),
            use_reg=True,
        ).to(device)
        logger.info('Built HybridDynamicsModel + EnhancedDynamicsLoss(use_reg=True)')
    else:
        raise ValueError(f"Unknown MODEL_TYPE='{cfg.training.MODEL_TYPE}'")
    return model, criterion


def main():
    # Windows multiprocessing safety
    if platform.system() == 'Windows':
        multiprocessing.set_start_method('spawn', force=True)

    # CLI: only window_size override
    parser = argparse.ArgumentParser(description='Train network branch')
    parser.add_argument('--window_size', type=int, help='Override config.training.WINDOW_SIZE')
    args = parser.parse_args()

    # Config
    cfg = Config()
    if args.window_size:
        cfg.training.WINDOW_SIZE = args.window_size
    # normalize MODEL_TYPE
    cfg.training.MODEL_TYPE = cfg.training.MODEL_TYPE.replace('-', '_').lower()
    cfg.print_config()

    # Seeds
    torch.manual_seed(42)
    np.random.seed(42)

    # Device
    device = torch.device(cfg.device.DEVICE if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        logger.warning('CUDA not available, using CPU')
    logger.info(f'Using device: {device}')

    # Thrust matrix if needed
    T = None
    if cfg.training.MODEL_TYPE in ('direct', 'hybrid'):
        arr = load_thrust_allocation_matrix(cfg.paths.THRUST_MATRIX_FILE)
        T = torch.tensor(arr, dtype=torch.float32, device=device)
        logger.info(f'Loaded thrust matrix: {T.shape}')

    # Build model + loss
    model, criterion = build_model_and_loss(cfg, device, T)

    # Data
    ds = PreprocessedDataset(
        cfg.paths.TRAIN_FEATURES_FILE,
        cfg.paths.TRAIN_ACCEL_LABELS_FILE,
        cfg.paths.TRAIN_ANGULAR_ACCEL_LABELS_FILE,
        cfg.paths.TRAIN_THRUST_LABELS_FILE,
        window_size=cfg.training.WINDOW_SIZE,
        clean_data=True,
    )
    check_dataset(ds)
    ds_tr, ds_val = split_dataset(ds, 0.8)
    dl_tr = torch.utils.data.DataLoader(
        ds_tr,
        batch_size=cfg.training.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.training.NUM_WORKERS,
        pin_memory=(device.type == 'cuda'),
    )
    dl_val = torch.utils.data.DataLoader(
        ds_val,
        batch_size=cfg.training.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.training.NUM_WORKERS,
        pin_memory=(device.type == 'cuda'),
    )
    logger.info('DataLoaders ready')

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.LEARNING_RATE,
        weight_decay=cfg.training.WEIGHT_DECAY,
    )
    for g in optimizer.param_groups:
        g['initial_lr'] = g['lr']

    # Scheduler
    scheduler, step_per_batch = make_scheduler(optimizer, cfg, len(dl_tr))

    # Train
    logger.info(f"Starting training [{cfg.training.MODEL_TYPE}] on {device}")
    history = custom_train_model(
        model,
        criterion,
        dl_tr,
        val_loader=dl_val,
        epochs=cfg.training.NUM_EPOCHS,
        optimizer=optimizer,
        scheduler=scheduler,
        warmup_steps=cfg.training.WARMUP_STEPS,
        grad_clip=cfg.training.CLIP_GRAD_NORM,
        amp_enabled=False,
        step_per_batch=step_per_batch,
    )

    # Save
    ckpt = os.path.join(cfg.paths.MODEL_DIR,f"model_{cfg.training.MODEL_TYPE}_full.pt")
    torch.save(model, ckpt)  # <-- 直接保存整个 nn.Module
    logger.info(f"Full model saved → {ckpt}")

    # Loss curve
    train_l = history['train_loss']
    val_l = history.get('val_loss') or None
    plot_loss_curve(
        train_l,
        val_losses=val_l,
        save_path=os.path.join(cfg.paths.SPLITS_DIR, 'loss_curve.png'),
    )

    # Validate & evaluate
    validate_model(model, criterion, dl_val, amp_enabled=(device.type == 'cuda'))
    evaluate_model(ckpt)

    # Visualize
    base = cfg.training.MODEL_TYPE
    out_dir = cfg.paths.SPLITS_DIR
    cmp_path = os.path.join(out_dir, f"compare_{base}.png")
    visualize_predictions(model, dl_val, device, num_batches=2, num_samples=6, save_path=cmp_path)
    logger.info(f"Saved compare plot → {cmp_path}")

    glob_path = os.path.join(out_dir, f"global_{base}.png")
    visualize_overall_accuracy(model, dl_val, device, save_path=glob_path, max_samples=1000)
    logger.info(f"Saved global plot → {glob_path}")

    seg_path = os.path.join(out_dir, f"segment_{base}.png")
    run_random_segment_fit(cfg, model, ds, device, dt=0.11, segment_duration=30, save_path=seg_path)
    logger.info(f"Saved segment plot → {seg_path}")

if __name__ == '__main__':
    main()