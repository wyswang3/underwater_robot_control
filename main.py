#!/usr/bin/env python
"""
main.py

Unified training script supporting 'pure_lstm', 'direct', 'hybrid' branches.
Optimized: renamed from train.py, with updated imports and dynamic scheduler/AMP CLI.
"""
import os
import platform
import multiprocessing
import argparse
import logging

import torch
import numpy as np
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    OneCycleLR, SequentialLR, CosineAnnealingLR, LinearLR, LambdaLR)

from config import Config
from utils.preprocessing import load_thrust_allocation_matrix
from utils.dataset import PreprocessedDataset, check_dataset, split_dataset
from utils.visualization import (
    plot_loss_curve, visualize_predictions, visualize_overall_accuracy)
from utils.random_segment_fit_utils import run_random_segment_fit
from utils.training import custom_train_model, validate_model
from evaluate import evaluate_model
from utils.log_helper import setup_logging

# Models
from models.pure_lstm_model import PureLSTMNet
from models.dynamics_net import (
    DirectMappingNet, EnhancedPhysicsNet,
    HybridDynamicsModel, EnhancedDynamicsLoss)

# -----------------------------------------------------------------------------
# Setup logging
# -----------------------------------------------------------------------------
setup_logging()
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Argument parser
# -----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description='Train ROV dynamics model')
    parser.add_argument('--window_size', type=int, help='Override sliding window size')
    parser.add_argument('--scheduler', choices=['onecyclelr', 'cosine', 'polynomial'],
                        help='Learning rate scheduler type')
    parser.add_argument('--amp', action='store_true', help='Enable mixed-precision training')
    return parser.parse_args()

# -----------------------------------------------------------------------------
# Scheduler factory
# -----------------------------------------------------------------------------

def make_scheduler(optimizer, cfg, steps_per_epoch):
    total_steps = cfg.training.NUM_EPOCHS * steps_per_epoch
    stype = cfg.training.SCHEDULER_TYPE.lower()
    if stype == 'onecyclelr':
        scheduler = OneCycleLR(
            optimizer,
            max_lr=cfg.training.MAX_LR,
            total_steps=total_steps,
            pct_start=cfg.training.PCT_START,
            final_div_factor=cfg.training.MAX_LR / cfg.training.MIN_LR
        )
        step_per_batch = True
    elif stype == 'cosine':
        warm = int(0.1 * cfg.training.NUM_EPOCHS)
        scheduler = SequentialLR(
            optimizer,
            schedulers=[
                LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warm),
                CosineAnnealingLR(optimizer, T_max=cfg.training.NUM_EPOCHS - warm,
                                  eta_min=cfg.training.MIN_LR)
            ],
            milestones=[warm]
        )
        step_per_batch = False
    else:
        warm = cfg.training.WARMUP_STEPS or int(0.05 * total_steps)
        def poly_lr(step):
            if step < warm:
                return (step + 1) / warm
            return (1 - (step - warm) / (total_steps - warm)) ** 2
        scheduler = LambdaLR(optimizer, lr_lambda=poly_lr)
        step_per_batch = True
    return scheduler, step_per_batch

# -----------------------------------------------------------------------------
# Model & loss builder
# -----------------------------------------------------------------------------

def build_model_and_loss(cfg, device, T=None):
    mtype = cfg.training.MODEL_TYPE.lower()
    if mtype == 'pure_lstm':
        model = PureLSTMNet(
            input_size=cfg.training.INPUT_DIM,
            hidden_size=cfg.training.PURE_LSTM_HIDDEN_DIM,
            num_layers=cfg.training.PURE_LSTM_LAYERS,
            dropout=cfg.training.PURE_LSTM_DROPOUT,
            output_size=cfg.training.PURE_LSTM_OUTPUT_DIM
        ).to(device)
        loss_fn = torch.nn.MSELoss().to(device)
        logger.info('Model: PureLSTMNet + MSELoss')
    elif mtype == 'direct':
        model = DirectMappingNet(
            p_hidden=cfg.training.DIRECT_P_HIDDEN,
            i_hidden=cfg.training.DIRECT_I_HIDDEN,
            fusion_hidden=cfg.training.FUSION_HIDDEN_DIM
        ).to(device)
        loss_fn = EnhancedDynamicsLoss(
            beta=cfg.training.REG_WEIGHT,
            linear_w=cfg.training.LINEAR_WEIGHT,
            use_reg=False
        ).to(device)
        logger.info('Model: DirectMappingNet + EnhancedDynamicsLoss(use_reg=False)')
    elif mtype == 'hybrid':
        phys = EnhancedPhysicsNet(
            T=T,
            enc_hidden=cfg.training.PHY_ENCODER_HIDDEN,
            layers=cfg.training.PHY_LAYERS,
            dropout=cfg.training.PHY_DROPOUT
        ).to(device)
        direct = DirectMappingNet(
            p_hidden=cfg.training.DIRECT_P_HIDDEN,
            i_hidden=cfg.training.DIRECT_I_HIDDEN,
            fusion_hidden=cfg.training.FUSION_HIDDEN_DIM
        ).to(device)
        model = HybridDynamicsModel(direct, phys).to(device)
        loss_fn = EnhancedDynamicsLoss(
            beta=cfg.training.REG_WEIGHT,
            linear_w=cfg.training.LINEAR_WEIGHT,
            use_reg=True
        ).to(device)
        logger.info('Model: HybridDynamicsModel + EnhancedDynamicsLoss(use_reg=True)')
    else:
        raise ValueError(f"Unsupported MODEL_TYPE '{cfg.training.MODEL_TYPE}'")
    return model, loss_fn

# -----------------------------------------------------------------------------
# Dataloader factory
# -----------------------------------------------------------------------------

def create_dataloaders(cfg, device):
    dataset = PreprocessedDataset(
        features_file=cfg.paths.TRAIN_FEATURES_FILE,
        accel_file=cfg.paths.TRAIN_ACCEL_LABELS_FILE,
        angular_accel_file=cfg.paths.TRAIN_ANGULAR_ACCEL_LABELS_FILE,  # 新增这一行
        thrust_file=cfg.paths.TRAIN_THRUST_LABELS_FILE,
        window_size=cfg.training.WINDOW_SIZE,
        clean_data=True,
    )
    check_dataset(dataset)
    ds_tr, ds_val = split_dataset(dataset, 0.8)
    loader_args = dict(
        batch_size=cfg.training.BATCH_SIZE,
        num_workers=cfg.training.NUM_WORKERS,
        pin_memory=(device.type == 'cuda'),
    )
    dl_tr = torch.utils.data.DataLoader(ds_tr, shuffle=True, **loader_args)
    dl_val = torch.utils.data.DataLoader(ds_val, shuffle=False, **loader_args)
    return dl_tr, dl_val


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    if platform.system() == 'Windows':
        multiprocessing.set_start_method('spawn', force=True)

    args = parse_args()
    cfg = Config()
    if args.window_size:
        cfg.training.WINDOW_SIZE = args.window_size
    if args.scheduler:
        cfg.training.SCHEDULER_TYPE = args.scheduler
    cfg.print_config()

    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device(cfg.device.DEVICE)
    logger.info(f'Using device: {device}')

    T = None
    if cfg.training.MODEL_TYPE in ('direct', 'hybrid'):
        arr = load_thrust_allocation_matrix(cfg.paths.THRUST_MATRIX_FILE)
        T = torch.tensor(arr, dtype=torch.float32, device=device)
        logger.info(f'Loaded thrust allocation matrix: {T.shape}')

    model, loss_fn = build_model_and_loss(cfg, device, T)
    dl_tr, dl_val = create_dataloaders(cfg, device)

    optimizer = AdamW(
        model.parameters(),
        lr=cfg.training.LEARNING_RATE,
        weight_decay=cfg.training.WEIGHT_DECAY
    )
    scheduler, step_per_batch = make_scheduler(optimizer, cfg, len(dl_tr))

    history = custom_train_model(
        model, loss_fn, dl_tr,
        val_loader=dl_val,
        epochs=cfg.training.NUM_EPOCHS,
        optimizer=optimizer,
        scheduler=scheduler,
        warmup_steps=cfg.training.WARMUP_STEPS,
        grad_clip=cfg.training.CLIP_GRAD_NORM,
        amp_enabled=args.amp,
        step_per_batch=step_per_batch
    )

    ckpt_path = cfg.paths.MODEL_DIR / f"model_{cfg.training.MODEL_TYPE}.pt"
    torch.save(model.state_dict(), ckpt_path)
    logger.info(f'Model state_dict saved at {ckpt_path}')

    plot_loss_curve(
        history['train_loss'],
        val_losses=history.get('val_loss'),
        save_path=cfg.paths.SPLITS_DIR / 'loss_curve.png'
    )

    validate_model(model, loss_fn, dl_val, amp_enabled=args.amp)
    evaluate_model(str(ckpt_path))

    visualize_predictions(
        model, dl_val, device,
        num_batches=2, num_samples=6,
        save_path=cfg.paths.SPLITS_DIR / 'compare.png'
    )
    visualize_overall_accuracy(
        model, dl_val, device,
        save_path=cfg.paths.SPLITS_DIR / 'global.png'
    )

    run_random_segment_fit(
        cfg, model, dl_tr.dataset, device,
        dt=1.0 / cfg.training.WINDOW_SIZE,
        segment_duration=30,
        save_path=cfg.paths.SPLITS_DIR / 'segment.png'
    )
    logger.info('All outputs generated')

if __name__ == '__main__':
    main()
