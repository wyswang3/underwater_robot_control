#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
utils/predict_and_compare.py

Pipeline:
  1) Load HybridDynamicsModel checkpoint
  2) Predict on full PreprocessedDataset
  3) Save 6-D preds vs true to CSV
  4) Produce global fit plot (scatter+histogram)
  5) Produce time-series comparison plot
  6) Produce random-segment fit analysis & plot
  7) Produce error-histograms
"""

import os
import argparse
import logging

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from config import Config
from models.dynamics_net import EnhancedPhysicsNet, DirectMappingNet, HybridDynamicsModel
from utils.preprocessing import load_thrust_allocation_matrix
from utils.dataset import PreprocessedDataset
from utils.visualization import (
    visualize_global_accuracy,
    visualize_time_series,
    visualize_error_histograms
)
from utils.random_segment_fit_utils import run_random_segment_fit

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Core Functions
# -----------------------------------------------------------------------------
def load_model(cfg: Config, checkpoint: str, device: torch.device) -> torch.nn.Module:
    """Build and load the HybridDynamicsModel from checkpoint."""
    T_np = load_thrust_allocation_matrix(cfg.paths.THRUST_MATRIX_FILE)
    T = torch.tensor(T_np, dtype=torch.float32, device=device)

    phys = EnhancedPhysicsNet(
        thrust_matrix=T,
        window_size=cfg.training.WINDOW_SIZE,
        hidden_dim=cfg.training.PHYSICS_HIDDEN_DIM
    ).to(device)
    e2e = DirectMappingNet(
        window_size=cfg.training.WINDOW_SIZE,
        hidden_dim=cfg.training.E2E_HIDDEN_DIM
    ).to(device)

    model = HybridDynamicsModel(phys, e2e).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()
    logger.info("Loaded checkpoint: %s", checkpoint)
    return model

def predict_all(model: torch.nn.Module,
                loader: DataLoader,
                device: torch.device) -> (np.ndarray, np.ndarray):
    """
    Run model over entire dataset loader, collect predictions and ground truths.
    Returns:
      preds: (N,6) numpy array
      tgts:  (N,6) numpy array
    """
    all_preds, all_tgts = [], []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting", unit="batch"):
            # move all tensors in batch to device
            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.to(device)
            out = model(batch['power_window'], batch['imu_window'])
            pred = out['accel_pred'] if isinstance(out, dict) else out
            all_preds.append(pred.cpu().numpy())
            all_tgts.append(batch['accel'].cpu().numpy())

    preds = np.vstack(all_preds)
    tgts  = np.vstack(all_tgts)
    logger.info("Finished prediction: %d samples", preds.shape[0])
    return preds, tgts

def save_to_csv(preds: np.ndarray, tgts: np.ndarray, save_path: str) -> None:
    """Save predictions vs ground truth into a CSV file."""
    D = preds.shape[1]
    cols = [f'pred_{i}' for i in range(D)] + [f'true_{i}' for i in range(D)]
    df = pd.DataFrame(np.hstack([preds, tgts]), columns=cols)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    logger.info("Saved CSV: %s", save_path)

# -----------------------------------------------------------------------------
# Main Pipeline
# -----------------------------------------------------------------------------
def main():
    # --- parse args ---
    cfg = Config()
    device = torch.device(cfg.device.DEVICE if torch.cuda.is_available() else 'cpu')

    default_ckpt = os.path.join(cfg.paths.MODEL_DIR, 'model_checkpoint.pt')
    default_csv  = os.path.join(cfg.paths.SPLITS_DIR, 'pred_vs_true.csv')
    default_vis  = os.path.join(cfg.paths.SPLITS_DIR, 'global_fit.png')
    default_ts   = os.path.join(cfg.paths.SPLITS_DIR, 'time_series.png')
    default_rs   = os.path.join(cfg.paths.SPLITS_DIR, 'random_segment_fit.png')
    hist_prefix  = os.path.join(cfg.paths.SPLITS_DIR, 'error_hist')

    parser = argparse.ArgumentParser(description="Predict, save CSV & visualize")
    parser.add_argument('-c','--checkpoint',  default=default_ckpt, help="Model checkpoint (.pt)")
    parser.add_argument('-o','--out_csv',     default=default_csv,  help="Output CSV")
    parser.add_argument('-v','--vis',         default=default_vis,  help="Global fit plot")
    parser.add_argument('-t','--time_series', default=default_ts,   help="Time series plot")
    parser.add_argument('-r','--random_seg',  default=default_rs,   help="Random segment plot")
    args = parser.parse_args()

    # --- ensure output dirs exist ---
    for path in [args.out_csv, args.vis, args.time_series, args.random_seg, f"{hist_prefix}_linear.png", f"{hist_prefix}_angular.png"]:
        os.makedirs(os.path.dirname(path), exist_ok=True)

    # 1) load model
    model = load_model(cfg, args.checkpoint, device)

    # 2) prepare dataset and loader
    ds = PreprocessedDataset(
        features_file=cfg.paths.TRAIN_FEATURES_FILE,
        accel_file=cfg.paths.TRAIN_ACCEL_LABELS_FILE,
        angular_accel_file=cfg.paths.TRAIN_ANGULAR_ACCEL_LABELS_FILE,
        thrust_file=cfg.paths.TRAIN_THRUST_LABELS_FILE,
        window_size=cfg.training.WINDOW_SIZE
    )
    loader = DataLoader(
        ds,
        batch_size=cfg.training.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.training.NUM_WORKERS
    )

    # 3) predict all
    preds, tgts = predict_all(model, loader, device)

    # 4) save CSV
    save_to_csv(preds, tgts, args.out_csv)

    # 5) visualize global fit
    visualize_global_accuracy(
        model, loader, device,
        save_path=args.vis,
        max_samples=len(ds)
    )
    logger.info("Saved global fit plot: %s", args.vis)

    # 6) visualize time series
    visualize_time_series(
        preds, tgts,
        time_step=0.2,
        save_path=args.time_series
    )
    logger.info("Saved time series plot: %s", args.time_series)

    # 7) random-segment fit analysis & viz
      # 使用 cfg, model, ds, device，dt=0.2s，片段总时长30s
    run_random_segment_fit(
            cfg = cfg,
            model = model,
            dataset = ds,
            device = device,
            dt = 0.2,
            segment_duration = 25,
            save_path = args.random_seg
                             )
    logger.info("Saved random-segment fit plot: %s", args.random_seg)

    # 8) error histograms
    visualize_error_histograms(
        preds, tgts,
        save_prefix=hist_prefix,
        linear_bins=40,
        angular_bins=40,
        tick_num=10
    )
    logger.info("Saved linear-error histogram:  %s_linear.png", hist_prefix)
    logger.info("Saved angular-error histogram: %s_angular.png", hist_prefix)


if __name__ == '__main__':
    main()
