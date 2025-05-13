#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
utils/predict_and_compare_lstm.py

Convenience script supporting pure_lstm, direct, and hybrid models:
  1. Instantiate appropriate model based on cfg.training.MODEL_TYPE
  2. Load checkpoint
  3. Predict on PreprocessedDataset
  4. Save predictions vs ground truth to CSV
  5. Visualize global fit (scatter + histogram)
  6. Visualize time series comparison
  7. Visualize random-segment fit
  8. Plot error histograms
  9. Generate per-axis error table and time-series plots

Usage:
    python utils/predict_and_compare_lstm.py
Or override with CLI flags.
"""
import os
import argparse
import logging

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from utils.preprocessing import load_thrust_allocation_matrix
from utils.dataset import PreprocessedDataset
from utils.visualization import (
    visualize_overall_accuracy,
    visualize_time_series,
    visualize_error_histograms
)
from utils.random_segment_fit_utils import run_random_segment_fit

# Model imports
from models.pure_lstm_model import PureLSTMNet
from models.dynamics_net import (
    DirectMappingNet_LSTM_Fusion,
    EnhancedPhysicsNet_LSTM,
    HybridDynamicsModel
)

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_model(cfg: Config, checkpoint: str, device: torch.device) -> torch.nn.Module:
    """
    Instantiate and load the model according to cfg.training.MODEL_TYPE.
    """
    mtype = cfg.training.MODEL_TYPE.replace('-', '_').lower()
    if mtype == 'pure_lstm':
        model = PureLSTMNet(
            input_size=cfg.training.INPUT_DIM,
            hidden_size=cfg.training.PURE_LSTM_HIDDEN_DIM,
            num_layers=cfg.training.PURE_LSTM_LAYERS,
            dropout=cfg.training.PURE_LSTM_DROPOUT,
            output_size=cfg.training.PURE_LSTM_OUTPUT_DIM
        ).to(device)
        logger.info("Instantiated PureLSTMNet.")
    else:
        T_arr = load_thrust_allocation_matrix(cfg.paths.THRUST_MATRIX_FILE)
        T = torch.tensor(T_arr, dtype=torch.float32, device=device)
        if mtype == 'direct':
            model = DirectMappingNet_LSTM_Fusion(
                hidden=int(cfg.training.HIDDEN_DIM * cfg.training.E2E_HIDDEN_FACTOR),
                layers=cfg.training.LSTM_LAYERS,
                fusion_hidden=cfg.training.FUSION_HIDDEN_DIM,
                debug=cfg.DEBUG
            ).to(device)
            logger.info("Instantiated DirectMappingNet_LSTM_Fusion.")
        elif mtype == 'hybrid':
            phys = EnhancedPhysicsNet_LSTM(
                T=T,
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
            logger.info("Instantiated HybridDynamicsModel.")
        else:
            raise ValueError(f"Unknown MODEL_TYPE='{cfg.training.MODEL_TYPE}'")
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()
    logger.info(f"Loaded checkpoint: {checkpoint}")
    return model


def predict_all(model: torch.nn.Module, loader: DataLoader, device: torch.device):
    """
    Collect predictions and ground truths for entire dataset.
    """
    preds_list, tgts_list = [], []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting"):
            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.to(device)
            out = model(batch['power_window'], batch.get('imu_window'))
            pred = out['accel_pred'] if isinstance(out, dict) else out
            preds_list.append(pred.cpu().numpy())
            tgts_list.append(batch['accel'].cpu().numpy())
    preds = np.vstack(preds_list)
    tgts = np.vstack(tgts_list)
    logger.info(f"Prediction complete: {preds.shape[0]} samples")
    return preds, tgts


def save_to_csv(preds: np.ndarray, tgts: np.ndarray, save_path: str) -> None:
    """
    Save predictions vs true values into CSV.
    """
    D = preds.shape[1]
    cols = [f'pred_{i}' for i in range(D)] + [f'true_{i}' for i in range(D)]
    df = pd.DataFrame(np.hstack([preds, tgts]), columns=cols)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    logger.info(f"Saved CSV → {save_path}")


if __name__ == '__main__':
    cfg = Config()
    cfg.training.MODEL_TYPE = cfg.training.MODEL_TYPE.replace('-', '_').lower()
    device = torch.device(cfg.device.DEVICE if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    parser = argparse.ArgumentParser(description="Predict & compare models")
    parser.add_argument('-c','--checkpoint', default=os.path.join(cfg.paths.MODEL_DIR, f"model_{cfg.training.MODEL_TYPE}.pt"))
    parser.add_argument('-o','--out_csv',     default=os.path.join(cfg.paths.SPLITS_DIR,'pred_vs_true.csv'))
    parser.add_argument('-v','--vis',         default=os.path.join(cfg.paths.SPLITS_DIR,'global_fit.png'))
    parser.add_argument('-t','--time_series', default=os.path.join(cfg.paths.SPLITS_DIR,'time_series.png'))
    parser.add_argument('--time_step', type=float, default=0.11)
    parser.add_argument('-r','--random_seg',  default=os.path.join(cfg.paths.SPLITS_DIR,'random_segment.png'))
    args = parser.parse_args()

    model = load_model(cfg, args.checkpoint, device)
    ds = PreprocessedDataset(
        cfg.paths.TRAIN_FEATURES_FILE,
        cfg.paths.TRAIN_ACCEL_LABELS_FILE,
        cfg.paths.TRAIN_ANGULAR_ACCEL_LABELS_FILE,
        cfg.paths.TRAIN_THRUST_LABELS_FILE,
        window_size=cfg.training.WINDOW_SIZE
    )
    loader = DataLoader(ds, batch_size=cfg.training.BATCH_SIZE, shuffle=False, num_workers=cfg.training.NUM_WORKERS)

    preds, tgts = predict_all(model, loader, device)
    save_to_csv(preds, tgts, args.out_csv)

    # Visualizations
    visualize_overall_accuracy(model, loader, device, save_path=args.vis, max_samples=len(ds))
    visualize_time_series(preds, tgts, time_step=args.time_step, save_path=args.time_series)
    run_random_segment_fit(cfg, model, ds, device, dt=args.time_step, segment_duration=30, save_path=args.random_seg)
    hist_pref = os.path.join(cfg.paths.SPLITS_DIR, 'error_hist')
    visualize_error_histograms(preds, tgts, save_prefix=hist_pref, linear_bins=40, angular_bins=40, tick_num=10)

    # 9) Per-axis error table & time-series plots
    errors = preds - tgts
    abs_errors = np.abs(errors)
    df_err = pd.DataFrame(
        abs_errors,
        columns=[
            'LinearAccel_X_err','LinearAccel_Y_err','LinearAccel_Z_err',
            'AngularAccel_X_err','AngularAccel_Y_err','AngularAccel_Z_err'
        ]
    )
    csv_err = os.path.join(cfg.paths.SPLITS_DIR, 'axis_errors.csv')
    df_err.to_csv(csv_err, index=False)
    logger.info(f"Saved per-axis error table → {csv_err}")

    # Plot error time-series in 3x2 layout
    fig, axes = plt.subplots(3, 2, figsize=(12, 12), sharex=True)
    axes = axes.flatten()
    axis_names = [
        'Linear Accel X','Linear Accel Y','Linear Accel Z',
        'Angular Accel X','Angular Accel Y','Angular Accel Z'
    ]
    times = np.arange(len(df_err)) * args.time_step
    for i, ax in enumerate(axes):
        ax.plot(times, df_err.iloc[:, i], linestyle='-', linewidth=1)
        ax.set_title(f"{axis_names[i]} Error")
        ax.set_ylabel("Error")
        ax.grid(True)
        if i >= 4:
            ax.set_xlabel("Time (s)")
    plt.tight_layout()
    err_plot = os.path.join(cfg.paths.SPLITS_DIR, 'axis_errors_timeseries.png')
    os.makedirs(os.path.dirname(err_plot), exist_ok=True)
    plt.savefig(err_plot, dpi=300, bbox_inches='tight')
    logger.info(f"Saved axis-error time series plot → {err_plot}")
    plt.close()
