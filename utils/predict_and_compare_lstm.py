#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
utils/predict_and_compare_lstm.py

LSTM branch convenience script:
  1. Load HybridDynamicsModel checkpoint (default from config)
  2. Predict on full PreprocessedDataset
  3. Save 6-D preds vs true to CSV
  4. Produce global fit visualization (scatter + histogram)
  5. Produce time-series comparison (2×3 subplots)

Usage:
    python utils/predict_and_compare_lstm.py
Or override:
    python utils/predict_and_compare_lstm.py \
      -c path/to/checkpoint.pt \
      -o path/to/pred_vs_true.csv \
      -v path/to/global_fit.png \
      -t path/to/time_series.png \
      --time_step 0.1
"""
import os
import argparse
from typing import Tuple

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from config import Config
from models.dynamics_net import (
    EnhancedPhysicsNet_LSTM,
    DirectMappingNet_LSTM_Fusion,
    HybridDynamicsModel,
)
from utils.preprocessing import load_thrust_allocation_matrix
from utils.dataset import PreprocessedDataset
from utils.visualization import visualize_overall_accuracy, visualize_time_series


def load_model(
    cfg: Config,
    checkpoint: str,
    device: torch.device
) -> torch.nn.Module:
    """Build and load the hybrid LSTM model."""
    T_np = load_thrust_allocation_matrix(cfg.paths.THRUST_MATRIX_FILE)
    T = torch.tensor(T_np, dtype=torch.float32, device=device)
    phys = EnhancedPhysicsNet_LSTM(
        T=T,
        hidden=cfg.training.HIDDEN_DIM,
        layers=cfg.training.LSTM_LAYERS,
        debug=cfg.DEBUG
    ).to(device)
    e2e = DirectMappingNet_LSTM_Fusion(
        hidden=int(cfg.training.HIDDEN_DIM * getattr(cfg.training, 'E2E_HIDDEN_FACTOR', 0.5)),
        layers=cfg.training.LSTM_LAYERS,
        fusion_hidden=getattr(cfg.training, 'FUSION_HIDDEN_DIM', 256),
        debug=cfg.DEBUG
    ).to(device)
    model = HybridDynamicsModel(phys, e2e, debug=cfg.DEBUG).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded checkpoint: {checkpoint}")
    return model


def predict_all(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    """Run model on loader and return (preds, tgts) arrays of shape (N,6)."""
    all_preds, all_tgts = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.to(device)
            out = model(batch['power_window'], batch['imu_window'])
            pred = out['accel_pred'] if isinstance(out, dict) else out
            all_preds.append(pred.cpu().numpy())
            all_tgts.append(batch['accel'].cpu().numpy())
    preds = np.vstack(all_preds)
    tgts = np.vstack(all_tgts)
    return preds, tgts


def save_to_csv(
    preds: np.ndarray,
    tgts: np.ndarray,
    path: str
) -> None:
    """Save preds and tgts to CSV with columns pred_0..5, true_0..5."""
    D = preds.shape[1]
    cols = [f'pred_{i}' for i in range(D)] + [f'true_{i}' for i in range(D)]
    df = pd.DataFrame(np.hstack([preds, tgts]), columns=cols)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved CSV to {path}")


def main():
    cfg = Config()
    device = torch.device(cfg.device.DEVICE if torch.cuda.is_available() else 'cpu')

    default_ckpt = os.path.join(cfg.paths.MODEL_DIR, 'model.pt')
    default_csv = os.path.join(cfg.paths.SPLITS_DIR, 'pred_vs_true.csv')
    default_vis = os.path.join(cfg.paths.SPLITS_DIR, 'global_fit.png')
    default_ts = os.path.join(cfg.paths.SPLITS_DIR, 'time_series.png')

    parser = argparse.ArgumentParser(description="Predict & compare (LSTM branch)")
    parser.add_argument('-c', '--checkpoint', default=default_ckpt,
                        help='Model checkpoint path')
    parser.add_argument('-o', '--out_csv', default=default_csv,
                        help='CSV output path')
    parser.add_argument('-v', '--vis', default=default_vis,
                        help='Global fit plot path')
    parser.add_argument('-t', '--time_series', default=default_ts,
                        help='Time series plot path')
    parser.add_argument('--time_step', type=float, default=0.11,
                        help='Time interval for time-series plot')
    args = parser.parse_args()

    model = load_model(cfg, args.checkpoint, device)

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

    preds, tgts = predict_all(model, loader, device)
    save_to_csv(preds, tgts, args.out_csv)

    visualize_overall_accuracy(
        model, loader, device,
        save_path=args.vis,
        max_samples=len(ds)
    )
    print(f"Saved global fit plot to {args.vis}")

    visualize_time_series(
        preds, tgts,
        time_step=args.time_step,
        save_path=args.time_series
    )
    print(f"Saved time series plot to {args.time_series}")


if __name__ == '__main__':
    main()
