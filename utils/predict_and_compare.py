#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
utils/predict_and_compare.py

MLP‑based branch:
  - Load HybridDynamicsModel checkpoint (default from config)
  - Predict on full PreprocessedDataset
  - Save 6‑D preds vs true to CSV in splits dir
  - Produce global fit plot (scatter+histogram) in splits dir
  - Produce time‑series comparison plot (pred vs true over time)

Usage (defaults):
    python utils/predict_and_compare.py
Or override:
    python utils/predict_and_compare.py \
      -c path/to/model.pt \
      -o path/to/pred_vs_true.csv \
      -v path/to/global_fit.png \
      -t path/to/time_series.png
"""

import os
import argparse

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from config import Config
from models.dynamics_net import EnhancedPhysicsNet, DirectMappingNet, HybridDynamicsModel
from utils.preprocessing import load_thrust_allocation_matrix
from utils.dataset import PreprocessedDataset
from utils.visualization import visualize_global_accuracy,visualize_error_histograms,visualize_time_series


def load_model(cfg: Config, checkpoint: str, device: torch.device) -> torch.nn.Module:
    """
    构建 MLP‑based 强物理网络 + e2e 网络，然后加载 checkpoint。
    """
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
    print(f">> Loaded checkpoint: {checkpoint}")
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
        for batch in loader:
            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.to(device)
            out = model(batch['power_window'], batch['imu_window'])
            pred = out['accel_pred'] if isinstance(out, dict) else out
            all_preds.append(pred.cpu().numpy())
            all_tgts.append(batch['accel'].cpu().numpy())

    preds = np.vstack(all_preds)
    tgts  = np.vstack(all_tgts)
    return preds, tgts


def save_to_csv(preds: np.ndarray,
                tgts: np.ndarray,
                save_path: str) -> None:
    """
    Save predictions vs ground truth into a CSV file:
    columns: pred_0,...,pred_5, true_0,...,true_5
    """
    D = preds.shape[1]
    cols = [f'pred_{i}' for i in range(D)] + [f'true_{i}' for i in range(D)]
    df = pd.DataFrame(np.hstack([preds, tgts]), columns=cols)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f">> Saved CSV: {save_path}")

def main():
    cfg = Config()
    device = torch.device(cfg.device.DEVICE if torch.cuda.is_available() else 'cpu')

    default_ckpt = os.path.join(cfg.paths.MODEL_DIR, 'model_checkpoint.pt')
    default_csv  = os.path.join(cfg.paths.SPLITS_DIR, 'pred_vs_true.csv')
    default_vis  = os.path.join(cfg.paths.SPLITS_DIR, 'global_fit.png')
    default_ts   = os.path.join(cfg.paths.SPLITS_DIR, 'time_series.png')

    p = argparse.ArgumentParser(
        description='Predict, save CSV, visualize global fit and time series'
    )
    p.add_argument('-c', '--checkpoint', default=default_ckpt,
                   help=f'Model checkpoint (.pt), default: {default_ckpt}')
    p.add_argument('-o', '--out_csv', default=default_csv,
                   help=f'Output CSV, default: {default_csv}')
    p.add_argument('-v', '--vis', default=default_vis,
                   help=f'Global fit plot, default: {default_vis}')
    p.add_argument('-t', '--time_series', default=default_ts,
                   help=f'Time series plot, default: {default_ts}')
    args = p.parse_args()

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
    loader = DataLoader(ds,
                        batch_size=cfg.training.BATCH_SIZE,
                        shuffle=False,
                        num_workers=cfg.training.NUM_WORKERS)

    # 3) predict all
    preds, tgts = predict_all(model, loader, device)

    # 4) save CSV
    save_to_csv(preds, tgts, args.out_csv)

    # 5) visualize global fit
    os.makedirs(os.path.dirname(args.vis), exist_ok=True)
    visualize_global_accuracy(model, loader, device,
                              save_path=args.vis,
                              max_samples=len(ds))
    print(f">> Saved global fit plot: {args.vis}")

    # 6) visualize time series
    visualize_time_series(preds, tgts,
                          time_step=0.2,
                          save_path=args.time_series)
    print(f">> Saved time series plot: {args.time_series}")
    # 生成误差分布直方图
    hist_prefix = os.path.join(cfg.paths.SPLITS_DIR, "error_hist")
    visualize_error_histograms(
        preds, tgts,
        save_prefix=hist_prefix,
        linear_bins=40,  # 线性误差 10 个箱
        angular_bins=40,  # 角误差 40 个箱（更细）
        tick_num=10
    )
    print(f">> Saved linear-error histogram:  {hist_prefix}_linear.png")
    print(f">> Saved angular-error histogram: {hist_prefix}_angular.png")


if __name__ == '__main__':
    main()
