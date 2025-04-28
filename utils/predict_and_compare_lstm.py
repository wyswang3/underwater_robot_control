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
  6. (新增) Produce random-segment fit analysis & plot

Usage:
    python utils/predict_and_compare_lstm.py
Or override:
    python utils/predict_and_compare_lstm.py \
      -c path/to/checkpoint.pt \
      -o path/to/pred_vs_true.csv \
      -v path/to/global_fit.png \
      -t path/to/time_series.png \
      --time_step 0.1 \
      --random_seg random_segment.png
"""
import os
import argparse
import logging

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from models.dynamics_net import (
    EnhancedPhysicsNet_LSTM,
    DirectMappingNet_LSTM_Fusion,
    HybridDynamicsModel,
)
from utils.preprocessing import load_thrust_allocation_matrix
from utils.dataset import PreprocessedDataset
from utils.visualization import (
    visualize_overall_accuracy,
    visualize_time_series,
    visualize_error_histograms
)
from utils.random_segment_fit_utils import run_random_segment_fit

# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

def load_model(cfg: Config, checkpoint: str, device: torch.device) -> torch.nn.Module:
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

    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    logger.info("Loaded checkpoint: %s", checkpoint)
    return model

def predict_all(model: torch.nn.Module, loader: DataLoader, device: torch.device):
    all_preds, all_tgts = [], []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting"):
            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.to(device)
            out = model(batch['power_window'], batch['imu_window'])
            pred = out['accel_pred'] if isinstance(out, dict) else out
            all_preds.append(pred.cpu().numpy())
            all_tgts.append(batch['accel'].cpu().numpy())
    preds = np.vstack(all_preds)
    tgts  = np.vstack(all_tgts)
    logger.info("Prediction complete: %d samples", preds.shape[0])
    return preds, tgts

def save_to_csv(preds: np.ndarray, tgts: np.ndarray, path: str):
    D = preds.shape[1]
    cols = [f'pred_{i}' for i in range(D)] + [f'true_{i}' for i in range(D)]
    df = pd.DataFrame(np.hstack([preds, tgts]), columns=cols)
    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Saved CSV to %s", path)

def main():
    cfg = Config()
    device = torch.device(cfg.device.DEVICE if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser(description="Predict & compare (LSTM branch)")
    parser.add_argument('-c','--checkpoint',  default=os.path.join(cfg.paths.MODEL_DIR,'model.pt'))
    parser.add_argument('-o','--out_csv',     default=os.path.join(cfg.paths.SPLITS_DIR,'pred_vs_true.csv'))
    parser.add_argument('-v','--vis',         default=os.path.join(cfg.paths.SPLITS_DIR,'global_fit.png'))
    parser.add_argument('-t','--time_series', default=os.path.join(cfg.paths.SPLITS_DIR,'time_series.png'))
    parser.add_argument('--time_step', type=float, default=0.11)
    parser.add_argument('-r','--random_seg',  default=os.path.join(cfg.paths.SPLITS_DIR,'random_segment.png'))
    args = parser.parse_args()

    # 1) Load model
    model = load_model(cfg, args.checkpoint, device)

    # 2) Prepare dataset & loader
    ds = PreprocessedDataset(
        features_file=cfg.paths.TRAIN_FEATURES_FILE,
        accel_file=cfg.paths.TRAIN_ACCEL_LABELS_FILE,
        angular_accel_file=cfg.paths.TRAIN_ANGULAR_ACCEL_LABELS_FILE,
        thrust_file=cfg.paths.TRAIN_THRUST_LABELS_FILE,
        window_size=cfg.training.WINDOW_SIZE
    )
    loader = DataLoader(ds, batch_size=cfg.training.BATCH_SIZE,
                        shuffle=False, num_workers=cfg.training.NUM_WORKERS)

    # 3) Predict
    preds, tgts = predict_all(model, loader, device)

    # 4) Save CSV
    save_to_csv(preds, tgts, args.out_csv)

    # 5) Global fit
    visualize_overall_accuracy(model, loader, device,
                               save_path=args.vis,
                               max_samples=len(ds))
    logger.info("Saved global fit to %s", args.vis)

    # 6) Time-series
    visualize_time_series(preds, tgts,
                          time_step=args.time_step,
                          save_path=args.time_series)
    logger.info("Saved time-series plot to %s", args.time_series)

    # 7) Random-segment fit
    run_random_segment_fit(
        cfg=cfg,
        model=model,
        dataset=ds,
        device=device,
        dt=args.time_step,
        segment_duration=30,
        save_path=args.random_seg
    )
    logger.info("Saved random-segment fit to %s", args.random_seg)

    # 8) Error histograms
    hist_pref = os.path.join(cfg.paths.SPLITS_DIR, "error_hist")
    visualize_error_histograms(
        preds, tgts,
        save_prefix=hist_pref,
        linear_bins=40,
        angular_bins=40,
        tick_num=10
    )
    logger.info("Saved error histograms to %s_[linear|angular].png", hist_pref)

if __name__ == '__main__':
    main()
