#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Verification-ONNX-and-PyTorch.py

This script does the following:
  1) Loads Config to get WINDOW_SIZE, INPUT_DIM, MODEL_TYPE, and data paths
  2) Loads & checks the ONNX model structure
  3) Runs a forward pass in ONNX Runtime on the first data window
  4) Instantiates the corresponding PyTorch model, loads its .pt weights
  5) Runs a forward pass in PyTorch on the same window
  6) Prints both outputs and their maximum absolute difference
"""
import os
import numpy as np
import onnx
import onnxruntime as ort
import torch

from config import Config
from models.pure_lstm_model import PureLSTMNet
from models.dynamics_net import (
    DirectMappingNet_LSTM_Fusion,
    EnhancedPhysicsNet_LSTM,
    HybridDynamicsModel,
    EnhancedDynamicsLoss
)
from utils.preprocessing import load_thrust_allocation_matrix


def load_pytorch_model(cfg: Config, device: torch.device, ckpt_path: str):
    mtype = cfg.training.MODEL_TYPE.replace('-', '_').lower()
    if mtype == 'pure_lstm':
        model = PureLSTMNet(
            input_size=cfg.training.INPUT_DIM,
            hidden_size=cfg.training.PURE_LSTM_HIDDEN_DIM,
            num_layers=cfg.training.PURE_LSTM_LAYERS,
            dropout=cfg.training.PURE_LSTM_DROPOUT,
            output_size=cfg.training.PURE_LSTM_OUTPUT_DIM
        ).to(device)
    elif mtype == 'direct':
        model = DirectMappingNet_LSTM_Fusion(
            hidden=int(cfg.training.HIDDEN_DIM * cfg.training.E2E_HIDDEN_FACTOR),
            layers=cfg.training.LSTM_LAYERS,
            fusion_hidden=cfg.training.FUSION_HIDDEN_DIM,
            debug=cfg.DEBUG
        ).to(device)
    else:  # hybrid
        T_np = load_thrust_allocation_matrix(cfg.paths.THRUST_MATRIX_FILE)
        T = torch.tensor(T_np, dtype=torch.float32, device=device)
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

    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def main():
    # 1) load config
    cfg = Config()
    W = cfg.training.WINDOW_SIZE  # e.g. 9
    P = cfg.training.INPUT_DIM  # e.g. 14
    mtype = cfg.training.MODEL_TYPE.replace('-', '_').lower()

    # 2) ONNX model path
    onnx_path = os.path.abspath("model.onnx")
    if not os.path.isfile(onnx_path):
        raise FileNotFoundError(f"Cannot find ONNX model at {onnx_path}")

    # 3) check ONNX legality
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("✅ ONNX model is valid!")

    # 4) launch ONNX Runtime session
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    print("Model inputs :", {inp.name: inp.shape for inp in sess.get_inputs()})
    print("Model outputs:", {out.name: out.shape for out in sess.get_outputs()})

    # 5) load and reshape the first data window
    feat_file = cfg.paths.TRAIN_FEATURES_FILE
    feats = np.load(feat_file)  # shape (N, W*P)
    assert feats.ndim == 2 and feats.shape[1] == W * P, \
        f"Expected (N, {W * P}), got {feats.shape}"
    flat = feats[0].astype(np.float32)  # (W*P,)
    window = flat.reshape(W, P)  # (W, P)
    power_window = window[np.newaxis, ...]  # (1, W, P)

    # 6) construct ONNX inputs
    inputs = {"power_window": power_window}
    if mtype in ("direct", "hybrid"):
        imu_file = cfg.paths.TRAIN_ANGULAR_ACCEL_LABELS_FILE
        imus = np.load(imu_file)  # shape (N,6)
        imu_flat = imus[0].astype(np.float32)  # (6,)
        # assume same value across time steps if no full window saved
        imu_window = imu_flat.reshape(1, 1, 6)
        imu_window = np.tile(imu_window, (1, W, 1))  # (1, W, 6)
        inputs["imu_window"] = imu_window

    # 7) run ONNX inference
    onnx_outs = sess.run(None, inputs)
    onnx_pred = onnx_outs[0]  # (1,6)
    print("ONNX output shape:", onnx_pred.shape)
    print("ONNX output:", onnx_pred)

    # 8) load PyTorch model & checkpoint
    ckpt_path = os.path.join(cfg.paths.MODEL_DIR, f"model_{cfg.training.MODEL_TYPE}.pt")
    device = torch.device(cfg.device.DEVICE if torch.cuda.is_available() else "cpu")
    pt_model = load_pytorch_model(cfg, device, ckpt_path)

    # 9) prepare PyTorch input tensors
    pw_t = torch.from_numpy(power_window).to(device)
    if mtype in ("direct", "hybrid"):
        imu_t = torch.from_numpy(inputs["imu_window"]).to(device)
        with torch.no_grad():
            pt_out_t = pt_model(pw_t, imu_t)
    else:
        with torch.no_grad():
            pt_out_t = pt_model(pw_t)  # pure_lstm only uses power_window

    pt_pred = pt_out_t.cpu().numpy()
    print("PyTorch output shape:", pt_pred.shape)
    print("PyTorch output:", pt_pred)

    # 10) compare
    diff = np.abs(pt_pred - onnx_pred)
    print("Max absolute difference:", diff.max())


if __name__ == "__main__":
    main()
