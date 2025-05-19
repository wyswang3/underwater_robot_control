#!/usr/bin/env python
# export_onnx.py

import os
import torch
import argparse
from typing import Optional

from config import Config
from utils.preprocessing import load_thrust_allocation_matrix

# 导入所有可能的模型类
from models.pure_lstm_model import PureLSTMNet
from models.dynamics_net import (
    DirectMappingNet_LSTM_Fusion,
    EnhancedPhysicsNet_LSTM,
    HybridDynamicsModel,
)

def build_model(
    cfg: Config,
    device: torch.device,
    T: Optional[torch.Tensor]
) -> torch.nn.Module:
    """
    根据 cfg.training.MODEL_TYPE 构建并返回对应模型。
    如果是 direct 或 hybrid 分支，T 必须传入推力矩阵张量；否则可传 None。
    """
    mtype = cfg.training.MODEL_TYPE.replace("-", "_").lower()

    if mtype == "pure_lstm":
        model = PureLSTMNet(
            input_size=cfg.training.INPUT_DIM,
            hidden_size=cfg.training.PURE_LSTM_HIDDEN_DIM,
            num_layers=cfg.training.PURE_LSTM_LAYERS,
            dropout=cfg.training.PURE_LSTM_DROPOUT,
            output_size=cfg.training.PURE_LSTM_OUTPUT_DIM,
        ).to(device)

    elif mtype == "direct":
        model = DirectMappingNet_LSTM_Fusion(
            hidden=int(cfg.training.HIDDEN_DIM * cfg.training.E2E_HIDDEN_FACTOR),
            layers=cfg.training.LSTM_LAYERS,
            fusion_hidden=cfg.training.FUSION_HIDDEN_DIM,
            debug=cfg.DEBUG,
        ).to(device)

    elif mtype == "hybrid":
        # T 在此分支下不能为空
        phys = EnhancedPhysicsNet_LSTM(
            T=T,
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

    else:
        raise ValueError(f"Unknown MODEL_TYPE '{cfg.training.MODEL_TYPE}'")

    return model

def main():
    parser = argparse.ArgumentParser(description="Export model to ONNX")
    parser.add_argument(
        "-c", "--checkpoint",
        default=None,
        help="Path to your trained .pt checkpoint (default: models/checkpoints/model_<type>.pt)"
    )
    parser.add_argument(
        "-o", "--onnx",
        default="model.onnx",
        help="Output ONNX filename"
    )
    args = parser.parse_args()

    # 1) 加载配置
    cfg = Config()
    cfg.training.MODEL_TYPE = cfg.training.MODEL_TYPE.replace("-", "_").lower()
    device = torch.device(cfg.device.DEVICE if torch.cuda.is_available() else "cpu")

    # 2) 如果需要物理网络，加载推力矩阵
    T = None
    if cfg.training.MODEL_TYPE in ("direct", "hybrid"):
        arr = load_thrust_allocation_matrix(cfg.paths.THRUST_MATRIX_FILE)
        T = torch.tensor(arr, dtype=torch.float32, device=device)

    # 3) 构建模型并加载权重
    model = build_model(cfg, device, T)
    ckpt = args.checkpoint or os.path.join(
        cfg.paths.MODEL_DIR, f"model_{cfg.training.MODEL_TYPE}.pt"
    )
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()
    print(f"Loaded checkpoint from {ckpt}")

    # 4) 准备示例输入：batch=1，time=WINDOW_SIZE
    W = cfg.training.WINDOW_SIZE
    dummy_pw = torch.randn(1, W, cfg.training.INPUT_DIM, device=device)
    dummy_imu = torch.randn(1, W, 6, device=device)

    # 5) 导出 ONNX
    if cfg.training.MODEL_TYPE == "pure_lstm":
        inputs = (dummy_pw,)
        names = ["power_window"]
    else:
        inputs = (dummy_pw, dummy_imu)
        names = ["power_window", "imu_window"]

    torch.onnx.export(
        model,
        inputs,
        args.onnx,
        opset_version=11,
        input_names=names,
        output_names=["accel_pred"],
        dynamic_axes={
            "power_window": {0: "batch_size", 1: "time_steps"},
            **({"imu_window": {0: "batch_size", 1: "time_steps"}}
               if cfg.training.MODEL_TYPE != "pure_lstm" else {}),
            "accel_pred": {0: "batch_size"},
        },
    )
    print(f"Exported ONNX model to {args.onnx}")

if __name__ == "__main__":
    main()
