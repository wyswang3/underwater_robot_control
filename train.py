#!/usr/bin/env python
import os, platform, multiprocessing, logging, argparse
import torch, numpy as np
from torch.optim.lr_scheduler import LambdaLR, OneCycleLR

# ── 项目内部模块 ─────────────────────────────────────────────
from config import Config
from models.dynamics_net import (
    EnhancedPhysicsNet_LSTM,
    EnhancedDynamicsLoss,
    DirectMappingNet_LSTM_Fusion,
    HybridDynamicsModel
)
from utils.preprocessing import load_thrust_allocation_matrix
from utils.dataset import PreprocessedDataset, check_dataset, split_dataset
from utils.visualization import plot_loss_curve, visualize_predictions
from utils.random_segment_fit_utils import run_random_segment_fit
from utils.training import custom_train_model, validate_model
from evaluate import evaluate_model
from utils.log_helper import setup_logging
setup_logging()

import logging
logger = logging.getLogger(__name__)   # 仅这一行即可



# ╭──────────────────────────────╮
# │           主函数             │
# ╰──────────────────────────────╯
def main():
    # 1) 多进程
    if platform.system() == "Windows":
        multiprocessing.set_start_method("spawn", force=True)

    # 2) 配置 & 随机种子
    cfg = Config()
    cfg.print_config()
    torch.manual_seed(42);  np.random.seed(42)

    # 3) 设备
    device = torch.device(cfg.device.DEVICE if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        logger.warning("CUDA 不可用，使用 CPU")
    logger.info(f"Using device: {device}")

    # 4) 推力矩阵
    T = torch.tensor(load_thrust_allocation_matrix(cfg.paths.THRUST_MATRIX_FILE),
                     dtype=torch.float32, device=device)
    logger.info(f"Thrust‑matrix {T.shape}")

    # 5) 网络 -------------------------------------------------------------
    physics_net = EnhancedPhysicsNet_LSTM(
        T=T,  # ← thrust_matrix → T
        hidden=cfg.training.HIDDEN_DIM,  # ← lstm_hidden_dim → hidden
        layers=cfg.training.LSTM_LAYERS,  # ← num_layers   → layers
        debug=cfg.DEBUG
    ).to(device)

    e2e_hidden_factor = getattr(cfg.training, "E2E_HIDDEN_FACTOR", 0.5)
    fusion_hidden_dim = getattr(cfg.training, "FUSION_HIDDEN_DIM", 256)

    e2e_net = DirectMappingNet_LSTM_Fusion(
        hidden=int(cfg.training.HIDDEN_DIM * e2e_hidden_factor),
        layers=cfg.training.LSTM_LAYERS,
        fusion_hidden=fusion_hidden_dim,
        debug=cfg.DEBUG
    ).to(device)

    model = HybridDynamicsModel(physics_net, e2e_net, debug=cfg.DEBUG).to(device)
    logger.info("Hybrid model ready.")

    # 6) 损失 -------------------------------------------------------------
    criterion = EnhancedDynamicsLoss(
        beta=cfg.training.LAMBDA_PHY,  # 正则化权重
        linear_w=getattr(cfg.training, "LIN_WEIGHT", 0.7)
    ).to(device)

    # 7) 数据
    ds_full = PreprocessedDataset(
        cfg.paths.TRAIN_FEATURES_FILE,
        cfg.paths.TRAIN_ACCEL_LABELS_FILE,
        cfg.paths.TRAIN_ANGULAR_ACCEL_LABELS_FILE,
        cfg.paths.TRAIN_THRUST_LABELS_FILE,
        window_size=cfg.training.WINDOW_SIZE,
        clean_data=True
    )
    check_dataset(ds_full)
    ds_tr, ds_val = split_dataset(ds_full, 0.8)

    dl_tr = torch.utils.data.DataLoader(
        ds_tr, batch_size=cfg.training.BATCH_SIZE, shuffle=True,
        num_workers=cfg.training.NUM_WORKERS, pin_memory=device.type == "cuda")
    dl_val = torch.utils.data.DataLoader(
        ds_val, batch_size=cfg.training.BATCH_SIZE, shuffle=False,
        num_workers=cfg.training.NUM_WORKERS, pin_memory=device.type == "cuda")
    logger.info("DataLoader ready.")

    # 8) 优化器 + 调度器
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.MAX_LR,                 # OneCycle 的峰值 lr
        weight_decay=cfg.training.WEIGHT_DECAY,
        eps=1e-6
    )
    for g in opt.param_groups:                 # 供 warm‑up
        g["initial_lr"] = g["lr"]

    scheduler, step_per_batch = None, False
    if cfg.training.LR_SCHEDULER:
        if cfg.training.LR_SCHEDULER_TYPE.lower() == "onecyclelr":
            total_steps = cfg.training.NUM_EPOCHS * len(dl_tr)
            scheduler = OneCycleLR(
                opt,
                max_lr=cfg.training.MAX_LR,
                total_steps=total_steps,
                pct_start=cfg.training.LR_SCHEDULER_PCT_START,
                anneal_strategy="linear",
                final_div_factor=cfg.training.MAX_LR / cfg.training.MIN_LR
            )
            step_per_batch = True
            logger.info("OneCycleLR scheduler on.")
        else:   # 简单 warm‑up
            warm_steps = cfg.training.WARMUP_STEPS
            scheduler = LambdaLR(opt, lambda s: min(1., (s+1)/warm_steps))
            step_per_batch = True
            logger.info("LambdaLR warm‑up on.")

    # 9) 训练
    logger.info("=== Train ===")
    history = custom_train_model(
        model, criterion, dl_tr,
        epochs=cfg.training.NUM_EPOCHS,
        optimizer=opt,
        scheduler=scheduler,
        warmup_steps=cfg.training.WARMUP_STEPS,
        grad_clip=cfg.training.CLIP_GRAD_NORM,
        amp_enabled=(device.type == "cuda"),
        step_per_batch=step_per_batch
    )

    # 10) 保存
    ckpt = os.path.join(cfg.paths.MODEL_DIR, "model.pt")
    os.makedirs(cfg.paths.MODEL_DIR, exist_ok=True)
    torch.save(model.state_dict(), ckpt)
    logger.info(f"Checkpoint → {ckpt}")

    plot_loss_curve(history,
        save_path=os.path.join(cfg.paths.SPLITS_DIR, "train_loss.png"))

    # 11) 验证
    validate_model(model, criterion, dl_val,
                   amp_enabled=(device.type == "cuda"))

    # 12) 评估脚本
    evaluate_model(ckpt)

    # 13) 可视化
    visualize_predictions(
        model,
        dl_val,  # 直接传验证集 DataLoader
        device,
        num_batches=2,  # 取前 2 个 batch
        num_samples=5,  # 每 batch 可视化 5 条
        save_path=os.path.join(cfg.paths.SPLITS_DIR, "prediction_cmp.png")
    )

    # 14) 随机片段拟合
    run_random_segment_fit(
        cfg,
        model,
        ds_full,  # Training or validation dataset object (should support indexing and len())
        device,
        dt=0.2,
        segment_duration=30,
        save_path=os.path.join(cfg.paths.SPLITS_DIR, "segment_cmp.png")
    )

if __name__ == "__main__":
    main()
