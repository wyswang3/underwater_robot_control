#!/usr/bin/env python
import os, platform, multiprocessing, logging, argparse
import torch, numpy as np
from torch.optim.lr_scheduler import (
    OneCycleLR, CosineAnnealingLR, SequentialLR, LinearLR, LambdaLR
)
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
from utils.visualization import plot_loss_curve, visualize_predictions,visualize_overall_accuracy
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
        linear_w=getattr(cfg.training, "LIN_WEIGHT", 0.7),
        use_reg=True
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

    # 8) optimizer -----------------------------------------------------------------
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.MAX_LR,  # 初始即 MAX_LR
        weight_decay=cfg.training.WEIGHT_DECAY,
        eps=1e-6
    )
    for g in opt.param_groups:  # 供 warm‑up 手动调整
        g["initial_lr"] = g["lr"]

    # 8.1) scheduler ----------------------------------------------------------------
    scheduler, step_per_batch = None, False

    if cfg.training.LR_SCHEDULER:
        sched_type = cfg.training.LR_SCHEDULER_TYPE.lower()

        # ------------------------------------------------------------------ #
        # 1. OneCycleLR（原方案，保持兼容）                                   #
        # ------------------------------------------------------------------ #
        if sched_type == "onecyclelr":
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

        # ------------------------------------------------------------------ #
        # 2. CosineAnnealingLR + Linear warm‑up（前 10 % epoch 升 LR）        #
        # ------------------------------------------------------------------ #
        elif sched_type == "cosine":
            warm_epochs = int(0.10 * cfg.training.NUM_EPOCHS)  # 10 % warm‑up
            cosine_epochs = cfg.training.NUM_EPOCHS - warm_epochs

            scheduler = SequentialLR(
                opt,
                schedulers=[
                    # Linear warm‑up from 0.1·MAX_LR → MAX_LR
                    LinearLR(opt, start_factor=0.1, end_factor=1.0,
                             total_iters=warm_epochs),
                    # Cosine decay to MIN_LR
                    CosineAnnealingLR(opt, T_max=cosine_epochs,
                                      eta_min=cfg.training.MIN_LR)
                ],
                milestones=[warm_epochs]
            )
            step_per_batch = False  # 每 epoch 调度
            logger.info("Warm‑up + CosineAnnealingLR scheduler on.")

        # ------------------------------------------------------------------ #
        # 3. Polynomial decay (p=2) + batch warm‑up                          #
        # ------------------------------------------------------------------ #
        elif sched_type == "poly":
            total_steps = cfg.training.NUM_EPOCHS * len(dl_tr)
            warm_steps = cfg.training.WARMUP_STEPS
            if warm_steps <= 0:  # 若配置为 0 / None
                warm_steps = int(0.05 * total_steps)  # 5 % total
            power = 2.0  # (1‑p)^power

            def poly_decay(step: int):
                if step < warm_steps:  # warm‑up
                    return (step + 1) / warm_steps
                progress = (step - warm_steps) / max(1, total_steps - warm_steps)
                return (1.0 - progress) ** power  # (1‑p)^power

            scheduler = LambdaLR(opt, lr_lambda=poly_decay)
            step_per_batch = True
            logger.info(f"Poly decay (power={power}) + warm‑up on. "
                        f"warm_steps={warm_steps}, total_steps={total_steps}")

        else:
            logger.warning(f"Unknown LR_SCHEDULER_TYPE='{cfg.training.LR_SCHEDULER_TYPE}', "
                           "no scheduler used.")

    # 9) 训练
    logger.info("=== Train ===")
    history = custom_train_model(
        model, criterion, dl_tr,
        epochs=cfg.training.NUM_EPOCHS,
        optimizer=opt,
        scheduler=scheduler,
        warmup_steps=cfg.training.WARMUP_STEPS,
        grad_clip=cfg.training.CLIP_GRAD_NORM,
        amp_enabled=False,
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
    # Prediction comparison visualization for selected batches:
    prediction_cmp_path = os.path.join(cfg.paths.SPLITS_DIR, "prediction_comparison.png")
    visualize_predictions(
        model,
        dl_val,  # Use the validation DataLoader directly
        device,
        num_batches=2,  # Process the first 2 batches
        num_samples=6,  # Visualize 6 samples per batch
        save_path=prediction_cmp_path
    )
    print(f"Prediction comparison visualization saved -> {prediction_cmp_path}")

    # Global fitting accuracy visualization:
    # This function aggregates predictions from up to max_samples samples from the validation set,
    # and then produces a scatter plot (or subplots for multi-dim outputs) along with a histogram of RMSE distribution.
    global_accuracy_path = os.path.join(cfg.paths.SPLITS_DIR, "global_fitting_accuracy.png")
    visualize_overall_accuracy(
        model,
        dl_val,
        device,
        save_path=global_accuracy_path,
        max_samples=1000  # You can adjust max_samples as needed
    )
    print(f"Global fitting accuracy visualization saved -> {global_accuracy_path}")

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
