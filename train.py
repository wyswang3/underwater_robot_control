# train.py
import os
import argparse
import multiprocessing
import torch
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# ── 项目内部模块 ────────────────────────────────────────────────
from config                 import Config
from models.dynamics_net    import BetterHydroNet, PhysicsAwareLoss
from utils.preprocessing    import load_thrust_allocation_matrix
from utils.dataset          import PreprocessedDataset, check_dataset
from utils.visualization    import plot_loss_curve, visualize_predictions
from utils.random_segment_fit_utils import run_random_segment_fit
from utils.training         import custom_train_model, validate_model
import evaluate

# ── 划分训练与验证数据集 ─────────────────────────────────────────
def split_dataset(dataset, ratio: float = 0.8):
    """按 ratio 拆分为训练集与验证集。"""
    n = len(dataset)
    n_train = int(n * ratio)
    n_val = n - n_train
    return torch.utils.data.random_split(dataset, [n_train, n_val])


def main():
    # -------- 0. 多进程设置（Windows 兼容） ---------------------
    multiprocessing.set_start_method("spawn", force=True)

    # -------- 1. 读取配置 & 设置随机种子 ------------------------
    cfg = Config()
    cfg.print_config()
    torch.manual_seed(42)
    np.random.seed(42)
    # 根据配置确定设备：先检查配置指定的设备是否可用
    if cfg.device.DEVICE.startswith("cuda"):
        if not torch.cuda.is_available():
            print(f"[WARN] 配置中指定的 {cfg.device.DEVICE} 不可用，自动切换到 cpu")
            device = torch.device("cpu")
        else:
            device = torch.device(cfg.device.DEVICE)
    else:
        device = torch.device("cpu")
    print(f"[INFO] Using device: {device}")

    # -------- 2. 推力分配矩阵（仅调试，如模型已不使用可忽略） ----
    thrust_matrix = load_thrust_allocation_matrix(cfg.paths.THRUST_MATRIX_FILE)
    T = torch.tensor(thrust_matrix, device=device)
    print(f"[INFO] Thrust allocation matrix shape: {T.shape}")

    # -------- 3. 构建模型 --------------------------------------
    model = BetterHydroNet(
        window_size = cfg.training.WINDOW_SIZE,
        input_dim   = cfg.training.INPUT_DIM,
        hidden_dim  = cfg.training.HIDDEN_DIM,
        lstm_layers = cfg.training.LSTM_LAYERS,
    ).to(device)
    print("[INFO] Model constructed.")

    # -------- 4. 定义损失函数 ----------------------------------
    # 从配置文件中读取物理损失相关参数；如没有则使用默认值
    lambda_phy = getattr(cfg.training, "LAMBDA_PHY", 0.4)
    criterion = PhysicsAwareLoss(lambda_phy=lambda_phy).to(device)
    print(f"[INFO] Loss function (PhysicsAwareLoss) created with lambda_phy={lambda_phy}.")

    # -------- 5. 构建数据集与 DataLoader -----------------------
    ds_full = PreprocessedDataset(
        features_file       = cfg.paths.TRAIN_FEATURES_FILE,
        accel_file          = cfg.paths.TRAIN_ACCEL_LABELS_FILE,
        angular_accel_file  = cfg.paths.TRAIN_ANGULAR_ACCEL_LABELS_FILE,
        thrust_file         = cfg.paths.TRAIN_THRUST_LABELS_FILE,
        window_size         = cfg.training.WINDOW_SIZE,
    )
    check_dataset(ds_full, accel_threshold=1)
    ds_train, ds_val = split_dataset(ds_full, 0.8)
    print(f"[INFO] Dataset split: {len(ds_train)} training samples, {len(ds_val)} validation samples.")

    dl_train = torch.utils.data.DataLoader(
        ds_train,
        batch_size  = cfg.training.BATCH_SIZE,
        shuffle     = True,
        num_workers = cfg.training.NUM_WORKERS,
        pin_memory  = (device.type == "cuda"),
    )
    dl_val = torch.utils.data.DataLoader(
        ds_val,
        batch_size  = cfg.training.BATCH_SIZE,
        shuffle     = False,
        num_workers = cfg.training.NUM_WORKERS,
        pin_memory  = (device.type == "cuda"),
    )
    print("[INFO] DataLoader created.")

    # -------- 6. 配置学习率调度器（可选） --------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = cfg.training.LEARNING_RATE,
        weight_decay = cfg.training.WEIGHT_DECAY,
    )
    scheduler = (
        CosineAnnealingWarmRestarts(
            optimizer,
            T_0     = cfg.training.T_0,
            T_mult  = cfg.training.T_MULT,
            eta_min = cfg.training.ETA_MIN,
        )
        if cfg.training.LR_SCHEDULER else None
    )
    if scheduler:
        print("[INFO] Learning rate scheduler configured.")

    # -------- 7. 模型训练 -------------------------------------
    print("\n=== Training ===")
    model, train_losses = custom_train_model(
        model, criterion, dl_train,
        epochs       = cfg.training.NUM_EPOCHS,
        lr           = cfg.training.LEARNING_RATE,
        weight_decay = cfg.training.WEIGHT_DECAY,
        grad_clip    = cfg.training.CLIP_GRAD_NORM,
        scheduler    = scheduler,
    )

    # -------- 8. 保存模型与 loss 曲线 --------------------------
    ckpt_path = os.path.join(cfg.paths.MODEL_DIR, "model_checkpoint.pt")
    os.makedirs(cfg.paths.MODEL_DIR, exist_ok=True)
    torch.save(model.state_dict(), ckpt_path)
    print(f"[INFO] Model saved to {ckpt_path}")

    plot_loss_curve(
        train_losses,
        save_path = os.path.join(cfg.paths.SPLITS_DIR, "training_loss.png")
    )
    print("[INFO] Training loss curve saved.")

    # -------- 9. 模型验证 --------------------------------------
    val_loss = validate_model(model, criterion, dl_val)
    print(f"[INFO] Final validation loss: {val_loss:.6f}")

    # -------- 10. 调用评估模块 ---------------------------------
    print("\n=== Evaluation ===")
    evaluate.main(argparse.Namespace(checkpoint=ckpt_path))

    # -------- 11. 可视化预测结果 -------------------------------
    visualize_predictions(
        model, dl_val, device,
        save_path = os.path.join(cfg.paths.SPLITS_DIR, "prediction_comparison.png")
    )
    print("[INFO] Prediction comparison visualization saved.")

    # -------- 12. 随机片段拟合对比 -----------------------------
    run_random_segment_fit(
        cfg, model, ds_full, device,
        dt               = 0.2,
        segment_duration = 20,
        save_path        = os.path.join(cfg.paths.SPLITS_DIR, "random_segment_comparison.png")
    )
    print("[INFO] Random segment fitting visualization saved.")


if __name__ == "__main__":
    main()
