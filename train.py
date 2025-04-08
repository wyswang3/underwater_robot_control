import os
import torch
import numpy as np
import argparse
import multiprocessing
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.cuda.amp import autocast, GradScaler

# 项目内部模块导入
from config import Config
from models.dynamics_net import BetterHydroNet, HydroParamEstimator, PhysicsAwareLoss
from utils.preprocessing import load_thrust_allocation_matrix
from utils.visualization import plot_loss_curve, visualize_predictions
from utils.random_segment_fit_utils import run_random_segment_fit
from utils.dataset import PreprocessedDataset, check_dataset
import evaluate  # 外部评估脚本
from utils.training import custom_train_model, validate_model

def split_dataset(dataset, train_ratio=0.8):
    """将数据集划分为训练集和验证集。"""
    total = len(dataset)
    train_size = int(total * train_ratio)
    val_size = total - train_size
    return torch.utils.data.random_split(dataset, [train_size, val_size])

def main():
    # 设置多进程启动模式
    multiprocessing.set_start_method('spawn', force=True)

    # 1) 加载配置并打印
    cfg = Config()
    cfg.print_config()
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device(cfg.device.DEVICE)
    print(f"Using device: {device}")

    # 2) 加载推力分配矩阵 (6x8)
    # 如果新网络不直接依赖推力数据，可以传入占位Tensor
    thrust_matrix_np = load_thrust_allocation_matrix(cfg.paths.THRUST_MATRIX_FILE)
    thrust_matrix = torch.tensor(thrust_matrix_np, device=device)

    # 3) 初始化模型：使用 BetterHydroNet
    model = BetterHydroNet(
        window_size=cfg.training.WINDOW_SIZE,
        input_dim=cfg.training.INPUT_DIM,
        hidden_dim=cfg.training.HIDDEN_DIM,
        lstm_layers=cfg.training.LSTM_LAYERS
    ).to(device)
   # print("Model architecture:")
   # print(model)

    # 4) 定义物理感知损失函数
    # 注意：这里的损失函数依赖于推力数据以及后续 HydroParamEstimator 输出的参数
    criterion = PhysicsAwareLoss(
        inertia=torch.eye(3).to(device),
        lambda_phy=0.4
    )

    # 5) 配置优化器和学习率调度器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.LEARNING_RATE,
        weight_decay=cfg.training.WEIGHT_DECAY
    )
    scheduler = None
    if cfg.training.LR_SCHEDULER:
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=cfg.training.T_0,
            T_mult=cfg.training.T_MULT,
            eta_min=cfg.training.ETA_MIN
        )

    # 6) 加载预处理数据集
    # 数据集文件需包含 'power_window', 'imu_window', 'accel', 'thrust'
    full_dataset = PreprocessedDataset(
        features_file=cfg.paths.TRAIN_FEATURES_FILE,
        accel_file=cfg.paths.TRAIN_ACCEL_LABELS_FILE,
        angular_accel_file=cfg.paths.TRAIN_ANGULAR_ACCEL_LABELS_FILE,
        thrust_file=cfg.paths.TRAIN_THRUST_LABELS_FILE,
        window_size=cfg.training.WINDOW_SIZE
    )
    print("=== Checking dataset for zeros ===")
    check_dataset(full_dataset, accel_threshold=1)
    print("Dataset check complete.")

    # 7) 划分数据集并构建 DataLoader
    train_dataset, val_dataset = split_dataset(full_dataset, train_ratio=0.8)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.training.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.training.NUM_WORKERS,
        pin_memory=True if device.type == 'cuda' else False
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.training.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.training.NUM_WORKERS,
        pin_memory=True if device.type == 'cuda' else False
    )

    # 8) 训练模型
    print("=== Starting Training ===")
    model, train_losses = custom_train_model(
        model, criterion, train_loader,
        epochs=cfg.training.NUM_EPOCHS,
        lr=cfg.training.LEARNING_RATE,
        weight_decay=cfg.training.WEIGHT_DECAY,
        grad_clip=cfg.training.CLIP_GRAD_NORM,
        scheduler=scheduler
    )

    # 9) 保存训练好的模型
    checkpoint_path = os.path.join(cfg.paths.MODEL_DIR, "model_checkpoint.pt")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Model saved -> {checkpoint_path}")

    # 10) 绘制训练损失曲线
    loss_plot_path = os.path.join(cfg.paths.SPLITS_DIR, "training_loss.png")
    plot_loss_curve(train_losses, save_path=loss_plot_path)
    print(f"Training loss curve saved -> {loss_plot_path}")

    # 11) 在验证集上评估模型
    val_loss = validate_model(model, criterion, val_loader)
    print(f"Validation Loss (final): {val_loss:.6f}")

    # 12) 调用评估脚本
    eval_args = argparse.Namespace(checkpoint=checkpoint_path)
    print("=== Starting Evaluation ===")
    evaluate.main(eval_args)

    # 13) 可视化预测结果
    pred_plot_path = os.path.join(cfg.paths.SPLITS_DIR, "prediction_comparison.png")
    visualize_predictions(model, val_loader, device, save_path=pred_plot_path)
    print(f"Prediction visualization saved -> {pred_plot_path}")

    # 14) 随机数据段预测对比
    random_seg_plot_path = os.path.join(cfg.paths.SPLITS_DIR, "random_segment_comparison.png")
    run_random_segment_fit(cfg, model, full_dataset, device, dt=0.11, segment_duration=20,
                             save_path=random_seg_plot_path)
    print(f"Random segment comparison saved -> {random_seg_plot_path}")

if __name__ == "__main__":
    main()
