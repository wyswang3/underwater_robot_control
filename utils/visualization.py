# utils/visualization.py
import os
from typing import List, Optional, Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

# 统一英文字体
matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans"]
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["axes.unicode_minus"] = False


# ╭────────────────────────────────────────────╮
# │                损失曲线                    │
# ╰────────────────────────────────────────────╯
def plot_loss_curve(
    train_losses: Sequence[float],
    val_losses: Optional[Sequence[float]] = None,
    save_path: Optional[str] = None,
) -> None:
    epochs = np.arange(1, len(train_losses) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="Train", marker="o")
    if val_losses is not None:
        plt.plot(epochs, val_losses, label="Val", marker="o")
    plt.xlabel("Epoch");  plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.grid(True);  plt.legend();  plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.show();  plt.close()

# ╭────────────────────────────────────────────╮
# │            预测 vs 真值 可视化             │
# ╰────────────────────────────────────────────╯
@torch.no_grad()
def visualize_predictions(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_batches: int = 1,
    num_samples: int = 6,
    save_path: Optional[str] = None,
) -> None:
    """
    从 dataloader 取前 num_batches 个 batch，
    对每个 batch 取前 num_samples 条样本，画出 6 维加速度预测对比，并打印 RMSE。
    """
    model.eval()
    model.to(device)

    all_pred, all_gt = [], []
    it = iter(dataloader)
    for _ in range(num_batches):
        try:
            batch = next(it)
        except StopIteration:
            break

        # ---- to device ----
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device, non_blocking=True)

        # ---- forward ----
        out = model(batch["power_window"], batch["imu_window"])
        if isinstance(out, dict):
            out = out["accel_pred"]
        all_pred.append(out.cpu())
        all_gt.append(batch["accel"].cpu())

    if not all_pred:
        print("❌ DataLoader 空，无法可视化！")
        return

    preds = torch.cat(all_pred)         # (N, D)
    gts   = torch.cat(all_gt)           # (N, D)
    N, D  = preds.size()
    N     = min(N, num_samples)
    dims  = np.arange(D)

    # ---- 误差指标 ----
    rmse_dim = torch.sqrt(torch.mean((preds - gts) ** 2, dim=0))
    mae_dim  = torch.mean(torch.abs(preds - gts), dim=0)
    overall_rmse = torch.sqrt(torch.mean((preds - gts) ** 2)).item()

    print("RMSE per dim :", torch.round(rmse_dim, decimals=4).numpy())
    print("MAE  per dim :", torch.round(mae_dim,  decimals=4).numpy())
    print("Overall RMSE :", overall_rmse)

    # ---- 绘图 ----
    fig, axes = plt.subplots(N, 1, figsize=(10, 3 * N), sharex=True)
    axes = axes if isinstance(axes, np.ndarray) else np.array([axes])

    for i in range(N):
        gt_i = gts[i]
        pr_i = preds[i]
        rmse_i = torch.sqrt(torch.mean((pr_i - gt_i) ** 2)).item()

        ax = axes[i]
        ax.plot(dims, gt_i.numpy(), "o-", label="GT")
        ax.plot(dims, pr_i.numpy(), "x--", label="Pred")
        ax.set_ylabel("Value")
        ax.set_title(f"Sample {i+1}  |  RMSE={rmse_i:.3f}")
        ax.grid(True)
        ax.legend()

    axes[-1].set_xlabel("Accel Dimension")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.show();  plt.close()

