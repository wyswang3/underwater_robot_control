import os
from typing import Optional, Sequence, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

# Use English fonts
matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans"]
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["axes.unicode_minus"] = False


def batch_to_device(batch, device):
    """
    Recursively move tensors in batch (dict/list/tuple) to device.
    """
    if isinstance(batch, dict):
        return {k: batch_to_device(v, device) for k, v in batch.items()}
    elif isinstance(batch, (list, tuple)):
        return type(batch)(batch_to_device(v, device) for v in batch)
    elif torch.is_tensor(batch):
        return batch.to(device, non_blocking=True)
    return batch


def plot_loss_curve(
    train_losses: Sequence[float],
    val_losses: Optional[Sequence[float]] = None,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot training and validation loss.
    """
    epochs = np.arange(1, len(train_losses) + 1)
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_losses, label="Training Loss", marker="o")
    if val_losses is not None:
        plt.plot(epochs, val_losses, label="Validation Loss", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.show()
    plt.close()


@torch.no_grad()
def aggregate_predictions(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    max_samples: int = 1000,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Aggregate model outputs and ground truths from dataloader up to max_samples.
    Returns:
      preds: (N, D), gts: (N, D)
    """
    model.eval()
    all_preds, all_gts = [], []
    count = 0
    for batch in dataloader:
        batch = batch_to_device(batch, device)
        out = model(batch['power_window'], batch['imu_window'])
        preds = out['accel_pred'] if isinstance(out, dict) else out
        preds = preds.detach().cpu()
        gts = batch['accel'].detach().cpu()
        all_preds.append(preds)
        all_gts.append(gts)
        count += preds.size(0)
        if count >= max_samples:
            break
    if not all_preds:
        raise RuntimeError("No data aggregated for predictions!")
    preds = torch.cat(all_preds, dim=0)
    gts   = torch.cat(all_gts, dim=0)
    return preds, gts


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
    Plot first num_samples from num_batches: dimwise true vs pred.
    """
    preds, gts = aggregate_predictions(model, dataloader, device, max_samples=num_batches * num_samples)
    N, D = preds.shape
    N = min(N, num_samples)
    dims = np.arange(D)
    # errors
    rmse_dim = torch.sqrt(torch.mean((preds - gts)**2, dim=0))
    mae_dim  = torch.mean(torch.abs(preds - gts), dim=0)
    overall  = torch.sqrt(torch.mean((preds - gts)**2)).item()
    print(f"Overall RMSE: {overall:.4f}")
    print("RMSE per dim:", rmse_dim.round(4).numpy())
    # plot samples
    fig, axes = plt.subplots(N, 1, figsize=(10, 3*N), sharex=True)
    axes = np.atleast_1d(axes)
    for i in range(N):
        ax = axes[i]
        ax.plot(dims, gts[i].numpy(), 'o-', label='True')
        ax.plot(dims, preds[i].numpy(), 'x--', label='Pred')
        ax.set_ylabel('Value')
        ax.set_title(f"Sample {i+1} | RMSE={((preds[i]-gts[i]).pow(2).mean().sqrt().item()):.3f}")
        ax.grid(True)
        ax.legend()
    axes[-1].set_xlabel('Dimension')
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.show()
    plt.close()


@torch.no_grad()
def visualize_overall_accuracy(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    save_path: Optional[str] = None,
    max_samples: int = 1000,
) -> None:
    """
    Scatter vs true & histogram of RMSE.
    """
    preds, gts = aggregate_predictions(model, dataloader, device, max_samples)
    N, D = preds.shape
    rmse_overall = torch.sqrt(torch.mean((preds-gts)**2)).item()
    per_rmse = torch.sqrt(torch.mean((preds-gts)**2, dim=1)).numpy()
    # scatter dims
    fig, axes = plt.subplots(1, D, figsize=(5*D,4))
    axes = np.atleast_1d(axes)
    for i, ax in enumerate(axes):
        ax.scatter(gts[:,i].numpy(), preds[:,i].numpy(), alpha=0.5)
        mn, mx = float(gts[:,i].min()), float(gts[:,i].max())
        ax.plot([mn,mx],[mn,mx],'r--')
        ax.set_title(f"Dim{i+1} RMSE={( (preds[:,i]-gts[:,i]).pow(2).mean().sqrt().item()):.3f}")
        ax.set_xlabel('True')
        ax.set_ylabel('Pred')
        ax.grid(True)
    plt.suptitle(f"Global RMSE: {rmse_overall:.3f}")
    plt.tight_layout(rect=[0,0,1,0.95])
    if save_path:
        base, ext = os.path.splitext(save_path)
        scatter_path = base + '_scatter' + ext
        os.makedirs(os.path.dirname(scatter_path), exist_ok=True)
        plt.savefig(scatter_path, dpi=300)
    plt.show()
    plt.close()
    # histogram
    plt.figure(figsize=(8,6))
    plt.hist(per_rmse, bins=30, edgecolor='k', alpha=0.7)
    plt.xlabel('Sample RMSE')
    plt.ylabel('Frequency')
    plt.title(f"RMSE Distribution (Overall {rmse_overall:.3f})")
    if save_path:
        base, ext = os.path.splitext(save_path)
        hist_path = base + '_hist' + ext
        os.makedirs(os.path.dirname(hist_path), exist_ok=True)
        plt.savefig(hist_path, dpi=300)
    plt.show()
    plt.close()


@torch.no_grad()
def visualize_time_series(
    preds: np.ndarray,
    tgts: np.ndarray,
    time_step: float,
    save_path: str
) -> None:
    """
    Plot time-series of prediction errors (pred - true) for each dimension.
    x-axis: time steps; layout 2 columns x 3 rows; red fine solid lines, no legend.
    """
    errors = preds - tgts
    axis_names = [
        "Linear Accel X", "Linear Accel Y", "Linear Accel Z",
        "Angular Accel X", "Angular Accel Y", "Angular Accel Z"
    ]
    times = np.arange(len(errors)) * time_step
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        # Red fine solid line for error
        ax.plot(
            times,
            errors[:, i],
            color='red',
            linestyle='-',
            linewidth=0.8
        )
        # Zero reference line
        ax.axhline(0, color='gray', linestyle=':', linewidth=0.8)
        ax.set_title(f"Error in {axis_names[i]}", fontsize=10)
        ax.set_ylabel('Error', fontsize=9)
        ax.grid(True, alpha=0.3)
        if i in (4, 5):
            ax.set_xlabel('Time (s)', fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


@torch.no_grad()
def visualize_error_histograms(
    preds: np.ndarray,
    tgts: np.ndarray,
    save_prefix: str,
    linear_bins: int = 20,
    angular_bins: int = 40,
    tick_num: int = 5
) -> None:
    """
    preds/tgts: (N,6)
    linear_bins:   number of bins for dimensions 0–2
    angular_bins:  number of bins for dimensions 3–5
    tick_num:      number of x‑axis ticks
    """
    errors = preds - tgts  # (N,6)

    def _plot_group(err_group, names, bins, suffix):
        bound = float(np.max(np.abs(err_group))) or 1.0
        edges = np.linspace(-bound, bound, bins + 1)
        ticks = np.linspace(-bound, bound, tick_num)
        fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 4))
        axes = np.atleast_1d(axes)
        for i, ax in enumerate(axes):
            ax.hist(err_group[:, i], bins=edges, edgecolor='black', alpha=0.7)
            ax.set_title(f"{names[i]} Error")
            ax.set_xlim(-bound, bound)
            ax.set_xticks(ticks)
            ax.set_xlabel("Error")
            ax.set_ylabel("Frequency")
            ax.grid(True)
        plt.tight_layout()
        path = f"{save_prefix}_{suffix}.png"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path, dpi=300)
        plt.show()
        plt.close()

    linear_names  = ["Linear Accel X","Linear Accel Y","Linear Accel Z"]
    angular_names = ["Angular Accel X","Angular Accel Y","Angular Accel Z"]

    _plot_group(errors[:, :3], linear_names,  linear_bins,   'linear')
    _plot_group(errors[:, 3:], angular_names, angular_bins, 'angular')