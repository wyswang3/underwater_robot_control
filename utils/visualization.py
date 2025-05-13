import os
from typing import Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

# Use English font configuration
matplotlib.rcParams["font.sans-serif"] = ["SimHei"]
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["axes.unicode_minus"] = False


def batch_to_device(batch, device):
    if isinstance(batch, dict):
        return {k: batch_to_device(v, device) for k, v in batch.items()}
    elif isinstance(batch, (list, tuple)):
        return type(batch)(batch_to_device(v, device) for v in batch)
    elif torch.is_tensor(batch):
        return batch.to(device, non_blocking=True)
    else:
        return batch


def plot_loss_curve(
        train_losses,
        val_losses=None,
        save_path: Optional[str] = None,
) -> None:
    epochs = np.arange(1, len(train_losses) + 1)
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_losses, label="Training Loss", marker="o")
    if val_losses is not None:
        plt.plot(epochs, val_losses, label="Validation Loss", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Curves")
    plt.legend()
    plt.grid(True)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close()


def visualize_predictions(
        model: torch.nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        num_batches: int = 1,
        num_samples: int = 5,
        save_path: Optional[str] = None,
) -> None:
    """
    Visualize model predictions vs targets.
    For sequence outputs (3D tensor), plot time-series comparison.
    For vector outputs (2D tensor), plot bar chart per sample.
    """
    model.eval()
    model.to(device)
    all_preds = []
    all_tgts = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= num_batches:
                break
            batch = batch_to_device(batch, device)
            out = model(batch['power_window'], batch.get('imu_window'))
            preds = out.get('accel_pred', out) if isinstance(out, dict) else out
            tgts = batch['accel']
            all_preds.append(preds.cpu())
            all_tgts.append(tgts.cpu())
    preds = torch.cat(all_preds, dim=0)
    tgts = torch.cat(all_tgts, dim=0)

    # Compute and print RMSE per dimension
    rmse_dim = torch.sqrt(torch.mean((preds - tgts) ** 2, dim=0))
    print("RMSE per dim:", np.round(rmse_dim.numpy(), 4))

    # Determine output shape
    if preds.dim() == 3:
        # sequence predictions: (N, T, D)
        samples = min(num_samples, preds.size(0))
        N, T, D = preds.size()
        fig, axes = plt.subplots(samples, D, figsize=(2 * D, 1.5 * samples), sharex=True)
        axes = np.atleast_2d(axes)
        for i in range(samples):
            for j in range(D):
                ax = axes[i][j]
                ax.plot(preds[i, :, j].numpy(), linestyle='-', linewidth=1)
                ax.plot(tgts[i, :, j].numpy(), linestyle='--', linewidth=1)
                if i == 0:
                    ax.set_title(f"Dim {j+1}")
                if j == 0:
                    ax.set_ylabel(f"Sample {i+1}")
                ax.grid(True, alpha=0.3)
        plt.tight_layout()
    elif preds.dim() == 2:
        # vector predictions: (N, D)
        samples = min(num_samples, preds.size(0))
        D = preds.size(1)
        fig, axes = plt.subplots(samples, 1, figsize=(5, 1.5 * samples), sharex=False)
        axes = np.atleast_1d(axes)
        dims = np.arange(1, D+1)
        for i in range(samples):
            ax = axes[i]
            ax.bar(dims - 0.2, preds[i].numpy(), width=0.4, label='pred')
            ax.bar(dims + 0.2, tgts[i].numpy(), width=0.4, label='true')
            ax.set_xticks(dims)
            ax.set_xlabel('Dimension')
            ax.set_ylabel('Value')
            ax.set_title(f"Sample {i+1}")
            ax.legend()
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
    else:
        raise ValueError(f"Unsupported preds dim: {preds.dim()}")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def visualize_overall_accuracy(
        model: torch.nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        save_path: Optional[str] = None,
        max_samples: int = 1000,
) -> None:
    from utils.visualization import aggregate_predictions
    preds, gts = aggregate_predictions(model, dataloader, device, max_samples)
    N, D = preds.size()
    overall_rmse = torch.sqrt(torch.mean((preds - gts) ** 2)).item()
    fig, axes = plt.subplots(1, D, figsize=(5 * D, 4))
    axes = np.atleast_1d(axes)
    for i, ax in enumerate(axes):
        p = preds[:, i].numpy()
        t = gts[:, i].numpy()
        ax.scatter(t, p, alpha=0.5, s=5)
        mn, mx = min(t.min(), p.min()), max(t.max(), p.max())
        ax.plot([mn, mx], [mn, mx], 'r--')
        ax.set_title(f"Dim {i+1} (RMSE: {torch.sqrt(torch.mean((preds[:,i]-gts[:,i])**2)).item():.3f})")
        ax.set_xlabel('True')
        ax.set_ylabel('Pred')
        ax.grid(True)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close()


def aggregate_predictions(
        model: torch.nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        max_samples: int = 1000,
) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    all_preds, all_gts = [], []
    with torch.no_grad():
        cnt = 0
        for batch in dataloader:
            batch = batch_to_device(batch, device)
            out = model(batch['power_window'], batch.get('imu_window'))
            preds = out.get('accel_pred', out) if isinstance(out, dict) else out
            gts = batch['accel']
            all_preds.append(preds.cpu())
            all_gts.append(gts.cpu())
            cnt += preds.size(0)
            if cnt >= max_samples:
                break
    return torch.cat(all_preds, 0), torch.cat(all_gts, 0)


def random_segment_plot_setup():
    pass  # placeholder if needed


def visualize_error_histograms(
    preds: np.ndarray,
    tgts: np.ndarray,
    save_prefix: str,
    linear_bins: int = 20,
    angular_bins: int = 40,
    tick_num: int = 5
) -> None:
    errors = preds - tgts
    def _plot_group(err, names, bins, suffix):
        bound = float(np.max(np.abs(err))) or 1.0
        edges = np.linspace(-bound, bound, bins+1)
        ticks = np.linspace(-bound, bound, tick_num)
        fig, axes = plt.subplots(1, len(names), figsize=(4*len(names), 4))
        axes = np.atleast_1d(axes)
        for i, ax in enumerate(axes):
            ax.hist(err[:,i], bins=edges, edgecolor='black', alpha=0.7)
            ax.set_title(f"{names[i]} Error")
            ax.set_xlim(-bound, bound)
            ax.set_xticks(ticks)
            ax.set_xlabel('Error')
            ax.set_ylabel('Freq')
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path = f"{save_prefix}_{suffix}.png"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path, dpi=300)
        plt.close()
    linear_names = ["Lin Acc X","Lin Acc Y","Lin Acc Z"]
    angular_names= ["Ang Acc X","Ang Acc Y","Ang Acc Z"]
    _plot_group(errors[:,:3], linear_names, linear_bins, 'linear')
    _plot_group(errors[:,3:], angular_names, angular_bins, 'angular')


def visualize_time_series(
    preds: np.ndarray,
    tgts: np.ndarray,
    time_step: float,
    save_path: str
) -> None:
    errors = preds - tgts
    axis_names = ["LinAcc X","LinAcc Y","LinAcc Z","AngAcc X","AngAcc Y","AngAcc Z"]
    times = np.arange(len(errors)) * time_step
    fig, axes = plt.subplots(3, 2, figsize=(12, 8), sharex=True)
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        ax.plot(times, errors[:,i], linewidth=0.8)
        ax.axhline(0, linestyle=':', linewidth=0.8)
        ax.set_title(axis_names[i])
        ax.set_ylabel('Error')
        ax.grid(True, alpha=0.3)
        if i >= 4:
            ax.set_xlabel('Time (s)')
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
