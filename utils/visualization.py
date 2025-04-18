import os
from typing import Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

# Use English font configuration; here still set to support Chinese if needed,
# but all labels and titles are in English.
matplotlib.rcParams["font.sans-serif"] = ["SimHei"]
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["axes.unicode_minus"] = False


def batch_to_device(batch, device):
    """
    Recursively move all tensors in a batch (dict/list/tuple) to the specified device.
    """
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
    """
    Plot the training and validation loss curves.

    Parameters:
      - train_losses: list or numpy array of training losses per epoch.
      - val_losses: list or numpy array of validation losses per epoch (optional).
      - save_path: str, if provided, save the figure to this path.
    """
    epochs = np.arange(1, len(train_losses) + 1)
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_losses, label="Training Loss", marker="o")
    if val_losses is not None:
        plt.plot(epochs, val_losses, label="Validation Loss", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Curve")
    plt.legend()
    plt.grid(True)
    if save_path is not None:
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
    Aggregate predictions and ground truth values from the provided dataloader.

    Parameters:
      - model: Trained network.
      - dataloader: DataLoader for evaluation.
      - device: torch.device for computation.
      - max_samples: Maximum number of samples to aggregate.

    Returns:
      - preds: Tensor of shape (N, D)
      - gts: Tensor of shape (N, D)
    """
    model.eval()
    model.to(device)
    all_preds = []
    all_gts = []
    sample_count = 0

    for batch in dataloader:
        batch = batch_to_device(batch, device)
        outputs = model(batch["power_window"], batch["imu_window"])
        if isinstance(outputs, dict):
            outputs = outputs.get("accel_pred", outputs)
        all_preds.append(outputs.detach().cpu())
        all_gts.append(batch["accel"].detach().cpu())
        sample_count += outputs.size(0)
        if sample_count >= max_samples:
            break

    if not all_preds:
        raise ValueError("No data aggregated from DataLoader!")
    preds = torch.cat(all_preds, dim=0)
    gts = torch.cat(all_gts, dim=0)
    return preds, gts


@torch.no_grad()
def visualize_global_accuracy(
        model: torch.nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        save_path: Optional[str] = None,
        max_samples: int = 1000,
) -> None:
    """
    Visualize the overall prediction accuracy of the network.

    This function aggregates predictions and ground truth values from the provided dataloader
    (up to max_samples samples) and creates scatter plots comparing predictions and ground truth.
    For multi-dimensional outputs (e.g., 6-dim acceleration), subplots for each dimension are created
    with a red dashed 45° line indicating perfect prediction.

    Parameters:
      - model: Trained network.
      - dataloader: DataLoader for evaluation.
      - device: torch.device for computation.
      - save_path: If provided, the figure is saved to this path.
      - max_samples: Maximum number of samples to aggregate.
    """
    preds, gts = aggregate_predictions(model, dataloader, device, max_samples)
    N, D = preds.size()
    overall_rmse = torch.sqrt(torch.mean((preds - gts) ** 2)).item()

    if D == 1:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(gts.numpy(), preds.numpy(), alpha=0.5)
        min_val = float(gts.min().item())
        max_val = float(gts.max().item())
        ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=2)
        ax.set_title(f"Global Prediction Accuracy (RMSE: {overall_rmse:.3f})")
        ax.set_xlabel("Ground Truth")
        ax.set_ylabel("Prediction")
        ax.grid(True)
    else:
        fig, axes = plt.subplots(1, D, figsize=(6 * D, 6))
        axes = np.atleast_1d(axes)
        for i in range(D):
            ax = axes[i]
            pred_i = preds[:, i].numpy()
            gt_i = gts[:, i].numpy()
            ax.scatter(gt_i, pred_i, alpha=0.5)
            min_val = float(gt_i.min())
            max_val = float(gt_i.max())
            ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=2)
            rmse_i = torch.sqrt(torch.mean((preds[:, i] - gts[:, i]) ** 2)).item()
            ax.set_title(f"Dimension {i + 1} (RMSE: {rmse_i:.3f})")
            ax.set_xlabel("Ground Truth")
            ax.set_ylabel("Prediction")
            ax.grid(True)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.show()
    plt.close()


@torch.no_grad()
def visualize_overall_error(
        model: torch.nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        save_path: Optional[str] = None,
) -> None:
    """
    Visualize the overall prediction error:
      1. Iterate through the entire dataloader, computing RMSE for each sample.
      2. Plot a scatter plot of sample index vs. RMSE.
      3. Plot a histogram to visualize the RMSE distribution.

    Parameters:
      - model: The trained network.
      - dataloader: DataLoader for evaluation.
      - device: torch.device for computation.
      - save_path: Base file path for saving plots; "_scatter" and "_histogram" will be appended.
    """
    model.eval()
    model.to(device)
    all_rmse = []
    sample_indices = []
    sample_counter = 0

    for batch in dataloader:
        batch = batch_to_device(batch, device)
        outputs = model(batch["power_window"], batch["imu_window"])
        if isinstance(outputs, dict):
            preds = outputs.get("accel_pred", None)
            if preds is None:
                print("Overall error visualization failed: 'accel_pred' not found in model output!")
                continue
        else:
            preds = outputs

        predictions = preds.detach().cpu().numpy()  # (B, D)
        targets = batch["accel"].detach().cpu().numpy()  # (B, D)
        B, dim_pred = predictions.shape
        _, dim_tar = targets.shape
        if dim_pred != dim_tar:
            print(f"Error: Prediction dimension ({dim_pred}) != Target dimension ({dim_tar})")
            continue

        for i in range(B):
            rmse = np.sqrt(np.mean((predictions[i] - targets[i]) ** 2))
            all_rmse.append(rmse)
            sample_indices.append(sample_counter)
            sample_counter += 1

    all_rmse = np.array(all_rmse)
    overall_avg_rmse = np.mean(all_rmse)

    # Scatter plot: Sample Index vs. RMSE
    plt.figure(figsize=(10, 6))
    plt.scatter(sample_indices, all_rmse, c="blue", alpha=0.6, edgecolors="k")
    plt.xlabel("Sample Index")
    plt.ylabel("RMSE")
    plt.title(f"Sample-wise RMSE (Overall Average RMSE: {overall_avg_rmse:.3f})")
    plt.grid(True)
    if save_path:
        base, ext = os.path.splitext(save_path)
        scatter_path = f"{base}_scatter{ext}"
        os.makedirs(os.path.dirname(scatter_path), exist_ok=True)
        plt.savefig(scatter_path, dpi=300)
    plt.show()
    plt.close()

    # Histogram plot: RMSE Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(all_rmse, bins=30, alpha=0.75, edgecolor="black")
    plt.xlabel("RMSE")
    plt.ylabel("Frequency")
    plt.title(f"RMSE Distribution (Overall Average RMSE: {overall_avg_rmse:.3f})")
    plt.grid(True)
    if save_path:
        base, ext = os.path.splitext(save_path)
        hist_path = f"{base}_histogram{ext}"
        os.makedirs(os.path.dirname(hist_path), exist_ok=True)
        plt.savefig(hist_path, dpi=300)
    plt.show()
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