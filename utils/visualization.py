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
