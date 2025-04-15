import os
from typing import List, Optional, Sequence, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

# Use an English font configuration.
matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans"]
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
        train_losses: Sequence[float],
        val_losses: Optional[Sequence[float]] = None,
        save_path: Optional[str] = None,
) -> None:
    """
    Plot the training and validation loss curves.

    Parameters:
      - train_losses: Sequence of training losses per epoch.
      - val_losses: Sequence of validation losses per epoch (optional).
      - save_path: If provided, the figure is saved to this path.
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
    Aggregate predictions and ground truth from the provided dataloader.

    Parameters:
      - model: Trained network.
      - dataloader: DataLoader for evaluation.
      - device: torch.device for computation.
      - max_samples: Maximum number of samples to aggregate.

    Returns:
      - preds: Tensor of shape (N, D), where N is the number of aggregated samples.
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
def visualize_predictions(
        model: torch.nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        num_batches: int = 1,
        num_samples: int = 6,
        save_path: Optional[str] = None,
) -> None:
    """
    Visualize the predictions versus ground truth for the first `num_batches` from the dataloader.
    For each batch, the first `num_samples` samples are plotted in individual subplots,
    with RMSE displayed in the title.
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

        batch = batch_to_device(batch, device)
        out = model(batch["power_window"], batch["imu_window"])
        if isinstance(out, dict):
            out = out["accel_pred"]
        all_pred.append(out.detach().cpu())
        all_gt.append(batch["accel"].detach().cpu())

    if not all_pred:
        print("❌ DataLoader is empty, visualization aborted!")
        return

    preds = torch.cat(all_pred)  # Shape: (N, D)
    gts = torch.cat(all_gt)  # Shape: (N, D)
    N_total, D = preds.size()
    N = min(N_total, num_samples)
    dims = np.arange(D)

    # Compute global error metrics
    rmse_dim = torch.sqrt(torch.mean((preds - gts) ** 2, dim=0))
    mae_dim = torch.mean(torch.abs(preds - gts), dim=0)
    overall_rmse = torch.sqrt(torch.mean((preds - gts) ** 2)).item()

    print("RMSE per dimension:", torch.round(rmse_dim, decimals=4).numpy())
    print("MAE per dimension:", torch.round(mae_dim, decimals=4).numpy())
    print("Overall RMSE:", overall_rmse)

    fig, axes = plt.subplots(N, 1, figsize=(10, 3 * N), sharex=True)
    axes = np.atleast_1d(axes)

    for i in range(N):
        gt_i = gts[i]
        pr_i = preds[i]
        rmse_i = torch.sqrt(torch.mean((pr_i - gt_i) ** 2)).item()

        ax = axes[i]
        ax.plot(dims, gt_i.numpy(), "o-", label="Ground Truth")
        ax.plot(dims, pr_i.numpy(), "x--", label="Prediction")
        ax.set_ylabel("Value")
        ax.set_title(f"Sample {i + 1} | RMSE = {rmse_i:.3f}")
        ax.grid(True)
        ax.legend()

    axes[-1].set_xlabel("Acceleration Dimension")
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
    Visualize the overall fitting accuracy of the network.

    This function aggregates predictions and ground truth from the provided dataloader
    (up to max_samples samples) and creates a two-part figure:
      1. Scatter plots comparing predicted and true values for each output dimension.
         A red dashed 45° line indicates perfect prediction.
      2. A histogram showing the distribution of per-sample RMSE values.

    The figure titles and axis labels are in English.

    Parameters:
      - model: Trained network.
      - dataloader: DataLoader for evaluation.
      - device: torch.device for computation.
      - save_path: If provided, the figure is saved to this path. If, for example, save_path is
                   "global_fitting_accuracy.png", a single file with both parts is generated.
      - max_samples: Maximum number of samples to aggregate.
    """
    # Aggregate data from the dataloader.
    preds, gts = aggregate_predictions(model, dataloader, device, max_samples)
    N, D = preds.size()

    # Compute overall RMSE and per-sample RMSE values.
    overall_rmse = torch.sqrt(torch.mean((preds - gts) ** 2)).item()
    per_sample_rmse = torch.sqrt(torch.mean((preds - gts) ** 2, dim=1)).numpy()

    # Create a figure with two rows: top for scatter plots, bottom for error distribution.
    if D == 1:
        # For 1-dim, use a single scatter plot.
        fig, (ax_scatter, ax_hist) = plt.subplots(2, 1, figsize=(8, 10))
        # Scatter plot:
        ax_scatter.scatter(gts.numpy(), preds.numpy(), alpha=0.5)
        min_val = float(gts.min().item())
        max_val = float(gts.max().item())
        ax_scatter.plot([min_val, max_val], [min_val, max_val], "r--", lw=2)
        ax_scatter.set_title(f"Global Fitting Accuracy (Overall RMSE: {overall_rmse:.3f})")
        ax_scatter.set_xlabel("Ground Truth")
        ax_scatter.set_ylabel("Prediction")
        ax_scatter.grid(True)
    else:
        # For multi-dim outputs, create subplots for each dimension in the first row.
        ncols = D
        fig, (axes_scatter, ax_hist) = plt.subplots(2, 1, figsize=(6 * ncols, 10))
        # Create scatter subplots for each dimension.
        fig_scatter, axes = plt.subplots(1, D, figsize=(6 * D, 6))
        axes = np.atleast_1d(axes)
        for i in range(D):
            pred_i = preds[:, i].numpy()
            gt_i = gts[:, i].numpy()
            axes[i].scatter(gt_i, pred_i, alpha=0.5)
            min_val = float(gt_i.min())
            max_val = float(gt_i.max())
            axes[i].plot([min_val, max_val], [min_val, max_val], "r--", lw=2)
            rmse_i = torch.sqrt(torch.mean((preds[:, i] - gts[:, i]) ** 2)).item()
            axes[i].set_title(f"Dimension {i + 1} (RMSE: {rmse_i:.3f})")
            axes[i].set_xlabel("Ground Truth")
            axes[i].set_ylabel("Prediction")
            axes[i].grid(True)
        # Combine scatter plots into one image.
        plt.tight_layout()
        # Save the scatter plot as a temporary buffer and then add it as top part.
        from io import BytesIO
        buf = BytesIO()
        fig_scatter.savefig(buf, dpi=300, bbox_inches='tight')
        plt.close(fig_scatter)
        buf.seek(0)
        # Read the scatter plot image and display it as the first row.
        import matplotlib.image as mpimg
        scatter_img = mpimg.imread(buf)
        ax_scatter = fig.add_subplot(211)
        ax_scatter.imshow(scatter_img)
        ax_scatter.axis("off")
    # Histogram plot for per-sample RMSE.
    ax_hist.hist(per_sample_rmse, bins=30, alpha=0.75, edgecolor="black")
    ax_hist.set_xlabel("RMSE")
    ax_hist.set_ylabel("Frequency")
    ax_hist.set_title(f"RMSE Distribution (Overall RMSE: {overall_rmse:.3f})")
    ax_hist.grid(True)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.show()
    plt.close()
