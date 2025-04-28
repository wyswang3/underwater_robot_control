# utils/random_segment_fit_utils.py

import os
import random
from typing import Tuple, List, Optional, Dict, Any

import numpy as np
import torch
import matplotlib.pyplot as plt


# ╭─────────────────────────────────────────╮
# │   1. Random Segment & Data Preparation  │
# ╰─────────────────────────────────────────╯
def select_random_segment(dataset: List[Dict[str, Any]], num_samples: int) -> Tuple[List[Dict[str, Any]], int]:
    """
    Randomly select a continuous segment of num_samples samples from the dataset.

    Args:
        dataset: List of sample dictionaries.
        num_samples: Number of continuous samples to select.

    Raises:
        ValueError: If the dataset does not contain enough samples.

    Returns:
        Tuple: (List of selected samples, starting index)
    """
    if len(dataset) < num_samples:
        raise ValueError("Not enough samples in the dataset.")
    start = random.randint(0, len(dataset) - num_samples)
    # Use list comprehension instead of slicing to avoid feature size mismatch errors
    segment = [dataset[i] for i in range(start, start + num_samples)]
    return segment, start


def stack_to_device(samples: List[Dict[str, Any]], device: torch.device) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert a list of dictionaries to batched tensors and move them to the specified device.

    Args:
        samples: Each dictionary must contain "power_window", "imu_window", and "accel" keys.
        device: The target device (e.g., torch.device("cuda") or torch.device("cpu")).

    Returns:
        Tuple: (power_window tensor, imu_window tensor, accel tensor)
    """
    keys = ["power_window", "imu_window", "accel"]
    out = {k: torch.stack([sample[k] for sample in samples]).to(device) for k in keys}
    return out["power_window"], out["imu_window"], out["accel"]


# ╭─────────────────────────────────────────╮
# │           2. Plotting Tools             │
# ╰─────────────────────────────────────────╯
def plot_segment(time_axis: np.ndarray,
                 gt: np.ndarray,
                 pred: np.ndarray,
                 save_path: Optional[str] = None) -> None:
    """
    Plot comparison for each acceleration axis:
      - Left: Measured vs. Predicted
      - Right: Residual (Pred - GT)
    All labels (xlabel/ylabel) fully aligned across subplots.
    """

    # Global style
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'font.size': 11
    })

    n_axes = gt.shape[1]
    axis_names = (
        ["Linear Accel X", "Linear Accel Y", "Linear Accel Z",
         "Angular Accel X", "Angular Accel Y", "Angular Accel Z"]
        if n_axes == 6 else
        [f"Axis {i + 1}" for i in range(n_axes)]
    )

    fig_height = 8 * n_axes / 6
    fig, axes = plt.subplots(
        n_axes, 2,
        figsize=(8, fig_height),
        sharex='col'
    )
    if n_axes == 1:
        axes = axes.reshape(1, 2)

    # Manually control margins to ensure alignment
    fig.subplots_adjust(left=0.12, right=0.95, hspace=0.4, wspace=0.3)

    for i in range(n_axes):
        # Left column
        ax_l = axes[i, 0]
        ax_l.plot(time_axis, gt[:, i], 'o-', color='red',
                  linewidth=1.5, markersize=3, markerfacecolor='none',
                  label='Measured')
        ax_l.plot(time_axis, pred[:, i], 'o-', color='blue',
                  linewidth=1.5, markersize=3, markerfacecolor='none',
                  label='Predicted')
        ax_l.set_ylabel(axis_names[i], fontsize=12)
        ax_l.tick_params(axis='both', labelsize=11)
        ax_l.grid(False)
        if i == 0:
            ax_l.legend(frameon=False, fontsize=10, loc='upper right')
        if i == n_axes - 1:
            ax_l.set_xlabel("Time (s)", fontsize=12)
        else:
            ax_l.tick_params(labelbottom=False)

        # Right column
        ax_r = axes[i, 1]
        resid = pred[:, i] - gt[:, i]
        ax_r.plot(time_axis, resid, 'o-', color='purple',
                  linewidth=1.5, markersize=4, markerfacecolor='none',
                  label='Residual')
        ax_r.axhline(0, color='red', linestyle='--', linewidth=1.2)
        ax_r.set_ylabel("Residual", fontsize=12)
        ax_r.tick_params(axis='both', labelsize=11)
        ax_r.grid(False)
        if i == 0:
            ax_r.legend(frameon=False, fontsize=10, loc='upper right')
        if i == n_axes - 1:
            ax_r.set_xlabel("Time (s)", fontsize=12)
        else:
            ax_r.tick_params(labelbottom=False)

    # Align left column y-labels
    fig.align_ylabels(axes[:, 0])

    # Align right column y-labels
    fig.align_ylabels(axes[:, 1])

    # Final layout tightening
    fig.tight_layout(pad=0.8)

    # Save if needed
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()
    plt.close(fig)

# ╭─────────────────────────────────────────╮
# │       3. Main Function: Random Segment Fit        │
# ╰─────────────────────────────────────────╯
@torch.no_grad()
def run_random_segment_fit(
        cfg: Any,
        model: torch.nn.Module,
        dataset: List[Dict[str, Any]],
        device: torch.device,
        dt: float = 0.2,
        segment_duration: float = 20.0,
        save_path: Optional[str] = None,
) -> None:
    """
    Randomly select a continuous data segment, run prediction, plot the results,
    and print error metrics. The figure layout shows for each acceleration axis:
      - Left column: Time series comparison of measured vs. predicted values.
      - Right column: Residual (predicted - measured) over time, with RMSE annotated in the title.

    If the data contains 6 channels, the first three are assumed to be linear acceleration (X/Y/Z)
    and the last three are angular acceleration (X/Y/Z).

    Args:
        cfg: Configuration object with attributes training.WINDOW_SIZE and paths.SPLITS_DIR.
        model: Prediction model.
        dataset: List of sample dictionaries.
        device: Target device.
        dt: Time interval per sample (seconds).
        segment_duration: Duration of the segment to select (seconds).
        save_path: Path to save the figure. If None, defaults to cfg.paths.SPLITS_DIR.
    """
    sample_time = cfg.training.WINDOW_SIZE * dt
    n_samples = max(1, int(segment_duration / sample_time))

    print(
        f"Each sample covers {sample_time:.2f}s; selecting {n_samples} samples (~{n_samples * sample_time:.1f}s total).")
    seg_samples, start = select_random_segment(dataset, n_samples)
    print(f"Segment starting index: {start}")

    pw, imu, accel_gt = stack_to_device(seg_samples, device)

    # Run model prediction
    model.eval()
    out = model(pw, imu)
    accel_pred = out["accel_pred"] if isinstance(out, dict) else out

    # Transfer to CPU and convert to numpy arrays
    accel_pred = accel_pred.cpu().numpy()
    accel_gt = accel_gt.cpu().numpy()

    # Calculate error metrics and print summary
    rmse_dim = np.sqrt(np.mean((accel_pred - accel_gt) ** 2, axis=0))
    mae_dim = np.mean(np.abs(accel_pred - accel_gt), axis=0)
    overall_rmse = np.sqrt(np.mean((accel_pred - accel_gt) ** 2))
    print("RMSE per dimension:", np.round(rmse_dim, 4))
    print("MAE per dimension:", np.round(mae_dim, 4))
    print("Overall RMSE:", overall_rmse)

    # Generate time axis for plotting
    t = np.arange(n_samples) * sample_time
    if save_path is None:
        save_path = os.path.join(cfg.paths.SPLITS_DIR, "random_segment_comparison.png")
    plot_segment(t, accel_gt, accel_pred, save_path)
