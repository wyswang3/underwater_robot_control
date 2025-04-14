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
def plot_segment(time_axis: np.ndarray, gt: np.ndarray, pred: np.ndarray, save_path: Optional[str] = None) -> None:
    """
    Plot time series comparison for each acceleration axis:
      - Left: Measured (GT) vs. Predicted (Pred) values.
      - Right: Residuals (Predicted - Measured) over time.

    If data has 6 channels, the first three are assumed to be linear acceleration (X/Y/Z) and
    the last three are angular acceleration (X/Y/Z), with corresponding titles.

    Args:
        time_axis: 1D numpy array of time points (shape: [n_samples]).
        gt: Ground truth data, a 2D numpy array (shape: [n_samples, n_axes]).
        pred: Predicted data, a 2D numpy array (shape: [n_samples, n_axes]).
        save_path: Path to save the figure. If None, the figure is shown only.
    """
    n_axes = gt.shape[1]
    if n_axes == 6:
        axis_names = [
            "Linear Accel X", "Linear Accel Y", "Linear Accel Z",
            "Angular Accel X", "Angular Accel Y", "Angular Accel Z"
        ]
    else:
        axis_names = [f"Axis{i}" for i in range(n_axes)]

    # Create a subplot grid with n_axes rows and 2 columns
    fig, axes = plt.subplots(n_axes, 2, figsize=(12, 3 * n_axes), sharex="col")

    # In case only one axis exists, ensure axes is 2D
    if n_axes == 1:
        axes = axes.reshape(1, 2)

    # Loop over each axis/dimension
    for i in range(n_axes):
        # Left subplot: Measured vs. Predicted
        ax_left = axes[i, 0]
        ax_left.plot(time_axis, gt[:, i], "o-", label="Measured", color="tab:blue")
        ax_left.plot(time_axis, pred[:, i], "s--", label="Predicted", color="tab:orange")
        ax_left.set_title(f"{axis_names[i]}: Measured vs Predicted")
        ax_left.grid(True)
        ax_left.legend()

        # Right subplot: Residual (Predicted - Measured)
        ax_right = axes[i, 1]
        residual = pred[:, i] - gt[:, i]
        ax_right.plot(time_axis, residual, "o-", label="Residual", color="purple")
        # Horizontal reference line at 0
        ax_right.axhline(0, color="red", linestyle="--", linewidth=1)
        rmse_i = np.sqrt(np.mean(residual ** 2))
        ax_right.set_title(f"{axis_names[i]} Residual (RMSE = {rmse_i:.3f})")
        ax_right.grid(True)
        ax_right.legend()

    # Set x-axis label for bottom subplots
    for col in range(2):
        axes[-1, col].set_xlabel("Time (s)")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.show()
    plt.close()


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

    print(f"Each sample covers {sample_time:.2f}s; selecting {n_samples} samples (~{n_samples * sample_time:.1f}s total).")
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
