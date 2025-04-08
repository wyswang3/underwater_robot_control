import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import matplotlib
from torch.utils.data import DataLoader
from typing import Optional, Union, List

# Set English fonts for consistency
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['axes.unicode_minus'] = False


def plot_loss_curve(train_losses: Union[List[float], np.ndarray],
                    val_losses: Optional[Union[List[float], np.ndarray]] = None,
                    save_path: Optional[str] = None) -> None:
    """
    Plot the training (and optional validation) loss curve.

    Params:
      - train_losses: List or numpy array of training losses (one value per epoch)
      - val_losses: List or numpy array of validation losses (optional)
      - save_path: If provided, the figure will be saved to the given path.
    """
    epochs = np.arange(1, len(train_losses) + 1)
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_losses, label='Train Loss', marker='o')
    if val_losses is not None:
        plt.plot(epochs, val_losses, label='Validation Loss', marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train and Validation Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.show()
    plt.clf()


def visualize_predictions(model: torch.nn.Module,
                          dataloader: DataLoader,
                          device: torch.device,
                          num_samples: int = 6,
                          save_path: Optional[str] = None) -> None:
    """
    Visualize the model's predictions on a batch and compute RMSE for each sample.
    Each subplot represents one sample, plotting ground truth vs. predictions with RMSE in the title.

    Params:
      - model: Trained model. Expected inputs: (B, window_size*14) and (B, 6)
      - dataloader: DataLoader that provides one batch of data.
      - device: The torch device on which the model is located.
      - num_samples: Number of samples from the batch to visualize (default 6).
      - save_path: If provided, the figure will be saved to the given path.
    """
    model.eval()
    with torch.no_grad():
        try:
            batch = next(iter(dataloader))
        except StopIteration:
            print("Visualization failed: DataLoader has no data!")
            return

        # Move all tensors in the batch to the target device
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)

        # Concatenate 'power_window' and 'imu_window'
        # Expected shapes: (B, W, 8) and (B, W, 6) -> (B, W, 14)
        features = torch.cat([batch['power_window'], batch['imu_window']], dim=2)
        features = features.view(features.size(0), -1)  # (B, W*14)
        thrust = batch['thrust']  # (B, 6)

        # Forward pass: 此处假设模型返回 (pred_lin, pred_ang, ...)，我们只关注前两个输出
        outputs = model(features, thrust)
        if isinstance(outputs, tuple) and len(outputs) >= 2:
            pred_lin, pred_ang = outputs[:2]
        else:
            print("Visualization failed: Unexpected model output format.")
            return

        preds = torch.cat([pred_lin, pred_ang], dim=1)  # (B, 6)
        predictions = preds.cpu().numpy()
        targets = batch['accel'].cpu().numpy()  # (B, 6)

    if predictions.shape[1] != targets.shape[1]:
        print(
            f"Visualization failed: Prediction dimension ({predictions.shape[1]}) != Target dimension ({targets.shape[1]})")
        return

    # Only visualize the first num_samples samples
    predictions = predictions[:num_samples]
    targets = targets[:num_samples]

    fig, axes = plt.subplots(nrows=num_samples, ncols=1, figsize=(12, 4 * num_samples))
    if num_samples == 1:
        axes = [axes]
    x_ticks = np.arange(predictions.shape[1])
    for i, ax in enumerate(axes):
        sample_pred = predictions[i]
        sample_target = targets[i]
        rmse = np.sqrt(np.mean((sample_pred - sample_target) ** 2))
        ax.plot(x_ticks, sample_target, label='Ground Truth', marker='o')
        ax.plot(x_ticks, sample_pred, label='Prediction', marker='x')
        ax.set_title(f"Sample {i + 1}: Prediction vs Ground Truth (RMSE: {rmse:.3f})")
        ax.set_xlabel("Dimension")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True)
    plt.tight_layout()
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.show()
    plt.clf()
