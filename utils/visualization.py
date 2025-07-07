#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
utils/visualization.py

Visualization utilities with bilingual support (English/中文双语).
- USE_CHINESE 开关控制中文或英文标签。
- 字体回退机制：英文模式下缺失中文字形会自动回退到 SimHei。
"""
import os
from typing import Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

# —— 中英文模式开关 ——#
USE_CHINESE = False  # True: 中文模式；False: English mode

# —— 字体配置（含回退）——#
if USE_CHINESE:
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
else:
    matplotlib.rcParams['font.family'] = 'serif'
    # 英文模式：Times New Roman，缺失中文回退到 SimHei
    matplotlib.rcParams['font.serif'] = ['Times New Roman', 'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False  # 负号正常显示


def batch_to_device(batch, device):
    """
    Recursively move tensors in batch (dict/list/tuple) to device.
    递归将 batch 中的张量移至指定设备。
    """
    if isinstance(batch, dict):
        return {k: batch_to_device(v, device) for k, v in batch.items()}
    elif isinstance(batch, (list, tuple)):
        return type(batch)(batch_to_device(v, device) for v in batch)
    elif torch.is_tensor(batch):
        return batch.to(device, non_blocking=True)
    return batch


def plot_loss_curve(
        train_losses,
        val_losses=None,
        save_path: Optional[str] = None,
) -> None:
    """
    Plot training and validation loss curves.
    绘制训练与验证损失曲线。
    """
    epochs = np.arange(1, len(train_losses) + 1)
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_losses,
             label=("训练损失" if USE_CHINESE else "Training Loss"), marker="o")
    if val_losses is not None:
        plt.plot(epochs, val_losses,
                 label=("验证损失" if USE_CHINESE else "Validation Loss"), marker="o")
    plt.xlabel(("轮次" if USE_CHINESE else "Epoch"))
    plt.ylabel(("损失" if USE_CHINESE else "Loss"))
    plt.title(("训练与验证损失曲线" if USE_CHINESE else "Training & Validation Loss"))
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
    Plot model predictions vs targets:
    - 3D outputs: time-series comparison
    - 2D outputs: per-sample bar charts
    """
    model.eval()
    model.to(device)
    all_preds, all_tgts = [], []
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            if idx >= num_batches:
                break
            batch = batch_to_device(batch, device)
            out = model(batch['power_window'], batch.get('imu_window'))
            preds = out.get('accel_pred', out) if isinstance(out, dict) else out
            tgts = batch['accel']
            all_preds.append(preds.cpu())
            all_tgts.append(tgts.cpu())
    preds = torch.cat(all_preds, dim=0)
    tgts = torch.cat(all_tgts, dim=0)

    rmse_dim = torch.sqrt(torch.mean((preds - tgts)**2, dim=0))
    print(("各维度RMSE:" if USE_CHINESE else "RMSE per dim:"), np.round(rmse_dim.numpy(), 4))

    if preds.dim() == 3:
        # Time-series plots
        samples, _, T, D = min(num_samples, preds.size(0)), *preds.size()
        fig, axes = plt.subplots(samples, D, figsize=(2*D, 1.5*samples), sharex=True)
        axes = np.atleast_2d(axes)
        for i in range(samples):
            for j in range(D):
                ax = axes[i][j]
                ax.plot(preds[i, :, j].numpy(), '-', lw=1,
                        label=("预测" if USE_CHINESE and j==0 else None))
                ax.plot(tgts[i, :, j].numpy(), '--', lw=1,
                        label=("真实" if USE_CHINESE and j==0 else None))
                if i==0:
                    ax.set_title((f"维度{j+1}" if USE_CHINESE else f"Dim {j+1}"))
                if j==0:
                    ax.set_ylabel((f"样本{i+1}" if USE_CHINESE else f"Sample {i+1}"))
                ax.grid(True, alpha=0.3)
        if USE_CHINESE:
            fig.legend(["真实","预测"], loc='upper right')
        plt.tight_layout()
    elif preds.dim() == 2:
        # Bar charts
        samples = min(num_samples, preds.size(0))
        D = preds.size(1)
        dims = np.arange(1, D+1)
        fig, axes = plt.subplots(samples, 1, figsize=(5,1.5*samples), sharex=False)
        axes = np.atleast_1d(axes)
        for i in range(samples):
            ax = axes[i]
            ax.bar(dims-0.2, preds[i].numpy(), width=0.4,
                   label=("预测" if USE_CHINESE else 'Pred'))
            ax.bar(dims+0.2, tgts[i].numpy(), width=0.4,
                   label=("真实" if USE_CHINESE else 'True'))
            ax.set_xticks(dims)
            ax.set_xlabel(("维度" if USE_CHINESE else 'Dimension'))
            ax.set_ylabel(("数值" if USE_CHINESE else 'Value'))
            ax.set_title((f"样本{i+1}" if USE_CHINESE else f"Sample {i+1}"))
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
    """
    Scatter plot vs true and RMSE histogram.
    绘制散点图与 RMSE 分布直方图。
    """
    from utils.visualization import aggregate_predictions
    preds, gts = aggregate_predictions(model, dataloader, device, max_samples)
    N, D = preds.size()
    overall_rmse = torch.sqrt(torch.mean((preds-gts)**2)).item()
    fig, axes = plt.subplots(1, D, figsize=(5*D,4))
    axes = np.atleast_1d(axes)
    for i, ax in enumerate(axes):
        p = preds[:,i].numpy(); t = gts[:,i].numpy()
        ax.scatter(t, p, alpha=0.5, s=5)
        mn, mx = min(t.min(),p.min()), max(t.max(),p.max())
        ax.plot([mn,mx],[mn,mx],'r--')
        ax.set_title((f"维度{i+1} RMSE={torch.sqrt(torch.mean((preds[:,i]-gts[:,i])**2)).item():.3f}" if USE_CHINESE else f"Dim{i+1} RMSE={torch.sqrt(torch.mean((preds[:,i]-gts[:,i])**2)).item():.3f}"))
        ax.set_xlabel(("真实" if USE_CHINESE else 'True'))
        ax.set_ylabel(("预测" if USE_CHINESE else 'Pred'))
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
    """
    Aggregate up to max_samples preds & gts.
    聚合最多 max_samples 个样本的预测和真实值。
    """
    model.eval()
    all_preds, all_gts = [], []
    with torch.no_grad():
        cnt=0
        for batch in dataloader:
            batch = batch_to_device(batch, device)
            out = model(batch['power_window'], batch.get('imu_window'))
            preds = out.get('accel_pred', out) if isinstance(out, dict) else out
            gts   = batch['accel']
            all_preds.append(preds.cpu()); all_gts.append(gts.cpu())
            cnt += preds.size(0)
            if cnt>=max_samples: break
    return torch.cat(all_preds,0), torch.cat(all_gts,0)


def visualize_error_histograms(
    preds: np.ndarray,
    tgts: np.ndarray,
    save_prefix: str,
    linear_bins: int = 20,
    angular_bins: int = 40,
    tick_num: int = 5
) -> None:
    """
    Plot histograms for linear & angular error.
    线性和角度误差直方图。
    """
    errors = preds - tgts
    def _plot_group(err, names, bins, suffix):
        bound = float(np.max(np.abs(err))) or 1.0
        edges = np.linspace(-bound,bound,bins+1); ticks=np.linspace(-bound,bound,tick_num)
        fig, axes = plt.subplots(1,len(names),figsize=(4*len(names),4))
        axes = np.atleast_1d(axes)
        for i, ax in enumerate(axes):
            ax.hist(err[:,i], bins=edges, edgecolor='black', alpha=0.7)
            ax.set_title((f"{names[i]} 误差" if USE_CHINESE else f"{names[i]} Error"))
            ax.set_xlim(-bound,bound); ax.set_xticks(ticks)
            ax.set_xlabel(("误差" if USE_CHINESE else 'Error'))
            ax.set_ylabel(("频率" if USE_CHINESE else 'Freq'))
            ax.grid(True,alpha=0.3)
        plt.tight_layout()
        path=f"{save_prefix}_{suffix}.png"
        os.makedirs(os.path.dirname(path),exist_ok=True); plt.savefig(path,dpi=300)
        plt.close()
    linear_names = (["线加速度X","线加速度Y","线加速度Z"] if USE_CHINESE else ["Lin Acc X","Lin Acc Y","Lin Acc Z"])
    angular_names= (["角加速度X","角加速度Y","角加速度Z"] if USE_CHINESE else ["Ang Acc X","Ang Acc Y","Ang Acc Z"])
    _plot_group(errors[:,:3], linear_names, linear_bins, 'linear')
    _plot_group(errors[:,3:], angular_names, angular_bins, 'angular')


def visualize_time_series(
    preds: np.ndarray,
    tgts: np.ndarray,
    time_step: float,
    save_path: str
) -> None:
    """
    Plot time-series of errors for each axis.
    绘制每个轴的误差时间序列。
    """
    errors = preds - tgts
    axis_names = (["线加速度X","线加速度Y","线加速度Z","角加速度X","角加速度Y","角加速度Z"]
                  if USE_CHINESE else ["LinAcc X","LinAcc Y","LinAcc Z","AngAcc X","AngAcc Y","AngAcc Z"])
    times = np.arange(len(errors))*time_step
    fig, axes = plt.subplots(3,2,figsize=(12,8),sharex=True)
    axes=axes.flatten()
    for i, ax in enumerate(axes):
        ax.plot(times, errors[:,i], lw=0.8)
        ax.axhline(0, ls=':', lw=0.8)
        ax.set_title(axis_names[i])
        ax.set_ylabel(("误差" if USE_CHINESE else 'Error'))
        ax.grid(True,alpha=0.3)
        if i>=4:
            ax.set_xlabel(("时间 (s)" if USE_CHINESE else 'Time (s)'))
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path),exist_ok=True); plt.savefig(save_path,dpi=300,bbox_inches='tight')
    plt.close()
