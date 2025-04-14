#!/usr/bin/env python
#models/dynamics_net.py
import logging
from typing import Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.torch_helper import eye_like, zeros_like_shape

# -----------------------------------------------------------------------------
# logging
# -----------------------------------------------------------------------------
from utils.log_helper import setup_logging
setup_logging()
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# helpers (dtype / device safe)
# -----------------------------------------------------------------------------

class Swish(nn.SiLU):
    """Alias kept for backward‑compatibility."""
    pass


def skew_symmetric(v: torch.Tensor) -> torch.Tensor:
    """Batch‑wise 3×3 skew‑symmetric matrices from (B,3) vectors."""
    B, dtype, device = v.size(0), v.dtype, v.device
    vx, vy, vz = v[:, 0], v[:, 1], v[:, 2]
    z = torch.zeros_like(vx)
    return torch.stack(
        [z, -vz, vy,
         vz, z, -vx,
         -vy, vx, z], 1).view(B, 3, 3).to(dtype)


# -----------------------------------------------------------------------------
# safety utils
# -----------------------------------------------------------------------------

def safe_slice(t: torch.Tensor, *slices, name: str = "tensor") -> torch.Tensor:
    out = t.__getitem__(slices)
    if out.dim() == 0:
        logger.error(f"[SAFE] {name} became 0‑d, auto‑unsqueeze → shape (1,)")
        out = out.unsqueeze(0)
    return out


def assert_tensor2d(t: torch.Tensor, where: str = "") -> None:
    assert t.dim() == 2, f"[ASSERT] expect 2‑D tensor in {where}, got {t.shape}"


# -----------------------------------------------------------------------------
# matrix nets
# -----------------------------------------------------------------------------
class SPDMatrixNet(nn.Module):
    """Outputs a symmetric positive‑definite matrix via *LL^T*."""

    def __init__(self, in_dim: int, n: int, diag_eps: float = 1e-4):
        super().__init__()
        self.n, self.eps = n, diag_eps
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 256), Swish(), nn.LayerNorm(256),
            nn.Linear(256, n * (n + 1) // 2), nn.Tanh(),
        )
        for m in self.fc:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

        idx = torch.tril_indices(n, n)
        self.register_buffer("tri_idx", idx, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, dtype, device = x.size(0), x.dtype, x.device
        tril = self.fc(x).to(dtype)
        L = torch.zeros(B, self.n, self.n, dtype=dtype, device=device)
        L[:, self.tri_idx[0], self.tri_idx[1]] = tril

        diag = F.softplus(torch.diagonal(L, dim1=1, dim2=2)) + self.eps
        main_diag = torch.diagonal(L, dim1=1, dim2=2)  # offset=0
        L = L - torch.diag_embed(main_diag) + torch.diag_embed(diag)
        return L @ L.transpose(1, 2)


class DiagonalMatrixNet(nn.Module):
    def __init__(self, in_dim: int, n: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, n)
        nn.init.kaiming_normal_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.diag_embed(torch.exp(self.fc(x)))


# -----------------------------------------------------------------------------
# encoders
# -----------------------------------------------------------------------------
class LSTMEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int, layers: int = 2,
                 dropout: float = 0.2, bidir: bool = False, debug: bool = False):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, layers, batch_first=True,
                            dropout=dropout, bidirectional=bidir)
        self.dropout = nn.Dropout(dropout)
        self.out_dim = hidden * (2 if bidir else 1)
        self.in_dim = in_dim
        self.debug = debug

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.size(-1) == self.in_dim, "LSTMEncoder input dim mismatch"
        if self.debug:
            logger.debug(f"[LSTMEncoder] in {x.shape}")
        out, _ = self.lstm(x)
        final = self.dropout(out[:, -1, :])
        if self.debug:
            logger.debug(f"[LSTMEncoder] out {final.shape}")
        return final


class ThrustLSTM(nn.Module):
    def __init__(self, hidden: int = 64, debug: bool = False):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, batch_first=True)
        self.fc = nn.Sequential(nn.ReLU(), nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))
        self.debug = debug

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.debug:
            logger.debug(f"[ThrustLSTM] in {x.shape}")
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


# -----------------------------------------------------------------------------
# direct mapping
# -----------------------------------------------------------------------------
class DirectMappingNet_LSTM_Fusion(nn.Module):
    def __init__(self, hidden: int = 256, layers: int = 2,
                 fusion_hidden: int = 256, debug: bool = False):
        super().__init__()
        self.p_enc = LSTMEncoder(8, hidden, layers, debug=debug)
        self.i_enc = LSTMEncoder(6, hidden, layers, debug=debug)
        self.fc = nn.Sequential(
            nn.Linear(hidden * 2, fusion_hidden), nn.ReLU(),
            nn.Linear(fusion_hidden, 6)
        )
        self.debug = debug

    def forward(self, pw: torch.Tensor, imu: torch.Tensor) -> torch.Tensor:
        feat = torch.cat([self.p_enc(pw), self.i_enc(imu)], -1)
        out = self.fc(feat)
        if self.debug:
            logger.debug(f"[DirectFusion] out {out.shape}")
        return out


# -----------------------------------------------------------------------------
# physics net
# -----------------------------------------------------------------------------
class EnhancedPhysicsNet_LSTM(nn.Module):
    def __init__(self, T: torch.Tensor, hidden: int = 512, layers: int = 2, debug: bool = False):
        super().__init__()
        self.register_buffer("T", T)          # (6,8)
        self.enc = LSTMEncoder(14, hidden, layers, debug=debug)
        self.thrust_nets = nn.ModuleList([ThrustLSTM(debug=debug) for _ in range(8)])
        self.Mnet = SPDMatrixNet(hidden, 3)
        self.Jnet = SPDMatrixNet(hidden, 3)
        self.Dnet = DiagonalMatrixNet(hidden, 6)
        self.reg = nn.Parameter(torch.tensor(1e-4))
        self.debug = debug

    def forward(self, pw: torch.Tensor, imu: torch.Tensor) -> dict[str, torch.Tensor]:
        x   = torch.cat([pw, imu], -1)
        enc = self.enc(x)

        thrusts = torch.stack([
            net(pw[..., i].unsqueeze(-1)) for i, net in enumerate(self.thrust_nets)
        ], 1)  # (B,8,1)
        tau = (self.T @ thrusts).squeeze(-1)  # (B,6)

        M = self.Mnet(enc) + self.reg * eye_like(enc, 3)
        J = self.Jnet(enc) + self.reg * eye_like(enc, 3)
        D = self.Dnet(enc)

        v_lin, omega = imu[:, -1, :3], imu[:, -1, 3:]
        v6 = torch.cat([v_lin, omega], 1)

        B = pw.size(0)
        In = zeros_like_shape((B, 6, 6), enc)
        In[:, :3, :3] = M
        In[:, 3:, 3:] = J
        In += 1e-6 * eye_like(enc, 6)

        C = torch.zeros_like(In)
        C[:, :3, 3:] = -skew_symmetric(v_lin)
        C[:, 3:, 3:] = -skew_symmetric(omega)

        rhs = tau.unsqueeze(-1) - (C + D) @ v6.unsqueeze(-1)
        acc = torch.linalg.solve(In, rhs).squeeze(-1)
        if self.debug:
            logger.debug(f"[Physics] acc {acc.shape}")
        return {"accel_pred": acc, "M": M, "J": J, "encoder_features": enc}


# -----------------------------------------------------------------------------
# hybrid
# -----------------------------------------------------------------------------
class HybridDynamicsModel(nn.Module):
    def __init__(self, pnet: nn.Module, enet: nn.Module, debug: bool = False):
        super().__init__()
        self.pnet, self.enet = pnet, enet
        self.gate = nn.Sequential(nn.Linear(70, 64), Swish(), nn.Linear(64, 1), nn.Sigmoid())
        self.debug = debug

    def forward(self, pw: torch.Tensor, imu: torch.Tensor) -> torch.Tensor:
        p_out = self.pnet(pw, imu)
        e_out = self.enet(pw, imu)
        feat = safe_slice(p_out["encoder_features"], ..., slice(None, 64), name="enc_feat")
        gate_in = torch.cat([feat, p_out["accel_pred"]], 1)
        assert_tensor2d(gate_in, "gate_in")
        g = self.gate(gate_in)
        if self.debug:
            logger.debug(f"[Hybrid] gate={g.mean().item():.3f}")
        return g * p_out["accel_pred"] + (1 - g) * e_out


# -----------------------------------------------------------------------------
# loss
# -----------------------------------------------------------------------------
class EnhancedDynamicsLoss(nn.Module):
    def __init__(self, beta: float = 0.1, linear_w: float = 0.7):
        super().__init__()
        self.beta, self.lw = beta, linear_w

    def forward(self, out: Any, tgt: dict[str, torch.Tensor]) -> torch.Tensor:
        if isinstance(out, dict):
            acc = out["accel_pred"]
            reg = -(torch.log(torch.det(out["M"]).clamp_min(1e-7)).mean() +
                    torch.log(torch.det(out["J"]).clamp_min(1e-7)).mean())
        else:
            acc, reg = out, 0.0

        lin = F.mse_loss(acc[:, :3], tgt["accel"][:, :3])
        ang = F.mse_loss(acc[:, 3:], tgt["accel"][:, 3:])
        total_loss=self.lw * lin + (1 - self.lw) * ang + self.beta * reg
        return total_loss

#############################################
# 说明
#############################################
"""
【整体架构说明】
1. 输入数据：
   - 功率数据: (B, window_size, 8)
   - IMU 数据:   (B, window_size, 6)
   这两部分数据在时间步上保持时序信息，并在特征维度拼接成 14 维输入。
2. DirectMappingNet_LSTM_Fusion:
   - 分别对功率与 IMU 数据使用 LSTMEncoder 进行编码，
   - 然后将编码的特征拼接经过全连接网络映射到 6 维输出。
3. EnhancedPhysicsNet_LSTM:
   - 拼接输入数据，经 LSTMEncoder 提取全局特征（并缓存 encoder_features），
   - 对 8 个电机分别使用 ThrustLSTM 预测推力，
   - 利用全局特征生成物理参数矩阵 M、J、D，并结合物理公式求解加速度预测。
4. HybridDynamicsModel:
   - 利用门控网络融合 EnhancedPhysicsNet_LSTM 与 DirectMappingNet_LSTM_Fusion 的预测，
   - 门控网络输入为物理网络缓存的 encoder_features（前64维）和其预测加速度，
   - 输出为最终预测加权融合结果。
5. EnhancedDynamicsLoss:
   - 分离计算线性加速度与角加速度误差，且对线性部分给予更高权重，
   - 同时结合推力损失与矩阵正则项组成总损失。

【调试与稳定性】：
   - 每个关键模块均加入断言检查与详细 DEBUG 日志，
   - 通过缓存 encoder 输出避免重复调用带来的异常，
   - 有助于定位“iteration over a 0-d tensor”错误。
"""