#!/usr/bin/env python
# models/dynamics_net.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.torch_helper import eye_like, zeros_like_shape

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
class Swish(nn.SiLU):
    pass

# Improved safe log-determinant

def safe_logdet(x: torch.Tensor, eps: float = 1e-6, min_val: float = -10.0) -> torch.Tensor:
    x_sym = (x + x.transpose(-1, -2)) * 0.5
    diag = torch.eye(x_sym.size(-1), device=x_sym.device, dtype=x_sym.dtype) * eps
    x_stable = x_sym + diag
    sign, logabs = torch.slogdet(x_stable)
    logabs = torch.clamp(logabs, min=min_val)
    return sign * logabs

# -----------------------------------------------------------------------------
# Thrust mapping: small MLP for all motors
# -----------------------------------------------------------------------------
class ThrustMLP(nn.Module):
    """Maps an 8-D power vector to an 8-D thrust vector."""
    def __init__(self, hidden_dim: int = 32, motors: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(motors, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, motors)
        )

    def forward(self, power: torch.Tensor) -> torch.Tensor:
        # power: (B, 8)
        return self.net(power)  # (B, 8)

# -----------------------------------------------------------------------------
# Encoders
# -----------------------------------------------------------------------------
class LSTMEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int, layers: int = 2,
                 dropout: float = 0.2, bidir: bool = False):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, layers,
                            batch_first=True, dropout=dropout,
                            bidirectional=bidir)
        self.dropout = nn.Dropout(dropout)
        self.out_dim = hidden * (2 if bidir else 1)
        self.in_dim = in_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.size(-1) == self.in_dim
        out, _ = self.lstm(x)
        return self.dropout(out[:, -1, :])

# -----------------------------------------------------------------------------
# Direct mapping fusion
# -----------------------------------------------------------------------------
class DirectMappingNet(nn.Module):
    def __init__(self, p_hidden: int = 128, i_hidden: int = 128,
                 fusion_hidden: int = 128):
        super().__init__()
        self.p_enc = LSTMEncoder(8, p_hidden)
        self.i_enc = LSTMEncoder(6, i_hidden)
        self.fc = nn.Sequential(
            nn.Linear(self.p_enc.out_dim + self.i_enc.out_dim, fusion_hidden),
            nn.ReLU(),
            nn.Linear(fusion_hidden, 6)
        )

    def forward(self, pw: torch.Tensor, imu: torch.Tensor) -> torch.Tensor:
        feat = torch.cat([self.p_enc(pw), self.i_enc(imu)], dim=-1)
        return self.fc(feat)

# -----------------------------------------------------------------------------
# Enhanced physics with residual
# -----------------------------------------------------------------------------
class EnhancedPhysicsNet(nn.Module):
    def __init__(self, T: torch.Tensor, enc_hidden: int = 128,
                 layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.register_buffer("T", T)  # (6,8)
        self.enc = LSTMEncoder(14, enc_hidden, layers, dropout)
        self.thrust_net = ThrustMLP(hidden_dim=32, motors=T.size(1))
        self.Mnet = nn.Linear(self.enc.out_dim, 6)
        self.Jnet = nn.Linear(self.enc.out_dim, 6)
        self.Dnet = nn.Linear(self.enc.out_dim, 6)
        self.res_fc = nn.Sequential(
            nn.Linear(self.enc.out_dim, 64), nn.ReLU(), nn.Linear(64, 6)
        )

    def forward(self, pw: torch.Tensor, imu: torch.Tensor) -> dict:
        # Concatenate inputs and encode
        x = torch.cat([pw, imu], dim=-1)
        enc = self.enc(x)  # (B, enc_dim)

        # Thrust estimation at last timestep
        power_last = pw[:, -1, :]            # (B,8)
        thrusts = self.thrust_net(power_last) # (B,8)
        tau = thrusts @ self.T.t()            # (B,6)

        # Build M, J, D
        M_diag = F.softplus(self.Mnet(enc)) + 1e-3
        J_diag = F.softplus(self.Jnet(enc)) + 1e-3
        D_diag = F.softplus(self.Dnet(enc)) + 1e-3
        M = torch.diag_embed(M_diag[:, :3])
        J = torch.diag_embed(J_diag[:, :3])
        D = torch.diag_embed(D_diag)

        # Dynamics solve
        v_lin = imu[:, -1, :3]
        omega = imu[:, -1, 3:]
        v6 = torch.cat([v_lin, omega], dim=-1).unsqueeze(-1)

        B = pw.size(0)
        In = torch.zeros(B, 6, 6, device=enc.device, dtype=enc.dtype)
        In[:, :3, :3] = M
        In[:, 3:, 3:] = J
        In += 1e-4 * eye_like(enc, 6)

        C = torch.zeros_like(In)
        def skew(v):
            return torch.tensor([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]],
                                device=v.device, dtype=v.dtype)
        for i in range(B):
            C[i, :3, 3:] = -skew(v_lin[i])
            C[i, 3:, 3:] = -skew(omega[i])

        rhs = tau.unsqueeze(-1) - (C + D) @ v6
        acc_phy = torch.linalg.solve(In, rhs).squeeze(-1)

        # Residual correction
        acc_res = self.res_fc(enc)
        acc = acc_phy + acc_res

        return {"accel_pred": acc, "M": M, "J": J, "encoder": enc}

# -----------------------------------------------------------------------------
# Hybrid with dynamic gate
# -----------------------------------------------------------------------------
class HybridDynamicsModel(nn.Module):
    def __init__(self, direct_net: nn.Module, phys_net: nn.Module):
        super().__init__()
        self.direct = direct_net
        self.phys = phys_net
        gate_in_dim = phys_net.enc.out_dim + 6
        self.gate = nn.Sequential(
            nn.Linear(gate_in_dim, 32), Swish(),
            nn.Linear(32, 1), nn.Sigmoid()
        )

    def forward(self, pw: torch.Tensor, imu: torch.Tensor) -> dict:
        # Physics branch output dict
        phys_out = self.phys(pw, imu)
        acc_phy = phys_out["accel_pred"]           # (B,6)
        enc = phys_out["encoder"]                  # (B, enc_dim)
        # Direct branch prediction
        pred_direct = self.direct(pw, imu)          # (B,6)

        # Compute gating
        gate_feat = torch.cat([enc, acc_phy], dim=-1)
        alpha = self.gate(gate_feat)                # (B,1)

        # Fused prediction
        acc_fused = alpha * acc_phy + (1 - alpha) * pred_direct
        # Override physics output's accel_pred with fused result
        phys_out["accel_pred"] = acc_fused
        return phys_out

# -----------------------------------------------------------------------------
# Loss with smooth log-det regularization
# -----------------------------------------------------------------------------
class EnhancedDynamicsLoss(nn.Module):
    def __init__(self, beta: float = 0.1, linear_w: float = 0.7, use_reg: bool = True):
        super().__init__()
        self.beta = beta
        self.lw = linear_w
        self.use_reg = use_reg

    def forward(self, out: dict, tgt: torch.Tensor) -> torch.Tensor:
        acc_pred = out["accel_pred"]                   # (B,6)
        lin_loss = F.mse_loss(acc_pred[:, :3], tgt[:, :3])
        ang_loss = F.mse_loss(acc_pred[:, 3:], tgt[:, 3:])
        total = self.lw * lin_loss + (1 - self.lw) * ang_loss
        if self.use_reg:
            reg_M = torch.mean(-safe_logdet(out["M"]))
            reg_J = torch.mean(-safe_logdet(out["J"]))
            total = total + self.beta * (reg_M + reg_J)
        return total
