import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.log_helper import setup_logging
setup_logging()

import logging
logger = logging.getLogger(__name__)   # 仅这一行即可
#############################################
# 基础激活函数及辅助函数
#############################################
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

def skew_symmetric(v):
    B = v.shape[0]
    vx, vy, vz = v[:, 0], v[:, 1], v[:, 2]
    zeros = torch.zeros_like(vx)
    return torch.stack([
        zeros, -vz, vy,
        vz, zeros, -vx,
        -vy, vx, zeros
    ], 1).view(B, 3, 3)

#############################################
# Safety helpers
#############################################

def safe_slice(t: torch.Tensor, *slices, name="tensor"):
    out = t.__getitem__(slices)
    if out.dim() == 0:
        logger.error(f"[SAFE] {name} became 0‑d, auto‑unsqueeze → shape (1,)")
        out = out.unsqueeze(0)
    return out

def assert_tensor2d(t, where=""):
    assert t.dim() == 2, f"[ASSERT] expect 2‑D tensor in {where}, got {t.shape}"

#############################################
# SPD / Diagonal nets
#############################################
class SPDMatrixNet(nn.Module):
    def __init__(self, input_dim, size, diag_eps=1e-4):
        super().__init__()
        self.size = size
        self.diag_eps = diag_eps
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256), Swish(), nn.LayerNorm(256),
            nn.Linear(256, size*(size+1)//2), nn.Tanh()
        )
        for m in self.fc:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        B = x.size(0)
        tril = self.fc(x)
        idx = torch.tril_indices(self.size, self.size, device=x.device)
        L = torch.zeros(B, self.size, self.size, device=x.device)
        L[:, idx[0], idx[1]] = tril
        diag = F.softplus(L.diagonal(dim1=1, dim2=2)) + self.diag_eps
        L = L - torch.diag_embed(L.diagonal(dim1=1, dim2=2)) + torch.diag_embed(diag)
        out = L @ L.transpose(1, 2)
        logger.debug(f"SPDMatrixNet output shape: {out.shape}")
        return out

class DiagonalMatrixNet(nn.Module):
    def __init__(self, input_dim, size):
        super().__init__()
        self.fc = nn.Linear(input_dim, size)
        nn.init.kaiming_normal_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        out = torch.diag_embed(torch.exp(self.fc(x)))
        logger.debug(f"DiagonalMatrixNet output shape: {out.shape}")
        return out

#############################################
# Encoders
#############################################
class LSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.2, bidirectional=False, debug=False):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True,
                            dropout=dropout, bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout)
        self.out_dim = hidden_dim * (2 if bidirectional else 1)
        self.input_dim = input_dim
        self.debug = debug

    def forward(self, x):
        assert x.size(-1) == self.input_dim, f"LSTMEncoder expect {self.input_dim}, got {x.size(-1)}"
        if self.debug:
            logger.debug(f"[LSTMEncoder] Input shape: {x.shape}")
        out, _ = self.lstm(x)
        if self.debug:
            logger.debug(f"[LSTMEncoder] LSTM output shape: {out.shape}")
        final = self.dropout(out[:, -1, :])
        if self.debug:
            logger.debug(f"[LSTMEncoder] Final output shape: {final.shape}")
        return final

class ThrustLSTM(nn.Module):
    def __init__(self, hidden_size=64, debug=False):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden_size, batch_first=True)
        self.fc = nn.Sequential(nn.ReLU(), nn.Linear(hidden_size, 32), nn.ReLU(), nn.Linear(32, 1))
        self.debug = debug

    def forward(self, x):
        if self.debug:
            logger.debug(f"[ThrustLSTM] Input shape: {x.shape}")
        out, _ = self.lstm(x)
        if self.debug:
            logger.debug(f"[ThrustLSTM] LSTM output shape: {out.shape}")
        out = self.fc(out[:, -1, :])
        if self.debug:
            logger.debug(f"[ThrustLSTM] FC output shape: {out.shape}")
        return out

#############################################
# Direct mapping net
#############################################
class DirectMappingNet_LSTM_Fusion(nn.Module):
    def __init__(self, hidden_dim=256, num_layers=2, fusion_hidden_dim=256, debug=False):
        super().__init__()
        self.power_enc = LSTMEncoder(8, hidden_dim, num_layers, debug=debug)
        self.imu_enc   = LSTMEncoder(6, hidden_dim, num_layers, debug=debug)
        self.fc = nn.Sequential(nn.Linear(hidden_dim*2, fusion_hidden_dim), nn.ReLU(), nn.Linear(fusion_hidden_dim, 6))
        self.debug = debug

    def forward(self, pw, imu):
        if self.debug:
            logger.debug(f"[DirectFusion] PW {pw.shape}, IMU {imu.shape}")
        feat = torch.cat([self.power_enc(pw), self.imu_enc(imu)], -1)
        if self.debug:
            logger.debug(f"[DirectFusion] fused {feat.shape}")
        out = self.fc(feat)
        if self.debug:
            logger.debug(f"[DirectFusion] out {out.shape}")
        return out

#############################################
# Physics net
#############################################
class EnhancedPhysicsNet_LSTM(nn.Module):
    def __init__(self, thrust_matrix, lstm_hidden_dim=512, num_layers=2, debug=False):
        super().__init__()
        self.register_buffer('T', thrust_matrix)  # (6,8)
        self.enc = LSTMEncoder(14, lstm_hidden_dim, num_layers, debug=debug)
        self.thrust_nets = nn.ModuleList([ThrustLSTM(debug=debug) for _ in range(8)])
        self.Mnet = SPDMatrixNet(lstm_hidden_dim, 3)
        self.Jnet = SPDMatrixNet(lstm_hidden_dim, 3)
        self.Dnet = DiagonalMatrixNet(lstm_hidden_dim, 6)
        self.reg = nn.Parameter(torch.tensor(1e-4))
        self.debug = debug

    def forward(self, pw, imu):
        x = torch.cat([pw, imu], -1)
        enc = self.enc(x)
        thrusts = torch.stack([net(pw[..., i].unsqueeze(-1)) for i, net in enumerate(self.thrust_nets)], 1)  # (B,8,1)
        tau = (self.T @ thrusts).squeeze(-1)
        M = self.Mnet(enc) + self.reg * torch.eye(3, device=enc.device)
        J = self.Jnet(enc) + self.reg * torch.eye(3, device=enc.device)
        D = self.Dnet(enc)
        v_lin = imu[:, -1, :3]
        omega = imu[:, -1, 3:]
        v6 = torch.cat([v_lin, omega], 1)
        B = pw.size(0)
        In = torch.zeros(B, 6, 6, device=enc.device)
        In[:, :3, :3] = M
        In[:, 3:, 3:] = J
        In += 1e-6 * torch.eye(6, device=enc.device)
        C = torch.zeros_like(In)
        C[:, :3, 3:] = -skew_symmetric(v_lin)
        C[:, 3:, 3:] = -skew_symmetric(omega)
        acc = torch.linalg.solve(In, tau.unsqueeze(-1) - (C + D) @ v6.unsqueeze(-1)).squeeze(-1)
        if self.debug:
            logger.debug(f"[Physics] acc {acc.shape}")
        return {"accel_pred": acc, "M": M, "J": J, "encoder_features": enc}

#############################################
# Hybrid model
#############################################
class HybridDynamicsModel(nn.Module):
    def __init__(self, physics_net, e2e_net, debug=False):
        super().__init__()
        self.pnet, self.enet = physics_net, e2e_net
        self.gate = nn.Sequential(nn.Linear(70, 64), Swish(), nn.Linear(64, 1), nn.Sigmoid())
        self.debug = debug

    def forward(self, pw, imu):
        p_out = self.pnet(pw, imu)
        e_out = self.enet(pw, imu)
        # ... 相当于 Ellipsis，第二个参数用 slice(None, 64)
        feat = safe_slice(
            p_out["encoder_features"],
            ...,  # 等价于 Ellipsis
            slice(None, 64),  # 等价于 :64
            name="enc_feat"
        )
        gate_in = torch.cat([feat, p_out['accel_pred']], 1)
        assert_tensor2d(gate_in, "gate_in")
        g = self.gate(gate_in)
        if self.debug:
            logger.debug(f"[Hybrid] gate {g.mean().item():.3f}")
        return g * p_out['accel_pred'] + (1-g) * e_out

#############################################
# Loss
#############################################
class EnhancedDynamicsLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.1, linear_weight=0.7):
        super().__init__()
        self.alpha, self.beta, self.lw = alpha, beta, linear_weight

    def forward(self, out, tgt):
        if isinstance(out, dict):
            acc = out['accel_pred']
            reg = -(torch.log(torch.det(out['M']).clamp_min(1e-7)).mean() +
                    torch.log(torch.det(out['J']).clamp_min(1e-7)).mean())
        else:
            acc, reg = out, 0.0
        lin = F.mse_loss(acc[:, :3], tgt['accel'][:, :3])
        ang = F.mse_loss(acc[:, 3:], tgt['accel'][:, 3:])
        total_loss=self.lw*lin + (1-self.lw)*ang + self.beta*reg
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

