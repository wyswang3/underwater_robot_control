import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from config import Config

cfg = Config()


# -----------------------------------------------------------
# HydroParamEstimator：估计 6×6 质量矩阵 M 与阻尼矩阵 D，确保对称正定
# -----------------------------------------------------------
class HydroParamEstimator(nn.Module):
    def __init__(self,
                 in_dim: int,
                 hidden: int = getattr(cfg.training, "HYDRO_HIDDEN", 128),  # 隐藏层大小，默认128
                 dim: int = getattr(cfg.training, "MATRIX_DIM", 6),          # 矩阵维度，默认6
                 min_diag: float = getattr(cfg.training, "HYDRO_MIN_DIAG", 1e-2)  # 最小对角线值，默认1e-2
                 ) -> None:
        super().__init__()
        out_dim = dim * (dim + 1) // 2  # 下三角元素个数
        self.dim = dim
        self.min_diag = min_diag

        def branch() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.Dropout(0.3),  # 防止过拟合
                nn.GELU(),
                nn.Linear(hidden, out_dim)
            )

        self.mass_branch = branch()
        self.damping_branch = branch()

    def _build_spd(self, raw: torch.Tensor) -> torch.Tensor:
        """
        构造对称正定矩阵：
          1. 将 raw 填充到下三角；
          2. 对角线使用 softplus 激活并加上 min_diag；
          3. 通过 L Lᵀ + min_diag * I 构造 SPD 矩阵。
        """
        B, _ = raw.shape
        L = raw.new_zeros(B, self.dim, self.dim)
        idx = torch.tril_indices(self.dim, self.dim)
        L[:, idx[0], idx[1]] = raw

        # 处理对角线，确保大于 min_diag
        diag = F.softplus(L.diagonal(dim1=1, dim2=2)) + self.min_diag
        L = L.tril(-1) + torch.diag_embed(diag)

        eye = torch.eye(self.dim, device=raw.device, dtype=raw.dtype)
        return L @ L.transpose(1, 2) + self.min_diag * eye

    def forward(self, feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mass_raw = self.mass_branch(feat)
        damping_raw = self.damping_branch(feat)
        M = self._build_spd(mass_raw)
        D = self._build_spd(damping_raw)
        return M, D


# -----------------------------------------------------------
# BetterHydroNet：双向 LSTM编码 + 参数估计 + 速度预测 + 物理基线计算 + 残差补偿
# -----------------------------------------------------------
class BetterHydroNet(nn.Module):
    def __init__(self,
                 window_size: int = cfg.training.WINDOW_SIZE,  # 例如5
                 input_dim: int = cfg.training.INPUT_DIM,       # 例如14
                 hidden_dim: int = cfg.training.HIDDEN_DIM,       # 例如512
                 lstm_layers: int = cfg.training.LSTM_LAYERS,       # 例如3
                 lstm_dropout: float = getattr(cfg.training, "LSTM_DROPOUT", 0.3),
                 layer_dropout: float = getattr(cfg.training, "LAYER_DROPOUT", 0.4),
                 residual_dropout: float = getattr(cfg.training, "RESIDUAL_DROPOUT", 0.4)
                 ) -> None:
        super().__init__()
        self.win = window_size

        # ── 编码部分 ─────────────────────────────
        self.unflatten = nn.Unflatten(1, (window_size, input_dim))
        self.norm_in = nn.LayerNorm(input_dim)

        # 双向 LSTM 提取时序特征，支持 dropout
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim // 2,
            num_layers=lstm_layers,
            bidirectional=True,
            batch_first=True,
            dropout=lstm_dropout if lstm_layers > 1 else 0.0
        )
        self.norm_lstm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(layer_dropout)

        # ── 速度预测模块 ─────────────────────────
        # detach() 防止梯度反向传播影响前面编码层
        velocity_hidden = getattr(cfg.training, "VELOCITY_HIDDEN", 128)
        self.velocity_net = nn.Sequential(
            nn.Linear(hidden_dim + 6, velocity_hidden),
            nn.GELU(),
            nn.LayerNorm(velocity_hidden),
            nn.Dropout(0.3),
            nn.Linear(velocity_hidden, 6)
        )

        # ── 残差补偿网络 ─────────────────────────
        self.residual_net = nn.Sequential(
            nn.Linear(hidden_dim + 6, 256),
            nn.Dropout(residual_dropout),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 6)
        )

        # ── Hydro 参数估计器 ──────────────────────
        self.hydro_estimator = HydroParamEstimator(hidden_dim + 6)
        self.register_buffer("eye6", torch.eye(6), persistent=False)

    def forward(self, flat_feat: torch.Tensor, thrust: torch.Tensor,
                return_v_pred: bool = True
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] or Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        参数说明：
          flat_feat: 扁平化输入特征，形状 (B, window_size * input_dim)
          thrust: 6 维推力/力矩向量，形状 (B, 6)
          return_v_pred: 若为 True 则返回 5 项 (pred_lin, pred_ang, M, D, v_pred) 用于训练，
                         为 False 则返回 4 项 (pred_lin, pred_ang, M, D) 用于验证/评估。
        """
        # ── 特征编码 ─────────────────────────────
        x = self.norm_in(self.unflatten(flat_feat))  # (B, window_size, input_dim)
        lstm_out, _ = self.lstm(x)  # (B, window_size, hidden_dim)
        g = self.norm_lstm(self.dropout(lstm_out)[:, -1])  # 取最后时刻 (B, hidden_dim)
        fused = torch.cat([g, thrust], dim=1)  # (B, hidden_dim + 6)

        # ── 速度预测（分离梯度） ─────────────────
        v_pred = self.velocity_net(fused.detach())

        # ── Hydro 参数估计 ─────────────────────────
        M, D = self.hydro_estimator(fused)

        # ── 物理基线计算 ─────────────────────────
        eps = 1e-3
        with torch.amp.autocast("cuda", enabled=False):
            M_mod = M.float() + eps * self.eye6  # (B, 6, 6)
            D_float = D.float()
            # thrust: (B,6) -> (B,6,1); v_pred: (B,6) -> (B,6,1)
            tau_net = thrust.float().unsqueeze(-1) - D_float @ v_pred.unsqueeze(-1)  # (B,6,1)
            a_physics = torch.linalg.solve(M_mod, tau_net).squeeze(-1)
        a_physics = a_physics.to(M.dtype)

        # ── 残差补偿 ─────────────────────────────
        total_acc = a_physics + self.residual_net(fused)
        pred_lin, pred_ang = total_acc.chunk(2, dim=1)

        if return_v_pred:
            return pred_lin, pred_ang, M, D, v_pred
        else:
            return pred_lin, pred_ang, M, D


# -----------------------------------------------------------
# PhysicsAwareLoss：数据损失 + 物理一致性损失 + 正则项（阻尼矩阵正定约束）
# -----------------------------------------------------------
class PhysicsAwareLoss(nn.Module):
    def __init__(self,
                 lambda_phy: float = getattr(cfg.training, "LAMBDA_PHY", 0.4),
                 beta_reg: float = getattr(cfg.training, "BETA_REG", 0.1),
                 eps: float = getattr(cfg.training, "LOSS_EPS", 1e-6)
                 ) -> None:
        super().__init__()
        self.lambda_phy = lambda_phy
        self.beta_reg = beta_reg
        self.eps = eps
        self.mse = nn.MSELoss()

    def forward(self,
                pred_lin: torch.Tensor, true_lin: torch.Tensor,
                pred_ang: torch.Tensor, true_ang: torch.Tensor,
                tau: torch.Tensor, M: torch.Tensor, D: torch.Tensor, v_pred: torch.Tensor
                ) -> torch.Tensor:
        # 检查 tau 维度，如存在多余信息则只取前6个元素
        if tau.dim() == 2 and tau.size(1) != 6:
            tau = tau[:, :6]
        # 1) 数据损失：预测加速度与真实加速度的均方误差
        loss_data = 0.5 * (self.mse(pred_lin, true_lin) + self.mse(pred_ang, true_ang))

        # 2) 物理一致性损失：依据 a_theory = M⁻¹ (tau - D*v_pred)
        with torch.amp.autocast("cuda", enabled=False):
            M_f = M.float()
            D_f = D.float()
            tau_net = tau.float().unsqueeze(-1) - D_f @ v_pred.unsqueeze(-1)
            a_theory = torch.linalg.solve(M_f, tau_net).squeeze(-1)
        loss_phys = 0.5 * (self.mse(pred_lin, a_theory[:, :3]) + self.mse(pred_ang, a_theory[:, 3:6]))

        # 3) 正则项：对 D 对角线施加正定约束
        diag_D = D.diagonal(dim1=1, dim2=2)
        pd_reg = torch.relu(-diag_D.min(dim=1).values + 1e-3).mean()
        loss_reg = pd_reg

        # 4) 组合总损失
        total_loss = (1 - self.lambda_phy) * loss_data + \
                     self.lambda_phy * (loss_phys + self.beta_reg * loss_reg)
        return total_loss
