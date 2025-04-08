import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class HydroParamEstimator(nn.Module):
    """
    通过输入特征估计水动力学参数，例如质量矩阵和阻尼矩阵的上三角元素，
    然后通过构造函数恢复到对称正定矩阵。

    优化点：
      1. 独立设计两个分支分别估计质量和阻尼矩阵的上三角参数。
      2. 使用 Softplus 激活函数确保输出为正。
      3. 输出参数的数量调整为 21（6×6 矩阵上三角的元素数量），
         以便在构造矩阵时能正确赋值。
    """

    def __init__(self, in_dim: int, out_dim: int = 21):
        super().__init__()
        self.mass_net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.GELU(),
            nn.Linear(64, out_dim),
            nn.Softplus()  # 确保输出为正
        )
        self.damping_net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.GELU(),
            nn.Linear(64, out_dim),
            nn.Softplus()
        )
        self.min_val = 1e-3  # 最小对角线阈值，防止奇异

    def build_matrix(self, triu: torch.Tensor) -> torch.Tensor:
        # 修改矩阵构造方式为更稳定的Cholesky分解形式
        B = triu.size(0)
        L = torch.zeros(B, 6, 6, device=triu.device)
        idx = torch.tril_indices(6, 6)  # 改为下三角索引
        L[:, idx[0], idx[1]] = triu  # 假设triu实际应为下三角参数
        L.diagonal(dim1=1, dim2=2).clamp_(min=self.min_val)  # 对角线约束
        return L @ L.transpose(1, 2) + 1e-6 * torch.eye(6, device=L.device)

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        参数:
          - x: 融合的特征，形状为 (B, in_dim)
        返回:
          - M: 估计的质量矩阵，形状 (B, 6, 6)
          - D: 估计的阻尼矩阵，形状 (B, 6, 6)
        """
        mass_triu = self.mass_net(x)  # (B, 21)
        damping_triu = self.damping_net(x)  # (B, 21)
        M = self.build_matrix(mass_triu)
        D = self.build_matrix(damping_triu)
        return M, D

class BetterHydroNet(nn.Module):
    def __init__(self, window_size: int = 5, input_dim: int = 14, hidden_dim: int = 256, lstm_layers: int = 2) -> None:
        """
        改进版网络结构：利用多层双向 LSTM 从时序数据中提取全局特征，并将该特征与预处理得到的 Thrust 数据融合，
        然后通过一个 HydroParamEstimator 分支估计系统的水动力学参数 M 和 D，最后借助物理公式（在本例中简单用 M^{-1}*tau）
        和一个残差补偿网络预测最终的 6 维加速度（前 3 维为线性加速度，后 3 维为角加速度）。

        参数:
          - window_size: 时间窗口大小
          - input_dim: 每个时间步的输入特征数（例如 14 = 8 电机功率 + 6 IMU）
          - hidden_dim: LSTM 输出总维度（双向 LSTM 后实际为 hidden_dim）
          - lstm_layers: 堆叠的 LSTM 层数（默认2）
        """
        super().__init__()
        self.window_size = window_size

        # 输入还原与归一化
        self.unflatten = nn.Unflatten(1, (window_size, input_dim))
        self.input_norm = nn.LayerNorm(input_dim)

        # 多层双向 LSTM 提取时序特征
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim // 2,
            num_layers=lstm_layers,
            bidirectional=True,
            batch_first=True,
            dropout=0.2 if lstm_layers > 1 else 0.0
        )
        self.lstm_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.3)

        # 融合 LSTM 全局特征与预处理推力数据（6 维）
        # 这里融合后的特征尺寸为 (hidden_dim + 6)
        self.residual_net = nn.Sequential(
            nn.Linear(hidden_dim + 6, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 6)
        )

        # 水动力学参数估计模块
        self.hydro_estimator = HydroParamEstimator(in_dim=hidden_dim + 6)

    def forward(self, x: torch.Tensor, thrust: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播:
          - x: 输入平铺向量，形状 (B, window_size*input_dim)
          - thrust: 预处理得到的推力数据，形状 (B, 6)
        返回:
          - pred_lin: 预测的线性加速度 (B, 3)
          - pred_ang: 预测的角加速度 (B, 3)
          - M: 估计的质量矩阵 (B, 6, 6)
          - D: 估计的阻尼矩阵 (B, 6, 6)
        """
        # 1. 还原与归一化输入
        x = self.unflatten(x)  # (B, window_size, input_dim)
        x = self.input_norm(x)  # LayerNorm
        # 2. 提取时序特征（LSTM），取最后时间步输出
        lstm_out, _ = self.lstm(x)  # (B, window_size, hidden_dim)
        lstm_out = self.dropout(lstm_out)
        global_feature = lstm_out[:, -1, :]  # (B, hidden_dim)
        global_feature = self.lstm_norm(global_feature)
        # 3. 融合全局特征与推力数据
        fused_feature = torch.cat([global_feature, thrust], dim=1)  # (B, hidden_dim + 6)
        # 4. 残差补偿：利用融合特征预测加速度残差
        residual = self.residual_net(fused_feature)  # (B, 6)
        # 5. 物理计算：利用估计的水动力学参数计算理论加速度
        # 这里采用简单模型：理论加速度 = M^{-1} * tau
        # 注意：这部分可根据实际动力学方程进一步扩展
        # 尝试求解 M * a_physics = tau
        a_physics = torch.linalg.solve(self.hydro_estimator(x=fused_feature)[0], thrust)
        # 若希望同时利用 D, 可进行更复杂计算，例如 a = M^{-1}(tau - D * v_approx); 这里简化处理

        # 6. 将理论加速度与残差补偿相加
        total_acc = a_physics + residual
        pred_lin, pred_ang = total_acc[:, :3], total_acc[:, 3:]
        # 7. 同时输出估计的 M 和 D（供损失计算或后续分析使用）
        M, D = self.hydro_estimator(fused_feature)
        return pred_lin, pred_ang, M, D


class PhysicsAwareLoss(nn.Module):
    def __init__(self, lambda_phy: float = 0.5, inertia: torch.Tensor = torch.eye(3)) -> None:
        """
        物理感知损失函数，结合数据驱动损失和物理约束损失。

        参数:
          - lambda_phy: 物理约束损失权重（例如 0.5 表示 50%的权重）
          - inertia: 转动惯量矩阵，用于计算理论角加速度
        """
        super().__init__()
        self.lambda_phy = lambda_phy
        self.inertia = inertia
        self.mse = nn.MSELoss()

    def forward(self, pred_lin: torch.Tensor, true_lin: torch.Tensor,
                pred_ang: torch.Tensor, true_ang: torch.Tensor,
                tau: torch.Tensor, M: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
        # 数据驱动损失：计算预测与真实加速度之间的 MSE
        loss_data = (self.mse(pred_lin, true_lin) + self.mse(pred_ang, true_ang)) / 2.0

        # 物理约束损失：
        # 这里假设 tau 的前 3 维是用于计算理论线性加速度
        tau_lin = tau[:, :3]
        # 提取 M 中代表线性运动的 3×3 子矩阵，并计算有效质量：对角线均值
        M_lin = M[:, :3, :3]                               # (B, 3, 3)
        effective_mass = M_lin.diagonal(dim1=1, dim2=2).mean(dim=1, keepdim=True)  # (B, 1)
        # 利用提取的有效质量计算理论线性加速度
        theoretical_lin = tau_lin / effective_mass         # (B, 3)
        loss_phys = self.mse(pred_lin, theoretical_lin)

        # 可选：对水动力参数矩阵加约束，如 M 的对称性和 D 的正定性损失
        symmetry_loss = self.mse(M, M.transpose(1, 2))
        eigen_D = torch.linalg.eigvalsh(D)
        pd_loss = F.relu(-eigen_D[:, 0]).mean()  # 若最小特征值为负则产生惩罚
        loss_param = symmetry_loss + pd_loss

        total_phys_loss = loss_phys + 0.1 * loss_param

        # 最终总损失：数据驱动损失和物理约束损失的加权组合
        total_loss = (1.0 - self.lambda_phy) * loss_data + self.lambda_phy * total_phys_loss
        return total_loss