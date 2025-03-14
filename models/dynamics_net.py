import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import sqrt

# 从 config 导入配置 (可选，如果你需要使用 cfg.device 等参数)
from config import Config
cfg = Config()

######################################################
# Swish 激活函数
######################################################
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

######################################################
# skew_symmetric 函数，用于构造歪对称矩阵
######################################################
def skew_symmetric(v):
    """
    v: (batch,3) 向量
    return: (batch,3,3) 歪对称矩阵
    """
    B = v.shape[0]
    vx, vy, vz = v[:, 0], v[:, 1], v[:, 2]
    zeros = torch.zeros_like(vx)
    mat = torch.stack([
        zeros, -vz, vy,
        vz, zeros, -vx,
        -vy, vx, zeros
    ], dim=1).view(B,3,3)
    return mat

######################################################
# SPDMatrixNet (对称正定矩阵)
######################################################
class SPDMatrixNet(nn.Module):
    """
    输出对称正定矩阵: SPD = L*L^T
    L 的下三角由全连接层产生 (size*(size+1)//2),
    对角元素 softplus+eps，确保正定。
    """
    def __init__(self, input_dim, size, diag_eps=1e-4):
        super().__init__()
        self.size = size
        self.diag_eps = diag_eps
        self.fc = nn.Sequential(
            nn.Linear(input_dim, size*(size+1)//2),
            nn.Tanh()
        )

    def forward(self, x):
        B = x.size(0)
        tril_elements = self.fc(x)  # (B, size*(size+1)//2)
        L = torch.zeros(B, self.size, self.size, device=x.device, dtype=x.dtype)
        indices = torch.tril_indices(self.size, self.size)
        L[:, indices[0], indices[1]] = tril_elements

        diag = L.diagonal(dim1=1, dim2=2)
        diag = F.softplus(diag) + self.diag_eps
        # 用对角替换
        L = L - torch.diag_embed(L.diagonal(dim1=1, dim2=2)) + torch.diag_embed(diag)

        return L @ L.transpose(1,2)

######################################################
# DiagonalMatrixNet (对角矩阵)
######################################################
class DiagonalMatrixNet(nn.Module):
    """
    输出对角矩阵( size x size ),
    对角元 = exp(linear) 保证正值
    """
    def __init__(self, input_dim, size):
        super().__init__()
        self.fc = nn.Linear(input_dim, size)

    def forward(self, x):
        diag_vals = torch.exp(self.fc(x))
        return torch.diag_embed(diag_vals)

######################################################
# SymmetricMatrixNet (对称矩阵)
######################################################
class SymmetricMatrixNet(nn.Module):
    """
    输出对称矩阵( size x size ),
    主对角不要求 > 0，也不保证正定
    """
    def __init__(self, input_dim, size):
        super().__init__()
        self.size = size
        self.fc = nn.Linear(input_dim, size*(size+1)//2)

    def forward(self, x):
        B = x.size(0)
        triup = self.fc(x)
        M = torch.zeros(B, self.size, self.size, device=x.device, dtype=x.dtype)
        ind = torch.triu_indices(self.size, self.size)
        M[:, ind[0], ind[1]] = triup
        M = M + M.transpose(1,2) - torch.diag_embed(M.diagonal(dim1=1,dim2=2))
        return M

######################################################
# 动力学方程约束层
######################################################
class DynamicsConstraintLayer(nn.Module):
    """
    6 自由度方程:
    tau = [ M  0 ] [a_linear] + C(v)*v + D*v
          [ 0  J ] [a_angular]
    """
    def __init__(self):
        super().__init__()

    def forward(self, tau, M, J, D, imu_window):
        B = tau.size(0)
        # 取最后帧速度
        v_linear = imu_window[:, -1, :3]
        omega    = imu_window[:, -1, 3:]
        v_6 = torch.cat([v_linear, omega], dim=1)  # (B,6)

        # 组装 惯性矩阵
        inertia_block = torch.zeros(B,6,6, device=tau.device, dtype=tau.dtype)
        inertia_block[:, :3, :3] = M
        inertia_block[:, 3:, 3:] = J

        # 科氏力C
        C = torch.zeros_like(inertia_block)
        C[:, :3, 3:] = -skew_symmetric(v_linear)
        C[:, 3:, 3:] = -skew_symmetric(omega)

        # 合力
        cdv = (C + D) @ v_6.unsqueeze(-1)  # (B,6,1)
        net_force = tau.unsqueeze(-1) - cdv  # (B,6,1)

        # 求 a_6 = inertia^-1 * net_force
        a_6 = torch.linalg.solve(inertia_block, net_force)  # (B,6,1)
        return a_6.squeeze(-1)

######################################################
# 三层编码的物理网络
######################################################
class EnhancedPhysicsNet(nn.Module):
    """
    三层编码: Linear->Swish->LayerNorm×3
    + thrust_nets, mass_net, inertia_net, damping_net
    """
    def __init__(self, thrust_matrix, window_size=5, hidden_dim=256):
        super().__init__()
        self.register_buffer('thrust_matrix', thrust_matrix)

        input_dim = (8+6)*window_size
        # 三层 (Linear->Swish->LayerNorm)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            Swish(),
            nn.LayerNorm(hidden_dim),

            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.LayerNorm(hidden_dim),

            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.LayerNorm(hidden_dim)
        )

        # thrust_nets
        self.thrust_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(window_size, 32),
                Swish(),
                nn.Linear(32,1)
            ) for _ in range(8)
        ])

        # 质量矩阵,惯性矩阵,阻尼矩阵
        self.mass_net     = SPDMatrixNet(hidden_dim, 3)
        self.inertia_net  = SPDMatrixNet(hidden_dim, 3)
        self.damping_net  = DiagonalMatrixNet(hidden_dim, 6)
        self.physics_layer= DynamicsConstraintLayer()

        self.window_size = window_size
        self.hidden_dim  = hidden_dim

    def forward(self, power_window, imu_window):
        B, T, _ = power_window.shape
        # 拼接并flatten
        x = torch.cat([power_window, imu_window], dim=-1).flatten(1)
        feat = self.encoder(x)

        # thrust_nets
        motor_thrusts = []
        for i in range(8):
            seq = power_window[:, :, i]  # (B,T)
            thrust_i = self.thrust_nets[i](seq).squeeze(-1)  # (B,)
            motor_thrusts.append(thrust_i)
        motor_thrusts = torch.stack(motor_thrusts, dim=1) # (B,8)

        # 计算 tau
        tau = (self.thrust_matrix @ motor_thrusts.unsqueeze(-1)).squeeze(-1)

        # mass, inertia, damping
        M = self.mass_net(feat)
        J = self.inertia_net(feat)
        D = self.damping_net(feat)

        # 动力学层
        accel_pred = self.physics_layer(tau, M, J, D, imu_window)

        return {
            'tau': tau,
            'mass_matrix': M,
            'inertia_matrix': J,
            'damping_matrix': D,
            'accel_pred': accel_pred
        }

######################################################
# 增强型损失函数
######################################################
class EnhancedDynamicsLoss(nn.Module):
    """
    1. 加速度误差
    2. 推力损失
    3. 正则(对 mass/inertia行列式)
    """
    def __init__(self, alpha=1.0, beta=0.1, gamma=0.01):
        super().__init__()
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma

    def forward(self, outputs, targets):
        if isinstance(outputs, dict):
            accel_pred = outputs['accel_pred']
            tau_pred   = outputs.get('tau', None)
            # thrust loss
            thrust_loss = F.mse_loss(tau_pred, targets['thrust']) if tau_pred is not None else 0.0

            # 正则
            if 'mass_matrix' in outputs and 'inertia_matrix' in outputs:
                M_det = torch.det(outputs['mass_matrix']).clamp_min(1e-7)
                J_det = torch.det(outputs['inertia_matrix']).clamp_min(1e-7)
                reg_loss = -(torch.log(M_det).mean() + torch.log(J_det).mean())
            else:
                reg_loss = 0.0

        else:
            # 端到端输出
            accel_pred = outputs
            thrust_loss= 0.0
            reg_loss   = 0.0

        accel_loss = F.mse_loss(accel_pred, targets['accel'])
        total_loss = accel_loss + self.alpha*thrust_loss + self.beta*reg_loss
        return total_loss

######################################################
# 端到端网络(三层)
######################################################
class DirectMappingNet(nn.Module):
    def __init__(self, window_size=5, hidden_dim=256):
        super().__init__()
        self.window_size = window_size
        self.input_dim   = (8+6)*window_size

        # 三层 (Linear->ReLU->LayerNorm)
        self.encoder_layers = nn.ModuleList([
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        ])

        # decoder
        self.decoder_layers = nn.ModuleList([
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 6)
        ])

    def forward(self, power_window, imu_window):
        x = torch.cat([power_window, imu_window], dim=-1).flatten(1)

        # 依次通过 encoder_layers, 每 3个(Linear,ReLU,LayerNorm)
        idx = 0
        while idx < len(self.encoder_layers):
            linear = self.encoder_layers[idx]
            activation = self.encoder_layers[idx+1]
            norm = self.encoder_layers[idx+2]

            x = linear(x)
            x = activation(x)
            x = norm(x)
            idx += 3

        # decoder
        for layer in self.decoder_layers:
            x = layer(x)

        return x

######################################################
# 混合网络
######################################################
class HybridDynamicsModel(nn.Module):
    """
    在发现NaN或需要时切换物理->端到端
    """
    def __init__(self, physics_net, e2e_net):
        super().__init__()
        self.physics_net = physics_net
        self.e2e_net     = e2e_net
        self.active_net  = 'physics'  # 默认物理模式

    def forward(self, power_window, imu_window):
        if self.active_net == 'physics':
            return self.physics_net(power_window, imu_window)
        else:
            return self.e2e_net(power_window, imu_window)

    def switch_to_e2e(self):
        self.active_net = 'e2e'
