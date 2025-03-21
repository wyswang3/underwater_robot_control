import torch
import torch.nn as nn
import torch.nn.functional as F


#############################################
# 基础激活函数及辅助函数
#############################################
class Swish(nn.Module):
    """自研Swish激活函数，提供平滑非线性"""

    def forward(self, x):
        return x * torch.sigmoid(x)


def skew_symmetric(v):
    """
    构造歪对称矩阵
    v: (B,3) 向量
    return: (B,3,3) 歪对称矩阵
    """
    B = v.shape[0]
    vx, vy, vz = v[:, 0], v[:, 1], v[:, 2]
    zeros = torch.zeros_like(vx)
    mat = torch.stack([
        zeros, -vz, vy,
        vz, zeros, -vx,
        -vy, vx, zeros
    ], dim=1).view(B, 3, 3)
    return mat


#############################################
# 矩阵生成模块
#############################################
class SPDMatrixNet(nn.Module):
    """增强型对称正定矩阵生成网络"""

    def __init__(self, input_dim, size, diag_eps=1e-4):
        super().__init__()
        self.size = size
        self.diag_eps = diag_eps
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),  # 增加隐藏层
            Swish(),
            nn.LayerNorm(256),
            nn.Linear(256, size * (size + 1) // 2),
            nn.Tanh()
        )
        # 参数初始化
        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity='linear')
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        B = x.shape[0]
        tril_elements = self.fc(x)  # (B, size*(size+1)//2)
        indices = torch.tril_indices(self.size, self.size)
        L = torch.zeros(B, self.size, self.size, device=x.device)
        L[:, indices[0], indices[1]] = tril_elements
        diag = F.softplus(L.diagonal(dim1=1, dim2=2)) + self.diag_eps
        # 替换对角线部分
        L = L - torch.diag_embed(L.diagonal(dim1=1, dim2=2)) + torch.diag_embed(diag)
        return L @ L.transpose(1, 2)


class DiagonalMatrixNet(nn.Module):
    """对角矩阵生成网络，保证对角元为正"""

    def __init__(self, input_dim, size):
        super().__init__()
        self.fc = nn.Linear(input_dim, size)
        nn.init.kaiming_normal_(self.fc.weight, nonlinearity='linear')
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        diag_vals = torch.exp(self.fc(x))
        return torch.diag_embed(diag_vals)


#############################################
# 端到端映射网络（e2e 网络）
#############################################
class DirectMappingNet(nn.Module):
    """
    简单的端到端网络，用于直接从输入到输出的映射
    输入维度：(8 + 6) * window_size
    输出维度：6（如加速度或其他6自由度量）
    """

    def __init__(self, window_size=5, hidden_dim=256):
        super().__init__()
        self.window_size = window_size
        self.input_dim = (8 + 6) * window_size
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
        self.decoder_layers = nn.ModuleList([
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 6)
        ])

    def forward(self, power_window, imu_window):
        x = torch.cat([power_window, imu_window], dim=-1).flatten(1)
        # 顺序通过 encoder_layers (每3个为一组)
        idx = 0
        while idx < len(self.encoder_layers):
            linear = self.encoder_layers[idx]
            activation = self.encoder_layers[idx + 1]
            norm = self.encoder_layers[idx + 2]
            x = norm(activation(linear(x)))
            idx += 3
        # decoder部分
        for layer in self.decoder_layers:
            x = layer(x)
        return x


#############################################
# 增强物理约束网络
#############################################
class EnhancedPhysicsNet(nn.Module):
    """增强物理约束网络，用物理先验辅助学习"""

    def __init__(self, thrust_matrix, window_size=5, hidden_dim=512):
        super().__init__()
        self.register_buffer('thrust_matrix', thrust_matrix)
        input_dim = (8 + 6) * window_size

        # 增强编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            Swish(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.LayerNorm(hidden_dim)
        )

        # 改进推力预测网络，为每个电机构建独立子网络
        self.thrust_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(window_size, 64),
                Swish(),
                nn.Linear(64, 32),
                Swish(),
                nn.Linear(32, 1)
            ) for _ in range(8)
        ])

        # 物理参数矩阵生成网络
        self.mass_net = SPDMatrixNet(hidden_dim, 3)
        self.inertia_net = SPDMatrixNet(hidden_dim, 3)
        self.damping_net = DiagonalMatrixNet(hidden_dim, 6)

        # 可学习的正则系数，确保矩阵数值稳定
        self.reg_scale = nn.Parameter(torch.tensor([1e-4]))

    def forward(self, power_window, imu_window):
        B = power_window.shape[0]
        # 拼接并展平输入
        x = torch.cat([power_window, imu_window], dim=-1).flatten(1)
        feat = self.encoder(x)

        # 推力预测：对每个电机分别进行预测
        motor_thrusts = torch.stack([
            net(power_window[:, :, i]) for i, net in enumerate(self.thrust_nets)
        ], dim=1)  # motor_thrusts 的形状为 (B, 8, 1)
        # 直接进行矩阵乘法，不需要 unsqueeze
        tau = (self.thrust_matrix @ motor_thrusts).squeeze(-1)

        # 后续部分保持不变
        M = self.mass_net(feat) + self.reg_scale * torch.eye(3, device=feat.device)
        J = self.inertia_net(feat) + self.reg_scale * torch.eye(3, device=feat.device)
        D = self.damping_net(feat)
        v_linear = imu_window[:, -1, :3]
        omega = imu_window[:, -1, 3:]
        v_6 = torch.cat([v_linear, omega], dim=1)
        inertia_block = torch.zeros(B, 6, 6, device=feat.device)
        inertia_block[:, :3, :3] = M
        inertia_block[:, 3:, 3:] = J
        inertia_block += 1e-6 * torch.eye(6, device=inertia_block.device)
        C = torch.zeros_like(inertia_block)
        C[:, :3, 3:] = -skew_symmetric(v_linear)
        C[:, 3:, 3:] = -skew_symmetric(omega)
        accel_pred = torch.linalg.solve(
            inertia_block,
            (tau.unsqueeze(-1) - (C + D) @ v_6.unsqueeze(-1))
        ).squeeze(-1)

        return {'accel_pred': accel_pred, 'M': M, 'J': J}


#############################################
# 混合模型：物理约束与端到端融合
#############################################
class HybridDynamicsModel(nn.Module):
    """
    门控混合模型，通过动态门控机制融合物理网络与端到端网络的输出
    """

    def __init__(self, physics_net, e2e_net):
        super().__init__()
        self.physics_net = physics_net
        self.e2e_net = e2e_net
        # 假设物理网络编码器输出维度为 hidden_dim，取前 64 维作为特征，再加上物理预测（6维），门控输入总维度 70
        self.gate = nn.Sequential(
            nn.Linear(70, 64),
            Swish(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, power_window, imu_window):
        physics_out = self.physics_net(power_window, imu_window)
        e2e_out = self.e2e_net(power_window, imu_window)
        # 提取物理网络编码器中部分特征（前64维）
        physics_features = self.physics_net.encoder(
            torch.cat([power_window, imu_window], dim=-1).flatten(1)
        )[:, :64]
        # 拼接特征与物理预测加速度，作为门控输入
        gate_input = torch.cat([physics_features, physics_out['accel_pred']], dim=1)
        gate_value = self.gate(gate_input)
        # 融合输出
        out = gate_value * physics_out['accel_pred'] + (1 - gate_value) * e2e_out
        return out


#############################################
# 增强型损失函数
#############################################
class EnhancedDynamicsLoss(nn.Module):
    """
    增强型损失函数：
      1. 加速度误差
      2. 推力损失（如果可用）
      3. 矩阵正则项（对生成的质量矩阵 M 与惯性矩阵 J 的行列式进行正则化）
    """

    def __init__(self, alpha=1.0, beta=0.1, gamma=0.01):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, outputs, targets):
        # 当 outputs 为 dict 时（物理约束模式）
        if isinstance(outputs, dict):
            accel_pred = outputs.get('accel_pred')
            tau_pred = outputs.get('tau', None)
            # 推力损失：如果存在 tau_pred 且目标中包含 'thrust'
            thrust_loss = F.mse_loss(tau_pred, targets['thrust']) if (
                        tau_pred is not None and 'thrust' in targets) else 0.0
            # 正则：对质量矩阵 M 和惯性矩阵 J 的行列式求对数并取负
            if 'M' in outputs and 'J' in outputs:
                M_det = torch.det(outputs['M']).clamp_min(1e-7)
                J_det = torch.det(outputs['J']).clamp_min(1e-7)
                reg_loss = -(torch.log(M_det).mean() + torch.log(J_det).mean())
            else:
                reg_loss = 0.0
        else:
            # 端到端模式下，仅计算加速度误差
            accel_pred = outputs
            thrust_loss = 0.0
            reg_loss = 0.0

        accel_loss = F.mse_loss(accel_pred, targets['accel'])
        total_loss = accel_loss + self.alpha * thrust_loss + self.beta * reg_loss
        return total_loss
