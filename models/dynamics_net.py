import torch
import torch.nn as nn
import torch.nn.functional as F

#############################################
# 基础激活与工具函数
#############################################
class Swish(nn.Module):
    """稳定型 Swish 激活函数，缓解梯度消失"""
    def forward(self, x):
        return x * torch.sigmoid(x.clamp(-15, 15))


def skew_symmetric(v):
    """
    构造歪对称矩阵
    :param v: 张量，形状 (B, 3)
    :return: 张量，形状 (B, 3, 3)
    """
    B = v.shape[0]
    device = v.device
    mat = torch.zeros(B, 3, 3, device=device)
    mat[:, 0, 1] = -v[:, 2]
    mat[:, 0, 2] =  v[:, 1]
    mat[:, 1, 0] =  v[:, 2]
    mat[:, 1, 2] = -v[:, 0]
    mat[:, 2, 0] = -v[:, 1]
    mat[:, 2, 1] =  v[:, 0]
    return mat


#############################################
# SPDMatrixNet 与 DiagonalMatrixNet
#############################################
class SPDMatrixNet(nn.Module):
    """
    稳定生成对称正定矩阵：
      1. 利用全连接层提取下三角元素
      2. 通过 softplus 保证对角元素为正
      3. 利用 L @ L^T 得到正定矩阵
    """
    def __init__(self, input_dim, size, diag_eps=1e-4):
        super().__init__()
        self.size = size
        self.diag_eps = diag_eps
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            Swish(),
            nn.LayerNorm(256),
            nn.Linear(256, size * (size + 1) // 2),
            nn.Tanh()
        )
        # 正交初始化第一层
        nn.init.orthogonal_(self.fc[0].weight)
        nn.init.zeros_(self.fc[0].bias)

    def forward(self, x):
        B = x.shape[0]
        tril = torch.zeros(B, self.size, self.size, device=x.device)
        indices = torch.tril_indices(self.size, self.size)
        tril[:, indices[0], indices[1]] = self.fc(x)
        diag = F.softplus(tril.diagonal(dim1=1, dim2=2)) + self.diag_eps
        tril = tril - torch.diag_embed(tril.diagonal(dim1=1, dim2=2)) + torch.diag_embed(diag)
        return tril @ tril.transpose(1, 2) + 1e-6 * torch.eye(self.size, device=x.device)


class DiagonalMatrixNet(nn.Module):
    """
    对角矩阵生成网络，保证输出矩阵对角元素为正。
    """
    def __init__(self, input_dim, size):
        super().__init__()
        self.fc = nn.Linear(input_dim, size)
        nn.init.kaiming_normal_(self.fc.weight, nonlinearity='linear')
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        diag_vals = torch.exp(self.fc(x))
        return torch.diag_embed(diag_vals)


#############################################
# 方向感知推力预测网络
#############################################
class DirectionAwareThrustNet(nn.Module):
    """
    方向感知推力预测模块：
      1. 利用 LSTM 提取功率序列特征
      2. 拼接方向历史（3 维）
      3. 输出推力幅值
    输入：
      - power_seq: (B, T) 功率序列
      - dir_history: (B, 3) 方向历史
    输出：
      - 推力幅值: (B, 1)
    """
    def __init__(self, window_size=5):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,            # 每次仅输入一个功率值
            hidden_size=16,
            bidirectional=True,
            batch_first=True         # 输出形状 (B, T, 32)
        )
        self.fc = nn.Sequential(
            nn.Linear(32 + 3, 64),    # LSTM 输出 (32) + 方向历史 (3)
            Swish(),
            nn.LayerNorm(64),
            nn.Linear(64, 1)
        )

    def forward(self, power_seq, dir_history):
        lstm_input = power_seq.unsqueeze(-1)  # (B, T, 1)
        lstm_out, _ = self.lstm(lstm_input)     # (B, T, 32)
        lstm_feat = lstm_out[:, -1]             # 取最后一时刻特征 (B,32)
        combined = torch.cat([lstm_feat, dir_history], dim=1)  # (B,35)
        thrust_mag = self.fc(combined)          # (B,1)
        return thrust_mag


#############################################
# 动力学方程求解器
#############################################
class StableDynamicsSolver(nn.Module):
    """
    数值稳定的动力学方程求解器：
    尝试先使用 Cholesky 分解求解，如果 A 不是正定则退回 torch.linalg.solve。
    """
    def forward(self, M, J, D, v_linear, omega, tau):
        B = M.size(0)
        device = M.device

        # 增加正则化因子，确保惯性块较为稳定
        reg = 1e-3  # 可根据需要调大
        inertia_block = torch.zeros(B, 6, 6, device=device)
        inertia_block[:, :3, :3] = M + reg * torch.eye(3, device=device)
        inertia_block[:, 3:, 3:] = J + reg * torch.eye(3, device=device)

        # 构造科氏力矩阵
        C = torch.zeros_like(inertia_block)
        C[:, :3, 3:] = -skew_symmetric(v_linear)
        C[:, 3:, 3:] = -skew_symmetric(omega)

        A = inertia_block + C + D

        rhs = tau.unsqueeze(-1)  # (B,6,1)
        try:
            # 尝试 Cholesky 分解求解
            L = torch.linalg.cholesky(A)
            accel = torch.cholesky_solve(rhs, L)
        except RuntimeError as e:
            # 若 Cholesky 分解失败，退回通用求解器
            print("Cholesky分解失败，退回torch.linalg.solve:", e)
            accel = torch.linalg.solve(A, rhs)
        return accel.squeeze(-1)  # (B,6)

#############################################
# 增强型物理约束网络
#############################################
class EnhancedPhysicsNet(nn.Module):
    """
    增强型物理约束网络：
      1. 利用编码器提取高层特征
      2. 为每个电机预测推力（方向感知）
      3. 预测动力学参数（M, J, D）
      4. 利用动力学求解器计算系统加速度
    """
    def __init__(self, thrust_matrix, window_size=5, hidden_dim=512):
        super().__init__()
        # 调整推力分配矩阵，匹配预处理的方向标签
        self.register_buffer('thrust_matrix', self._adjust_thrust_matrix(thrust_matrix))
        self.window_size = window_size

        self.encoder = nn.Sequential(
            nn.Linear((8 + 6) * window_size, hidden_dim),
            Swish(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.LayerNorm(hidden_dim)
        )

        # 为每个电机构建方向感知推力预测网络
        self.thrust_nets = nn.ModuleList([
            DirectionAwareThrustNet(window_size) for _ in range(8)
        ])

        # 动力学参数网络
        self.mass_net = SPDMatrixNet(hidden_dim, 3)
        self.inertia_net = SPDMatrixNet(hidden_dim, 3)
        self.damping_net = DiagonalMatrixNet(hidden_dim, 6)

        # 动力学求解器
        self.dynamics_solver = StableDynamicsSolver()

    def _adjust_thrust_matrix(self, T):
        """
        根据机械结构调整推力分配矩阵的符号，
        例如下潜时将 motor6 和 motor7 对应列取反。
        """
        T_adj = T.clone()
        T_adj[:, 5] *= -1  # Motor6
        T_adj[:, 6] *= -1  # Motor7
        return T_adj

    def _get_direction_history(self, power_window):
        """
        提取最近 3 个时刻的方向信息，返回形状 (B, 3, 8)。
        """
        return torch.sign(power_window[:, -3:])

    def forward(self, power_window, imu_window):
        B = power_window.shape[0]
        # 拼接并 flatten 输入特征
        x = torch.cat([power_window, imu_window], dim=-1).flatten(1)
        feat = self.encoder(x)

        # 获取方向历史信息
        dir_history = self._get_direction_history(power_window)  # (B,3,8)
        motor_thrust_list = []
        for i, net in enumerate(self.thrust_nets):
            power_seq = power_window[:, :, i]        # (B,T)
            dir_feat = dir_history[:, :, i]           # (B,3)
            # 对功率序列取绝对值作为 LSTM 输入
            thrust_mag = net(torch.abs(power_seq), dir_feat)
            # 恢复符号：使用最后一时刻的功率符号
            sign = torch.sign(power_seq[:, -1]).unsqueeze(-1)  # (B,1)
            motor_thrust_list.append(thrust_mag * sign)
        motor_thrusts = torch.stack(motor_thrust_list, dim=1)  # (B,8,1)
        tau = (self.thrust_matrix @ motor_thrusts).squeeze(-1)  # (B,6)

        # 动力学参数预测
        M = self.mass_net(feat)
        J = self.inertia_net(feat)
        D = self.damping_net(feat)

        # 提取运动状态（假设 imu_window 最后时刻包含线速度与角速度信息）
        v_linear = imu_window[:, -1, :3]
        omega = imu_window[:, -1, 3:]

        # 利用动力学求解器计算加速度预测
        accel_pred = self.dynamics_solver(M, J, D, v_linear, omega, tau)

        return {'accel_pred': accel_pred, 'tau': tau}


#############################################
# 混合模型：物理与端到端融合
#############################################
class HybridDynamicsModel(nn.Module):
    """
    智能混合动力学模型：
      1. 物理分支（EnhancedPhysicsNet）
      2. 端到端分支（DirectMappingNet，需另外定义，此处假设其输出形状为 (B,6)）
      3. 动态门控网络融合两者输出
    """
    def __init__(self, physics_net, e2e_net):
        super().__init__()
        self.physics_net = physics_net
        self.e2e_net = e2e_net
        # 修改门控网络输入维度：物理特征 (64) + 最新 IMU (6) + 运动统计 (6+6=12) = 64+6+12 = 82
        self.gate_net = nn.Sequential(
            nn.Linear(82, 128),  # 修改此处为82维输入
            Swish(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def _motion_intensity(self, imu):
        """
        计算运动强度特征：
          - 最新 IMU 数据 (B,6)
          - 标准差 (B,6)
          - 最大绝对值 (B,6)
        拼接后得到 (B,18) 维向量。
        """
        std_feat = torch.std(imu, dim=1)          # (B,6)
        max_feat = torch.max(torch.abs(imu), dim=1)[0]  # (B,6)
        latest_imu = imu[:, -1]                   # (B,6)
        return torch.cat([latest_imu, std_feat, max_feat], dim=1)  # (B,18)

    def forward(self, power_window, imu_window):
        # 物理分支输出
        physics_out = self.physics_net(power_window, imu_window)
        # 端到端分支输出（假设 e2e_net 输出 (B,6)）
        e2e_out = self.e2e_net(power_window, imu_window)
        # 提取物理网络编码器特征（不参与梯度更新）
        with torch.no_grad():
            physics_feat = self.physics_net.encoder(
                torch.cat([power_window, imu_window], dim=-1).flatten(1)
            )[:, :64]
        # 计算运动强度特征
        motion_feat = self._motion_intensity(imu_window)  # (B,18)
        gate_input = torch.cat([physics_feat, motion_feat], dim=1)  # (B,82)
        gate_value = self.gate_net(gate_input)  # (B,1)
        # 动态融合两路预测
        accel_pred_mix = gate_value * physics_out['accel_pred'] + (1 - gate_value) * e2e_out
        return accel_pred_mix


#############################################
# 增强型损失函数
#############################################
class EnhancedDynamicsLoss(nn.Module):
    """
    增强型损失函数：
      1. 加速度 MSE 损失
      2. 推力方向损失（二元交叉熵）
      3. 物理正则约束（例如，保证质量矩阵最小特征值为正）
    """
    def __init__(self, alpha=1.0, beta=0.1, gamma=0.5, delta=0.1):
        super().__init__()
        self.alpha = alpha    # 推力方向损失权重
        self.beta = beta      # 物理正则权重
        self.gamma = gamma    # 可扩展其他损失权重
        self.delta = delta    # 方向损失权重

    def forward(self, outputs, targets):
        # 基础加速度 MSE 损失
        accel_loss = F.mse_loss(outputs['accel_pred'], targets['accel'])

        # 推力方向损失：如果目标中包含 'thrust'
        if 'tau' in outputs and 'thrust' in targets:
            true_sign = (targets['thrust'] > 0).float()
            pred_sign = (outputs['tau'] > 0).float()
            dir_loss = F.binary_cross_entropy(pred_sign, true_sign)
        else:
            dir_loss = 0.0

        # 物理正则约束：例如保证质量矩阵最小特征值为正
        reg_loss = 0.0
        if 'M' in outputs:
            M = outputs['M']
            min_eig = torch.linalg.eigvalsh(M).min(dim=1)[0]
            reg_loss = torch.mean(F.relu(-min_eig + 1e-4))

        total_loss = accel_loss + self.delta * dir_loss + self.beta * reg_loss
        return total_loss
