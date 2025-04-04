import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicsCore(nn.Module):
    def __init__(self, input_dim=14, hidden_dim=128, window_size=5):
        """
        网络主干：利用时间窗口内的电机功率和IMU数据提取时序特征，
        经过LSTM、Dropout和多头注意力，融合推力信息后预测下一时刻的线性和角加速度。

        参数:
          - input_dim: 每个时间步的特征数（14 = 8 电机功率 + 6 IMU数据）
          - hidden_dim: LSTM及后续层的隐藏维度（注意：LSTM为双向，所以输出为 hidden_dim）
          - window_size: 时间窗口大小
        """
        super().__init__()
        self.window_size = window_size

        # 将输入展平后还原为 (B, window_size, input_dim)
        self.unflatten = nn.Unflatten(1, (window_size, input_dim))

        # LSTM 层（双向，输出维度为 hidden_dim，由于 hidden_dim//2 每个方向）
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim // 2,
            bidirectional=True,
            batch_first=True
        )
        # Dropout 仅作用于 LSTM 输出
        self.dropout = nn.Dropout(0.3)

        # 多头注意力层，输入和输出均为 hidden_dim
        self.spatial_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=0.2
        )

        # 融合推力信息，输入维度为 hidden_dim + 6（6为推力特征维度），输出256维融合特征
        self.physics_fusion = nn.Sequential(
            nn.Linear(hidden_dim + 6, 256),
            nn.GELU(),
            nn.LayerNorm(256)
        )

        # 多任务预测头：分别预测线性加速度（3维）和角加速度（3维）
        self.accel_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, 3)
        )
        self.angular_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, 3)
        )

    def forward(self, x, thrust):
        """
        前向传播：
          - x: 输入特征，形状 (B, window_size*14)
          - thrust: 推力数据，形状 (B, 6)
        返回:
          - linear_acc: 预测的线性加速度 (B, 3)
          - angular_acc: 预测的角加速度 (B, 3)
        """
        # 将展平输入还原为时序格式： (B, window_size, 14)
        x = self.unflatten(x)
        # LSTM 提取时序特征，忽略隐状态
        lstm_out, _ = self.lstm(x)  # (B, window_size, hidden_dim)
        lstm_out = self.dropout(lstm_out)

        # 多头注意力
        # MultiheadAttention 需要输入形状为 (seq_len, batch, embed_dim)
        # 这里以整个时序的均值作为 query
        temporal_feat = lstm_out  # (B, window_size, hidden_dim)
        query = temporal_feat.mean(dim=1).unsqueeze(0)  # (1, B, hidden_dim)
        key = temporal_feat.transpose(0, 1)  # (window_size, B, hidden_dim)
        value = temporal_feat.transpose(0, 1)  # (window_size, B, hidden_dim)
        attn_feat, _ = self.spatial_attn(query, key, value)  # (1, B, hidden_dim)
        attn_feat = attn_feat.squeeze(0)  # (B, hidden_dim)

        # 融合推力信息
        fused = torch.cat([attn_feat, thrust], dim=1)  # (B, hidden_dim + 6)
        physics_feat = self.physics_fusion(fused)  # (B, 256)

        # 多任务输出
        linear_acc = self.accel_head(physics_feat)
        angular_acc = self.angular_head(physics_feat)
        return linear_acc, angular_acc


class PhysicsGuidedLoss(nn.Module):
    def __init__(self, mass=10.0, inertia=torch.eye(3)):
        """
        物理指导损失函数：结合数据驱动损失和物理约束，
        包括牛顿力学约束（利用推力计算加速度）和欧拉方程约束（利用推力计算角加速度）。
        """
        super().__init__()
        self.mass = mass
        self.inertia = inertia

    def forward(self, pred_acc, true_acc, thrust, ang_acc_pred, ang_acc_true):
        # 牛顿力学约束：利用推力前3个维度除以质量获得理论加速度
        physics_acc = thrust[:, :3] / self.mass
        physics_loss = F.mse_loss(pred_acc, physics_acc)

        # 欧拉方程约束：利用推力后3个维度计算理论角加速度
        torque = thrust[:, 3:]
        physics_ang_acc = torch.matmul(torch.inverse(self.inertia), torque.T).T
        ang_physics_loss = F.mse_loss(ang_acc_pred, physics_ang_acc)

        # 数据驱动损失：加权结合线性和角加速度的均方误差
        data_loss = 0.7 * F.mse_loss(pred_acc, true_acc) + \
                    0.3 * F.mse_loss(ang_acc_pred, ang_acc_true)

        return 0.5 * data_loss + 0.3 * physics_loss + 0.2 * ang_physics_loss


class RecurrentPredictor(nn.Module):
    def __init__(self, core_net, steps=10):
        """
        循环预测器：利用核心网络预测未来多个时刻的状态。

        参数:
          - core_net: DynamicsCore 模块，用于提取特征和预测加速度
          - steps: 预测步数
        """
        super().__init__()
        self.core = core_net
        self.steps = steps

        # 状态记忆单元：将上一时刻 IMU 状态（6维）和控制输入（8维）拼接后更新记忆
        self.memory_cell = nn.GRUCell(
            input_size=14,  # IMU (6) + 控制输入 (8) -> 注意此处根据实际输入调整
            hidden_size=256
        )

    def rollout(self, init_state, motor_commands):
        """
        滚动预测未来多个时刻的状态。

        参数:
          - init_state: (B, 6) 初始IMU状态 [AccX, AccY, AccZ, AsX, AsY, AsZ]
          - motor_commands: (B, steps, 8) 未来控制指令序列
        返回:
          - 预测状态序列: (B, steps, 6)
        """
        states = [init_state]
        hidden = None
        for t in range(self.steps):
            # 拼接当前状态与控制输入
            current_input = torch.cat([states[-1], motor_commands[:, t]], dim=1)
            hidden = self.memory_cell(current_input, hidden)
            # 利用核心网络的预测头得到下一时刻加速度预测
            next_acc = self.core.accel_head(hidden)
            next_ang = self.core.angular_head(hidden)
            next_state = torch.cat([next_acc, next_ang], dim=1)
            states.append(next_state)
        return torch.stack(states[1:], dim=1)  # (B, steps, 6)


class MultiScaleEncoder(nn.Module):
    def __init__(self, window_size, input_dim):
        """
        多尺度编码器：结合时域（CNN）、全局（Transformer）和频域信息对输入进行综合特征提取。

        参数:
          - window_size: 时间窗口大小
          - input_dim: 每个时间步的特征数
        """
        super().__init__()
        # 局部特征提取（CNN）
        self.local_conv = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(2)
        )
        # 全局特征提取（Transformer）
        self.global_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=64,
                nhead=4,
                dim_feedforward=256
            ),
            num_layers=3
        )
        # 频域特征提取：利用 FFT 后取幅值，再通过全连接层映射到64维
        self.freq_encoder = nn.Sequential(
            LambdaLayer(lambda x: torch.fft.rfft(x, dim=1).abs()),
            nn.Linear(window_size // 2 + 1, 64),
            nn.GELU()
        )

    def forward(self, x):
        """
        前向传播:
          - x: 输入数据，形状 (B, window_size, input_dim)
        返回:
          - 综合特征向量，形状 (B, 64+64+64=192)
        """
        B, W, F = x.size()
        # 时域CNN特征
        cnn_feat = self.local_conv(x.permute(0, 2, 1))  # (B, 64, T)
        cnn_feat = cnn_feat.mean(dim=2)  # (B, 64)
        # Transformer全局特征
        trans_feat = self.global_transformer(x)  # (B, window_size, F')
        trans_feat = trans_feat.mean(dim=1)  # (B, F')
        # 频域特征
        freq_feat = self.freq_encoder(x)  # (B, 64)
        # 特征融合
        combined = torch.cat([cnn_feat, trans_feat, freq_feat], dim=1)
        return combined


class LambdaLayer(nn.Module):
    def __init__(self, func):
        """
        Lambda 层：用于包装任意函数，使其作为 nn.Module 使用。
        """
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)
