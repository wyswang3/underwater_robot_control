import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class SimpleHydroNet(nn.Module):
    def __init__(self, window_size: int = 5, input_dim: int = 14, hidden_dim: int = 256) -> None:
        """
        简化版水下动力学网络：
          - 输入为平铺后的特征向量 (B, window_size*14)，其中 14 = 8 (电机功率) + 6 (IMU 数据)
          - 通过 Unflatten 将输入还原为 (B, window_size, 14)
          - 利用多尺度卷积块提取局部特征，再通过 LSTM 提取全局时序特征（取最后时间步输出）
          - 最后分别预测线性加速度和角加速度（各3维）

        参数:
          - window_size: 时间窗口大小
          - input_dim: 每个时间步的输入特征数（默认14）
          - hidden_dim: LSTM 输出的维度（这里为双向 LSTM，输出维度为 hidden_dim）
        """
        super(SimpleHydroNet, self).__init__()
        self.window_size = window_size

        # 将平铺输入还原为 (B, window_size, input_dim)
        self.unflatten = nn.Unflatten(1, (window_size, input_dim))

        # 多尺度卷积块：使用两层1D卷积提取局部特征
        self.conv_blocks = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.MaxPool1d(2)  # 降低时间维度
        )

        # 双向 LSTM 层：输入维度为 256，输出维度为 hidden_dim（双向）
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=hidden_dim // 2,
            bidirectional=True,
            batch_first=True
        )
        self.dropout = nn.Dropout(0.3)

        # 全连接预测头：直接从 LSTM 最后时间步输出预测线性和角加速度（各3维）
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 6)
        )

    def forward(self, x: torch.Tensor, thrust: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播：
          - x: 输入特征，形状 (B, window_size*14)
          - thrust: 推力数据，此处简化版不使用推力数据，可设置为 None
        返回：
          - linear_acc: 预测的线性加速度 (B, 3)
          - angular_acc: 预测的角加速度 (B, 3)
        """
        # 1. 还原为 (B, window_size, input_dim)
        x = self.unflatten(x)
        # 2. 转换为 (B, input_dim, window_size) 以适应 Conv1d
        x = x.transpose(1, 2)
        # 3. 卷积块提取局部特征：输出 (B, 256, T_out)
        x = self.conv_blocks(x)
        # 4. 转换回 (B, T_out, 256) 供 LSTM 使用
        x = x.transpose(1, 2)
        # 5. LSTM 提取全局时序特征
        lstm_out, _ = self.lstm(x)  # (B, T_out, hidden_dim)
        lstm_out = self.dropout(lstm_out)
        # 6. 取最后时间步输出作为全局特征
        global_feature = lstm_out[:, -1, :]  # (B, hidden_dim)
        # 7. 全连接层预测：输出为 (B, 6) -> 前3为线性加速度，后3为角加速度
        output = self.fc(global_feature)
        pred_lin, pred_ang = output[:, :3], output[:, 3:]
        return pred_lin, pred_ang
