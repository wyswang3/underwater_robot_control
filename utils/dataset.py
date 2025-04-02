import numpy as np
import torch
from torch.utils.data import Dataset
import os

class PreprocessedDataset(Dataset):
    """
    预处理数据集，用于加载预处理后的 .npy 文件，并生成训练样本。

    输入文件：
      - features_file: 预处理生成的 features 文件路径，形状为 (N, (8+6)*window_size)
        （包含电机功率和 IMU 数据，已融合最终处理结果）
      - accel_file: 线性加速度标签，形状 (N, 3)
      - angular_accel_file: 角加速度标签，形状 (N, 3)
      - thrust_file: 推力标签，形状 (N, 6)
      - motor_sign_file: （可选）电机反转标签文件，形状 (N, window_size, 8)
      - window_size: 时间窗口大小，与预处理时保持一致
    输出样本包含：
      - "power_window": (window_size, 8) 张量（电机功率，已包含反转信息）
      - "imu_window": (window_size, 6) 张量（IMU 数据）
      - "accel": (6,) 张量，拼接了线性加速度和角加速度（例如 [Acc_linear, Acc_angular]）
      - "thrust": (6,) 张量，推力标签
      - "motor_sign": (window_size, 8) 张量，原始的电机反转标签（如果提供）
    """

    def __init__(self, features_file: str, accel_file: str, angular_accel_file: str,
                 thrust_file: str, window_size: int, motor_sign_file: str = None):
        self.features = np.load(features_file)  # shape: (N, (8+6)*window_size)
        self.accel = np.load(accel_file)  # shape: (N, 3)
        self.angular_accel = np.load(angular_accel_file)  # shape: (N, 3)
        self.thrust = np.load(thrust_file)  # shape: (N, 6)
        self.window_size = window_size

        # 如果提供了 motor_sign_file，则加载电机反转标签
        if motor_sign_file is not None and os.path.exists(motor_sign_file):
            self.motor_sign = np.load(motor_sign_file)  # shape: (N, window_size, 8)
        else:
            self.motor_sign = None

        self.num_samples = self.features.shape[0]
        self.num_features = 14  # 8 (电机) + 6 (IMU)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index: int):
        # 还原 flatten 后的特征为 (window_size, 14)
        flat_feat = self.features[index]
        window = flat_feat.reshape(self.window_size, self.num_features)
        # 电机功率窗口 (window_size, 8)
        power_window = torch.tensor(window[:, :8], dtype=torch.float32)
        # IMU 数据窗口 (window_size, 6)
        imu_window = torch.tensor(window[:, 8:], dtype=torch.float32)

        # 拼接线性加速度与角加速度，生成6维加速度标签
        linear_accel = torch.tensor(self.accel[index], dtype=torch.float32)  # (3,)
        angular_accel = torch.tensor(self.angular_accel[index], dtype=torch.float32)  # (3,)
        accel_label = torch.cat([linear_accel, angular_accel], dim=0)  # (6,)

        thrust_label = torch.tensor(self.thrust[index], dtype=torch.float32)  # (6,)

        sample = {
            "power_window": power_window,  # (T,8)
            "imu_window": imu_window,  # (T,6)
            "accel": accel_label,  # (6,)
            "thrust": thrust_label  # (6,)
        }
        # 如果有 motor_sign 文件，则添加到 sample 中
        if self.motor_sign is not None:
            motor_sign = torch.tensor(self.motor_sign[index], dtype=torch.float32)  # (T,8)
            sample["motor_sign"] = motor_sign

        return sample
#git checkout -b motor-reversal-adjustment
#git status
#git add .
#git commit -m "调整预处理代码，添加电机反转标签，确保推力方向正确"
#git push -u origin motor-reversal-adjustment