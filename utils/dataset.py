import numpy as np
import torch

class PreprocessedDataset:
    """
    加载预处理后的 numpy 数据文件，并将平铺的特征向量恢复为原始时间窗口格式。

    参数:
      - features_file: 预处理生成的 features 文件路径，形状 (N, window_size * 14)，其中 14 = 8 (电机功率) + 6 (IMU 数据)
      - accel_file: numpy 文件路径，线性加速度标签，形状 (N, 3)
      - angular_accel_file: numpy 文件路径，角加速度标签，形状 (N, 3)
      - thrust_file: numpy 文件路径，推力标签，形状 (N, 6)
      - window_size: 时间窗口大小，与预处理时保持一致
    """
    def __init__(self, features_file: str, accel_file: str, angular_accel_file: str, thrust_file: str, window_size: int):
        self.features = np.load(features_file)
        self.accel = np.load(accel_file)
        self.angular_accel = np.load(angular_accel_file)
        self.thrust = np.load(thrust_file)
        self.window_size = window_size
        self.num_samples = self.features.shape[0]
        self.num_features = 14  # (8 + 6)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index: int):
        feat = self.features[index]
        feat_reshaped = feat.reshape(self.window_size, self.num_features)
        power_window = torch.tensor(feat_reshaped[:, :8], dtype=torch.float32)
        imu_window = torch.tensor(feat_reshaped[:, 8:], dtype=torch.float32)
        accel_linear = torch.tensor(self.accel[index], dtype=torch.float32)
        accel_angular = torch.tensor(self.angular_accel[index], dtype=torch.float32)
        accel_combined = torch.cat([accel_linear, accel_angular], dim=0)
        thrust = torch.tensor(self.thrust[index], dtype=torch.float32)
        return {
            'power_window': power_window,
            'imu_window': imu_window,
            'accel': accel_combined,
            'thrust': thrust
        }
#git status
#git add .