import numpy as np
import torch
from torch.utils.data import Dataset

class PreprocessedDataset(Dataset):
    """
    加载预处理后的 numpy 数据文件，并将平铺的特征向量恢复为原始时间窗口格式。

    参数:
      - features_file: 预处理生成的 features 文件路径，形状 (N, window_size * 14)，其中 14 = 8 (电机功率) + 6 (IMU 数据)
      - accel_file: numpy 文件路径，线性加速度标签，形状 (N, 3)
      - angular_accel_file: numpy 文件路径，角加速度标签，形状 (N, 3)
      - thrust_file: numpy 文件路径，推力标签，形状 (N, 6)
      - window_size: 时间窗口大小，与预处理时保持一致
    """
    def __init__(self, features_file: str, accel_file: str, angular_accel_file: str, thrust_file: str,
                 window_size: int):
        super().__init__()
        self.features = np.load(features_file)
        self.accel = np.load(accel_file)
        self.angular_accel = np.load(angular_accel_file)
        self.thrust = np.load(thrust_file)
        self.window_size = window_size
        self.num_samples = self.features.shape[0]
        self.num_features = 14  # 8 (电机功率) + 6 (IMU 数据)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int):
        # 获取样本的平铺特征并检查尺寸
        flat_features = self.features[index]
        expected_size = self.window_size * self.num_features
        if flat_features.size != expected_size:
            raise ValueError(f"样本 {index} 的特征大小不匹配：期望 {expected_size}，实际 {flat_features.size}")

        # 将平铺特征恢复为 (window_size, num_features)
        features_reshaped = flat_features.reshape(self.window_size, self.num_features)

        # 分割电机功率（前8维）和 IMU 数据（后6维），转换为 torch.Tensor
        power_window = torch.from_numpy(features_reshaped[:, :8]).float()   # (window_size, 8)
        imu_window   = torch.from_numpy(features_reshaped[:, 8:]).float()    # (window_size, 6)

        # 加载加速度标签并转换为 torch.Tensor
        accel_linear = torch.from_numpy(self.accel[index]).float()         # (3,)
        accel_angular = torch.from_numpy(self.angular_accel[index]).float()   # (3,)
        # 合并为一个 6 维向量：前3为线性加速度，后3为角加速度
        accel_combined = torch.cat([accel_linear, accel_angular], dim=0)      # (6,)

        thrust_tensor = torch.from_numpy(self.thrust[index]).float()         # (6,)

        return {
            'power_window': power_window,
            'imu_window': imu_window,
            'accel': accel_combined,   # (6,)
            'thrust': thrust_tensor
        }


def _max_consecutive_zeros(arr: np.ndarray) -> int:
    """
    辅助函数：计算一维数组中最大连续零值个数
    """
    max_count = 0
    count = 0
    for val in arr:
        if val == 0:
            count += 1
            max_count = max(max_count, count)
        else:
            count = 0
    return max_count


def check_dataset(dataset: Dataset, accel_threshold: float = 1):
    """
    检查数据集中是否存在以下异常情况：
      当电机8通道功率数据（'power_window'，形状：(window_size, 8)）全部为0时，
      如果对应的系统线性加速度（'accel' 前3个数值）的范数大于 accel_threshold，
      则认为该样本异常。

    参数:
      - dataset: PreprocessedDataset 实例
      - accel_threshold: 线性加速度判断阈值，默认 0.1
    """
    problematic_samples = []
    num_samples = len(dataset)

    for i in range(num_samples):
        sample = dataset[i]
        # 检查电机功率是否全为0（使用 np.allclose 判断接近0的情况）
        power_data = sample['power_window'].numpy()  # shape: (window_size, 8)
        if np.allclose(power_data, 0, atol=1e-6):
            # 获取线性加速度：取 'accel' 前3个数值
            accel_data = sample['accel'].numpy()  # shape: (6,)
            linear_accel = accel_data[:3]
            norm_linear = np.linalg.norm(linear_accel)
            if norm_linear > accel_threshold:
                problematic_samples.append((i, norm_linear))

    if problematic_samples:
        print(f"警告: 在 {len(problematic_samples)} 个样本中检测到异常情况：")
        #for idx, norm_val in problematic_samples:
          #  print(f"  样本 {idx}: 电机8通道功率全为0，但线性加速度范数为 {norm_val:.3f} (阈值 {accel_threshold})")
    else:
        print("数据集检查通过，没有发现电机全零但加速度异常的情况。")

#git checkout -b motor-reversal-adjustment
#git status
#git add .
#git commit -m "设计了新的网络结构"
#git push -u origin motor-reversal-adjustment