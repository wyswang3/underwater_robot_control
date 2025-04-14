import numpy as np
import torch
from torch.utils.data import Dataset
from utils.log_helper import setup_logging
setup_logging()

import logging
logger = logging.getLogger(__name__)   # 仅这一行即可


class PreprocessedDataset(Dataset):
    """
    加载预处理后的 numpy 数据文件，并将平铺的特征向量恢复为原始时间窗口格式。

    参数:
      - features_file: 预处理生成的 features 文件路径，形状 (N, window_size * 14)，
                       其中 14 = 8 (电机功率) + 6 (IMU 数据)
      - accel_file: numpy 文件路径，线性加速度标签，形状 (N, 3)
      - angular_accel_file: numpy 文件路径，角加速度标签，形状 (N, 3)
      - thrust_file: numpy 文件路径，推力标签，形状 (N, 6)
      - window_size: 时间窗口大小，与预处理时保持一致
      - clean_data: 若为 True，则在加载时清除异常样本
      - accel_threshold: 线性加速度的判断阈值
    """

    def __init__(self,
                 features_file: str,
                 accel_file: str,
                 angular_accel_file: str,
                 thrust_file: str,
                 window_size: int,
                 clean_data: bool = False,
                 accel_threshold: float = 1.0):
        super().__init__()
        # 加载数据，加入异常检测：检查 NaN 或 Inf
        self.features = np.load(features_file)
        self.accel = np.load(accel_file)
        self.angular_accel = np.load(angular_accel_file)
        self.thrust = np.load(thrust_file)

        # 检查是否存在 NaN/Inf（可以根据具体需求调整）
        if np.isnan(self.features).any() or np.isinf(self.features).any():
            logger.warning("Features 文件中包含 NaN 或 Inf")
        if np.isnan(self.accel).any() or np.isinf(self.accel).any():
            logger.warning("Accel 文件中包含 NaN 或 Inf")
        if np.isnan(self.angular_accel).any() or np.isinf(self.angular_accel).any():
            logger.warning("Angular accel 文件中包含 NaN 或 Inf")
        if np.isnan(self.thrust).any() or np.isinf(self.thrust).any():
            logger.warning("Thrust 文件中包含 NaN 或 Inf")

        self.window_size = window_size
        self.num_features = 14  # 8 (电机功率) + 6 (IMU 数据)
        self.num_samples = self.features.shape[0]

        if clean_data:
            self._clean_dataset(accel_threshold)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int):
        flat_features = self.features[index]
        expected_size = self.window_size * self.num_features

        # 检查样本是否为空，以及尺寸是否符合预期
        if flat_features.size != expected_size:
            raise ValueError(
                f"样本 {index} 的特征大小不匹配：期望 {expected_size}，实际 {flat_features.size}"
            )
        if np.isnan(flat_features).any() or np.isinf(flat_features).any():
            raise ValueError(f"样本 {index} 包含 NaN 或 Inf 值！")

        # 重塑为 (window_size, num_features)
        features_reshaped = flat_features.reshape(self.window_size, self.num_features)
        power_window = torch.from_numpy(features_reshaped[:, :8]).float()  # (window_size, 8)
        imu_window = torch.from_numpy(features_reshaped[:, 8:]).float()  # (window_size, 6)

        accel_linear = torch.from_numpy(self.accel[index]).float()  # (3,)
        accel_angular = torch.from_numpy(self.angular_accel[index]).float()  # (3,)
        accel_combined = torch.cat([accel_linear, accel_angular], dim=0)  # (6,)
        thrust_tensor = torch.from_numpy(self.thrust[index]).float()  # (6,)

        return {
            'power_window': power_window,
            'imu_window': imu_window,
            'accel': accel_combined,
            'thrust': thrust_tensor
        }

    def _clean_dataset(self, accel_threshold: float):
        """
        清理异常样本：
          - 如果某个样本的特征尺寸不正确或包含 NaN/Inf，则剔除。
          - 如果某个样本的电机8通道功率数据全为 0，
            且其对应的线性加速度（accel 标签前3个数值）的范数大于 accel_threshold，
            则认为该样本异常，需要移除。
        """
        valid_indices = []
        total = self.num_samples
        for i in range(total):
            flat_features = self.features[i]
            expected_size = self.window_size * self.num_features

            # 检查尺寸
            if flat_features.size != expected_size:
                logger.debug(f"样本 {i} 尺寸不匹配：期望 {expected_size}，实际 {flat_features.size}")
                continue

            # 检查 NaN 或 Inf
            if np.isnan(flat_features).any() or np.isinf(flat_features).any():
                logger.debug(f"样本 {i} 包含 NaN 或 Inf，剔除")
                continue

            features_reshaped = flat_features.reshape(self.window_size, self.num_features)
            power_window = features_reshaped[:, :8]
            # 如果功率数据全为 0，检测加速度是否异常
            if np.allclose(power_window, 0, atol=1e-6):
                accel_linear = self.accel[i]
                norm_linear = np.linalg.norm(accel_linear)
                if norm_linear > accel_threshold:
                    logger.debug(f"样本 {i} 异常：功率全零且加速度范数 {norm_linear} 超过阈值 {accel_threshold}")
                    continue

            valid_indices.append(i)

        num_removed = total - len(valid_indices)
        if num_removed > 0:
            print(f"清理异常数据：共 {num_removed} 个异常样本被剔除。")
        else:
            print("数据清理：未检测到异常样本。")

        # 根据有效索引过滤数据
        self.features = self.features[valid_indices]
        self.accel = self.accel[valid_indices]
        self.angular_accel = self.angular_accel[valid_indices]
        self.thrust = self.thrust[valid_indices]
        self.num_samples = len(valid_indices)


def _max_consecutive_zeros(arr: np.ndarray) -> int:
    """
    辅助函数：计算一维数组中最大连续零值个数。
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


def split_dataset(dataset, ratio=0.8):
    """按给定的 ratio 拆分数据集为训练集和验证集。"""
    n = len(dataset)
    n_train = int(n * ratio)
    n_val = n - n_train
    return torch.utils.data.random_split(dataset, [n_train, n_val])


def check_dataset(dataset: Dataset, accel_threshold: float = 1):
    """
    检查数据集中是否存在异常样本：
      - 若电机8通道功率数据全为 0，且对应的加速度（前3个值）的范数超过 accel_threshold，则视为异常。
    """
    problematic_samples = []
    num_samples = len(dataset)
    for i in range(num_samples):
        sample = dataset[i]
        power_data = sample['power_window'].numpy()  # shape: (window_size, 8)
        if np.allclose(power_data, 0, atol=1e-6):
            accel_data = sample['accel'].numpy()  # shape: (6,)
            norm_linear = np.linalg.norm(accel_data[:3])
            if norm_linear > accel_threshold:
                problematic_samples.append((i, norm_linear))
    if problematic_samples:
        print(f"警告: 检测到 {len(problematic_samples)} 个异常样本:")
       # for idx, norm_val in problematic_samples:
          #  print(f"  样本 {idx}: 线性加速度范数为 {norm_val:.3f} (阈值 {accel_threshold})")
    else:
        print("数据集检查通过，未发现异常样本。")
