import numpy as np
import torch
from torch.utils.data import Dataset, random_split
from utils.log_helper import setup_logging

# Initialize logging
setup_logging()
import logging
logger = logging.getLogger(__name__)

class PreprocessedDataset(Dataset):
    """
    加载预处理后的 numpy 数据文件，并将平铺的特征向量恢复为原始时间窗口格式。

    参数:
      - features_file: numpy 文件路径，shape (N, window_size*14)
      - accel_file: numpy 文件路径，shape (N,3)
      - angular_accel_file: numpy 文件路径，shape (N,3)
      - thrust_file: numpy 文件路径，shape (N,6)
      - window_size: 时间窗口大小
      - clean_data: 是否清理异常样本
      - accel_threshold: 加速度阈值，用于清理
    """
    def __init__(self,
                 features_file,
                 accel_file,
                 angular_accel_file,
                 thrust_file,
                 window_size,
                 clean_data=False,
                 accel_threshold=1.0):
        super(PreprocessedDataset, self).__init__()
        # 加载数据
        self.features = np.load(features_file)
        self.accel = np.load(accel_file)
        self.angular_accel = np.load(angular_accel_file)
        self.thrust = np.load(thrust_file)

        # 基本属性
        self.window_size = window_size
        self.num_features = 14  # 8 power + 6 IMU
        self.num_samples = self.features.shape[0]

        # 形状校验
        N = self.num_samples
        assert self.accel.shape == (N, 3), \
            f"accel shape mismatch: expected ({N},3), got {self.accel.shape}"
        assert self.angular_accel.shape == (N, 3), \
            f"angular_accel shape mismatch: expected ({N},3), got {self.angular_accel.shape}"
        assert self.thrust.shape == (N, 6), \
            f"thrust shape mismatch: expected ({N},6), got {self.thrust.shape}"

        # 合并标签
        self.labels = np.hstack((self.accel, self.angular_accel))  # (N,6)

        # 可选清洗
        if clean_data:
            self._clean_dataset(accel_threshold)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        flat = self.features[index]
        expected = self.window_size * self.num_features
        if flat.size != expected:
            raise ValueError(
                f"样本 {index} 尺寸不符: 期望 {expected}, 实际 {flat.size}"
            )
        # 重塑序列
        seq = flat.reshape(self.window_size, self.num_features)
        power = torch.from_numpy(seq[:, :8]).float()    # (W,8)
        imu   = torch.from_numpy(seq[:, 8:]).float()     # (W,6)
        accel = torch.from_numpy(self.labels[index]).float()  # (6,)
        thrust= torch.from_numpy(self.thrust[index]).float() # (6,)

        return {
            'power_window': power,
            'imu_window': imu,
            'accel': accel,
            'thrust': thrust,
        }

    def _clean_dataset(self, accel_threshold):
        """
        清理异常样本：
          - 包含 NaN/Inf
          - 功率全零且线性加速度过大
        """
        mask = np.ones(self.num_samples, dtype=bool)

        # 去除 NaN/Inf
        mask &= ~np.isnan(self.features).any(axis=1)
        mask &= ~np.isinf(self.features).any(axis=1)
        mask &= ~np.isnan(self.labels).any(axis=1)
        mask &= ~np.isinf(self.labels).any(axis=1)

        # 检测功率全零样本
        reshaped = self.features.reshape(-1, self.window_size, self.num_features)
        # np.isclose then all over axes
        is_zero = np.isclose(reshaped[:, :, :8], 0, atol=1e-6)
        power_zero = np.all(is_zero, axis=(1,2))
        # 高加速度样本
        lin_norm = np.linalg.norm(self.accel, axis=1)
        high_accel = lin_norm > accel_threshold
        # 标记需剔除样本
        mask &= ~(power_zero & high_accel)

        removed = np.count_nonzero(~mask)
        logger.info(f"清理异常样本: 移除 {removed}/{self.num_samples} 个样本")

        # 应用掩码更新数据
        self.features = self.features[mask]
        self.labels = self.labels[mask]
        self.accel = self.accel[mask]
        self.angular_accel = self.angular_accel[mask]
        self.thrust = self.thrust[mask]
        self.num_samples = self.features.shape[0]


def split_dataset(dataset, ratio=0.8):
    """按比例拆分为训练集和验证集"""
    N = len(dataset)
    train_n = int(N * ratio)
    val_n = N - train_n
    return random_split(dataset, [train_n, val_n])


def check_dataset(dataset, accel_threshold=1.0):
    """检查异常样本并打印警告"""
    problems = []
    for idx in range(len(dataset)):
        sample = dataset[idx]
        power = sample['power_window'].numpy()
        if np.allclose(power, 0, atol=1e-6):
            lin = sample['accel'].numpy()[:3]
            if np.linalg.norm(lin) > accel_threshold:
                problems.append(idx)
    if problems:
        logger.warning(f"检测到 {len(problems)} 个异常样本 (功率全零, 高加速度)")
    else:
        logger.info("数据集检查通过，无异常样本。")
