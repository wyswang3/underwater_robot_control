# utils/filters.py
import numpy as np
from scipy.signal import butter, lfilter


def unwrap_angles(angles):
    """
    增量式角度解缠算法（修正版）
    改进点：基于前一个解缠后的角度计算跳变
    """
    unwrapped = np.zeros_like(angles)
    unwrapped[0] = angles[0]
    cumulative_offset = 0  # 累积的偏移量

    for i in range(1, len(angles)):
        # 计算当前原始角度与前一个解缠后角度的差值
        delta = angles[i] - (unwrapped[i - 1] - cumulative_offset)

        # 检测是否需要补偿2π
        if delta > np.pi:
            cumulative_offset -= 2 * np.pi
        elif delta < -np.pi:
            cumulative_offset += 2 * np.pi

        # 应用累积偏移
        unwrapped[i] = angles[i] + cumulative_offset

    return unwrapped


class EnhancedIMUFilter:
    def __init__(self, static_start=0.0, static_duration=1.0, fs=100):
        """
        增强型IMU滤波器
        :param static_start: 静止阶段开始时间(s)
        :param static_duration: 静止阶段持续时间(s)
        :param fs: 采样频率(Hz)
        """
        self.static_start = static_start
        self.static_duration = static_duration
        self.fs = fs

        # 初始化状态
        self.cumulative_offset = 0.0
        self.last_unwrapped = None
        self.bias = np.zeros(3)

        # 低通滤波器参数（截止频率5Hz）
        self.cutoff = 5.0
        self.b, self.a = butter(4, self.cutoff / (0.5 * fs), btype='low')
        self.filter_state = None  # 滤波器状态保持

    def _lowpass_filter(self, data):
        """实时低通滤波（保持滤波器状态）"""
        if self.filter_state is None:
            filtered, self.filter_state = lfilter(self.b, self.a, data, zi=np.zeros(max(len(self.b), len(self.a)) - 1))
        else:
            filtered, self.filter_state = lfilter(self.b, self.a, data, zi=self.filter_state)
        return filtered

    def process(self, timestamps, gyro_data, yaw_data):
        """
        完整IMU数据处理流程
        :param timestamps: 时间戳数组 (shape: [N])
        :param gyro_data: 原始角速度数据 (shape: [N, 3])
        :param yaw_data: 原始航向角数据 (shape: [N])
        :return: 处理后的航向角、角速度、零偏估计
        """
        # 输入校验
        assert len(timestamps) == len(gyro_data) == len(yaw_data), "输入数据长度不一致"

        # Step 1: 航向角解缠
        yaw_unwrapped = unwrap_angles(yaw_data)

        # Step 2: 低通滤波（角速度）
        gyro_filtered = np.apply_along_axis(self._lowpass_filter, 0, gyro_data)

        # Step 3: 零偏估计（静态阶段）
        static_mask = (timestamps >= self.static_start) & \
                      (timestamps <= self.static_start + self.static_duration)
        if np.sum(static_mask) > 10:  # 至少有10个采样点
            self.bias = np.mean(gyro_filtered[static_mask], axis=0)

        # Step 4: 零偏补偿
        gyro_compensated = gyro_filtered - self.bias

        # Step 5: 动态零偏修正（卡尔曼滤波）
        # 此处可扩展为完整卡尔曼滤波器实现（见下方补充代码）

        return yaw_unwrapped, gyro_compensated, self.bias


# ---------------------------
# 补充代码：完整的卡尔曼滤波实现
# ---------------------------
class GyroBiasKalmanFilter:
    def __init__(self, initial_bias, process_noise=1e-6, meas_noise=1e-4):
        """
        陀螺仪零偏卡尔曼滤波器
        :param initial_bias: 初始零偏估计 (shape: [3])
        """
        # 状态向量：[bias_x, bias_y, bias_z]
        self.state = initial_bias.reshape(3, 1)
        self.P = np.eye(3)  # 协方差矩阵
        self.Q = process_noise * np.eye(3)  # 过程噪声
        self.R = meas_noise * np.eye(3)  # 测量噪声

    def predict(self):
        # 过程模型：假设零偏缓慢变化 x_{k+1} = x_k + w
        self.P += self.Q

    def update(self, gyro_meas, is_static=False):
        """
        :param gyro_meas: 当前角速度测量值 (shape: [3])
        :param is_static: 是否处于静止状态（强制零偏不变）
        """
        if is_static:
            # 静止时观测模型：z = x + v，真实角速度应为0
            H = np.eye(3)
            z = gyro_meas.reshape(3, 1)  # 测量值=零偏+噪声
        else:
            # 运动时不更新零偏（假设无其他传感器数据）
            return

        # 卡尔曼增益
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)

        # 状态更新
        y = z - H @ self.state
        self.state += K @ y
        self.P = (np.eye(3) - K @ H) @ self.P


# ---------------------------
# 改进后的EnhancedIMUFilter（集成卡尔曼滤波）
# ---------------------------
class AdvancedIMUFilter(EnhancedIMUFilter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.kf = None

    def process(self, timestamps, gyro_data, yaw_data):
        yaw_unwrapped, gyro_compensated, bias = super().process(timestamps, gyro_data, yaw_data)

        # 初始化卡尔曼滤波器
        if self.kf is None:
            self.kf = GyroBiasKalmanFilter(initial_bias=bias)

        # 处理每一帧数据
        dt = 1.0 / self.fs
        corrected_gyro = []
        for i in range(len(timestamps)):
            is_static = self.static_start <= timestamps[i] <= (self.static_start + self.static_duration)

            self.kf.predict()
            self.kf.update(gyro_data[i], is_static=is_static)

            # 应用最新零偏估计
            current_bias = self.kf.state.flatten()
            corrected = gyro_data[i] - current_bias
            corrected_gyro.append(corrected)

        return yaw_unwrapped, np.array(corrected_gyro), self.kf.state.flatten()