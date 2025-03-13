# utils/filters.py
import numpy as np
from scipy.signal import butter, lfilter, lfilter_zi


class AngleUnwrapper:
    """实时角度解缠类，维护状态"""

    def __init__(self):
        self.cumulative_offset = 0.0
        self.last_unwrapped = None

    def unwrap(self, angle):
        if self.last_unwrapped is None:
            self.last_unwrapped = angle
            return angle

        delta = angle - (self.last_unwrapped - self.cumulative_offset)
        if delta > np.pi:
            self.cumulative_offset -= 2 * np.pi
        elif delta < -np.pi:
            self.cumulative_offset += 2 * np.pi

        unwrapped = angle + self.cumulative_offset
        self.last_unwrapped = unwrapped
        return unwrapped
class GyroBiasKalmanFilter:
    def __init__(self, initial_bias, process_noise=1e-6, meas_noise=1e-4):
        """
        陀螺仪零偏卡尔曼滤波器
        :param initial_bias: 初始零偏 (shape: [3])
        """
        # 将初始偏置转换为列向量
        self.state = initial_bias.reshape(3, 1)
        self.P = np.eye(3)  # 状态协方差矩阵
        self.Q = process_noise * np.eye(3)  # 过程噪声
        self.R = meas_noise * np.eye(3)  # 测量噪声

    def predict(self):
        # 过程模型：假设零偏缓慢变化 x_{k+1} = x_k + w
        self.P += self.Q

    def update(self, gyro_meas, is_static=False):
        """
        使用当前角速度测量更新卡尔曼滤波器状态
        :param gyro_meas: 当前角速度测量 (shape: [3])
        :param is_static: 是否处于静止状态（若静止，则更新零偏）
        """
        if is_static:
            H = np.eye(3)
            z = gyro_meas.reshape(3, 1)
        else:
            # 如果非静止，则不更新零偏
            return

        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        y = z - H @ self.state
        self.state += K @ y
        self.P = (np.eye(3) - K @ H) @ self.P


class EnhancedIMUFilter:
    def __init__(self, static_start=0.0, static_duration=1.0, fs=100):
        self.static_start = static_start
        self.static_duration = static_duration
        self.fs = fs
        self.cutoff = 5.0
        self.b, self.a = butter(4, self.cutoff / (0.5 * fs), btype='low')
        self.filter_states = [None] * 3  # 各轴独立状态
        self.bias = np.zeros(3)
        self.angle_unwrapper = AngleUnwrapper()  # 角度解缠实例

    def _lowpass_filter(self, data, axis):
        """处理单个轴的滤波，维护独立状态"""
        if self.filter_states[axis] is None:
            zi = lfilter_zi(self.b, self.a) * data[0]
            filtered, zo = lfilter(self.b, self.a, data, zi=zi)
            self.filter_states[axis] = zo
        else:
            filtered, zo = lfilter(self.b, self.a, data, zi=self.filter_states[axis])
            self.filter_states[axis] = zo
        return filtered

    def process(self, timestamps, gyro_data, yaw_data):
        # Step 1: 实时解缠角度
        yaw_unwrapped = np.array([self.angle_unwrapper.unwrap(angle) for angle in yaw_data])

        # Step 2: 各轴独立低通滤波
        gyro_filtered = np.zeros_like(gyro_data)
        for axis in range(3):
            gyro_filtered[:, axis] = self._lowpass_filter(gyro_data[:, axis], axis)

        # Step 3: 静态阶段零偏估计
        static_mask = (timestamps >= self.static_start) & (timestamps <= self.static_start + self.static_duration)
        if np.sum(static_mask) > 10:
            self.bias = np.mean(gyro_filtered[static_mask], axis=0)

        # 零偏补偿
        gyro_compensated = gyro_filtered - self.bias
        return yaw_unwrapped, gyro_compensated, self.bias


class AdvancedIMUFilter(EnhancedIMUFilter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.kf = None

    def process(self, timestamps, gyro_data, yaw_data):
        yaw_unwrapped, gyro_compensated, _ = super().process(timestamps, gyro_data, yaw_data)

        if self.kf is None:
            self.kf = GyroBiasKalmanFilter(initial_bias=self.bias)

        corrected_gyro = []
        for i in range(len(timestamps)):
            is_static = self.static_start <= timestamps[i] <= (self.static_start + self.static_duration)
            self.kf.predict()
            # 使用滤波后的数据进行零偏更新
            self.kf.update(gyro_compensated[i], is_static=is_static)
            corrected = gyro_compensated[i] - self.kf.state.flatten()
            corrected_gyro.append(corrected)

        return yaw_unwrapped, np.array(corrected_gyro), self.kf.state.flatten()