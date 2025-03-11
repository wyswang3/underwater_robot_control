import os
import pandas as pd
import numpy as np
from openpyxl import load_workbook
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.signal import butter, lfilter
import logging
from scipy.integrate import trapezoid  # 推荐使用

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_thrust_data(thrust_path):
    """
    读取单电机推力-功率测试数据（Excel格式），返回包含 'power'(W) 和 'thrust_N'(牛顿) 的 DataFrame。
    假设 Excel 中列: [电流(A), power(W), thrust(KG), 实际推力(力臂1.3)]。
    """
    if not os.path.exists(thrust_path):
        raise FileNotFoundError(f"推力测试文件不存在: {os.path.abspath(thrust_path)}")
    if not thrust_path.endswith('.xlsx'):
        raise ValueError("推力测试文件必须是 .xlsx 格式")
    try:
        wb = load_workbook(thrust_path)
        ws = wb.active

        data = []
        for row in ws.iter_rows(min_row=2, values_only=True):
            current, power_w, thrust_kg, actual_thrust = row
            if power_w is None or thrust_kg is None:
                continue
            try:
                pw = float(power_w)
                tk = float(thrust_kg)
            except ValueError:
                continue
            thrust_n = tk * 9.80665  # 将千克力转换为牛顿
            data.append([pw, thrust_n])
        thrust_df = pd.DataFrame(data, columns=['power', 'thrust_N'])
        thrust_df.dropna(subset=['power', 'thrust_N'], inplace=True)
        return thrust_df
    except Exception as e:
        raise RuntimeError(f"加载推力测试文件失败: {thrust_path}。错误详情: {str(e)}")


def fit_thrust_model(thrust_df):
    """
    拟合推力-功率幂律模型 F = alpha * P^beta，返回 [alpha, beta]
    """

    def func(p, alpha, beta):
        return alpha * (p ** beta)

    try:
        popt, _ = curve_fit(func, thrust_df['power'], thrust_df['thrust_N'])
        return popt  # [alpha, beta]
    except Exception as e:
        raise RuntimeError(f"推力模型拟合失败。错误详情: {str(e)}")


def align_motor_imu(power_path, imu_path):
    """
    从 CSV 文件加载电机功率与IMU数据，按时间戳对齐（最近邻），返回合并后的 DataFrame。
    假设 Timestamp 格式: YYYY-MM-DD HH:MM:SS.fff
    """
    if not os.path.exists(power_path):
        raise FileNotFoundError(f"电机功率文件不存在: {os.path.abspath(power_path)}")
    if not os.path.exists(imu_path):
        raise FileNotFoundError(f"IMU 文件不存在: {os.path.abspath(imu_path)}")
    try:
        power_df = pd.read_csv(power_path)
        imu_df = pd.read_csv(imu_path)
        power_df['timestamp'] = pd.to_datetime(power_df['Timestamp'], format='%Y-%m-%d %H:%M:%S.%f')
        imu_df['timestamp'] = pd.to_datetime(imu_df['Timestamp'], format='%Y-%m-%d %H:%M:%S.%f')
        merged = pd.merge_asof(
            power_df.sort_values('timestamp'),
            imu_df.sort_values('timestamp'),
            on='timestamp',
            direction='nearest',
            tolerance=pd.Timedelta("100ms")
        )
        return merged
    except Exception as e:
        raise RuntimeError(f"对齐电机功率与IMU数据失败。错误详情: {str(e)}")


def butter_lowpass_filter(data, cutoff=5, fs=50, order=4):
    """
    使用 Butterworth 低通滤波器对 1D 数组 data 滤波。
    默认设置: cutoff=5Hz, fs=50Hz, order=4
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = lfilter(b, a, data)
    return y


def apply_lowpass_filter(df, cols, cutoff=5, fs=50, order=4):
    """
    对 df 的指定列的数据进行原地低通滤波。
    """
    for col in cols:
        df[col] = butter_lowpass_filter(df[col].values, cutoff=cutoff, fs=fs, order=order)
    return df


def convert_accel_units(df, accel_cols=['AccX', 'AccY', 'AccZ']):
    """
    将加速度从 g 转换为 m/s² (1g = 9.80665 m/s²)
    """
    df[accel_cols] = df[accel_cols] * 9.80665
    return df


def load_thrust_allocation_matrix(matrix_path=None):
    """
    加载推力分配矩阵 T (6x8)。
    如果 matrix_path 提供且存在，则从文件加载；否则返回默认的 T。
    默认 T 为：
        [[ 0.7071,  0.7071, -0.7071, -0.7071,  0,      0,      0,      0     ],
         [-0.7071,  0.7071, -0.7071,  0.7071,  0,      0,      0,      0     ],
         [ 0,       0,       0,       0,     -1,      1,      1,     -1     ],
         [ 0,       0,       0,       0,      0.218,  0.218, -0.218, -0.218 ],
         [ 0,       0,       0,       0,      0.12,  -0.12,   0.12,  -0.12  ],
         [-0.1888,  0.1888,  0.1888, -0.1888,   0,      0,      0,      0     ]]
    """
    if matrix_path is not None and os.path.exists(matrix_path):
        if matrix_path.endswith('.npy'):
            T = np.load(matrix_path)
        else:
            T = pd.read_csv(matrix_path).values
    else:
        T = np.array([
            [0.7071, 0.7071, -0.7071, -0.7071, 0, 0, 0, 0],
            [-0.7071, 0.7071, -0.7071, 0.7071, 0, 0, 0, 0],
            [0, 0, 0, 0, -1, 1, 1, -1],
            [0, 0, 0, 0, 0.218, 0.218, -0.218, -0.218],
            [0, 0, 0, 0, 0.12, -0.12, 0.12, -0.12],
            [-0.1888, 0.1888, 0.1888, -0.1888, 0, 0, 0, 0]
        ])
    if T.shape != (6, 8):
        raise ValueError(f"推力分配矩阵应为6x8，当前形状为 {T.shape}")
    return T.astype(np.float32)


def compute_angular_acceleration(gyro_data, dt):
    """
    通过陀螺仪数据计算角加速度 (rad/s²)，使用 np.gradient 方法
    :param gyro_data: 角速度数组 (N,3) [rad/s]
    :param dt: 采样间隔 (s)
    :return: 角加速度数组 (N,3)
    """
    return np.gradient(gyro_data, dt, axis=0)


def create_sequences(
        df,
        input_cols,
        label_cols_acc,
        label_cols_thrust,
        motor_cols,
        thrust_matrix,
        window_size=5,
        step=1,
        dt=0.1
):
    """
    将数据切成时间窗口样本，并生成:
      - features: 窗口内所有输入特征 (flatten后的向量)
      - accel_labels: 窗口最后一帧的加速度 (3,)
      - thrust_labels: 利用推力分配矩阵将8维电机推力转换为6维推力 (6,)
      - velocity_labels: 窗口最后时刻速度 (3,) （对加速度积分，起始速度=0）
      - angular_accel_labels: 窗口最后时刻角加速度 (3,) （利用陀螺仪数据计算）
    """
    features = []
    accel_labels = []
    thrust_labels = []
    velocity_labels = []
    angular_accel_labels = []

    for i in range(0, len(df) - window_size + 1, step):
        window = df.iloc[i: i + window_size]
        # 如果窗口内存在 NaN，则跳过此窗口
        if window.isnull().any().any():
            continue

        feat = window[input_cols].values.flatten()

        # 标签：窗口最后一帧加速度 (3,)
        a_label = window[label_cols_acc].iloc[-1].values

        # 推力：窗口最后一帧电机推力（由插值结果获得） -> 8维
        motor_thrusts = np.array([window[f'Motor{i}_Thrust'].iloc[-1] for i in range(1, 9)])
        # 通过推力分配矩阵转换为6维推力 tau (6,)
        tau = thrust_matrix @ motor_thrusts

        # 速度：对窗口内加速度积分 (使用梯形积分)
        time_points = np.arange(window_size) * dt
        acc_window = window[label_cols_acc].values  # (window_size, 3)
        vel_window = trapezoid(acc_window, x=time_points, axis=0)  # (3,)

        # 角加速度：利用窗口内陀螺仪数据计算 (假设陀螺仪列名为 ["AsX","AsY","AsZ"])
        gyro_cols = ["AsX", "AsY", "AsZ"]
        gyro_window = window[gyro_cols].values  # (window_size, 3)
        ang_accel = compute_angular_acceleration(gyro_window, dt)  # (window_size, 3)
        ang_accel_label = ang_accel[-1]  # (3,)

        features.append(feat)
        accel_labels.append(a_label)
        thrust_labels.append(tau)
        velocity_labels.append(vel_window)
        angular_accel_labels.append(ang_accel_label)

    return (
        np.array(features),           # (num_samples, window_size * len(input_cols))
        np.array(accel_labels),         # (num_samples, 3)
        np.array(thrust_labels),        # (num_samples, 6)
        np.array(velocity_labels),      # (num_samples, 3)
        np.array(angular_accel_labels)  # (num_samples, 3)
    )


def full_pipeline(config):
    """
    完整预处理流程:
      1) 读取推力测试Excel, 拟合幂律模型
      2) 对齐电机功率CSV与IMU CSV
      3) 将加速度从 g 转 m/s²
      4) 用插值函数将 MotorX_Power 转为 MotorX_Thrust
      5) 对指定IMU列低通滤波
      6) 清洗数据：填充或删除缺失值
      7) 切分时间窗口，生成 features, accel_labels, thrust_labels, velocity_labels, angular_accel_labels
      8) 保存 .npy 文件
      返回: features, accel_labels, thrust_labels, velocity_labels, angular_accel_labels, thrust_params
    """
    logging.info("开始预处理流程...")

    # 1) 检查文件
    required_files = [config['thrust_path'], config['power_path'], config['imu_path']]
    for f in required_files:
        if not os.path.exists(f):
            raise FileNotFoundError(f"文件不存在: {os.path.abspath(f)}")

    # 2) 加载并拟合推力测试数据
    logging.info("加载推力测试数据...")
    thrust_df = load_thrust_data(config['thrust_path'])
    thrust_params = fit_thrust_model(thrust_df)  # [alpha, beta]

    # 3) 对齐电机功率与IMU数据
    logging.info("对齐电机功率与IMU数据...")
    merged = align_motor_imu(config['power_path'], config['imu_path'])

    # 4) 加速度单位转换: 将 [AccX,AccY,AccZ] 从 g 转为 m/s²
    logging.info("转换加速度单位 (g->m/s²)...")
    merged = convert_accel_units(merged, accel_cols=["AccX", "AccY", "AccZ"])

    # 5) 用插值函数将 MotorX_Power 转为 MotorX_Thrust
    logging.info("插值计算推力...")
    motor_cols = [f"Motor{i}_Power" for i in range(1, 9)]
    for col in motor_cols:
        thr_col = col.replace("Power", "Thrust")
        merged[thr_col] = config['thrust_interp'](merged[col])

    # 6) 对 IMU 列低通滤波 (示例: AccX,AccY,AccZ,AsX,AsY,AsZ)
    logging.info("对IMU数据进行低通滤波...")
    merged = apply_lowpass_filter(
        merged,
        cols=["AccX", "AccY", "AccZ", "AsX", "AsY", "AsZ"],
        cutoff=5, fs=50, order=4
    )

    # 6.5) 清洗数据：如果存在缺失值，则采用前向填充，后向填充补全
    if merged.isnull().any().any():
        logging.warning("检测到缺失值，进行前向填充...")
        merged.fillna(method='ffill', inplace=True)
        merged.fillna(method='bfill', inplace=True)
        if merged.isnull().any().any():
            logging.error("数据中仍存在缺失值，请检查数据源。")
            raise ValueError("数据清洗后仍存在缺失值。")

    # 7) 加载推力分配矩阵
    logging.info("加载推力分配矩阵...")
    thrust_matrix = load_thrust_allocation_matrix(config.get('thrust_matrix_path', None))

    # 8) 切分时间窗口
    logging.info("切分时间窗口...")
    input_cols = motor_cols + ["AccX", "AccY", "AccZ", "AsX", "AsY", "AsZ"]
    label_cols_acc = ["AccX", "AccY", "AccZ"]
    label_cols_thrust = [col.replace("Power", "Thrust") for col in motor_cols]
    dt = config.get('dt', 0.1)

    feats, accs, thrs, vels, ang_accels = create_sequences(
        merged,
        input_cols,
        label_cols_acc,
        label_cols_thrust,
        motor_cols,
        thrust_matrix,
        window_size=config.get('window_size', 5),
        step=1,
        dt=dt
    )

    # 9) 保存预处理结果
    logging.info("保存预处理结果...")
    save_dir = config['save_dir']
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, "train_features.npy"), feats)
    np.save(os.path.join(save_dir, "train_accel_labels.npy"), accs)
    np.save(os.path.join(save_dir, "train_thrust_labels.npy"), thrs)
    np.save(os.path.join(save_dir, "train_velocity_labels.npy"), vels)
    np.save(os.path.join(save_dir, "train_angular_accel_labels.npy"), ang_accels)

    logging.info("预处理完成。")
    return feats, accs, thrs, vels, ang_accels, thrust_params


if __name__ == "__main__":
    from scipy.interpolate import interp1d

    # 获取项目根目录 (假设本文件在 underwater_robot_control/utils/ )
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # 构造推力测试数据路径
    thrust_path = os.path.join(PROJECT_ROOT, "data", "raw", "thrust_test.xlsx")

    logging.info("加载推力测试数据并生成插值函数...")
    local_thrust_df = load_thrust_data(thrust_path)
    p_vals = local_thrust_df['power'].values
    f_vals = local_thrust_df['thrust_N'].values
    local_interp = interp1d(p_vals, f_vals, kind='linear', fill_value='extrapolate')

    config = {
        'thrust_path': thrust_path,
        'power_path': os.path.join(PROJECT_ROOT, "data", "raw", "new_motor_power_data_0112.csv"),
        'imu_path': os.path.join(PROJECT_ROOT, "data", "raw", "imu_data0112_cleaned.csv"),
        'thrust_matrix_path': os.path.join(PROJECT_ROOT, "data", "raw", "thrust_allocation_matrix.csv"),
        'save_dir': os.path.join(PROJECT_ROOT, "data", "processed"),
        'window_size': 5,
        'dt': 0.1,
        'thrust_interp': local_interp
    }

    feats, accs, thrs, vels, ang_accels, params = full_pipeline(config)
    logging.info("预处理结果:")
    logging.info(f"Features shape: {feats.shape}")
    logging.info(f"Accel labels shape: {accs.shape}")
    logging.info(f"Thrust labels shape: {thrs.shape}")
    logging.info(f"Velocity labels shape: {vels.shape}")
    logging.info(f"Angular accel labels shape: {ang_accels.shape}")
    logging.info(f"Fitted thrust params [alpha, beta]: {params}")
