import os
import pandas as pd
import numpy as np
import logging
from openpyxl import load_workbook
from scipy.optimize import curve_fit
from scipy.integrate import trapezoid
import matplotlib.pyplot as plt
import matplotlib

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

#############################################
# 1) 推力数据处理
#############################################
def load_thrust_data(thrust_path):
    """
    从 Excel 读取单电机推力数据, 返回 DataFrame，
    包含 'power'(W) 和 'thrust_N'(N) 两列。
    """
    if not os.path.exists(thrust_path):
        raise FileNotFoundError(f"推力测试文件不存在: {os.path.abspath(thrust_path)}")
    if not thrust_path.endswith('.xlsx'):
        raise ValueError("推力测试文件必须是 .xlsx 格式")
    try:
        logging.debug(f"[DEBUG] 开始读取 Excel: {thrust_path}")
        wb = load_workbook(thrust_path)
        ws = wb.active
        data = []
        for row in ws.iter_rows(min_row=2, values_only=True):
            if not row or len(row) < 3:
                continue
            power_w = row[1]
            thrust_kg = row[2]
            if power_w is None or thrust_kg is None:
                continue
            try:
                pw = float(power_w)
                tk = float(thrust_kg)
            except ValueError:
                continue
            thrust_n = tk * 9.80665
            data.append([pw, thrust_n])
        thrust_df = pd.DataFrame(data, columns=['power', 'thrust_N'])
        logging.debug(f"[DEBUG] thrust_df.shape={thrust_df.shape}")
        if not thrust_df.empty:
            logging.debug(f"[DEBUG] thrust_df head:\n{thrust_df.head().to_string(index=False)}")
        else:
            logging.warning("[WARN] 读取后 thrust_df 为空, 可能导致后续拟合失败.")
        return thrust_df
    except Exception as e:
        raise RuntimeError(f"加载推力测试文件失败: {thrust_path}, 错误详情: {e}")

def fit_thrust_model(thrust_df):
    """
    拟合推力-功率幂律: F = alpha * P^beta 。
    """
    if thrust_df.empty:
        raise ValueError("thrust_df为空, 无法拟合推力模型")
    def func(p, alpha, beta):
        return alpha * (p ** beta)
    try:
        popt, _ = curve_fit(func, thrust_df['power'], thrust_df['thrust_N'])
        logging.debug(f"[DEBUG] 拟合得到 popt={popt}")
        return popt
    except Exception as e:
        raise RuntimeError(f"推力模型拟合失败: {e}")

#############################################
# 2) 电机功率 & IMU 数据对齐
#############################################
def parse_motor_timestamp(ts_str):
    """
    解析电机数据的 Timestamp 字符串（完整日期时间格式），
    返回该时间的分钟、秒和微秒转换成的总秒数（忽略小时和日期）。
    例如 "2024-06-18 04:38:40.241276" -> 38*60 + 40.241276 = 2320.241276 秒。
    """
    try:
        dt = pd.to_datetime(ts_str, errors='coerce')
        if pd.isnull(dt):
            return np.nan
        return dt.minute * 60 + dt.second + dt.microsecond / 1e6
    except Exception as e:
        return np.nan

def extract_imu_time_in_seconds(timestamp_series):
    """
    解析 IMU 数据的 Timestamp（完整日期时间字符串），
    只提取分钟和秒部分，转换为总秒数。
    例如 "2024-06-18 04:38:40.453737" -> 38*60 + 40.453737 = 2320.453737 秒。
    """
    dt_series = pd.to_datetime(timestamp_series, format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
    dt_series = dt_series.fillna(pd.to_datetime(timestamp_series, format='%Y-%m-%d %H:%M:%S', errors='coerce'))
    return dt_series.apply(lambda dt: dt.minute * 60 + dt.second + dt.microsecond / 1e6 if pd.notnull(dt) else np.nan)

def align_motor_imu(power_path, imu_path):
    """
    读取电机功率 CSV 和 IMU CSV，并基于时间戳对齐。
    由于时间戳具体到微秒，而传感器数据难以实现微秒级对齐，
    只要两组数据的时间戳差值低于0.1秒，就认为它们在同一时刻采集。
    电机数据假设 Timestamp 为完整日期时间字符串；
    IMU 数据同样为完整日期时间字符串。
    返回对齐后的 DataFrame。
    """
    if not os.path.exists(power_path):
        raise FileNotFoundError(f"电机功率文件不存在: {os.path.abspath(power_path)}")
    if not os.path.exists(imu_path):
        raise FileNotFoundError(f"IMU 文件不存在: {os.path.abspath(imu_path)}")
    try:
        logging.debug(f"[DEBUG] 读取电机功率 CSV: {power_path}")
        power_df = pd.read_csv(power_path)
        logging.debug(f"[DEBUG] power_df 原始 shape={power_df.shape}")
        logging.debug(f"[DEBUG] power_df head:\n{power_df.head(3).to_string(index=False)}")
        logging.debug(f"[DEBUG] 读取 IMU CSV: {imu_path}")
        imu_df = pd.read_csv(imu_path)
        logging.debug(f"[DEBUG] imu_df 原始 shape={imu_df.shape}")
        logging.debug(f"[DEBUG] imu_df head:\n{imu_df.head(3).to_string(index=False)}")
        if 'Timestamp' not in power_df.columns:
            raise ValueError("电机 CSV 中缺少 'Timestamp' 列")
        if 'Timestamp' not in imu_df.columns:
            raise ValueError("IMU CSV 中缺少 'Timestamp' 列")
        logging.debug("[DEBUG] 解析电机数据时间戳为秒数...")
        power_df['time_in_seconds'] = power_df['Timestamp'].apply(parse_motor_timestamp)
        logging.debug(f"[DEBUG] 电机数据时间示例: {power_df['time_in_seconds'].head(3).tolist()}")
        logging.debug("[DEBUG] 提取 IMU 数据中的分钟和秒...")
        imu_df['time_in_seconds'] = extract_imu_time_in_seconds(imu_df['Timestamp'])
        logging.debug(f"[DEBUG] IMU 数据时间示例: {imu_df['time_in_seconds'].head(3).tolist()}")
        power_df.dropna(subset=['time_in_seconds'], inplace=True)
        imu_df.dropna(subset=['time_in_seconds'], inplace=True)
        logging.debug(f"[DEBUG] dropna后 power_df shape={power_df.shape}, imu_df shape={imu_df.shape}")
        power_df.sort_values('time_in_seconds', inplace=True)
        imu_df.sort_values('time_in_seconds', inplace=True)
        merged = pd.merge_asof(
            power_df, imu_df,
            on='time_in_seconds',
            direction='nearest',
            tolerance=0.1
        )
        logging.debug(f"[DEBUG] merged shape={merged.shape}")
        if merged.empty:
            raise RuntimeError("对齐后数据为空, 无法继续")
        logging.debug(f"[DEBUG] merged head:\n{merged.head(5).to_string(index=False)}")
        return merged
    except Exception as e:
        raise RuntimeError(f"对齐电机功率与IMU数据失败: {e}")

#############################################
# 3) 常用函数
#############################################
def butter_lowpass_filter(data, cutoff=5, fs=50, order=4):
    from scipy.signal import butter, lfilter
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return lfilter(b, a, data)

def apply_lowpass_filter(df, cols, cutoff=5, fs=50, order=4):
    for col in cols:
        logging.debug(f"[DEBUG] 对 {col} 做低通滤波.")
        df[col] = butter_lowpass_filter(df[col].values, cutoff=cutoff, fs=fs, order=order)
    return df

def convert_accel_units(df, accel_cols=['AccX', 'AccY', 'AccZ']):
    """
    将加速度单位从 g 转换为 m/s²。
    对X、Y轴直接转换；对Z轴先减去1（补偿重力），再转换。
    """
    df[['AccX', 'AccY']] = df[['AccX', 'AccY']] * 9.80665
    df['AccZ'] = (df['AccZ'] - 1) * 9.80665
    return df

def load_thrust_allocation_matrix(matrix_path=None):
    if matrix_path and os.path.exists(matrix_path):
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
        raise ValueError(f"推力分配矩阵应为 6x8, 当前形状={T.shape}")
    return T.astype(np.float32)

def compute_angular_acceleration(gyro_data, dt):
    return np.gradient(gyro_data, dt, axis=0)

#############################################
# 4) 窗口切分
#############################################
def create_sequences(df, input_cols, label_cols_acc, label_cols_thrust,
                     motor_cols, thrust_matrix, window_size=5, step=1, dt=0.1):
    logging.debug(f"[DEBUG] 准备进行时间窗口切分: window_size={window_size}, dt={dt}, step={step}")
    feats, accs, thrs, vels, ang_accels = [], [], [], [], []

    for i in range(0, len(df) - window_size + 1, step):
        window = df.iloc[i:i + window_size]
        if window.isnull().any().any():
            continue

        feat = window[input_cols].values.flatten()
        a_label = window[label_cols_acc].iloc[-1].values

        # 使用电机功率列名，要求已经被重命名为 MotorX_Power
        motor_thrusts = [window[f'Motor{k}_Power'].iloc[-1] for k in range(1, 9)]
        motor_thrusts = np.array(motor_thrusts, dtype=np.float32)
        tau = thrust_matrix @ motor_thrusts

        time_points = np.arange(window_size) * dt
        acc_window = window[label_cols_acc].values
        vel_window = trapezoid(acc_window, x=time_points, axis=0)

        gyro_cols = ['AsX', 'AsY', 'AsZ']
        gyro_window = window[gyro_cols].values
        ang_accel = compute_angular_acceleration(gyro_window, dt)
        ang_accel_label = ang_accel[-1]

        feats.append(feat)
        accs.append(a_label)
        thrs.append(tau)
        vels.append(vel_window)
        ang_accels.append(ang_accel_label)

    logging.debug(f"[DEBUG] create_sequences 结果: feats={len(feats)}, accs={len(accs)}")
    return (
        np.array(feats),
        np.array(accs),
        np.array(thrs),
        np.array(vels),
        np.array(ang_accels)
    )

#############################################
# 5) 主流程 full_pipeline
#############################################
def full_pipeline(config):
    logging.info("开始预处理流程...")

    # 1) 检查必需文件
    req_files = [config['thrust_path'], config['power_path'], config['imu_path']]
    for f in req_files:
        if not os.path.exists(f):
            raise FileNotFoundError(f"文件不存在: {os.path.abspath(f)}")

    # 2) 加载并拟合推力数据
    logging.info("加载并拟合推力测试数据...")
    thrust_df = load_thrust_data(config['thrust_path'])
    popt = fit_thrust_model(thrust_df)
    if popt is None or len(popt) < 2:
        raise RuntimeError("拟合推力模型失败: popt无效 (None 或长度<2)")
    alpha, beta = popt
    logging.info(f"拟合得 alpha={alpha:.4f}, beta={beta:.4f}")

    def thrust_interp_func(power):
        # 根据理论 F = C * P^(2/3)；此处beta应接近2/3
        return alpha * (power ** beta)

    # 3) 对齐电机功率与 IMU 数据
    logging.info("对齐电机功率与IMU数据...")
    merged = align_motor_imu(config['power_path'], config['imu_path'])
    if merged is None or merged.empty:
        raise RuntimeError("对齐后数据为空, 无法继续")
    logging.debug(f"[DEBUG] merged after rename columns:\n{merged.head(3).to_string(index=False)}")

    # 4) 加速度单位转换：将加速度从 g 转换为 m/s²；对Z轴先减去1补偿重力
    logging.info("加速度单位从 g->m/s²...")
    merged = convert_accel_units(merged, ['AccX', 'AccY', 'AccZ'])

    # 5) 电机功率 -> 推力转换
    logging.info("插值得到电机推力...")
    # 假设原始电机数据列已重命名为 Motor1_Power ~ Motor8_Power
    motor_cols = [f'Motor{i}_Power' for i in range(1, 9)]
    for col in motor_cols:
        merged[col] = thrust_interp_func(merged[col])

    # 5.1 根据补偿后 AccZ 判断上浮或下潜，调整电机5~8的推力符号
    # 下潜：AccZ > 0 -> Motor5, Motor8 正转 (+1); Motor6, Motor7 反转 (-1)
    # 上浮：AccZ < 0 -> Motor5, Motor8 反转 (-1); Motor6, Motor7 正转 (+1)
    diving_mask = merged['AccZ'] > 0
    ascending_mask = merged['AccZ'] < 0
    merged.loc[diving_mask, 'Motor6_Power'] *= -1
    merged.loc[diving_mask, 'Motor7_Power'] *= -1
    merged.loc[ascending_mask, 'Motor5_Power'] *= -1
    merged.loc[ascending_mask, 'Motor8_Power'] *= -1
    logging.info("根据加速度调整电机推力方向完成。")

    # 6) 低通滤波步骤暂时移除，因为数据已预先滤波
    # 7) 缺失值填充
    # 检查缺失值
    n_missing = merged.isnull().sum().sum()
    if n_missing > 0:
        logging.warning(f"发现 {n_missing} 个缺失值，进行前向填充和后向填充...")
        merged.ffill(inplace=True)
        merged.bfill(inplace=True)
        n_missing_after = merged.isnull().sum().sum()
        if n_missing_after > 0:
            logging.error(f"填充后仍存在 {n_missing_after} 个缺失值, 请检查源数据.")
            raise ValueError("仍有NaN.")
        else:
            logging.info("缺失值已成功填充。")
    else:
        logging.info("未发现缺失值。")

    # 8) 加载推力分配矩阵 (6x8)
    logging.info("载入推力分配矩阵(6x8)...")
    T = load_thrust_allocation_matrix(config.get('thrust_matrix_path', None))

    # 9) 时间窗口切分
    logging.info("开始窗口切分...")
    dt = config.get('dt', 0.1)
    input_cols = motor_cols + ['AccX', 'AccY', 'AccZ', 'AsX', 'AsY', 'AsZ']
    label_cols_acc = ['AccX', 'AccY', 'AccZ']
    label_cols_thrust = motor_cols  # 使用调整后的电机功率列
    feats, accs, thrs, vels, ang_accels = create_sequences(
        merged,
        input_cols,
        label_cols_acc,
        label_cols_thrust,
        motor_cols,
        T,
        window_size=config.get('window_size', 5),
        step=1,
        dt=dt
    )

    logging.info(f"窗口切分后: feats={feats.shape}, accs={accs.shape}, thrs={thrs.shape}, vels={vels.shape}, ang_acc={ang_accels.shape}")

    # 10) 保存处理结果 .npy
    logging.info("保存处理结果 .npy...")
    save_dir = config['save_dir']
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, "train_features.npy"), feats)
    np.save(os.path.join(save_dir, "train_accel_labels.npy"), accs)
    np.save(os.path.join(save_dir, "train_thrust_labels.npy"), thrs)
    np.save(os.path.join(save_dir, "train_velocity_labels.npy"), vels)
    np.save(os.path.join(save_dir, "train_angular_accel_labels.npy"), ang_accels)

    logging.info("预处理完成.")
    return feats, accs, thrs, vels, ang_accels, (alpha, beta)


############################
# 测试入口
############################
if __name__=="__main__":
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config = {
        'thrust_path': os.path.join(PROJECT_ROOT, "data", "raw", "thrust_test.xlsx"),
        'power_path': os.path.join(PROJECT_ROOT, "data", "raw", "table1_motor_data0331.csv"),
        'imu_path': os.path.join(PROJECT_ROOT, "data", "raw", "imu_data_downsampled_9hz0331.csv"),
        # 'thrust_matrix_path': os.path.join(PROJECT_ROOT, "data", "raw", "thrust_allocation_matrix.csv"),
        'save_dir': os.path.join(PROJECT_ROOT, "data", "processed"),
        'window_size': 5,
        'dt': 0.5
    }

    feats, accs, thrs, vels, ang_accels, params = full_pipeline(config)
    alpha, beta = params
    logging.info(f"Features shape={feats.shape}, Acc shape={accs.shape}, Thrust shape={thrs.shape}")
    logging.info(f"Velocity shape={vels.shape}, AngularAccel shape={ang_accels.shape}")
    logging.info(f"Fitted thrust params: alpha={alpha:.4f}, beta={beta:.4f}")
