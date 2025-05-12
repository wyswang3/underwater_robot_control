import os
import pandas as pd
import numpy as np
import logging
from openpyxl import load_workbook
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.integrate import trapezoid

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

############################
# 1) 推力数据处理
############################
def load_thrust_data(thrust_path):
    """
    从 Excel 读取单电机推力数据, 返回含 'power'(W) 和 'thrust_N'(N).
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

        thrust_df = pd.DataFrame(data, columns=['power','thrust_N'])
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
    拟合推力-功率幂律: F=alpha*P^beta
    """
    if thrust_df.empty:
        raise ValueError("thrust_df为空, 无法拟合推力模型")

    def func(p, alpha, beta):
        return alpha*(p**beta)

    try:
        popt, _ = curve_fit(func, thrust_df['power'], thrust_df['thrust_N'])
        logging.debug(f"[DEBUG] 拟合得到 popt={popt}")
        return popt
    except Exception as e:
        raise RuntimeError(f"推力模型拟合失败: {e}")

############################
# 2) 电机功率&IMU数据对齐
############################
def standardize_timestamp(ts_series: pd.Series) -> pd.Series:
    """
    将时间戳统一格式化为只保留秒后两位小数的字符串。
    例如 "2024-06-18 04:38:40.239" 会被格式化为 "2024-06-18 04:38:40.23"
    """
    dt_series = pd.to_datetime(ts_series, errors='coerce')
    return dt_series.apply(lambda dt: dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:22] if pd.notnull(dt) else None)


def extract_imu_time_in_seconds(timestamp_series: pd.Series) -> pd.Series:
    """
    解析 IMU 数据的 Timestamp（完整日期时间字符串），只提取分钟和秒部分，
    转换为总秒数。例如 "2024-06-18 04:31:33.50" 提取后为 31*60 + 33.50 = 1893.5 秒。
    """
    dt_series = pd.to_datetime(timestamp_series, format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
    dt_series = dt_series.fillna(pd.to_datetime(timestamp_series, format='%Y-%m-%d %H:%M:%S', errors='coerce'))
    return dt_series.apply(lambda dt: dt.minute * 60 + dt.second + dt.microsecond / 1e6 if pd.notnull(dt) else np.nan)


def extract_motor_time_in_seconds(timestamp_series: pd.Series) -> pd.Series:
    """
    解析电机数据的 Timestamp（完整日期时间字符串），只提取分钟和秒部分，
    转换为总秒数。例如 "2024-06-18 04:38:40.24" 转换后为 38*60 + 40.24 = 2320.24 秒。
    """
    dt_series = pd.to_datetime(timestamp_series, errors='coerce')
    return dt_series.apply(lambda dt: dt.minute * 60 + dt.second + dt.microsecond / 1e6 if pd.notnull(dt) else np.nan)


def align_one_to_one(power_df: pd.DataFrame, imu_df: pd.DataFrame, tolerance: float = 0.06,
                     time_col: str = 'time_in_seconds') -> pd.DataFrame:
    """
    对两个已按 time_in_seconds 排序的 DataFrame 进行一对一匹配：
      - 如果两侧当前行的时间差小于等于 tolerance，则认为匹配成功；
      - 匹配成功后，双方行都“消耗掉”，不再参与后续匹配；
      - 如果时间差超过 tolerance，则跳过较早的一侧。
    返回对齐后的 DataFrame，同时记录未匹配行的数量。
    """
    matches = []
    i, j = 0, 0
    n_power = len(power_df)
    n_imu = len(imu_df)
    while i < n_power and j < n_imu:
        p_time = power_df.iloc[i][time_col]
        imu_time = imu_df.iloc[j][time_col]
        diff = abs(p_time - imu_time)
        if diff <= tolerance:
            merged_row = {**power_df.iloc[i].to_dict(), **imu_df.iloc[j].to_dict()}
            matches.append(merged_row)
            i += 1
            j += 1
        else:
            if p_time < imu_time:
                i += 1
            else:
                j += 1
    matched_count = len(matches)
    unmatched_power = n_power - matched_count
    unmatched_imu = n_imu - matched_count
    logging.info(f"未匹配到的电机数据行数: {unmatched_power}, 未匹配到的 IMU 数据行数: {unmatched_imu}")
    return pd.DataFrame(matches)


def align_motor_imu(power_path: str, imu_path: str, tolerance: float = 0.06) -> pd.DataFrame:
    """
    读取电机功率 CSV 和 IMU CSV，并基于时间戳对齐（只比较分钟和秒部分）。
    使用自定义的一对一匹配算法：如果两行时间差小于等于 tolerance（单位秒），
    则视为匹配成功，并且已经匹配的数据不再参与后续匹配。
    在对齐前，先统一两个 CSV 文件中 Timestamp 的格式（只保留秒后两位小数）。
    返回对齐后的 DataFrame。
    """
    if not os.path.exists(power_path):
        raise FileNotFoundError(f"电机功率文件不存在: {os.path.abspath(power_path)}")
    if not os.path.exists(imu_path):
        raise FileNotFoundError(f"IMU 文件不存在: {os.path.abspath(imu_path)}")

    try:
        logging.debug(f"[DEBUG] 读取电机 CSV: {power_path}")
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

        # 统一两个数据集的 Timestamp 格式（只保留秒后两位小数）
        power_df['Timestamp'] = standardize_timestamp(power_df['Timestamp'])
        imu_df['Timestamp'] = standardize_timestamp(imu_df['Timestamp'])
        logging.debug(
            f"[DEBUG] 统一格式后，power_df Timestamp head:\n{power_df['Timestamp'].head(3).to_string(index=False)}")
        logging.debug(
            f"[DEBUG] 统一格式后，imu_df Timestamp head:\n{imu_df['Timestamp'].head(3).to_string(index=False)}")

        # 解析时间戳，转换为总秒数（只比较分钟、秒和微秒）
        logging.debug("[DEBUG] 解析电机数据时间戳为秒数...")
        power_df['time_in_seconds'] = extract_motor_time_in_seconds(power_df['Timestamp'])
        logging.debug(f"[DEBUG] 电机数据时间示例: {power_df['time_in_seconds'].head(3).tolist()}")

        logging.debug("[DEBUG] 提取 IMU 数据中的分钟和秒...")
        imu_df['time_in_seconds'] = extract_imu_time_in_seconds(imu_df['Timestamp'])
        logging.debug(f"[DEBUG] IMU 数据时间示例: {imu_df['time_in_seconds'].head(3).tolist()}")

        power_df.dropna(subset=['time_in_seconds'], inplace=True)
        imu_df.dropna(subset=['time_in_seconds'], inplace=True)
        logging.debug(f"[DEBUG] dropna后 power_df shape={power_df.shape}, imu_df shape={imu_df.shape}")

        power_df.sort_values('time_in_seconds', inplace=True)
        imu_df.sort_values('time_in_seconds', inplace=True)

        merged = align_one_to_one(power_df, imu_df, tolerance=tolerance, time_col='time_in_seconds')
        logging.debug(f"[DEBUG] 一对一匹配后 merged shape={merged.shape}")

        if merged.empty:
            error_msg = "对齐后的数据为空！"
            logging.error(error_msg)
            raise RuntimeError(error_msg)

        start_time = merged['time_in_seconds'].min()
        end_time = merged['time_in_seconds'].max()
        logging.info(f"对齐数据的起始时间戳: {start_time} 秒, 终止时间戳: {end_time} 秒")

        logging.debug(f"[DEBUG] merged head:\n{merged.head(5).to_string(index=False)}")

        return merged
    except Exception as e:
        raise RuntimeError(f"对齐电机功率与IMU数据失败: {e}")
############################
# 3) 常用函数
############################
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

def convert_accel_units(df, accel_cols=['AccX','AccY','AccZ']):
    # X 和 Y 轴直接乘以 9.80665 转换单位
    df['AccX'] = df['AccX'] * 9.80665
    df['AccY'] = df['AccY'] * 9.80665
    # Z 轴：先减去 1g，再乘以 9.80665
    df['AccZ'] = (df['AccZ']) * 9.80665
    return df


def load_thrust_allocation_matrix(matrix_path=None):
    if matrix_path and os.path.exists(matrix_path):
        if matrix_path.endswith('.npy'):
            T = np.load(matrix_path)
        else:
            T = pd.read_csv(matrix_path).values
    else:
        T = np.array([
            [0.7071, 0.7071, -0.7071, -0.7071,   0,     0,     0,     0],
            [-0.7071, 0.7071, -0.7071, 0.7071,    0,     0,     0,     0],
            [0,       0,       0,       0,       -1,     1,     1,    -1],
            [0,       0,       0,       0,        0.218, 0.218,-0.218,-0.218],
            [0,       0,       0,       0,        0.12, -0.12, 0.12, -0.12],
            [-0.1888, 0.1888,  0.1888, -0.1888,   0,     0,     0,     0]
        ])
    if T.shape != (6,8):
        raise ValueError(f"推力分配矩阵应为 6x8, 当前形状={T.shape}")
    return T.astype(np.float32)

def compute_angular_acceleration(gyro_data, dt):
    return np.gradient(gyro_data, dt, axis=0)

############################
# 4) 窗口切分
############################
def create_sequences(df, input_cols, label_cols_acc, label_cols_thrust,
                     motor_cols, thrust_matrix, window_size=9, step=1, dt=0.11):
    logging.debug(f"[DEBUG] 准备进行时间窗口切分: window_size={window_size}, dt={dt}, step={step}")
    feats, accs, thrs, vels, ang_accels = [], [], [], [], []

    for i in range(0, len(df) - window_size + 1, step):
        window = df.iloc[i:i+window_size]
        if window.isnull().any().any():
            continue

        feat = window[input_cols].values.flatten()
        a_label = window[label_cols_acc].iloc[-1].values

        motor_thrusts = [window[f'Motor{k}_Thrust'].iloc[-1] for k in range(1,9)]
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

############################
# 5) 主流程 full_pipeline
############################
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
        return alpha * (power ** beta)

    # 3) 对齐电机功率与 IMU
    logging.info("对齐电机功率与IMU数据...")
    merged = align_motor_imu(config['power_path'], config['imu_path'])
    if merged is None or merged.empty:
        raise RuntimeError("对齐后数据为空, 无法继续")
    logging.debug(f"[DEBUG] merged.shape={merged.shape}\n{merged.head(5).to_string(index=False)}")

    # 4) 加速度单位转换
    logging.info("加速度单位从 g->m/s^2...")
    merged = convert_accel_units(merged, ['AccX','AccY','AccZ'])

    # 5) 电机功率 -> 推力
    logging.info("插值得到电机推力...")
    motor_cols = [f'Motor{i}_Power' for i in range(1,9)]
    for col in motor_cols:
        thr_col = col.replace('Power', 'Thrust')
        merged[thr_col] = thrust_interp_func(merged[col])

    # 6) 低通滤波步骤暂时移除，因为数据已在前面过滤
    # logging.info("IMU数据低通滤波...")
    # merged = apply_lowpass_filter(merged, ['AccX','AccY','AccZ','AsX','AsY','AsZ'], 5, 50, 4)

    # 缺失值填充
    # 缺失值处理：检查合并后的数据，若存在缺失值则丢弃含缺失值的行
    n_missing = merged.isnull().sum().sum()
    if n_missing > 0:
        logging.warning(f"发现 {n_missing} 个缺失值，正在丢弃含缺失值的行...")
        merged = merged.dropna()
        if merged.empty:
            raise ValueError("丢弃缺失值后数据集为空，请检查原始数据。")
        else:
            logging.info(f"丢弃缺失值后，剩余数据量: {merged.shape[0]} 行。")
    else:
        logging.info("未发现缺失值。")

    # 7) 加载推力分配矩阵
    logging.info("载入推力分配矩阵(6x8)...")
    T = load_thrust_allocation_matrix(config.get('thrust_matrix_path', None))

    # 8) 时间窗口切分
    logging.info("开始窗口切分...")
    dt = config.get('dt', 0.11)
    input_cols = motor_cols + ['AccX','AccY','AccZ','AsX','AsY','AsZ']
    label_cols_acc = ['AccX','AccY','AccZ']
    label_cols_thrust = [m.replace('Power','Thrust') for m in motor_cols]

    feats, accs, thrs, vels, ang_accels = create_sequences(
        merged,
        input_cols,
        label_cols_acc,
        label_cols_thrust,
        motor_cols,
        T,
        window_size=config.get('window_size', 9),
        step=1,
        dt=dt
    )

    logging.info(f"窗口切分后: feats={feats.shape}, accs={accs.shape}, thrs={thrs.shape}, vels={vels.shape}, ang_acc={ang_accels.shape}")

    # 9) 保存处理结果 .npy
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
        'power_path' : os.path.join(PROJECT_ROOT, "data", "raw", "motor_data_0331.csv"),
        'imu_path'   : os.path.join(PROJECT_ROOT, "data", "raw", "imu_data_0331.csv"),
        # 'thrust_matrix_path': os.path.join(PROJECT_ROOT, "data", "raw", "thrust_allocation_matrix.csv"),
        'save_dir'   : os.path.join(PROJECT_ROOT, "data", "processed"),
        'window_size': 9,
        'dt': 0.5
    }

    feats, accs, thrs, vels, ang_accels, params = full_pipeline(config)
    alpha, beta = params
    logging.info(f"Features shape={feats.shape}, Acc shape={accs.shape}, Thrust shape={thrs.shape}")
    logging.info(f"Velocity shape={vels.shape}, AngularAccel shape={ang_accels.shape}")
    logging.info(f"Fitted thrust params: alpha={alpha:.4f}, beta={beta:.4f}")