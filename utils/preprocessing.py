import os
import logging
import pandas as pd
import numpy as np
from openpyxl import load_workbook
from scipy.optimize import curve_fit
from scipy.integrate import trapezoid
import matplotlib.pyplot as plt

# 设置日志级别，正式运行建议 INFO，调试时可设置为 DEBUG
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


#############################################
# 1. 推力测试数据读取与拟合
#############################################
def load_thrust_data(thrust_path):
    """
    从 Excel 读取推力测试数据，返回 DataFrame，
    包含 'power' (W) 和 'thrust_N' (N) 两列。
    """
    if not os.path.exists(thrust_path):
        raise FileNotFoundError(f"推力测试文件不存在: {os.path.abspath(thrust_path)}")
    if not thrust_path.endswith(".xlsx"):
        raise ValueError("推力测试文件必须是 .xlsx 格式")

    data = []
    try:
        wb = load_workbook(thrust_path)
        ws = wb.active
        for row in ws.iter_rows(min_row=2, values_only=True):
            if len(row) < 3:
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
            # 转换单位：kgf -> N
            thrust_n = tk * 9.80665
            data.append([pw, thrust_n])
        df = pd.DataFrame(data, columns=["power", "thrust_N"])
        logging.debug(f"load_thrust_data: 读取数据 shape={df.shape}")
        return df
    except Exception as e:
        raise RuntimeError(f"加载推力测试文件失败: {thrust_path}, 错误: {e}")


def fit_thrust_model(thrust_df):
    """
    拟合推力-功率幂律: F = alpha * P^beta，
    返回拟合参数 (alpha, beta)。
    """
    if thrust_df.empty:
        raise ValueError("推力测试数据为空，无法拟合")

    def func(p, alpha, beta):
        return alpha * (p ** beta)

    popt, _ = curve_fit(func, thrust_df["power"], thrust_df["thrust_N"])
    logging.debug(f"fit_thrust_model: 拟合参数 popt={popt}")
    return popt  # (alpha, beta)


def load_thrust_allocation_matrix(matrix_path=None):
    """
    加载推力分配矩阵，要求矩阵形状为 (6,8)。
    若未提供，则使用默认矩阵。
    """
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


#############################################
# 2. 电机功率与IMU数据对齐
#############################################
def parse_timestamp(ts_str):
    """
    解析完整日期时间字符串（例如 "2024-06-18 04:38:40.241276"），
    返回（分钟+秒+微秒）之和（忽略小时和日期）。
    例如：04:38:40.241276 -> 38*60 + 40.241276 = 2320.241276 秒
    """
    dt = pd.to_datetime(ts_str, errors="coerce")
    if pd.isnull(dt):
        return np.nan
    return dt.minute * 60 + dt.second + dt.microsecond / 1e6


def align_motor_imu(power_path, imu_path, tolerance=0.1):
    """
    读取电机功率 CSV 与 IMU CSV，并基于时间戳对齐。
    将时间戳解析为 (分钟+秒+微秒) 的数值，
    并使用 merge_asof 对齐，若时间差小于 tolerance，则视为同一采样时刻。
    """
    if not os.path.exists(power_path):
        raise FileNotFoundError(f"电机数据文件不存在: {os.path.abspath(power_path)}")
    if not os.path.exists(imu_path):
        raise FileNotFoundError(f"IMU 文件不存在: {os.path.abspath(imu_path)}")

    power_df = pd.read_csv(power_path)
    imu_df   = pd.read_csv(imu_path)

    power_df["time_in_seconds"] = power_df["Timestamp"].apply(parse_timestamp)
    imu_df["time_in_seconds"]   = imu_df["Timestamp"].apply(parse_timestamp)

    power_df.dropna(subset=["time_in_seconds"], inplace=True)
    imu_df.dropna(subset=["time_in_seconds"], inplace=True)
    power_df.sort_values("time_in_seconds", inplace=True)
    imu_df.sort_values("time_in_seconds", inplace=True)

    merged = pd.merge_asof(
        power_df, imu_df,
        on="time_in_seconds",
        direction="nearest",
        tolerance=tolerance
    )
    if merged.empty:
        raise RuntimeError("对齐后数据为空，无法继续")
    logging.debug(f"align_motor_imu: merged shape={merged.shape}")
    return merged


#############################################
# 3. 加速度单位转换与角加速度计算
#############################################
def convert_accel_units(df):
    """
    将加速度单位从 g 转换为 m/s²：
      - 对 AccX, AccY 直接乘 9.80665；
      - 对 AccZ 先减 1（重力补偿）再乘 9.80665。
    """
    df[["AccX", "AccY"]] = df[["AccX", "AccY"]] * 9.80665
    df["AccZ"] = (df["AccZ"] - 1) * 9.80665
    return df


def compute_angular_acceleration(gyro_data, dt):
    """
    计算角加速度，使用 numpy 的梯度方法（对每个时间步计算数值梯度），返回最后一帧角加速度。
    """
    return np.gradient(gyro_data, dt, axis=0)


#############################################
# 4. 电机功率拆分：生成绝对值与方向标签
#############################################
def generate_motor_sign_by_state(df):
    """
    根据重力补偿后的 AccZ 判断上浮/下潜，生成各电机方向标签。
    规则：
      - 下潜（AccZ > 0）：设置 Motor6_Sign 与 Motor7_Sign 为 -1；
      - 上浮（AccZ < 0）：设置 Motor5_Sign 与 Motor8_Sign 为 -1；
      - 其他电机默认 +1。
    """
    for i in range(1, 9):
        df[f"Motor{i}_Sign"] = 1
    diving_mask = (df["AccZ"] > 0)
    ascending_mask = (df["AccZ"] < 0)
    df.loc[diving_mask, "Motor6_Sign"] = -1
    df.loc[diving_mask, "Motor7_Sign"] = -1
    df.loc[ascending_mask, "Motor5_Sign"] = -1
    df.loc[ascending_mask, "Motor8_Sign"] = -1
    return df


def preprocess_motor_power(df, motor_cols=None):
    """
    对电机功率列生成两个新列：
      - Motor{i}_AbsPower：保存原始正值功率；
      - Motor{i}_Sign：保存方向标签（若未生成则默认 +1）。
    """
    if motor_cols is None:
        motor_cols = [col for col in df.columns if col.startswith("Motor") and col.endswith("_Power")]
    for col in motor_cols:
        abs_col = col.replace("_Power", "_AbsPower")
        sign_col = col.replace("_Power", "_Sign")
        df[abs_col] = df[col]
        if sign_col not in df.columns:
            df[sign_col] = 1
    return df


def apply_thrust_inference(df, alpha, beta, motor_cols=None):
    """
    对每个 Motor{i}_AbsPower 列，根据公式 F = alpha * (AbsPower^beta) 计算推力幅值，
    然后乘以对应的 Motor{i}_Sign 恢复电机功率的正负，
    结果写回 Motor{i}_Power 列。
    """
    if motor_cols is None:
        motor_cols = [col for col in df.columns if col.startswith("Motor") and col.endswith("_Power")]
    def thrust_func(p):
        return alpha * (p ** beta)
    for col in motor_cols:
        abs_col = col.replace("_Power", "_AbsPower")
        sign_col = col.replace("_Power", "_Sign")
        df[col] = thrust_func(df[abs_col]) * df[sign_col]
    return df


#############################################
# 5. 窗口切分
#############################################
def create_sequences(df, input_cols, label_cols_acc, label_cols_thrust,
                     motor_cols, thrust_matrix, window_size=5, step=1, dt=0.1):
    """
    将连续的时间窗口切分为训练样本：
      - 对每个窗口，将 input_cols 进行 flatten 生成特征向量 feats；
      - 以窗口最后一帧的加速度作为标签 accs；
      - 以窗口最后一帧的电机功率（包含方向信息）计算推力 tau 作为推力标签；
      - 同时计算速度（使用梯形积分）和角加速度（数值梯度）。
    """
    feats, accs, thrs, vels, ang_accels = [], [], [], [], []
    for i in range(0, len(df) - window_size + 1, step):
        window = df.iloc[i:i + window_size]
        if window.isnull().any().any():
            continue

        # 将窗口内指定列 flatten 为 1D 向量
        feat = window[input_cols].values.flatten()
        # 加速度标签：取窗口最后一行的加速度（通常为 [AccX,AccY,AccZ]）
        a_label = window[label_cols_acc].iloc[-1].values

        # 推力标签：取窗口最后一行的电机功率（已含方向信息），计算 tau = thrust_matrix @ motor_power
        motor_thrusts = [window[f"{m}"].iloc[-1] for m in motor_cols]
        motor_thrusts = np.array(motor_thrusts, dtype=np.float32)
        tau = thrust_matrix @ motor_thrusts  # 结果为 (6,)

        # 计算速度标签（利用加速度窗口进行梯形积分）
        time_points = np.arange(window_size) * dt
        acc_window = window[label_cols_acc].values
        vel_window = trapezoid(acc_window, x=time_points, axis=0)

        # 计算角加速度标签（取梯度的最后一帧）
        gyro_cols = ['AsX', 'AsY', 'AsZ']
        gyro_window = window[gyro_cols].values
        ang_accel = np.gradient(gyro_window, dt, axis=0)[-1]

        feats.append(feat)
        accs.append(a_label)
        thrs.append(tau)
        vels.append(vel_window)
        ang_accels.append(ang_accel)

    feats = np.array(feats, dtype=np.float32)
    accs = np.array(accs, dtype=np.float32)
    thrs = np.array(thrs, dtype=np.float32)
    vels = np.array(vels, dtype=np.float32)
    ang_accels = np.array(ang_accels, dtype=np.float32)

    logging.debug(f"create_sequences: feats={len(feats)}, accs={len(accs)}")
    return feats, accs, thrs, vels, ang_accels

#############################################
# 6. 主流程 full_pipeline
#############################################
def full_pipeline(config):
    logging.info("开始预处理流程...")

    # 检查必需文件
    req_files = [config['thrust_path'], config['power_path'], config['imu_path']]
    for f in req_files:
        if not os.path.exists(f):
            raise FileNotFoundError(f"文件不存在: {os.path.abspath(f)}")

    # 1) 加载并拟合推力测试数据
    logging.info("加载并拟合推力测试数据...")
    thrust_df = load_thrust_data(config["thrust_path"])
    if thrust_df.empty:
        raise RuntimeError("推力测试数据为空，无法继续。")
    alpha, beta = fit_thrust_model(thrust_df)
    logging.info(f"拟合得 α={alpha:.4f}, β={beta:.4f}")

    # 定义推力插值函数（理论公式 F = α * P^β）
    def thrust_interp_func(power):
        return alpha * (power ** beta)

    # 2) 对齐电机功率与IMU数据
    logging.info("对齐电机功率与IMU数据...")
    merged = align_motor_imu(config["power_path"], config["imu_path"], tolerance=0.1)
    if merged.empty:
        raise RuntimeError("对齐后数据为空，无法继续。")

    # 3) 加速度单位转换（g -> m/s²，补偿重力）
    logging.info("加速度单位转换 (g -> m/s², 补偿重力)...")
    merged = convert_accel_units(merged)

    # 4) 根据 AccZ 状态生成电机方向标签
    logging.info("生成电机方向标签...")
    merged = generate_motor_sign_by_state(merged)

    # 5) 拆分电机功率：生成 AbsPower 与方向标签
    logging.info("拆分电机功率为 AbsPower 与方向标签...")
    motor_cols = [f"Motor{i}_Power" for i in range(1, 9)]
    merged = preprocess_motor_power(merged, motor_cols)

    # 6) 推力插值：根据公式计算推力，并恢复方向
    logging.info("插值得到电机推力...")
    merged = apply_thrust_inference(merged, alpha, beta, motor_cols)

    # 7) 缺失值填充
    n_missing = merged.isnull().sum().sum()
    if n_missing > 0:
        logging.warning(f"发现 {n_missing} 个缺失值，执行前向与后向填充...")
        merged.ffill(inplace=True)
        merged.bfill(inplace=True)
        if merged.isnull().sum().sum() > 0:
            raise ValueError("填充后仍存在缺失值")
        else:
            logging.info("缺失值填充成功。")
    else:
        logging.info("未发现缺失值。")

    # 8) 加载推力分配矩阵 (6x8)
    logging.info("载入推力分配矩阵 (6x8)...")
    T = load_thrust_allocation_matrix(config.get("thrust_matrix_path", None))

    # 9) 窗口切分
    logging.info("开始窗口切分...")
    dt = config.get("dt", 0.11)
    # 输入列包括处理后的电机推力和IMU数据
    input_cols = motor_cols + ["AccX", "AccY", "AccZ", "AsX", "AsY", "AsZ"]
    label_cols_acc = ["AccX", "AccY", "AccZ"]
    # 这里使用 motor_cols 作为推力标签的列名（已包含正负信息）
    feats, accs, thrs, vels, ang_accels = create_sequences(
        merged,
        input_cols,
        label_cols_acc,
        motor_cols,
        motor_cols,
        T,
        window_size=config.get("window_size", 5),
        step=1,
        dt=dt
    )
    logging.info(f"窗口切分结果: feats={feats.shape}, accs={accs.shape}, thrs={thrs.shape}, vels={vels.shape}, ang_acc={ang_accels.shape}")

    # 10) 保存预处理结果
    logging.info("保存预处理结果到 .npy 文件...")
    save_dir = config["save_dir"]
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, "train_features.npy"), feats)
    np.save(os.path.join(save_dir, "train_accel_labels.npy"), accs)
    np.save(os.path.join(save_dir, "train_thrust_labels.npy"), thrs)
    np.save(os.path.join(save_dir, "train_velocity_labels.npy"), vels)
    np.save(os.path.join(save_dir, "train_angular_accel_labels.npy"), ang_accels)

    logging.info("预处理流程完成。")
    return feats, accs, thrs, vels, ang_accels, (alpha, beta)

#############################################
# 测试入口
#############################################
if __name__ == "__main__":
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config = {
        "thrust_path": os.path.join(PROJECT_ROOT, "data", "raw", "thrust_test.xlsx"),
        "power_path": os.path.join(PROJECT_ROOT, "data", "raw", "table1_motor_data0331.csv"),
        "imu_path": os.path.join(PROJECT_ROOT, "data", "raw", "imu_data_downsampled_9hz0331.csv"),
        "thrust_matrix_path": os.path.join(PROJECT_ROOT, "data", "raw", "thrust_allocation_matrix.csv"),
        "save_dir": os.path.join(PROJECT_ROOT, "data", "processed"),
        "window_size": 5,
        "dt": 0.11
    }

    feats, accs, thrs, vels, ang_accels, params = full_pipeline(config)
    alpha, beta = params
    logging.info(f"Features shape: {feats.shape}, Acc shape: {accs.shape}, Thrust shape: {thrs.shape}")
    logging.info(f"Velocity shape: {vels.shape}, AngularAccel shape: {ang_accels.shape}")
    logging.info(f"Fitted thrust params: α={alpha:.4f}, β={beta:.4f}")
