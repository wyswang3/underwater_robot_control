import os
import torch
from dataclasses import dataclass, field


@dataclass
class PathsConfig:
    """
    路径配置：定义项目中数据、模型、日志等相关文件和目录的路径。
    在 __post_init__ 中根据 PROJECT_ROOT 动态生成各个文件和目录的完整路径，
    并自动创建模型、日志和数据切分目录，确保运行时文件夹存在。
    """
    PROJECT_ROOT: str = field(default_factory=lambda: os.path.abspath(os.path.dirname(__file__)))

    # 数据文件
    TRAIN_FEATURES_FILE: str = field(init=False)
    TRAIN_ACCEL_LABELS_FILE: str = field(init=False)
    TRAIN_THRUST_LABELS_FILE: str = field(init=False)
    TRAIN_VELOCITY_LABELS_FILE: str = field(init=False)
    TRAIN_ANGULAR_ACCEL_LABELS_FILE: str = field(init=False)

    # 推力分配矩阵（可选）
    THRUST_MATRIX_FILE: str = field(init=False)

    # 输出目录：模型检查点、日志、数据划分
    MODEL_DIR: str = field(init=False)
    LOG_DIR: str = field(init=False)
    SPLITS_DIR: str = field(init=False)

    def __post_init__(self) -> None:
        self.TRAIN_FEATURES_FILE = os.path.join(self.PROJECT_ROOT, "data", "processed", "train_features.npy")
        self.TRAIN_ACCEL_LABELS_FILE = os.path.join(self.PROJECT_ROOT, "data", "processed", "train_accel_labels.npy")
        self.TRAIN_THRUST_LABELS_FILE = os.path.join(self.PROJECT_ROOT, "data", "processed", "train_thrust_labels.npy")
        self.TRAIN_VELOCITY_LABELS_FILE = os.path.join(self.PROJECT_ROOT, "data", "processed",
                                                       "train_velocity_labels.npy")
        self.TRAIN_ANGULAR_ACCEL_LABELS_FILE = os.path.join(self.PROJECT_ROOT, "data", "processed",
                                                            "train_angular_accel_labels.npy")

        self.THRUST_MATRIX_FILE = os.path.join(self.PROJECT_ROOT, "data", "raw", "thrust_allocation_matrix.csv")

        self.MODEL_DIR = os.path.join(self.PROJECT_ROOT, "models", "checkpoints")
        self.LOG_DIR = os.path.join(self.PROJECT_ROOT, "logs")
        self.SPLITS_DIR = os.path.join(self.PROJECT_ROOT, "data", "splits")

        for d in [self.MODEL_DIR, self.LOG_DIR, self.SPLITS_DIR]:
            os.makedirs(d, exist_ok=True)


@dataclass
class TrainingConfig:
    """
    训练超参数配置：包含数据预处理、网络结构、训练策略等参数。
    """
    # 数据与网络输入
    WINDOW_SIZE: int = 5  # 时间窗口大小
    INPUT_DIM: int = 14  # 每个时间步特征数（8 电机功率 + 6 IMU）

    # 新网络结构相关参数（针对 DeepHydroNet）
    PHYSICS_HIDDEN_DIM: int = 512  # 用于 LSTM 的输出维度
    USE_DYNAMIC_WIDTH: bool = True  # 是否启用动态宽度调整
    BASE_WIDTH: int = 512  # 动态宽度模块的基宽（通常与 PHYSICS_HIDDEN_DIM 保持一致）
    MAX_WIDTH: int = 1024  # 动态宽度模块扩展后的最大宽度

    # 训练参数
    NUM_EPOCHS: int = 15
    BATCH_SIZE: int = 32
    LEARNING_RATE: float = 5e-4
    WEIGHT_DECAY: float = 1e-4
    CLIP_GRAD_NORM: float = 1.0
    NUM_WORKERS: int = 4

    # 损失函数权重（数据驱动损失与物理约束损失之间的权重平衡）
    ALPHA: float = 1.0
    BETA: float = 0.1
    GAMMA: float = 0.01
    DELTA: float = 0.1

    # 学习率调度器参数（CosineAnnealingWarmRestarts）
    LR_SCHEDULER: bool = True
    T_0: int = 10
    T_MULT: int = 2
    ETA_MIN: float = 1e-6


@dataclass
class DeviceConfig:
    """
    设备配置：自动检测 GPU 可用性，若有 GPU 则使用 GPU，否则使用 CPU。
    可通过环境变量 GPU_ID 指定 GPU 序号（例如 "0"、"1"）。
    """
    DEVICE: str = field(default_factory=lambda: (
        f"cuda:{os.environ.get('GPU_ID', '0')}" if torch.cuda.is_available() else "cpu"
    ))


@dataclass
class Config:
    """
    主配置：整合路径、训练、设备等配置，便于统一管理各模块超参数。
    未来可扩展为支持 YAML/JSON 配置文件加载。
    """
    paths: PathsConfig = field(default_factory=PathsConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    device: DeviceConfig = field(default_factory=DeviceConfig)
    DEBUG: bool = False

    def print_config(self) -> None:
        print("========== Config ==========")
        print(f"PROJECT_ROOT: {self.paths.PROJECT_ROOT}")
        print(f"TRAIN_FEATURES_FILE: {self.paths.TRAIN_FEATURES_FILE}")
        print(f"TRAIN_ACCEL_LABELS_FILE: {self.paths.TRAIN_ACCEL_LABELS_FILE}")
        print(f"TRAIN_THRUST_LABELS_FILE: {self.paths.TRAIN_THRUST_LABELS_FILE}")
        print(f"TRAIN_VELOCITY_LABELS_FILE: {self.paths.TRAIN_VELOCITY_LABELS_FILE}")
        print(f"TRAIN_ANGULAR_ACCEL_LABELS_FILE: {self.paths.TRAIN_ANGULAR_ACCEL_LABELS_FILE}")
        print(f"THRUST_MATRIX_FILE: {self.paths.THRUST_MATRIX_FILE}")
        print(f"MODEL_DIR: {self.paths.MODEL_DIR}")
        print(f"LOG_DIR: {self.paths.LOG_DIR}")
        print(f"SPLITS_DIR: {self.paths.SPLITS_DIR}\n")

        print("=== Training Config ===")
        print(f"WINDOW_SIZE: {self.training.WINDOW_SIZE}")
        print(f"INPUT_DIM: {self.training.INPUT_DIM}")
        print(f"PHYSICS_HIDDEN_DIM: {self.training.PHYSICS_HIDDEN_DIM}")
        print(f"USE_DYNAMIC_WIDTH: {self.training.USE_DYNAMIC_WIDTH}")
        if self.training.USE_DYNAMIC_WIDTH:
            print(f"BASE_WIDTH: {self.training.BASE_WIDTH}")
            print(f"MAX_WIDTH: {self.training.MAX_WIDTH}")
        print(f"NUM_EPOCHS: {self.training.NUM_EPOCHS}")
        print(f"BATCH_SIZE: {self.training.BATCH_SIZE}")
        print(f"LEARNING_RATE: {self.training.LEARNING_RATE}")
        print(f"WEIGHT_DECAY: {self.training.WEIGHT_DECAY}")
        print(f"CLIP_GRAD_NORM: {self.training.CLIP_GRAD_NORM}")
        print(f"NUM_WORKERS: {self.training.NUM_WORKERS}")
        print(f"ALPHA: {self.training.ALPHA}")
        print(f"BETA: {self.training.BETA}")
        print(f"GAMMA: {self.training.GAMMA}")
        print(f"DELTA: {self.training.DELTA}\n")

        print("=== Scheduler Config ===")
        print(f"LR_SCHEDULER: {self.training.LR_SCHEDULER}")
        print(f"T_0: {self.training.T_0}")
        print(f"T_MULT: {self.training.T_MULT}")
        print(f"ETA_MIN: {self.training.ETA_MIN}\n")

        print("=== Device Config ===")
        print(f"DEVICE: {self.device.DEVICE}")
        print(f"DEBUG: {self.DEBUG}")


if __name__ == "__main__":
    cfg = Config()
    cfg.print_config()
