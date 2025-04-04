import os
import torch
from dataclasses import dataclass, field

@dataclass
class PathsConfig:
    """
    存放项目中各类文件与目录的路径配置。
    在 __post_init__ 中根据 PROJECT_ROOT 动态生成具体路径。
    """
    PROJECT_ROOT: str = field(default_factory=lambda: os.path.abspath(os.path.dirname(__file__)))

    # 训练数据及标签文件
    TRAIN_FEATURES_FILE: str = field(init=False)
    TRAIN_ACCEL_LABELS_FILE: str = field(init=False)
    TRAIN_THRUST_LABELS_FILE: str = field(init=False)
    TRAIN_VELOCITY_LABELS_FILE: str = field(init=False)
    TRAIN_ANGULAR_ACCEL_LABELS_FILE: str = field(init=False)

    # 推力分配矩阵文件（如无则使用默认）
    THRUST_MATRIX_FILE: str = field(init=False)

    # 模型、日志和数据切分目录
    MODEL_DIR: str = field(init=False)
    LOG_DIR: str = field(init=False)
    SPLITS_DIR: str = field(init=False)

    def __post_init__(self):
        # 设置训练数据及标签文件路径
        self.TRAIN_FEATURES_FILE = os.path.join(self.PROJECT_ROOT, "data", "processed", "train_features.npy")
        self.TRAIN_ACCEL_LABELS_FILE = os.path.join(self.PROJECT_ROOT, "data", "processed", "train_accel_labels.npy")
        self.TRAIN_THRUST_LABELS_FILE = os.path.join(self.PROJECT_ROOT, "data", "processed", "train_thrust_labels.npy")
        self.TRAIN_VELOCITY_LABELS_FILE = os.path.join(self.PROJECT_ROOT, "data", "processed", "train_velocity_labels.npy")
        self.TRAIN_ANGULAR_ACCEL_LABELS_FILE = os.path.join(self.PROJECT_ROOT, "data", "processed", "train_angular_accel_labels.npy")

        # 设置推力分配矩阵文件路径
        self.THRUST_MATRIX_FILE = os.path.join(self.PROJECT_ROOT, "data", "raw", "thrust_allocation_matrix.csv")

        # 设置模型、日志和数据切分目录
        self.MODEL_DIR = os.path.join(self.PROJECT_ROOT, "models", "checkpoints")
        self.LOG_DIR = os.path.join(self.PROJECT_ROOT, "logs")
        self.SPLITS_DIR = os.path.join(self.PROJECT_ROOT, "data", "splits")

        # 自动创建目录
        for d in [self.MODEL_DIR, self.LOG_DIR, self.SPLITS_DIR]:
            os.makedirs(d, exist_ok=True)

@dataclass
class TrainingConfig:
    """
    训练相关超参数配置，包括网络结构、训练轮数、batch大小、学习率等参数，
    以及带热重启余弦退火调度器的参数和损失函数权重。
    """
    # 基本参数
    WINDOW_SIZE: int = 5
    PHYSICS_HIDDEN_DIM: int = 512
    E2E_HIDDEN_DIM: int = 256

    NUM_EPOCHS: int = 10
    BATCH_SIZE: int = 32
    LEARNING_RATE: float = 2e-3
    WEIGHT_DECAY: float = 1e-4
    CLIP_GRAD_NORM: float = 1.0
    NUM_WORKERS: int = 4

    # 损失函数权重（数据驱动与物理约束之间的权重平衡）
    ALPHA: float = 1.0
    BETA: float = 0.1
    GAMMA: float = 0.01
    DELTA: float = 0.1

    # 学习率调度器参数（CosineAnnealingWarmRestarts）
    LR_SCHEDULER: bool = True
    T_0: int = 10           # 初始重启周期（以 epoch 计）
    T_MULT: int = 2         # 重启周期乘数
    ETA_MIN: float = 1e-5   # 最低学习率

@dataclass
class DeviceConfig:
    """
    设备配置：如果 torch.cuda.is_available() 为 True，则使用 GPU，
    否则使用 CPU。可通过环境变量 GPU_ID 指定 GPU 序号（例如 "0"、"1"）。
    """
    DEVICE: str = field(default_factory=lambda: (
        f"cuda:{os.environ.get('GPU_ID', '0')}"
        if torch.cuda.is_available() else "cpu"
    ))

@dataclass
class Config:
    """
    主配置，包含路径、训练、设备等子配置，以及调试开关。
    你可以根据需要进一步扩展，例如支持从 YAML/JSON 文件加载配置。
    """
    paths: PathsConfig = field(default_factory=PathsConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    device: DeviceConfig = field(default_factory=DeviceConfig)
    DEBUG: bool = False  # 调试开关

    def print_config(self):
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
        print(f"PHYSICS_HIDDEN_DIM: {self.training.PHYSICS_HIDDEN_DIM}")
        print(f"E2E_HIDDEN_DIM: {self.training.E2E_HIDDEN_DIM}")
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
