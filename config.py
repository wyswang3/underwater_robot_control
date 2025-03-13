import os
from dataclasses import dataclass, field

@dataclass
class PathsConfig:
    # 项目根目录（动态获取当前文件所在目录）
    PROJECT_ROOT: str = field(default_factory=lambda: os.path.abspath(os.path.dirname(__file__)))
    # 预处理数据文件路径（将在 __post_init__ 中初始化）
    TRAIN_FEATURES_FILE: str = field(init=False)
    TRAIN_ACCEL_LABELS_FILE: str = field(init=False)
    TRAIN_THRUST_LABELS_FILE: str = field(init=False)
    TRAIN_VELOCITY_LABELS_FILE: str = field(init=False)
    TRAIN_ANGULAR_ACCEL_LABELS_FILE: str = field(init=False)
    # 推力分配矩阵文件（可选，如未提供则网络中使用默认）
    THRUST_MATRIX_FILE: str = field(init=False)
    # 模型和日志保存目录
    MODEL_DIR: str = field(init=False)
    LOG_DIR: str = field(init=False)
    # 用于保存训练/评估图片等结果的目录
    SPLITS_DIR: str = field(init=False)

    def __post_init__(self):
        self.TRAIN_FEATURES_FILE = os.path.join(self.PROJECT_ROOT, "data", "processed", "train_features.npy")
        self.TRAIN_ACCEL_LABELS_FILE = os.path.join(self.PROJECT_ROOT, "data", "processed", "train_accel_labels.npy")
        self.TRAIN_THRUST_LABELS_FILE = os.path.join(self.PROJECT_ROOT, "data", "processed", "train_thrust_labels.npy")
        self.TRAIN_VELOCITY_LABELS_FILE = os.path.join(self.PROJECT_ROOT, "data", "processed", "train_velocity_labels.npy")
        self.TRAIN_ANGULAR_ACCEL_LABELS_FILE = os.path.join(self.PROJECT_ROOT, "data", "processed", "train_angular_accel_labels.npy")
        self.THRUST_MATRIX_FILE = os.path.join(self.PROJECT_ROOT, "data", "raw", "thrust_allocation_matrix.csv")
        self.MODEL_DIR = os.path.join(self.PROJECT_ROOT, "models", "checkpoints")
        self.LOG_DIR = os.path.join(self.PROJECT_ROOT, "logs")
        self.SPLITS_DIR = os.path.join(self.PROJECT_ROOT, "data", "splits")
        # 自动创建保存目录
        os.makedirs(self.MODEL_DIR, exist_ok=True)
        os.makedirs(self.LOG_DIR, exist_ok=True)
        os.makedirs(self.SPLITS_DIR, exist_ok=True)

@dataclass
class TrainingConfig:
    # 数据窗口与网络结构
    WINDOW_SIZE: int = 6
    HIDDEN_DIM: int = 512

    # 训练超参数
    NUM_EPOCHS: int = 1
    BATCH_SIZE: int = 32
    LEARNING_RATE: float = 2e-5
    WEIGHT_DECAY: float = 1e-5
    CLIP_GRAD_NORM: float = 1.0
    NUM_WORKERS: int = 4

    # 损失函数权重（例如 EnhancedDynamicsLoss(alpha, beta, gamma)）
    ALPHA: float = 1.0   # 物理约束/推力损失权重
    BETA: float = 0.1    # 矩阵正则损失权重
    GAMMA: float = 0.01  # 其他扩展项的权重

@dataclass
class DeviceConfig:
    # 设备选择：优先使用第三张 GPU（可通过环境变量 USE_CUDA 控制），否则使用 CPU
    DEVICE: str = field(default_factory=lambda: "cuda:1" if os.environ.get("USE_CUDA", "1") == "1" and os.path.exists("/dev/nvidia0") else "cpu")

@dataclass
class Config:
    paths: PathsConfig = field(default_factory=PathsConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    device: DeviceConfig = field(default_factory=DeviceConfig)
    DEBUG: bool = False

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
        print(f"WINDOW_SIZE: {self.training.WINDOW_SIZE}")
        print(f"HIDDEN_DIM: {self.training.HIDDEN_DIM}")
        print(f"NUM_EPOCHS: {self.training.NUM_EPOCHS}")
        print(f"BATCH_SIZE: {self.training.BATCH_SIZE}")
        print(f"LEARNING_RATE: {self.training.LEARNING_RATE}")
        print(f"WEIGHT_DECAY: {self.training.WEIGHT_DECAY}")
        print(f"CLIP_GRAD_NORM: {self.training.CLIP_GRAD_NORM}")
        print(f"NUM_WORKERS: {self.training.NUM_WORKERS}")
        print(f"ALPHA: {self.training.ALPHA}")
        print(f"BETA: {self.training.BETA}")
        print(f"GAMMA: {self.training.GAMMA}\n")
        print(f"DEVICE: {self.device.DEVICE}")
        print(f"DEBUG: {self.DEBUG}")

if __name__ == "__main__":
    cfg = Config()
    cfg.print_config()
