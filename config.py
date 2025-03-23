import os
from dataclasses import dataclass, field

@dataclass
class PathsConfig:
    """
    存放项目中各类文件与目录的路径配置。
    在 __post_init__ 中根据 PROJECT_ROOT 动态生成。
    """
    PROJECT_ROOT: str = field(default_factory=lambda: os.path.abspath(os.path.dirname(__file__)))

    TRAIN_FEATURES_FILE: str = field(init=False)
    TRAIN_ACCEL_LABELS_FILE: str = field(init=False)
    TRAIN_THRUST_LABELS_FILE: str = field(init=False)
    TRAIN_VELOCITY_LABELS_FILE: str = field(init=False)
    TRAIN_ANGULAR_ACCEL_LABELS_FILE: str = field(init=False)

    THRUST_MATRIX_FILE: str = field(init=False)

    MODEL_DIR: str = field(init=False)
    LOG_DIR: str = field(init=False)
    SPLITS_DIR: str = field(init=False)

    def __post_init__(self):
        # 1) 训练数据 & 标签文件
        self.TRAIN_FEATURES_FILE = os.path.join(self.PROJECT_ROOT, "data", "processed", "train_features.npy")
        self.TRAIN_ACCEL_LABELS_FILE = os.path.join(self.PROJECT_ROOT, "data", "processed", "train_accel_labels.npy")
        self.TRAIN_THRUST_LABELS_FILE = os.path.join(self.PROJECT_ROOT, "data", "processed", "train_thrust_labels.npy")
        self.TRAIN_VELOCITY_LABELS_FILE = os.path.join(self.PROJECT_ROOT, "data", "processed", "train_velocity_labels.npy")
        self.TRAIN_ANGULAR_ACCEL_LABELS_FILE = os.path.join(self.PROJECT_ROOT, "data", "processed", "train_angular_accel_labels.npy")

        # 2) 推力矩阵（如果有的话），否则网络中使用默认值
        self.THRUST_MATRIX_FILE = os.path.join(self.PROJECT_ROOT, "data", "raw", "thrust_allocation_matrix.csv")

        # 3) 模型与日志目录
        self.MODEL_DIR = os.path.join(self.PROJECT_ROOT, "models", "checkpoints")
        self.LOG_DIR   = os.path.join(self.PROJECT_ROOT, "logs")
        self.SPLITS_DIR= os.path.join(self.PROJECT_ROOT, "data", "splits")

        # 自动创建保存目录
        os.makedirs(self.MODEL_DIR, exist_ok=True)
        os.makedirs(self.LOG_DIR,   exist_ok=True)
        os.makedirs(self.SPLITS_DIR, exist_ok=True)

@dataclass
class TrainingConfig:
    """
    训练相关的超参数，包括网络结构、训练轮数、batch大小、学习率等。
    """
    # 窗口大小 & 隐藏层维度（网络模型中，物理网络和端到端网络可分别设定，视情况调整）
    WINDOW_SIZE: int = 5
    # 例如 EnhancedPhysicsNet 的隐藏层维度，这里配置 512 与 DirectMappingNet 的 256 可分开设置
    PHYSICS_HIDDEN_DIM: int = 512
    E2E_HIDDEN_DIM: int = 256

    # 训练轮数、batch大小、学习率等
    NUM_EPOCHS: int = 3
    BATCH_SIZE: int = 32
    LEARNING_RATE: float = 2e-5
    WEIGHT_DECAY: float = 1e-5
    CLIP_GRAD_NORM: float = 1.0
    NUM_WORKERS: int = 4

    # 损失函数权重：alpha 为推力损失、beta 为矩阵正则项、gamma 为其他扩展项
    ALPHA: float = 1.0
    BETA: float  = 0.1
    GAMMA: float = 0.01

@dataclass
class DeviceConfig:
    """
    设备配置：如果环境变量 USE_CUDA=1 且存在 /dev/nvidia0，
    则使用 'cuda:1' (表征第三张GPU)，否则使用 CPU。
    """
    DEVICE: str = field(default_factory=lambda: (
        "cuda:0" if os.environ.get("USE_CUDA", "1") == "1"
        and os.path.exists("/dev/nvidia0")
        else "cpu"
    ))

@dataclass
class Config:
    """
    主配置，包含路径、训练、设备等子配置。
    同时可加入调试标志 DEBUG 用于控制日志详细程度。
    """
    paths: PathsConfig = field(default_factory=PathsConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    device: DeviceConfig = field(default_factory=DeviceConfig)
    DEBUG: bool = False  # 调试标志

    def print_config(self):
        """打印配置信息，便于调试。"""
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
        print(f"GAMMA: {self.training.GAMMA}\n")

        print(f"DEVICE: {self.device.DEVICE}")
        print(f"DEBUG: {self.DEBUG}")

if __name__ == "__main__":
    cfg = Config()
    cfg.print_config()
