import os
from dataclasses import dataclass, field


@dataclass
class PathsConfig:
    """
    存放项目中各类文件与目录的路径配置。
    根据 PROJECT_ROOT 自动生成各类路径，并创建所需目录。
    """
    PROJECT_ROOT: str = field(default_factory=lambda: os.path.abspath(os.path.dirname(__file__)))

    # 训练数据及标签文件
    TRAIN_FEATURES_FILE: str = field(init=False)
    TRAIN_ACCEL_LABELS_FILE: str = field(init=False)
    TRAIN_THRUST_LABELS_FILE: str = field(init=False)
    TRAIN_VELOCITY_LABELS_FILE: str = field(init=False)
    TRAIN_ANGULAR_ACCEL_LABELS_FILE: str = field(init=False)

    # 推力矩阵文件（如无则使用默认）
    THRUST_MATRIX_FILE: str = field(init=False)

    # 模型、日志和数据切分目录
    MODEL_DIR: str = field(init=False)
    LOG_DIR: str = field(init=False)
    SPLITS_DIR: str = field(init=False)

    def __post_init__(self):
        # 数据文件路径
        self.TRAIN_FEATURES_FILE = os.path.join(self.PROJECT_ROOT, "data", "processed", "train_features.npy")
        self.TRAIN_ACCEL_LABELS_FILE = os.path.join(self.PROJECT_ROOT, "data", "processed", "train_accel_labels.npy")
        self.TRAIN_THRUST_LABELS_FILE = os.path.join(self.PROJECT_ROOT, "data", "processed", "train_thrust_labels.npy")
        self.TRAIN_VELOCITY_LABELS_FILE = os.path.join(self.PROJECT_ROOT, "data", "processed",
                                                       "train_velocity_labels.npy")
        self.TRAIN_ANGULAR_ACCEL_LABELS_FILE = os.path.join(self.PROJECT_ROOT, "data", "processed",
                                                            "train_angular_accel_labels.npy")

        # 推力矩阵文件
        self.THRUST_MATRIX_FILE = os.path.join(self.PROJECT_ROOT, "data", "raw", "thrust_allocation_matrix.csv")

        # 目录路径
        self.MODEL_DIR = os.path.join(self.PROJECT_ROOT, "models", "checkpoints")
        self.LOG_DIR = os.path.join(self.PROJECT_ROOT, "logs")
        self.SPLITS_DIR = os.path.join(self.PROJECT_ROOT, "data", "splits")

        # 自动创建目录
        self._make_dirs([self.MODEL_DIR, self.LOG_DIR, self.SPLITS_DIR])

    def _make_dirs(self, dirs):
        """创建目录列表中的所有目录（若不存在则创建）。"""
        for d in dirs:
            os.makedirs(d, exist_ok=True)


@dataclass
class TrainingConfig:
    """
    训练相关的超参数配置，包括网络结构、训练轮数、batch 大小、学习率等。
    """
    WINDOW_SIZE: int = 5
    PHYSICS_HIDDEN_DIM: int = 512  # 物理网络隐藏层维度
    E2E_HIDDEN_DIM: int = 256  # 端到端网络隐藏层维度

    NUM_EPOCHS: int = 5
    BATCH_SIZE: int = 32
    LEARNING_RATE: float = 2e-5
    WEIGHT_DECAY: float = 1e-5
    CLIP_GRAD_NORM: float = 1.0
    NUM_WORKERS: int = 4

    # 损失函数权重设置
    ALPHA: float = 1.0  # 推力损失权重
    BETA: float = 0.1  # 矩阵正则项权重
    GAMMA: float = 0.01  # 其他扩展损失权重
    DELTA: float = 0.1  # 方向损失权重


@dataclass
class DeviceConfig:
    """
    设备配置：
      - 如果环境变量 USE_CUDA=1 且存在 /dev/nvidia0，则使用 'cuda:GPU_ID'（通过环境变量 GPU_ID 指定，默认 '0'）；
      - 否则使用 CPU。
    """
    DEVICE: str = field(default_factory=lambda: (
        "cuda:" + os.environ.get("GPU_ID", "0")
        if os.environ.get("USE_CUDA", "0") == "1" and os.path.exists("/dev/nvidia0")
        else "cpu"
    ))


@dataclass
class Config:
    """
    主配置，包含路径、训练和设备子配置，同时提供调试开关 DEBUG。
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

        print("=== Device Config ===")
        print(f"DEVICE: {self.device.DEVICE}")
        print(f"DEBUG: {self.DEBUG}")


if __name__ == "__main__":
    cfg = Config()
    cfg.print_config()
