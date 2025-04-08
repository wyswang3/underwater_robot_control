import os
import torch
from dataclasses import dataclass, field

@dataclass
class PathsConfig:
    """
    PathsConfig defines the file and directory paths for the project, including data, models, and logs.
    The __post_init__ method dynamically generates complete paths based on PROJECT_ROOT and creates
    directories if they don't exist.
    """
    PROJECT_ROOT: str = field(default_factory=lambda: os.path.abspath(os.path.dirname(__file__)))

    # Data files
    TRAIN_FEATURES_FILE: str = field(init=False)
    TRAIN_ACCEL_LABELS_FILE: str = field(init=False)
    TRAIN_THRUST_LABELS_FILE: str = field(init=False)
    TRAIN_VELOCITY_LABELS_FILE: str = field(init=False)
    TRAIN_ANGULAR_ACCEL_LABELS_FILE: str = field(init=False)

    # Thrust allocation matrix (optional)
    THRUST_MATRIX_FILE: str = field(init=False)

    # Output directories: model checkpoints, logs, and data splits
    MODEL_DIR: str = field(init=False)
    LOG_DIR: str = field(init=False)
    SPLITS_DIR: str = field(init=False)

    def __post_init__(self) -> None:
        self.TRAIN_FEATURES_FILE = os.path.join(self.PROJECT_ROOT, "data", "processed", "train_features.npy")
        self.TRAIN_ACCEL_LABELS_FILE = os.path.join(self.PROJECT_ROOT, "data", "processed", "train_accel_labels.npy")
        self.TRAIN_THRUST_LABELS_FILE = os.path.join(self.PROJECT_ROOT, "data", "processed", "train_thrust_labels.npy")
        self.TRAIN_VELOCITY_LABELS_FILE = os.path.join(self.PROJECT_ROOT, "data", "processed", "train_velocity_labels.npy")
        self.TRAIN_ANGULAR_ACCEL_LABELS_FILE = os.path.join(self.PROJECT_ROOT, "data", "processed", "train_angular_accel_labels.npy")

        self.THRUST_MATRIX_FILE = os.path.join(self.PROJECT_ROOT, "data", "raw", "thrust_allocation_matrix.csv")

        self.MODEL_DIR = os.path.join(self.PROJECT_ROOT, "models", "checkpoints")
        self.LOG_DIR = os.path.join(self.PROJECT_ROOT, "logs")
        self.SPLITS_DIR = os.path.join(self.PROJECT_ROOT, "data", "splits")

        for d in [self.MODEL_DIR, self.LOG_DIR, self.SPLITS_DIR]:
            os.makedirs(d, exist_ok=True)


@dataclass
class TrainingConfig:
    """
    TrainingConfig specifies hyperparameters for data preprocessing, network structure, and training strategy.
    """
    # Data & network input parameters
    WINDOW_SIZE: int = 5             # Time window size
    INPUT_DIM: int = 14              # Features per time step (8 motor power + 6 IMU)

    # Network structure parameters
    HIDDEN_DIM: int = 512            # LSTM output dimension (for bidirectional LSTM, total dimension = HIDDEN_DIM)
    LSTM_LAYERS: int = 2             # Number of LSTM layers

    # Training hyperparameters
    NUM_EPOCHS: int = 45
    BATCH_SIZE: int = 32
    LEARNING_RATE: float = 5e-4
    WEIGHT_DECAY: float = 1e-4
    CLIP_GRAD_NORM: float = 1.0
    NUM_WORKERS: int = 4

    # Loss function weights (for physical constraints, can be tuned)
    ALPHA: float = 1.0   # (Optional, for further extension)
    BETA: float = 0.1
    GAMMA: float = 0.01
    DELTA: float = 0.1

    # Learning rate scheduler parameters (CosineAnnealingWarmRestarts)
    LR_SCHEDULER: bool = True
    T_0: int = 10
    T_MULT: int = 2
    ETA_MIN: float = 1e-6


@dataclass
class DeviceConfig:
    """
    DeviceConfig automatically selects GPU if available, otherwise CPU.
    You can specify a GPU by setting the environment variable GPU_ID.
    """
    DEVICE: str = field(default_factory=lambda: (
        f"cuda:{os.environ.get('GPU_ID', '0')}" if torch.cuda.is_available() else "cpu"
    ))


@dataclass
class Config:
    """
    Main configuration consolidates paths, training, and device configurations.
    It can be extended in the future to load settings from YAML/JSON files.
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
        print(f"SPLITS_DIR: {self.paths.SPLITS_DIR}")
        print("----------------------------------")
        print("=== Training Config ===")
        print(f"WINDOW_SIZE: {self.training.WINDOW_SIZE}")
        print(f"INPUT_DIM: {self.training.INPUT_DIM}")
        print(f"HIDDEN_DIM: {self.training.HIDDEN_DIM}")
        print(f"LSTM_LAYERS: {self.training.LSTM_LAYERS}")
        print(f"NUM_EPOCHS: {self.training.NUM_EPOCHS}")
        print(f"BATCH_SIZE: {self.training.BATCH_SIZE}")
        print(f"LEARNING_RATE: {self.training.LEARNING_RATE}")
        print(f"WEIGHT_DECAY: {self.training.WEIGHT_DECAY}")
        print(f"CLIP_GRAD_NORM: {self.training.CLIP_GRAD_NORM}")
        print(f"NUM_WORKERS: {self.training.NUM_WORKERS}")
        print(f"ALPHA: {self.training.ALPHA}")
        print(f"BETA: {self.training.BETA}")
        print(f"GAMMA: {self.training.GAMMA}")
        print(f"DELTA: {self.training.DELTA}")
        print("----------------------------------")
        print("=== Scheduler Config ===")
        print(f"LR_SCHEDULER: {self.training.LR_SCHEDULER}")
        print(f"T_0: {self.training.T_0}")
        print(f"T_MULT: {self.training.T_MULT}")
        print(f"ETA_MIN: {self.training.ETA_MIN}")
        print("----------------------------------")
        print("=== Device Config ===")
        print(f"DEVICE: {self.device.DEVICE}")
        print(f"DEBUG: {self.DEBUG}")


if __name__ == "__main__":
    cfg = Config()
    cfg.print_config()
