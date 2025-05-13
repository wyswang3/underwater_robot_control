import os
from pathlib import Path
from dataclasses import dataclass, field, asdict
import torch
from typing import Dict, Any

# ╭─────────────────────────────╮
# │ 1. 工具函数                 │
# ╰─────────────────────────────╯

def _p(*parts) -> Path:
    """跨平台路径拼接（返回 pathlib.Path）。"""
    return Path(*parts).expanduser().resolve()

def _mkdirs(*dirs: Path):
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

def _pretty_dict(d: Dict[str, Any], indent: int = 2) -> str:
    return "\n".join(f"{' '*indent}{k:<15}: {v}" for k, v in d.items())

# ╭─────────────────────────────╮
# │ 2. 路径配置                 │
# ╰─────────────────────────────╯
@dataclass
class PathsConfig:
    PROJECT_ROOT: Path = field(default_factory=lambda: _p(__file__).parent)
    TRAIN_FEATURES_FILE: Path = field(init=False)
    TRAIN_ACCEL_LABELS_FILE: Path = field(init=False)
    TRAIN_THRUST_LABELS_FILE: Path = field(init=False)
    TRAIN_VELOCITY_LABELS_FILE: Path = field(init=False)
    TRAIN_ANGULAR_ACCEL_LABELS_FILE: Path = field(init=False)
    THRUST_MATRIX_FILE: Path = field(init=False)
    MODEL_DIR: Path = field(init=False)
    LOG_DIR: Path = field(init=False)
    SPLITS_DIR: Path = field(init=False)

    def __post_init__(self):
        data_proc = self.PROJECT_ROOT / "data" / "processed"
        data_raw  = self.PROJECT_ROOT / "data" / "raw"

        self.TRAIN_FEATURES_FILE        = data_proc / "train_features.npy"
        self.TRAIN_ACCEL_LABELS_FILE    = data_proc / "train_accel_labels.npy"
        self.TRAIN_THRUST_LABELS_FILE   = data_proc / "train_thrust_labels.npy"
        self.TRAIN_VELOCITY_LABELS_FILE = data_proc / "train_velocity_labels.npy"
        self.TRAIN_ANGULAR_ACCEL_LABELS_FILE = data_proc / "train_angular_accel_labels.npy"
        self.THRUST_MATRIX_FILE = data_raw / "thrust_allocation_matrix.csv"

        self.MODEL_DIR  = self.PROJECT_ROOT / "models" / "checkpoints"
        self.LOG_DIR    = self.PROJECT_ROOT / "logs"
        self.SPLITS_DIR = self.PROJECT_ROOT / "data" / "splits"

        _mkdirs(self.MODEL_DIR, self.LOG_DIR, self.SPLITS_DIR)

# ╭─────────────────────────────╮
# │ 3. 训练超参数               │
# ╰─────────────────────────────╯
@dataclass
class TrainingConfig:
    # 选择模型分支：'hybrid', 'direct', 'pure_lstm'
    MODEL_TYPE: str = "pure-lstm"

    # 数据及模型相关参数
    WINDOW_SIZE: int = 9
    INPUT_DIM: int = 14

    # 通用LSTM参数
    HIDDEN_DIM: int = 1024
    LSTM_LAYERS: int = 1
    LSTM_DROPOUT: float = 0.2

    # 端到端(Direct)或Hybrid融合参数
    E2E_HIDDEN_FACTOR: float = 0.5
    FUSION_HIDDEN_DIM: int = 512

    # 物理网络相关参数
    HYDRO_HIDDEN: int = 256
    MATRIX_DIM: int = 6
    HYDRO_MIN_DIAG: float = 1e-2

    # 纯LSTM网络参数
    PURE_LSTM_HIDDEN_DIM: int = 400
    PURE_LSTM_LAYERS: int = 1
    PURE_LSTM_DROPOUT: float = 0.2
    PURE_LSTM_OUTPUT_DIM: int = 6

    # MLP网络参数
    MLP_HIDDEN_DIMS: list = field(default_factory=lambda: [256, 128])

    # 损失相关参数
    BASE_LOSS_TYPE: str = "mse"
    LAMBDA_PHY: float = 0.5
    BETA_REG: float = 0.1
    LOSS_EPS: float = 1e-6

    # 训练超参数
    NUM_EPOCHS: int = 2
    BATCH_SIZE: int = 32
    LEARNING_RATE: float = 1e-4
    WEIGHT_DECAY: float = 1e-4
    CLIP_GRAD_NORM: float = 3.0
    NUM_WORKERS: int = 2

    # Learning rate scheduler
    LR_SCHEDULER: bool = True
    LR_SCHEDULER_TYPE: str = "poly"
    LR_SCHEDULER_PCT_START: float = 0.3
    MIN_LR: float = 5e-7
    MAX_LR: float = 5e-4
    STEP_PER_BATCH: bool = True
    WARMUP_STEPS: int = 400

# ╭─────────────────────────────╮
# │ 4. 设备配置                 │
# ╰─────────────────────────────╯
@dataclass
class DeviceConfig:
    DEVICE: str = field(default_factory=lambda: (
        f"cuda:{os.getenv('GPU_ID', 0)}" if torch.cuda.is_available() else "cpu"
    ))

# ╭─────────────────────────────╮
# │ 5. 主配置                   │
# ╰─────────────────────────────╯
@dataclass
class Config:
    paths   : PathsConfig    = field(default_factory=PathsConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    device  : DeviceConfig   = field(default_factory=DeviceConfig)
    DEBUG   : bool           = False

    def print_config(self):
        print("\n========== Config ==========")
        print("• Paths")
        for k, v in asdict(self.paths).items():
            print(f"  {k:<25}: {v}")
        print("• Training")
        print(_pretty_dict(asdict(self.training)))
        print("• Device")
        print(f"  DEVICE              : {self.device.DEVICE}")
        print(f"  DEBUG               : {self.DEBUG}")
        print("============================\n")

# ╭─────────────────────────────╮
# │ 6. 快速测试                 │
# ╰─────────────────────────────╯
if __name__ == "__main__":
    cfg = Config()
    cfg.print_config()