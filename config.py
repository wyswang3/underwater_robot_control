# config.py  ── 完全替换原文件即可
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
    PROJECT_ROOT: Path = field(
        default_factory=lambda: _p(__file__).parent
    )

    # 以下字段在 __post_init__ 中生成
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
    # 原有字段…
    WINDOW_SIZE: int = 5
    INPUT_DIM: int = 14
    HIDDEN_DIM: int = 512
    LSTM_LAYERS: int = 2
    # 新增模型相关字段
    HYDRO_HIDDEN: int = 128
    MATRIX_DIM: int = 6
    HYDRO_MIN_DIAG: float = 1e-2
    LSTM_DROPOUT: float = 0.3
    LAYER_DROPOUT: float = 0.4
    RESIDUAL_DROPOUT: float = 0.4
    VELOCITY_HIDDEN: int = 128

    # 可选：损失相关参数
    LAMBDA_PHY: float = 0.4
    BETA_REG: float = 0.1
    LOSS_EPS: float = 1e-6
    # 训练
    NUM_EPOCHS  : int   = 120
    BATCH_SIZE  : int   = 32
    LEARNING_RATE : float = 1e-5
    WEIGHT_DECAY  : float = 1e-5
    CLIP_GRAD_NORM: float = 1.0
    NUM_WORKERS   : int   = 2

    # 物理损失权重（保留占位）
    ALPHA: float = 1.0
    BETA : float = 0.1
    GAMMA: float = 0.01
    DELTA: float = 0.1

    # Scheduler
    LR_SCHEDULER: bool  = True
    T_0   : int   = 10
    T_MULT: int   = 2
    ETA_MIN: float = 1e-6


# ╭─────────────────────────────╮
# │ 4. 设备配置                 │
# ╰─────────────────────────────╯
@dataclass
class DeviceConfig:
    DEVICE: str = field(
        default_factory=lambda: (
            f"cuda:{os.getenv('GPU_ID', 0)}"
            if torch.cuda.is_available()
            else "cpu"
        )
    )


# ╭─────────────────────────────╮
# │ 5. 主配置                   │
# ╰─────────────────────────────╯
@dataclass
class Config:
    paths   : PathsConfig    = field(default_factory=PathsConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    device  : DeviceConfig   = field(default_factory=DeviceConfig)
    DEBUG   : bool           = False

    # -------- 打印 --------
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
