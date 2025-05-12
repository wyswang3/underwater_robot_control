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
    # 数据及模型相关参数
    WINDOW_SIZE: int = 9
    INPUT_DIM: int = 14
    HIDDEN_DIM: int = 1024
    LSTM_LAYERS: int = 1

    # 网络结构参数
    # 例如混合网络中端到端网络使用的隐藏层数及其融合层尺寸
    E2E_HIDDEN_FACTOR: float = 0.5      # 例如端到端网络的隐藏层维度为 HIDDEN_DIM * E2E_HIDDEN_FACTOR
    FUSION_HIDDEN_DIM: int = 512          # 门控融合网络中的隐藏层尺寸

    # 模型相关（针对物理网络等）
    HYDRO_HIDDEN: int = 256
    MATRIX_DIM: int = 6
    HYDRO_MIN_DIAG: float = 1e-2
    LSTM_DROPOUT: float = 0.2
    LAYER_DROPOUT: float = 0.3
    RESIDUAL_DROPOUT: float = 0.3
    VELOCITY_HIDDEN: int = 128

    # 损失相关参数
    BASE_LOSS_TYPE: str = "mse"
    LAMBDA_PHY: float = 0.5
    BETA_REG: float = 0.1
    LOSS_EPS: float = 1e-6

    # 训练超参数
    NUM_EPOCHS: int = 420
    BATCH_SIZE: int = 32
    LEARNING_RATE: float = 1e-4
    WEIGHT_DECAY: float = 1e-4
    CLIP_GRAD_NORM: float = 3.0
    NUM_WORKERS: int = 2

    # Scheduler 参数（这里使用 OneCycleLR）
    LR_SCHEDULER: bool = True
    LR_SCHEDULER_TYPE: str = "poly"
    LR_SCHEDULER_PCT_START: float = 0.3
    MIN_LR: float = 5e-7
    MAX_LR: float = 5e-4
    STEP_PER_BATCH: bool = True      # ★ Poly 需要 batch 级更新
    # Warm up 参数（备用）
    WARMUP_STEPS: int = 400            # 只在 poly/自定义 Lambda 时读取

# ╭─────────────────────────────╮
# │ 4. 设备配置                 │
# ╰─────────────────────────────╯
@dataclass
class DeviceConfig:
    DEVICE: str = field(
        default_factory=lambda: (
            f"cuda:{os.getenv('GPU_ID', 4)}"
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
