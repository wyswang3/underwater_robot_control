# ROV Underwater Dynamics Identification (MLP Thrust Branch)

This branch (`feature/pure-lstm-mlp`) introduces a refined neural-network-based approach for system identification of an ROV's underwater dynamics. It replaces the original per-motor LSTM thrust estimators with a compact MLP, simplifies the physics branch, and updates data handling & evaluation scripts for clarity and performance.

## 🚀 Branch Highlights

* **MLP-Based Thrust Mapping**: Replaces `ThrustLSTM` with `ThrustMLP` to map the 8‑channel power vector directly to thrusts, improving training stability and reducing complexity.
* **Simplified Physics Net**: Updated `EnhancedPhysicsNet` to consume only the last timestep's power, generate inertia/drag matrices via learned diagonals, and solve accelerations with an analytical residual.
* **Modular Dataset & Evaluation**:

  * `dataset.py`: Enhanced `PreprocessedDataset` with robust NaN/Inf checks, sliding-window reshaping, and optional outlier cleaning.
  * `evaluate.py`: Unified evaluation script supporting all branches, with fallback RMSE computation.
* **Unified Training Entry**: `main.py` (formerly `train.py`) offers CLI flags for scheduler selection, AMP, and window-size overrides, plus structured logging & artifact saving.

## 📦 Installation

1. **Clone the repository** (ensure you’re on this branch):

   ```bash
   git clone https://github.com/wyswang3/underwater_robot_control.git
   cd underwater_robot_control
   git checkout feature/pure-lstm-mlp
   ```
2. **Create virtual environment & install dependencies**:

   ```bash
   python3.9 -m venv .venv
   source .venv/bin/activate    # Linux/macOS
   .\.venv\Scripts\activate   # Windows PowerShell
   pip install -r requirements.txt
   ```

## 🎯 Usage

### Training

```bash
python main.py \
  --scheduler=polynomial \  # onecyclelr | cosine | polynomial
  --amp \                     # optional mixed-precision
  --window_size=9             # override default
```

* Model checkpoints and logs will save under `models/checkpoints` and `logs/`.
* Plots (loss\_curve, compare, global accuracy, segment fit) appear in `data/splits`.

### Evaluation

```bash
python evaluate.py --checkpoint models/checkpoints/model_hybrid.pt
```

* Computes validation loss; if NaN, falls back to RMSE.

## 🔧 Configuration

All hyperparameters/functionality live in `config.py`:

* **`MODEL_TYPE`**: `pure_lstm` | `direct` | `hybrid`
* **Scheduler**: `polynomial` (default), `onecyclelr`, `cosine`
* **Window Size**, **LR**, **Batch Size**, **Epochs**, **Dropout**, etc.

## 📂 Directory Structure

```
├── config.py            # Global configurations
├── main.py              # Unified training script
├── evaluate.py          # Model evaluation with fallback
├── models/
│   └── dynamics_net.py  # Core networks & loss
├── utils/
│   ├── dataset.py       # Data loading & cleaning
│   ├── training.py      # Training loop & AMP support
│   ├── torch_helper.py  # Numeric utilities
│   └── ...
├── data/processed/      # Preprocessed .npy features & labels
├── data/raw/            # Raw thrust allocation matrix
└── data/splits/         # Generated plots & split info
```

## 🤝 Contributing

This branch is intended for experimental MLP thrust mapping. Feel free to:

* Tweak hyperparameters in `config.py`
* Extend `ThrustMLP` architecture
* Integrate additional sensors (e.g., PWM, sonar, SLAM outputs)
* Report issues or open PRs against this branch

---

*Branch maintained by wyswang3*
