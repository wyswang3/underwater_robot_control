import os
import torch
from torch.utils.data import DataLoader
import argparse
from config import Config

cfg = Config()

# 更新导入新的网络结构和损失函数
from models.dynamics_net import BetterHydroNet, PhysicsAwareLoss
from utils.preprocessing import load_thrust_allocation_matrix  # 如有需要，保留或删除
from utils.dataset import PreprocessedDataset
from utils.training import validate_model  # 假设该函数已更新，能够处理模型返回的 v_pred 参数

def main(args):
    device = torch.device(cfg.device.DEVICE)

    # 构造评估数据集
    full_dataset = PreprocessedDataset(
        features_file=cfg.paths.TRAIN_FEATURES_FILE,
        accel_file=cfg.paths.TRAIN_ACCEL_LABELS_FILE,
        angular_accel_file=cfg.paths.TRAIN_ANGULAR_ACCEL_LABELS_FILE,
        thrust_file=cfg.paths.TRAIN_THRUST_LABELS_FILE,
        window_size=cfg.training.WINDOW_SIZE
    )
    eval_loader = DataLoader(
        full_dataset,
        batch_size=cfg.training.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.training.NUM_WORKERS
    )

    # 构造网络模型（与训练时保持一致）
    model = BetterHydroNet(
        window_size=cfg.training.WINDOW_SIZE,
        input_dim=cfg.training.INPUT_DIM,
        hidden_dim=cfg.training.HIDDEN_DIM,
        lstm_layers=cfg.training.LSTM_LAYERS
    ).to(device)

    # 加载模型检查点
    checkpoint_path = args.checkpoint
    if not os.path.exists(checkpoint_path):
        print("Checkpoint file not found:", checkpoint_path)
        return
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)

    # 初始化物理感知损失函数
    # 使用配置文件中的参数（若不存在则使用默认值）
    criterion = PhysicsAwareLoss(
        lambda_phy=getattr(cfg.training, "LAMBDA_PHY", 0.4),
        beta_reg=getattr(cfg.training, "BETA_REG", 0.1),
        eps=getattr(cfg.training, "LOSS_EPS", 1e-6)
    )

    # 调用验证函数
    # 注意：新模型 forward 返回了 v_pred，因此 validate_model 必须提取到所有5项输出，
    # 并在调用损失函数时传入 v_pred。
    avg_loss = validate_model(model, criterion, eval_loader)
    print("Evaluation complete, average loss: {:.6f}".format(avg_loss))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate dynamics model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/checkpoints/model_checkpoint.pt",
        help="Path to model checkpoint"
    )
    args = parser.parse_args()
    main(args)
