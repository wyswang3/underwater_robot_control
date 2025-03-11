#analyze.py
import os
import argparse
import torch
from torch.utils.data import DataLoader

# 从评估模块中导入 evaluate_main（评估流程）
from evaluate import main as evaluate_main
# 导入可视化函数（新版：针对每个维度绘制子图）
from utils.visualization import visualize_predictions, plot_loss_curve
# 导入配置
from config import Config
cfg = Config()
# 导入模型结构和预处理工具
from models.dynamics_net import HybridDynamicsModel, EnhancedPhysicsNet, DirectMappingNet
from utils.preprocessing import load_thrust_allocation_matrix
# 导入数据集类（已拆分到 utils/dataset.py）
from utils.dataset import PreprocessedDataset

def load_model(device):
    """
    根据配置加载模型结构并加载权重
    """
    checkpoint_path = r"D:\神经网络训练\underwater_robot_control\model_checkpoint.pt"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"模型检查点不存在: {checkpoint_path}")

    # 加载推力分配矩阵
    thrust_matrix_np = load_thrust_allocation_matrix(cfg.paths.THRUST_MATRIX_FILE)
    thrust_matrix = torch.tensor(thrust_matrix_np, device=device)

    # 初始化物理引导网络和端到端网络，并构成混合网络
    physics_net = EnhancedPhysicsNet(
        thrust_matrix,
        window_size=cfg.training.WINDOW_SIZE,
        hidden_dim=cfg.training.HIDDEN_DIM
    )
    e2e_net = DirectMappingNet(
        window_size=cfg.training.WINDOW_SIZE,
        hidden_dim=cfg.training.HIDDEN_DIM
    )
    model = HybridDynamicsModel(physics_net, e2e_net).to(device)

    # 加载权重
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"已加载模型权重：{checkpoint_path}. 当前网络模式: {model.active_net}")
    return model

def get_dataloader():
    """
    构造并返回评估用数据集的 DataLoader
    """
    dataset = PreprocessedDataset(
        features_file=cfg.paths.TRAIN_FEATURES_FILE,
        accel_file=cfg.paths.TRAIN_ACCEL_LABELS_FILE,
        angular_accel_file=cfg.paths.TRAIN_ANGULAR_ACCEL_LABELS_FILE,
        thrust_file=cfg.paths.TRAIN_THRUST_LABELS_FILE,
        window_size=cfg.training.WINDOW_SIZE
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.training.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.training.NUM_WORKERS
    )
    return loader

def main():
    parser = argparse.ArgumentParser(description="整合评估与可视化分析的脚本")
    parser.add_argument("--evaluate", action="store_true", help="执行评估流程")
    parser.add_argument("--visualize", action="store_true", help="执行预测可视化分析")
    args = parser.parse_args()

    # 如果未指定参数，则默认同时执行评估和可视化
    if not (args.evaluate or args.visualize):
        args.evaluate = True
        args.visualize = True

    device = torch.device(cfg.device.DEVICE)

    # ===================== 执行评估 =====================
    if args.evaluate:
        eval_args = argparse.Namespace(
            checkpoint=r"D:\神经网络训练\underwater_robot_control\model_checkpoint.pt"
        )
        print(">>> 开始评估 <<<")
        evaluate_main(eval_args)
        print(">>> 评估完成 <<<")

    # ===================== 执行可视化分析 =====================
    if args.visualize:
        print(">>> 开始可视化分析 <<<")
        model = load_model(device)
        dataloader = get_dataloader()
        save_fig_path = os.path.join(cfg.paths.SPLITS_DIR, "model_prediction_by_dimension.png")
        visualize_predictions(model, dataloader, device, save_path=save_fig_path)
        print(f"可视化结果已保存到: {save_fig_path}")
        print(">>> 可视化分析完成 <<<")

if __name__ == "__main__":
    main()

