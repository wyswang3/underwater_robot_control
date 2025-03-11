import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
import numpy as np

# 从 config 导入配置
from config import Config
cfg = Config()

######################################################
#  Swish 激活函数
######################################################
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

######################################################
# skew_symmetric 函数，用于构造歪对称矩阵
######################################################
def skew_symmetric(v):
    B = v.shape[0]
    vx, vy, vz = v[:, 0], v[:, 1], v[:, 2]
    zeros = torch.zeros_like(vx)
    mat = torch.stack([
        zeros, -vz, vy,
        vz, zeros, -vx,
        -vy, vx, zeros
    ], dim=1).view(B, 3, 3)
    return mat

######################################################
# SPDMatrixNet (对称正定矩阵)
######################################################
class SPDMatrixNet(nn.Module):
    def __init__(self, input_dim, size, diag_eps=1e-4):
        super().__init__()
        self.size = size
        self.diag_eps = diag_eps
        self.fc = nn.Sequential(
            nn.Linear(input_dim, size * (size + 1) // 2),
            nn.Tanh()
        )

    def forward(self, x):
        B = x.shape[0]
        tril_elements = self.fc(x)
        L = torch.zeros(B, self.size, self.size, device=x.device, dtype=x.dtype)
        indices = torch.tril_indices(self.size, self.size)
        L[:, indices[0], indices[1]] = tril_elements
        diag = L.diagonal(dim1=1, dim2=2)
        diag = F.softplus(diag) + self.diag_eps
        L = L - torch.diag_embed(L.diagonal(dim1=1, dim2=2)) + torch.diag_embed(diag)
        return torch.matmul(L, L.transpose(1, 2))

######################################################
# DiagonalMatrixNet (对角矩阵)
######################################################
class DiagonalMatrixNet(nn.Module):
    def __init__(self, input_dim, size):
        super().__init__()
        self.fc = nn.Linear(input_dim, size)
        self.size = size

    def forward(self, x):
        diag_vals = torch.exp(self.fc(x))
        return torch.diag_embed(diag_vals)

######################################################
# SymmetricMatrixNet (对称矩阵)
######################################################
class SymmetricMatrixNet(nn.Module):
    def __init__(self, input_dim, size):
        super().__init__()
        self.size = size
        self.fc = nn.Linear(input_dim, size * (size + 1) // 2)

    def forward(self, x):
        B = x.size(0)
        triup = self.fc(x)
        M = torch.zeros(B, self.size, self.size, device=x.device, dtype=x.dtype)
        ind = torch.triu_indices(self.size, self.size)
        M[:, ind[0], ind[1]] = triup
        M = M + M.transpose(1, 2) - torch.diag_embed(M.diagonal(dim1=1, dim2=2))
        return M

######################################################
# 动力学方程约束层
######################################################
class DynamicsConstraintLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, tau, M, J, D, imu_window):
        B = tau.shape[0]
        v_linear = imu_window[:, -1, :3]
        omega = imu_window[:, -1, 3:]
        v_6 = torch.cat([v_linear, omega], dim=1)
        inertia_block = torch.zeros(B, 6, 6, device=tau.device, dtype=tau.dtype)
        inertia_block[:, :3, :3] = M
        inertia_block[:, 3:, 3:] = J
        C = torch.zeros_like(inertia_block)
        C[:, :3, 3:] = -skew_symmetric(v_linear)
        C[:, 3:, 3:] = -skew_symmetric(omega)
        cdv = (C + D).matmul(v_6.unsqueeze(-1))
        net_force = tau.unsqueeze(-1) - cdv
        a_6 = torch.linalg.solve(inertia_block, net_force)
        return a_6.squeeze(-1)

######################################################
# 增强型动力学网络
######################################################
class EnhancedPhysicsNet(nn.Module):
    def __init__(self, thrust_matrix, window_size=5, hidden_dim=256):
        super().__init__()
        self.register_buffer('thrust_matrix', thrust_matrix)
        input_dim = (8 + 6) * window_size
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            Swish(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.LayerNorm(hidden_dim)
        )
        self.thrust_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(window_size, 32),
                Swish(),
                nn.Linear(32, 1)
            ) for _ in range(8)
        ])
        self.mass_net = SPDMatrixNet(hidden_dim, 3)
        self.inertia_net = SPDMatrixNet(hidden_dim, 3)
        self.damping_net = DiagonalMatrixNet(hidden_dim, 6)
        self.physics_layer = DynamicsConstraintLayer()
        self.window_size = window_size
        self.hidden_dim = hidden_dim

    def forward(self, power_window, imu_window):
        B, T, _ = power_window.shape
        x = torch.cat([power_window, imu_window], dim=-1).flatten(1)
        feat = self.encoder(x)
        motor_thrusts = []
        for i in range(8):
            seq = power_window[:, :, i]
            thrust_i = self.thrust_nets[i](seq).squeeze(-1)
            motor_thrusts.append(thrust_i)
        motor_thrusts = torch.stack(motor_thrusts, dim=1)
        tau = torch.matmul(self.thrust_matrix, motor_thrusts.unsqueeze(-1)).squeeze(-1)
        M = self.mass_net(feat)
        J = self.inertia_net(feat)
        D = self.damping_net(feat)
        accel_pred = self.physics_layer(tau, M, J, D, imu_window)
        return {
            'tau': tau,
            'mass_matrix': M,
            'inertia_matrix': J,
            'damping_matrix': D,
            'accel_pred': accel_pred
        }

######################################################
# 增强型损失函数
######################################################
class EnhancedDynamicsLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.1, gamma=0.01):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, outputs, targets):
        if isinstance(outputs, dict):
            accel_pred = outputs.get('accel_pred', outputs)
            tau_pred = outputs.get('tau', None)
            thrust_loss = F.mse_loss(tau_pred, targets['thrust']) if tau_pred is not None else 0.0
            if 'mass_matrix' in outputs and 'inertia_matrix' in outputs:
                M_det = torch.det(outputs['mass_matrix']).clamp_min(1e-7)
                J_det = torch.det(outputs['inertia_matrix']).clamp_min(1e-7)
                reg_loss = -(torch.log(M_det).mean() + torch.log(J_det).mean())
            else:
                reg_loss = 0.0
        else:
            accel_pred = outputs
            thrust_loss = 0.0
            reg_loss = 0.0

        accel_loss = F.mse_loss(accel_pred, targets['accel'])
        total_loss = accel_loss + self.alpha * thrust_loss + self.beta * reg_loss
        return total_loss

######################################################
# 端到端网络
######################################################
class DirectMappingNet(nn.Module):
    def __init__(self, window_size=5, hidden_dim=256):
        super().__init__()
        self.window_size = window_size
        self.input_dim = (8 + 6) * window_size
        self.encoder_layers = nn.ModuleList([
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        ])
        self.decoder_layers = nn.ModuleList([
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 6)
        ])

    def forward(self, power_window, imu_window):
        x = torch.cat([power_window, imu_window], dim=-1).flatten(1)
        for layer in self.encoder_layers:
            x = layer(x)
        features = x
        for layer in self.decoder_layers:
            features = layer(features)
        return features

######################################################
# 混合网络
######################################################
class HybridDynamicsModel(nn.Module):
    def __init__(self, physics_net, e2e_net):
        super().__init__()
        self.physics_net = physics_net
        self.e2e_net = e2e_net
        self.active_net = 'physics'

    def forward(self, *args):
        if self.active_net == 'physics':
            return self.physics_net(*args)
        else:
            return self.e2e_net(*args)

    def switch_to_e2e(self):
        self.active_net = 'e2e'

######################################################
# 训练与验证
######################################################
def train_model(model, loss_fn, train_loader, epochs=100, lr=1e-4):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    model.train()
    device = torch.device(cfg.device.DEVICE)
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in train_loader:
            for k in batch:
                batch[k] = batch[k].to(device)
            outputs = model(batch['power_window'], batch['imu_window'])
            loss = loss_fn(outputs, batch)
            if torch.isnan(loss):
                model.switch_to_e2e()
                optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
                continue
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * batch['accel'].size(0)
        avg_loss = total_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.6f}")

def validate_model(model, loss_fn, val_loader):
    model.eval()
    device = torch.device(cfg.device.DEVICE)
    total_loss = 0.0
    total_accel_loss = 0.0
    total_thrust_loss = 0.0
    sample_count = 0
    with torch.no_grad():
        for batch in val_loader:
            for k in batch:
                batch[k] = batch[k].to(device)
            outputs = model(batch['power_window'], batch['imu_window'])
            loss = loss_fn(outputs, batch)
            batch_size = batch['accel'].size(0)
            total_loss += loss.item() * batch_size
            if isinstance(outputs, dict):
                accel_loss = F.mse_loss(outputs['accel_pred'], batch['accel']).item()
                thrust_loss = F.mse_loss(outputs['tau'], batch['thrust']).item()
                total_accel_loss += accel_loss * batch_size
                total_thrust_loss += thrust_loss * batch_size
            else:
                accel_loss = F.mse_loss(outputs, batch['accel']).item()
                total_accel_loss += accel_loss * batch_size
            sample_count += batch_size
    avg_total_loss = total_loss / sample_count
    avg_accel_loss = total_accel_loss / sample_count
    print(f"Validation Total Loss: {avg_total_loss:.6f}")
    print(f"Validation Acceleration Loss: {avg_accel_loss:.6f}")
    if isinstance(outputs, dict):
        avg_thrust_loss = total_thrust_loss / sample_count
        print(f"Validation Thrust Loss: {avg_thrust_loss:.6f}")
    return avg_total_loss
