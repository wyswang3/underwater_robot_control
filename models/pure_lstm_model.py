import torch
import torch.nn as nn

class PureLSTMNet(nn.Module):
    """
    Pure LSTM model for mapping concatenated power and IMU windows to accel predictions.
    Accepts two inputs: power_window and imu_window, each of shape (B, T, C).
    """
    def __init__(self, input_size, hidden_size, num_layers, dropout, output_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.output_size = output_size

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, power_window, imu_window=None):
        # Concatenate power and IMU along feature dim if both provided
        if imu_window is not None:
            # both have same T
            x = torch.cat([power_window, imu_window], dim=-1)
        else:
            x = power_window
        # x: (B, T, C)
        out, _ = self.lstm(x)
        # take last time step
        last = out[:, -1, :]
        return self.fc(last)
