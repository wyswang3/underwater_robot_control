import torch
import torch.nn as nn

class PureLSTMNet(nn.Module):
    """
    End-to-end LSTM network for sequence-to-value regression.
    Input: tensor of shape (batch, seq_len, input_size)
    Output: tensor of shape (batch, output_size)
    """
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float, output_size: int):
        super(PureLSTMNet, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, input_size]
        lstm_out, _ = self.lstm(x)
        # Use the last time step's output
        last_step = lstm_out[:, -1, :]
        out = self.fc(last_step)
        return out
