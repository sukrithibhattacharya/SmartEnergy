import torch
import torch.nn as nn

class BiLSTM_KAN(nn.Module):
    """
    BiLSTM + KAN model for multi-target load forecasting
    Based on BiKAN-LoadNet paper (IEEE Access, 2026)
    """
    def __init__(self, seq_dim=96, weather_dim=6, hidden_dim=128, num_layers=2):
        super().__init__()

        # BiLSTM Encoder - captures time patterns in load sequences
        self.bilstm = nn.LSTM(
            input_size=seq_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2,
            bidirectional=True
        )

        # Residual Connection - preserves recent fine-grained details
        self.residual_proj = nn.Linear(seq_dim, hidden_dim * 2)

        # Weather Encoder (MLP) - integrates weather features
        self.weather_mlp = nn.Sequential(
            nn.Linear(weather_dim, 64),
            nn.ReLU(),
            nn.Linear(64, hidden_dim * 2)
        )

        # KAN-like Output Layer
        self.kan = nn.Sequential(
            nn.Linear(hidden_dim * 4, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # 3 targets: avg, max, min
        )

    def forward(self, seq_input, weather_input):
        # seq_input: (batch, 7, 96)
        # weather_input: (batch, weather_dim)

        # BiLSTM processes load sequence
        lstm_out, _ = self.bilstm(seq_input)
        last_step = lstm_out[:, -1, :]

        # Residual connection
        residual = self.residual_proj(seq_input[:, -1, :])
        seq_feat = last_step + residual

        # Weather features
        weather_feat = self.weather_mlp(weather_input)

        # Fusion
        fusion = torch.cat([seq_feat, weather_feat], dim=1)

        # KAN output
        out = self.kan(fusion)

        return out