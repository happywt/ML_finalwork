import torch
import torch.nn as nn

class LSTMNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_steps):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
            bidirectional=True
        )
        self.head = nn.Linear(hidden_dim * 2, output_steps)

    def forward(self, x):
        output_seq, _ = self.rnn(x)
        last_timestep = output_seq[:, -1, :]
        return self.head(last_timestep)
