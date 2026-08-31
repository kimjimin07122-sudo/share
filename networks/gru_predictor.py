import torch.nn as nn


class GRUPredictor(nn.Module):
    """Minimal GRU forecasting baseline, inspired by TiTAD's GRU branch
    (Liu, Wang & Wu, 2025) but without the Transformer/time-invariant memory
    and Feature Fusion modules -- ablation in that paper shows GRU alone
    accounts for most of the detection performance (F1 0.842 -> 0.528 when
    GRU is removed vs. 0.842 -> 0.785 when the time-invariant module is
    removed), so this is the cheapest piece worth trying first.

    Input:  (batch, seq_len, num_channels) -- past `seq_len` steps
    Output: (batch, num_channels) -- predicted next step
    """

    def __init__(self, num_channels=16, hidden_size=64, num_layers=1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=num_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, num_channels)

    def forward(self, x):
        _, h_n = self.gru(x)
        last_hidden = h_n[-1]
        return self.head(last_hidden)
