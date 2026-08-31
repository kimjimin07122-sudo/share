import torch
import torch.nn as nn


class LSTMAutoencoder(nn.Module):
    """2-layer, 32-unit LSTM autoencoder with a fully-connected bottleneck,
    per the B-Spline UAV fault-detection paper (Sensors 2025, Section 6.5:
    "two LSTM layers with 32 units each, a fully connected bottleneck layer,
    and a symmetric decoder"). No public code was available at the time of
    writing, so this is a from-scratch reimplementation.

    Input:  (batch, seq_len, input_dim) -- input_dim = num_channels * 4
            (raw, first derivative, second derivative, energy per channel;
            see feature_engineering.py)
    Output: (batch, seq_len, input_dim) -- reconstruction of the input
    """

    def __init__(self, input_dim, hidden_size=32, num_layers=2, bottleneck_dim=16):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        # Fully connected bottleneck: compresses the encoder's final hidden
        # state across all layers down to bottleneck_dim and back up, giving
        # genuine dimensionality reduction between encoder and decoder.
        self.bottleneck_down = nn.Linear(num_layers * hidden_size, bottleneck_dim)
        self.bottleneck_up = nn.Linear(bottleneck_dim, num_layers * hidden_size)

        self.decoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.output_head = nn.Linear(hidden_size, input_dim)

    def forward(self, x):
        _, (h_n, c_n) = self.encoder(x)  # h_n, c_n: (num_layers, batch, hidden)
        batch = h_n.size(1)

        h_flat = h_n.transpose(0, 1).reshape(batch, self.num_layers * self.hidden_size)
        z = torch.relu(self.bottleneck_down(h_flat))
        h_recon = self.bottleneck_up(z)
        h_recon = h_recon.reshape(batch, self.num_layers, self.hidden_size).transpose(0, 1).contiguous()

        # Teacher forcing: decoder runs over the full input sequence in one
        # vectorized call, initialized from the bottlenecked encoder state.
        dec_out, _ = self.decoder(x, (h_recon, c_n))
        return self.output_head(dec_out)
