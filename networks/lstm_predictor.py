import torch.nn as nn


class LSTMPredictor(nn.Module):
    """2-layer, 32-unit LSTM forecaster: reads a past window of engineered
    features and predicts the single next timestep's feature vector (same
    input_dim = num_channels * 4 space as LSTMAutoencoder, so the downstream
    IF/fusion/channel-diagnosis code needs no changes). No decoder -- just an
    encoder LSTM followed by a linear prediction head on the final hidden
    state, matching last week's GRU predictor pattern but swapped to LSTM and
    kept in feature-engineered space instead of raw-channel space.

    Input:  (batch, window, input_dim)
    Output: (batch, input_dim) -- predicted next-timestep feature vector
    """

    def __init__(self, input_dim, hidden_size=32, num_layers=2):
        super().__init__()
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, input_dim)

    def forward(self, x):
        _, (h_n, _) = self.encoder(x)  # h_n: (num_layers, batch, hidden)
        last_layer_h = h_n[-1]  # (batch, hidden)
        return self.head(last_layer_h)
