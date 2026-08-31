import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config

class Conv1DVAE(nn.Module):
    def __init__(self, in_channels=16, latent_dim=32):
        super(Conv1DVAE, self).__init__()
        
        # Encoder (패딩 없이 차원 축소: 8 -> 6 -> 4 -> 2)
        self.conv1 = nn.Conv1d(in_channels, 8, kernel_size=3, padding=0)
        self.bn1 = nn.BatchNorm1d(8)
        self.conv2 = nn.Conv1d(8, 4, kernel_size=3, padding=0)
        self.bn2 = nn.BatchNorm1d(4)
        self.conv3 = nn.Conv1d(4, 2, kernel_size=3, padding=0)
        self.bn3 = nn.BatchNorm1d(2)
        
        self.flatten_dim = 2 * 2  # 4
        
        self.fc_enc = nn.Linear(self.flatten_dim, 100)
        self.bn_fc_enc = nn.BatchNorm1d(100)
        
        self.fc_mu = nn.Linear(100, latent_dim)
        self.fc_logvar = nn.Linear(100, latent_dim)
        
        # Decoder
        self.fc_dec1 = nn.Linear(latent_dim, 100)
        self.bn_dec1 = nn.BatchNorm1d(100)
        self.fc_dec2 = nn.Linear(100, 100)
        self.bn_dec2 = nn.BatchNorm1d(100)
        self.fc_dec3 = nn.Linear(100, self.flatten_dim)
        self.bn_dec3 = nn.BatchNorm1d(self.flatten_dim)
        
        # Transpose Conv (차원 복구: 2 -> 4 -> 5 -> 8)
        self.tconv1 = nn.ConvTranspose1d(2, 4, kernel_size=3, padding=0)
        self.bn_tconv1 = nn.BatchNorm1d(4)
        self.tconv2 = nn.ConvTranspose1d(4, 8, kernel_size=2, padding=0)
        self.bn_tconv2 = nn.BatchNorm1d(8)
        self.tconv3 = nn.ConvTranspose1d(8, in_channels, kernel_size=4, padding=0)

    def reparameterize(self, mu, logvar):
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Encoder
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.relu(self.bn2(self.conv2(h)))
        h = F.relu(self.bn3(self.conv3(h)))
        
        h = h.view(h.size(0), -1)
        h = F.relu(self.bn_fc_enc(self.fc_enc(h)))
        
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        
        # Decoder
        h_d = F.relu(self.bn_dec1(self.fc_dec1(z)))
        h_d = F.relu(self.bn_dec2(self.fc_dec2(h_d)))
        h_d = F.relu(self.bn_dec3(self.fc_dec3(h_d)))
        h_d = h_d.view(h_d.size(0), 2, 2)
        
        h_d = F.relu(self.bn_tconv1(self.tconv1(h_d)))
        h_d = F.relu(self.bn_tconv2(self.tconv2(h_d)))
        out = self.tconv3(h_d)

        if Config.SCALING_METHOD.lower() in {"minmax", "min_max", "normalization"}:
            x_recon = torch.sigmoid(out)
        else:
            x_recon = out

        # Diagnostic channel view from reconstruction residuals.
        x_ident = F.softmax(torch.abs(x - x_recon), dim=1)

        return x_recon, x_ident, mu, logvar
