import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock1D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = x
        x = self.act(self.conv1(x))
        x = self.conv2(x)
        x = self.act(x + r)
        return x


class ResidualChannelEstimator(nn.Module):
    def __init__(self, nfft: int, hidden_channels: int = 64, n_blocks: int = 3):
        super().__init__()
        self.nfft = nfft

        layers = []
        in_ch = 2
        ch = hidden_channels

        layers.append(nn.Conv1d(in_ch, ch, kernel_size=3, padding=1))
        layers.append(nn.ReLU(inplace=True))

        for _ in range(n_blocks):
            layers.append(ResBlock1D(ch))

        layers.append(nn.Conv1d(ch, 2, kernel_size=3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, Yp: torch.Tensor, Xp: torch.Tensor) -> torch.Tensor:
        H_ls = Yp / (Xp + 1e-8)
        H_ls_mean = H_ls.mean(dim=1)
        feat = torch.stack([H_ls_mean.real, H_ls_mean.imag], dim=1)
        delta = self.net(feat)
        delta_c = torch.complex(delta[:, 0, :], delta[:, 1, :])
        H_hat = H_ls_mean + delta_c
        return H_hat


class ResidualChannelEstimatorRobust(nn.Module):

    def __init__(self, nfft: int, hidden_channels: int = 64, n_blocks: int = 4):
        super().__init__()
        self.nfft = nfft

        in_ch = 4
        ch = hidden_channels

        layers = [
            nn.Conv1d(in_ch, ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        ]
        for _ in range(n_blocks):
            layers.append(ResBlock1D(ch))
        layers.append(nn.Conv1d(ch, 2, kernel_size=3, padding=1))

        self.net = nn.Sequential(*layers)

    def forward(self, Yp: torch.Tensor, Xp: torch.Tensor, snr_db: torch.Tensor) -> torch.Tensor:

        B = Yp.size(0)

        H_ls = Yp / (Xp + 1e-8)
        H_ls_mean = H_ls.mean(dim=1)

        H_rep = H_ls_mean.unsqueeze(1)
        pilot_residual = (Yp - Xp * H_rep).abs().mean(dim=1)

        if snr_db.dim() == 0:
            snr_db = snr_db.expand(B)
        elif snr_db.dim() == 1 and snr_db.size(0) == 1:
            snr_db = snr_db.expand(B)

        snr_feat = snr_db.view(B, 1).repeat(1, self.nfft) / 25.0

        feat = torch.stack(
            [
                H_ls_mean.real,
                H_ls_mean.imag,
                snr_feat,
                pilot_residual,
            ],
            dim=1)

        delta = self.net(feat)
        delta_c = torch.complex(delta[:, 0, :], delta[:, 1, :])

        H_hat = H_ls_mean + delta_c
        return H_hat