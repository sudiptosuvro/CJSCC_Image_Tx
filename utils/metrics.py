import torch
import torch.nn.functional as F
import math


def mse_torch(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    mse = F.mse_loss(x_hat, x, reduction="none")
    return mse.flatten(1).mean(dim=1)

def psnr_torch(x_hat: torch.Tensor, x: torch.Tensor, data_range: float = 1.0, eps: float = 1e-10) -> torch.Tensor:
    mse = mse_torch(x_hat, x)
    return 10.0 * torch.log10((data_range ** 2) / (mse + eps))

def _gaussian_1d(window_size: int, sigma: float, device, dtype):
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return g

def _create_gaussian_window(window_size: int, sigma: float, channel: int, device, dtype):
    g1 = _gaussian_1d(window_size, sigma, device, dtype).view(1, 1, 1, -1)
    g2 = _gaussian_1d(window_size, sigma, device, dtype).view(1, 1, -1, 1)
    window = (g2 @ g1).view(1, 1, window_size, window_size)
    window = window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim_torch(x_hat: torch.Tensor, x: torch.Tensor, data_range: float = 1.0, window_size: int = 11, 
               sigma: float = 1.5, k1: float = 0.01, k2: float = 0.03, eps: float = 1e-12) -> torch.Tensor:

    assert x_hat.dim() == 4 and x.dim() == 4, "Expected NCHW."
    B, C, H, W = x.shape
    device, dtype = x.device, x.dtype

    window = _create_gaussian_window(window_size, sigma, C, device, dtype)
    mu_x  = F.conv2d(x,     window, padding=window_size//2, groups=C)
    mu_y  = F.conv2d(x_hat, window, padding=window_size//2, groups=C)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sig_x2 = F.conv2d(x * x,         window, padding=window_size//2, groups=C) - mu_x2
    sig_y2 = F.conv2d(x_hat * x_hat, window, padding=window_size//2, groups=C) - mu_y2
    sig_xy = F.conv2d(x * x_hat,     window, padding=window_size//2, groups=C) - mu_xy

    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2

    num = (2 * mu_xy + c1) * (2 * sig_xy + c2)
    den = (mu_x2 + mu_y2 + c1) * (sig_x2 + sig_y2 + c2)

    ssim_map = num / (den + eps)
    return ssim_map.flatten(1).mean(dim=1)