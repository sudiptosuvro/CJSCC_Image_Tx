import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    def __init__(self, c: int, eps: float = 1e-6):
        super().__init__()
        self.ln = nn.LayerNorm(c, eps=eps)

    def forward(self, x):
        return self.ln(x.permute(0,2,3,1)).permute(0,3,1,2)


class DWSeparableConv(nn.Module):
    def __init__(self, c_in: int, c_out: int, k: int = 5, stride: int = 1):
        super().__init__()
        p = k // 2
        self.dw = nn.Conv2d(c_in, c_in, kernel_size=k, stride=stride, padding=p, groups=c_in, bias=False)
        self.pw = nn.Conv2d(c_in, c_out, kernel_size=1, bias=True)

    def forward(self, x):
        return self.pw(self.dw(x))


class SCFBEncoder(nn.Module):
    def __init__(self, c_in, c_out, k=5, stride=1,
                 gdn=None, act=None, use_res=True,
                 use_mix=True):
        super().__init__()
        p = k // 2
        self.conv = nn.Conv2d(c_in, c_out, kernel_size=k,
                              stride=stride, padding=p, bias=True)
        self.gdn  = gdn if gdn is not None else nn.Identity()
        self.act  = act if act is not None else nn.ReLU(inplace=True)
        self.mix  = nn.Conv2d(c_out, c_out, 1, bias=True) if use_mix else nn.Identity()
        self.use_res = use_res and (stride == 1) and (c_in == c_out)

    def forward(self, x):
        y = self.conv(x)
        y = self.gdn(y)
        y = self.act(y)
        y = self.mix(y)
        if self.use_res:
            y = y + x
        return y


class DSCFBEncoder(nn.Module):
    def __init__(self, c_in: int, c_out: int, k: int = 5, stride: int = 1,
                 gdn: nn.Module | None = None, act: nn.Module | None = None, use_res: bool = True):
        super().__init__()
        self.conv = DWSeparableConv(c_in, c_out, k=k, stride=stride)
        self.gdn = gdn if gdn is not None else nn.Identity()
        self.act = act if act is not None else nn.ReLU(inplace=True)

        self.use_res = use_res and (stride == 1) and (c_in == c_out)
        self.mix = nn.Conv2d(c_out, c_out, 1, bias=True)

    def forward(self, x):
        y = self.conv(x)
        y = self.gdn(y)
        y = self.act(y)
        y = self.mix(y)

        if self.use_res:
            y = y + x
        return y


class DSCFBDecoder(nn.Module):
    def __init__(self, c_in: int, c_out: int, k: int = 5, upsample: int = 1,
                 igdn: nn.Module | None = None, act: nn.Module | None = None, use_res: bool = True,
                 mode: str = "nearest", use_mix=True):
        super().__init__()
        self.upsample = upsample
        self.mode = mode
        self.conv = DWSeparableConv(c_in, c_out, k=k, stride=1)
        self.igdn = igdn if igdn is not None else nn.Identity()
        self.act = act if act is not None else nn.ReLU(inplace=True)

        self.use_res = use_res and (upsample == 1) and (c_in == c_out)
        self.mix = nn.Conv2d(c_out, c_out, 1, bias=True) if use_mix else nn.Identity()

    def forward(self, x):
        if self.upsample != 1:
            x = F.interpolate(x, scale_factor=self.upsample, mode=self.mode)

        y = self.conv(x)
        y = self.igdn(y)
        y = self.act(y)
        y = self.mix(y)

        if self.use_res:
            y = y + x
        return y