import torch
import torch.nn as nn
import math

def autopad(k, p = None, d = 1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

class ConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, k = 1,
                 s = 1, p = None, d = 1, g = 1, act = True):
        super().__init__()
        if p is None:
            p = autopad(k = k, d = d)

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = k, stride = s,
                              padding = p, groups = g, dilation = d, bias = False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace = True) if act else nn.Identity()

    def forward(self, x):
        x = self.act(self.bn(self.conv(x)))
        return x

class DWConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, k = 1, s = 1, p = None, d = 1,  act = True):
        super().__init__()
        assert in_channels == out_channels

        if p is None:
            p = autopad(k)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = k, stride = s,
                              padding = p, groups = in_channels, dilation= d, bias = False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace = True) if act else nn.Identity()

    def forward(self, x):
        x = self.act(self.bn(self.conv(x)))
        return x

class SPPF(nn.Module):
    def __init__(self, in_channels, out_channels, kernels = (5, 5, 5), act = True):
        super().__init__()
        assert in_channels == out_channels
        hidden_channels = max(1, in_channels // 2)

        self.conv1 = ConvBNAct(in_channels, hidden_channels, k = 1, s = 1, act = act)
        self.MaxPools = nn.ModuleList([
            nn.MaxPool2d(kernel_size = k, stride = 1, padding = k // 2) for k in kernels
        ])
        self.conv2 = ConvBNAct(hidden_channels * (len(kernels) + 1), out_channels, k = 1, s = 1, act = act)

    def forward(self, x):

        y = self.conv1(x)
        outs = [y]
        for max_pool in self.MaxPools:
            y = max_pool(y)
            outs.append(y)
        out = torch.cat(outs, dim = 1)
        out = self.conv2(out)
        return out

class LSKA(nn.Module):
    def __init__(self, in_channels, out_channels, kernels = [[1, 7], [7, 1]], act = True):
        super().__init__()
        assert in_channels == out_channels
        self.blocks = nn.Sequential(*[
            DWConvBNAct(in_channels, out_channels, k = k, s = 1, act = act)
            for k in kernels
        ])
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = 1, bias = False)

    def forward(self, x):
        identity = x
        x = self.blocks(x)
        out = self.conv(x) + identity
        return out

class SPFF_LSKA(nn.Module):
    def __init__(self, in_channels, out_channels, act = True, k_sppf = [5, 5, 5], k_lska = [[1, 7], [7, 1]]):
        super().__init__()
        self.sppf = SPPF(in_channels, in_channels,  kernels = k_sppf,act = act)
        self.lska = LSKA(in_channels, in_channels, kernels = k_lska, act = act)
        self.conv = ConvBNAct(in_channels, out_channels, k = 1, s = 1, act = act)

    def forward(self, x):
        out = self.sppf(x)
        out = self.lska(out)
        out = self.conv(out)
        return out

if __name__ == "__main__":
    x = torch.randn(1, 64, 224, 224)
    model = SPFF_LSKA(64, 64)
    out = model(x)

    print(f"input shape: {x.shape}\noutput shape: {out.shape}")
