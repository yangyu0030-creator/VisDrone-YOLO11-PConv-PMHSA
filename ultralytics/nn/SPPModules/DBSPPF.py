import torch
import torch.nn as nn
import math

def autopad(k, p = None, d = 1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else tuple(x // 2 for x in k)
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
        hidden_channels = max(1, in_channels // 2)

        self.conv1 = ConvBNAct(in_channels, hidden_channels, k = 1, s = 1, act = act)
        self.MaxPools = nn.ModuleList([
            nn.MaxPool2d(kernel_size = k, stride = 1, padding = k // 2) for k in kernels
        ])
        self.conv2 = ConvBNAct(hidden_channels * 4, out_channels, k = 1, s = 1, act = act)

    def forward(self, x):

        y = self.conv1(x)
        outs = [y]
        for max_pool in self.MaxPools:
            y = max_pool(y)
            outs.append(y)
        out = torch.cat(outs, dim = 1)
        out = self.conv2(out)
        return outs, out

class DBS_SPPF(nn.Module):
    def __init__(self, in_channels, out_channels,kernels = (5, 5, 5), act = True):
        super().__init__()
        hidden_channels = max(1, in_channels // 2)

        if isinstance(kernels, int):
            kernels = [kernels]
        self.kernels = kernels

        self.sppf = SPPF(in_channels, out_channels, kernels, act)
        self.conv = ConvBNAct(in_channels, hidden_channels, k = 1,act = act)
        self.blocks = nn.ModuleList([
            DWConvBNAct(hidden_channels, hidden_channels)
            for _ in range(len(kernels))
        ])
        self.conv1 = ConvBNAct(hidden_channels, out_channels, k = 1,act = act)
        self.conv2 = ConvBNAct(out_channels * 2, out_channels, k = 1,act = act)

    def forward(self, x):
        sppf_outs, cbs_out = self.sppf(x)
        x = self.conv(x)
        identity = x
        for n in range(len(self.kernels) - 1):
            x = self.blocks[n](x + sppf_outs[n] + sppf_outs[n + 1])
        x = self.blocks[-1](x + sppf_outs[-1])
        out = self.conv1(identity * x)
        out = torch.cat((out, cbs_out), dim = 1)
        return self.conv2(out)

if __name__ == "__main__":
    x = torch.randn(1, 64, 224, 224)
    model = DBS_SPPF(64, 64)
    out = model(x)

    print(f"input shape: {x.shape}\noutput shape: {out.shape}")

