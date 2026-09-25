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

class SENetV2Lite(nn.Module):
    def __init__(self, in_channels, out_channels, reduction = 16, branches = 4):
        super().__init__()
        assert branches >= 1
        hidden_channels = max(8, in_channels // reduction)
        self.branch = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels, hidden_channels, kernel_size = 1, stride = 1, bias = False),
                nn.ReLU(inplace = True),
            )
            for _ in range(branches)
        ])

        self.fuse = nn.Conv2d(hidden_channels * branches, out_channels, kernel_size = 1, stride = 1, bias = False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x

        outs = []
        for branch in self.branch:
            outs.append(branch(x))

        out = torch.cat(outs, dim = 1)
        out = self.fuse(out)
        out_attn = self.sigmoid(out)
        out = out_attn * identity

        return out

class SPPF(nn.Module):
    def __init__(self, in_channels, out_channels, kernels = (5, 5, 5), act = True):
        super().__init__()
        assert in_channels == out_channels
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
        return out

class SE_SPPF(nn.Module):
    def __init__(self, in_channels, out_channels, act = True):
        super().__init__()
        self.se = SENetV2Lite(in_channels, in_channels, reduction = 16)
        self.spp = SPPF(in_channels, in_channels, act = act)
        self.conv1 = ConvBNAct(in_channels * 2, out_channels, k = 1, s = 1, act = act)
        self.conv2 = ConvBNAct(out_channels, out_channels, k = 3, s = 1, act = act)

    def forward(self, x):
        x = self.se(x)
        y = self.spp(x)
        out = torch.cat([x, y], dim = 1)
        out = self.conv1(out)
        out = self.conv2(out)
        return out

if __name__ == "__main__":
    x = torch.randn(1, 64, 224, 224)
    model = SE_SPPF(64, 64)
    out = model(x)

    print(f"input shape: {x.shape}\noutput shape: {out.shape}")
