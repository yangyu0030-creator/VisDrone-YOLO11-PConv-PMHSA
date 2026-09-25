import torch
import torch.nn as nn
import torch.nn.functional as F
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
        self.act = nn.ReLU(inplace = True) if act else nn.Identity()

    def forward(self, x):
        x = self.act(self.bn(self.conv(x)))
        return x

class SepConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, k = 1, p = None, d = 1, act = True):
        super().__init__()
        assert in_channels == out_channels
        if p is None:
            p = autopad(k = k, d = d)

        self.SepConv = nn.Conv2d(in_channels, out_channels, kernel_size = k, stride = 1,
                                 padding = p, dilation = d, groups = in_channels, bias = False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace = True) if act else nn.Identity()

    def forward(self, x):
        out = self.act(self.bn(self.SepConv(x)))
        return out

class ASPP_Pool_Branch(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = 1, bias = False),
            nn.ReLU(inplace = True),
        )

    def forward(self, x):
        size = x.shape[-2:]
        x = self.pool(x)
        x = self.conv(x)
        x = F.interpolate(x, size = size, mode = "bilinear", align_corners = False)
        return x

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, k = 3,
                 ds = [12, 24, 36], act = True):
        super().__init__()

        self.pre_conv = ConvBNAct(in_channels, out_channels, k=1) if in_channels != out_channels else nn.Identity()
        in_channels = out_channels
        self.dw_branches = nn.ModuleList([
            SepConvBNAct(in_channels, out_channels, k, None, d, act) for d in ds
        ])
        self.Conv1 = ConvBNAct(in_channels, out_channels, k = 1, s = 1)
        self.pool_branch = ASPP_Pool_Branch(in_channels, out_channels)
        self.Conv2 = ConvBNAct(out_channels * (len(ds) + 2), out_channels, k = 1, s = 1)
        self.dropout = nn.Dropout2d(p = 0.2)

    def forward(self, x):
        x = self.pre_conv(x)
        y1 = self.Conv1(x)

        outs = [y1]
        for branch in self.dw_branches:
            outs.append(branch(x))
        outs.append(self.pool_branch(x))
        out = torch.cat(outs, dim = 1)
        out = self.Conv2(out)
        out = self.dropout(out)
        return out

if __name__ == "__main__":
    x = torch.randn(1, 256, 20, 20)
    model = ASPP(256, 256)
    out = model(x)

    print(f"input shape: {x.shape}\noutput shape: {out.shape}")