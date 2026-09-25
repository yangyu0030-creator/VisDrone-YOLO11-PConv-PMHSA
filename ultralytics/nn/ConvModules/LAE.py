import torch
import torch.nn as nn
import math

def autopad(k, pad = None, d = 1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if pad is None:
        pad = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return pad

class Rearrange(nn.Module):
    def __init__(self, scale = 2):
        super().__init__()
        self.scale = scale
        self.unshuffle = nn.PixelUnshuffle(scale)

    def forward(self, x):
        b, c, h, w = x.shape
        assert h % self.scale == 0 and w % self.scale == 0
        x = self.unshuffle(x)   #[b, c * (scale ** 2), h // scale, w // scale]
        return x.view(b, c, self.scale ** 2, h // self.scale, w // self.scale)

class ConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 1, stride=1, padding=None, dilation=1, groups = 1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              autopad(kernel_size, padding, dilation),
                              groups = groups, dilation = dilation)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class   shareGroupsMap(nn.Module):
    def __init__(self, c1, c2, scale = 2,group_div = 16, act = True):
        super().__init__()
        in_ch = c1 * (scale ** 2)
        out_ch = c2 * (scale ** 2)

        base_g = max(1, in_ch // group_div)
        groups = math.gcd(in_ch, out_ch)
        groups = math.gcd(groups, base_g)
        groups = max(groups, 1)

        self.map = ConvBNAct(in_ch, out_ch, groups = groups, act = act)
        self.c2 = c2

    def forward(self, x):
        b, c, n, h, w = x.shape
        x = x.view(b, c * n, h, w)
        x = self.map(x)
        x = x.view(b, self.c2, n, h, w)
        return x

class LAE(nn.Module):
    def __init__(self, in_channels, out_channels, pool_k = 3, scale = 2, group_div = 16, act = True):
        super().__init__()
        self.scale = scale
        self.avg_pool = nn.AvgPool2d(kernel_size = pool_k, stride = 1, padding = pool_k // 2)
        self.rearrange_feat = Rearrange(scale)
        self.rearrange_attn = Rearrange(scale)
        self.share_map = shareGroupsMap(in_channels, out_channels, group_div = group_div, act = act, scale = scale)
        self.softmax = nn.Softmax(dim = 2)

    def forward(self, x):
        b, c, h, w = x.shape
        x_feat = self.rearrange_feat(x) #[b, c, (scale ** 2), h // scale, w // scale]
        x_faet = self.share_map(x_feat) #[b, c2, (scale ** 2), h // scale, w // scale]

        x_pool = self.avg_pool(x)
        x_attn = self.rearrange_attn(x_pool)
        x_attn = self.softmax(self.share_map(x_attn))

        out = x_faet * x_attn   #[b, c2, (scale ** 2), h // scale, w // scale]
        out = out.sum(dim = 2)  #[b, c2, h // scale, w // scale]
        return out

if __name__ == "__main__":
    x = torch.randn(1, 64, 224, 224)
    model = LAE(in_channels=64, out_channels=32, pool_k = 3)
    out = model(x)

    print(f"input shape: {x.shape}\noutput shape: {out.shape}")
