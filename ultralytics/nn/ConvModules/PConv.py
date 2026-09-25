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
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size = 1,
                 stride = 1,
                 padding= None,
                 groups = 1,
                 dilation = 1,
                 act = True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size,
                              stride = stride, padding = autopad(kernel_size,padding, dilation),
                              groups = groups, dilation = dilation, bias = False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        x = self.act(self.bn(self.conv(x)))
        return x

class AsymPadConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size = 1,
                 stride = 1,
                 mode = 'h',
                 padding = (0, 0, 0, 0),
                 act = True):
        super().__init__()
        assert mode in ('h', 'v')

        self.pad = nn.ZeroPad2d(padding)

        if mode == 'h':
            kernel = (1, kernel_size)
        else:
            kernel = (kernel_size, 1)

        self.conv = ConvBNAct(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel, stride = stride, padding=0, act=act)

    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        return x

class PConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size = 3,
                 stride = 1,
                 act = True
    ):
        super().__init__()
        channels_branch = math.ceil(out_channels / 4)
        self.channels_branch = channels_branch

        self.b1 = AsymPadConv(in_channels, channels_branch, kernel_size, mode='h', padding=(0, kernel_size, 1, 0),act=act)
        self.b2 = AsymPadConv(in_channels, channels_branch, kernel_size, mode='v', padding=(0, 1, 0, kernel_size),act=act)
        self.b3 = AsymPadConv(in_channels, channels_branch, kernel_size, mode='h', padding=(kernel_size, 0, 0, 1),act=act)
        self.b4 = AsymPadConv(in_channels, channels_branch, kernel_size, mode='v', padding=(1, 0, kernel_size, 0),act=act)

        self.fuse = ConvBNAct(channels_branch * 4, out_channels, kernel_size = 2, stride = stride, padding= 0, act = act)

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)

        y = torch.cat([y1, y2, y3, y4], 1)
        out = self.fuse(y)
        return out

if __name__ == "__main__":
    x = torch.randn(1, 64, 224, 224)
    model = PConv(64, 64)
    out = model(x)

    print(f"input shape: {x.shape}\noutput shape: {out.shape}")
