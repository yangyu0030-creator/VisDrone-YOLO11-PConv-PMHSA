import torch
import torch.nn as nn

def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

class ConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=None, dilation=1, groups=1, bias=False, act = True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              autopad(kernel_size, padding, dilation),
                              dilation, groups, bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))

class SpaceToDepth(nn.Module):
    def __init__(self, scale:int = 2):
        super().__init__()
        self.scale = scale
        self.op = nn.PixelUnshuffle(scale)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        assert h % self.scale == 0 and w % self.scale == 0
        return self.op(x)


class SPDConv(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, kernel_size=3, scale: int = 2, act:bool = True):
        super().__init__()
        self.scale = scale
        self.spd = SpaceToDepth(scale)  #[b, c, h, w] -> [b, c * (scale ** 2), h // scale, w // scale]
        c_mid = in_channels * (scale ** 2)

        # [b, c * (scale ** 2), h // scale, w // scale] -> [b, out_channel, h // scale, w // scale]
        self.conv = ConvBNAct(c_mid, out_channels, kernel_size=kernel_size, act=act)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        return self.conv(self.spd(x))

if __name__ == "__main__":
    x = torch.randn(1, 64, 224, 224)
    model = SPDConv(in_channels=64, out_channels=15, kernel_size=3, scale=2, act=True)
    out = model(x)

    print(f"input shape: {x.shape}\noutput shape: {out.shape}")