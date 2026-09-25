import torch
import torch.nn as nn

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

class DWConvBNAct(nn.Module):
    """
        深度卷积块：DWConv + BN + ReLU6
    """
    def __init__(self, in_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels = in_channels,
                              out_channels = in_channels,
                              kernel_size = kernel_size,
                              stride = 1,
                              padding = autopad(kernel_size),
                              groups = in_channels,
                              bias = False)
        self.bn = nn.BatchNorm2d(in_channels)
        self.act = nn.ReLU6(inplace = True)

    def forward(self, x):
        x = self.act(self.bn(self.conv(x)))
        return x

class FCM(nn.Module):
    def __init__(self, c1, c2, alpha=0.5, act=True):
        super().__init__()
        assert c1 >= 2, "FCM requires c1 >= 2"
        assert 0.0 < alpha < 1.0, "FCM requires 0 < alpha < 1"

        self.c_sem = max(1, min(int(round(c1 * alpha)), c1 - 1))
        self.c_spa = c1 - self.c_sem

        self.conv_sem = ConvBNAct(self.c_sem, c2, kernel_size = 3, stride = 1, act = act)
        self.conv_spa = ConvBNAct(self.c_spa, c2, kernel_size = 1, stride = 1, act = act)

        self.DWC = DWConvBNAct(c2, kernel_size = 3)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_sigmoid = nn.Sigmoid()

        self.spa_conv = nn.Conv2d(in_channels = c2, out_channels = 1, kernel_size = 1, bias = False)
        self.spa_bn = nn.BatchNorm2d(1)
        self.spa_sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_sem, x_spa = torch.split(x, [self.c_sem, self.c_spa], dim = 1)

        x_sem = self.conv_sem(x_sem)
        x_spa = self.conv_spa(x_spa)

        x_sem_attn = self.channel_sigmoid(self.avg_pool(self.DWC(x_sem)))
        x_spa_attn = self.spa_sigmoid(self.spa_bn(self.spa_conv(x_spa)))

        out = x_sem_attn * x_spa + x_spa_attn * x_sem

        return out

if __name__ == "__main__":
    x = torch.randn(1, 64, 224, 224)
    model = FCM(c1=64, c2=64)
    out = model(x)

    print(f"input shape: {x.shape}\noutput shape: {out.shape}")
