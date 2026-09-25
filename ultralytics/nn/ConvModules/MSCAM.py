import torch
import torch.nn as nn

def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else tuple(x // 2 for x in k)
    return p

def channel_shuffle(x: torch.Tensor, groups: int) -> torch.Tensor:
    b, c, h, w = x.size()
    assert c % groups == 0, "channels must be divisible by groups"

    x = x.view(b, groups, c // groups, h, w)

    x = x.transpose(1, 2).contiguous()

    x = x.view(b, c, h, w)
    return x

class ConvBNAct(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride = 1,
                 padding = None,
                 dilation = 1,
                 groups = 1,
                 act = False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels = in_channels,
                              out_channels = out_channels,
                              kernel_size = kernel_size,
                              stride = stride,
                              padding = autopad(kernel_size, padding, dilation),
                              dilation = dilation,
                              groups = groups,
                              bias = False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        x = self.act(self.bn(self.conv(x)))
        return x

class CAB(nn.Module):
    def __init__(self, c1, c2, reduction=16):
        super().__init__()
        self.proj = ConvBNAct(in_channels = c1,
                 out_channels = c2,
                 kernel_size = 1,
                 act = False)   if c1 != c2 else nn.Identity()

        hidden = max(1, c2 // reduction)

        self.fc1 = nn.Conv2d(in_channels = c2,
                        out_channels = hidden,
                        kernel_size = 1,
                        bias = False)
        self.relu = nn.ReLU(inplace=True)

        self.fc2 = nn.Conv2d(in_channels = hidden,
                             out_channels = c2,
                             kernel_size = 1,
                             bias = False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.proj(x)    #[b, c1, h, w] -> [b, c2, h, w]

        max_feat = nn.AdaptiveMaxPool2d(1)(x)  #[b, c2, 1, 1]
        avg_feat = nn.AdaptiveAvgPool2d(1)(x)  #[b, c2, 1, 1]

        max_out = self.fc2(self.relu(self.fc1(max_feat)))
        avg_out = self.fc2(self.relu(self.fc1(avg_feat)))
        attn = self.sigmoid(max_out + avg_out)  #通道方向注意力

        out = x * attn
        return out

class SAB(nn.Module):
    def __init__(self, c1, c2, sab_k=7):
        super().__init__()
        self.proj = ConvBNAct(in_channels = c1,
                              out_channels = c2,
                              kernel_size = 1,
                              act = False)  if c1 != c2 else nn.Identity()
        self.conv = nn.Conv2d(in_channels = 2,
                             out_channels = 1,
                             kernel_size = sab_k,
                             padding = autopad(sab_k),
                             bias = False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.proj(x)

        channel_max = torch.max(x, dim = 1, keepdim=True)[0]    #[b, 1, h, w]
        channel_avg = torch.mean(x, dim = 1, keepdim = True)    #[b, 1, h, w]
        attn = torch.cat([channel_max, channel_avg], dim = 1)   #[b, 2, h, w]
        attn = self.sigmoid(self.conv(attn))    #[b, 1, h, w],  空间注意力

        out = x * attn
        return out

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

class MSDC(nn.Module):
    def __init__(self,
                 in_channels,
                 kernel_sizes = (1, 3, 5),
                 shuffle_g=2):
        super().__init__()
        self.blocks = nn.ModuleList([DWConvBNAct(in_channels, k) for k in kernel_sizes])
        self.shuffle_g = shuffle_g

    def forward(self, x):
        outs = [block(x) for block in self.blocks]

        out = torch.stack(outs, dim=0).sum(dim=0)

        out = channel_shuffle(out, groups = self.shuffle_g)

        return out

class MSCB(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 expansion = 2,
                 kernel_sizes = (1, 3, 5),
                 shuffle_g = 2,
                 shortcut = True):
        super().__init__()
        hidden = max(1, int(expansion * in_channels))

        self.pw1 = ConvBNAct(in_channels = in_channels,
                            out_channels = hidden,
                            kernel_size = 1)
        self.msdc = MSDC(in_channels = hidden)
        self.pw2 = ConvBNAct(in_channels = hidden,
                             out_channels = out_channels,
                             kernel_size = 1)
        self.shortcut = shortcut and in_channels == out_channels

    def forward(self, x):
        indentity = x

        x = self.pw1(x)
        x = self.msdc(x)
        x = self.pw2(x)

        if self.shortcut:
            x = x + indentity

        return x

class MSCAM(nn.Module):
    def __init__(self, c1,
                 c2,
                 reduction = 16,
                 sab_k = 7,
                 expansion=2,
                 kernel_sizes=(1, 3, 5),
                 shuffle_g=2,
                 shortcut=True
                 ):
        super().__init__()
        self.CAB = CAB(c1, c2, reduction)
        self.SAB =  SAB(c2, c2, sab_k=sab_k)
        self.MSCB = MSCB(in_channels = c2,
                 out_channels = c2,
                 expansion = expansion,
                 kernel_sizes = kernel_sizes,
                 shuffle_g = shuffle_g,
                 shortcut = shortcut)

    def forward(self, x):
        x = self.CAB(x)
        x = self.SAB(x)
        x = self.MSCB(x)
        return x

if __name__ == "__main__":
    x = torch.randn(1, 64, 224, 224)
    model = MSCAM(c1=64, c2=64)
    out = model(x)

    print(f"input shape: {x.shape}\noutput shape: {out.shape}")
