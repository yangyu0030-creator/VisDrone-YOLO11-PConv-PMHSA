import torch
import math
import torch.nn as nn

def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else tuple(x // 2 for x in k)
    return p

class ConvBNAct(nn.Module):
    """
        普通卷积块
    """
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

class DWConvBNAct(nn.Module):
    """
        深度卷积块：DWConv + BN + ReLU6
    """
    def __init__(self, in_channels, kernel_size, act = True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels = in_channels,
                              out_channels = in_channels,
                              kernel_size = kernel_size,
                              stride = 1,
                              padding = autopad(kernel_size),
                              groups = in_channels,
                              bias = False)
        self.bn = nn.BatchNorm2d(in_channels)
        self.act = nn.ReLU6(inplace = True) if act else nn.Identity()

    def forward(self, x):
        x = self.act(self.bn(self.conv(x)))
        return x

class MSBranchBlock(nn.Module):
    def __init__(self, c, k = 3, expansion = 2, act = True):
        super().__init__()
        hidden = max(1, int(expansion * c))

        self.cv1 = ConvBNAct(in_channels = c,
                             out_channels = hidden,
                             kernel_size = 1,
                             stride = 1,
                             act = act)
        self.dw = DWConvBNAct(in_channels = hidden,
                              kernel_size = k,
                              act = act)
        self.cv2 = ConvBNAct(in_channels = hidden,
                             out_channels = c,
                             kernel_size = 1,
                             stride = 1,
                             act = act)
    def forward(self, x):
        x = self.cv1(x)
        x = self.dw(x)
        x = self.cv2(x)
        return x

class MSBlock(nn.Module):
    def __init__(self, c1, c2, branches = 3,k = 3, expansion = 2, shortcut = True, act = True):
        super().__init__()
        assert branches >= 2, "The branch must be >= 2"

        self.branches = branches
        self.shortcut = shortcut and c1 == c2

        branch_channels = math.ceil(c2 / branches)
        hidden_total = branch_channels * branches

        self.cv1 = ConvBNAct(in_channels = c1, out_channels = hidden_total, kernel_size=1, act=act)

        self.blocks = nn.ModuleList([
            MSBranchBlock(branch_channels, k=k, expansion=expansion, act = act) for _ in range(branches - 1)
        ])

        self.cv2 = ConvBNAct(in_channels = hidden_total,out_channels = c2, kernel_size=1, act=act)

    def forward(self, x):
        identity = x

        x = self.cv1(x)
        xs = torch.chunk(x, self.branches, dim=1)

        ys = [xs[0]]

        for i in range(1, self.branches):
            cur = ys[i - 1] + xs[i]
            cur = self.blocks[i - 1](cur)
            ys.append(cur)

        out = torch.cat(ys, dim=1)
        out = self.cv2(out)

        if self.shortcut:
            out = out + identity

        return out

if __name__ == "__main__":
    x = torch.randn(1, 64, 224, 224)
    model = MSBlock(c1=64, c2=64)
    out = model(x)

    print(f"input shape: {x.shape}\noutput shape: {out.shape}")
