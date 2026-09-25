import torch
import torch.nn as nn
import math

def autopad(k, d = 1, p = None):
    if d > 1:
        k = (k - 1) * d + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

class ConvBnAct(nn.Module):
    def __init__(self, c1, c2,  k=1, s=1, p=None, d=1, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, dilation = d,
                              groups = g, padding = autopad(k, d, p), bias = False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        x = self.act(self.bn(self.conv(x)))
        return x

class RFEMBranch(nn.Module):
    def __init__(self, c1, c2, dilation = 3):
        super().__init__()
        self.conv1 = ConvBnAct(c1 = c1, c2 = c2, k = 1, s = 1)
        self.conv2 = ConvBnAct(c1 = c2, c2 = c2, k = 3, s = 1, d = dilation)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class RFEM(nn.Module):
    def __init__(self, c1, c2,dilations=(3, 5, 7),e = 0.5, short_cut=True, act=True):
        super().__init__()
        self.shortcut = short_cut
        self.cv_short = ConvBnAct(c1 = c1, c2 = c2, k = 1, s = 1, act = False)

        hidden_total= max(1, int(c2 * e))
        branch_num = len(dilations)
        branch_c = math.ceil(hidden_total / branch_num)

        self.branches = nn.ModuleList([RFEMBranch(c1, branch_c, d) for d in dilations])
        self.cv_fuse = ConvBnAct(branch_num * branch_c, c2, k = 1, s = 1, act = False)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        temp = self.cv_short(x) if self.shortcut else nn.Identity(x)

        ys = [branch(x) for branch in self.branches]
        y  = torch.cat(ys, dim = 1)
        y = self.cv_fuse(y)
        y = y + temp
        return self.act(y)

if __name__ == "__main__":
    x = torch.randn(1, 64, 224, 224)
    model = RFEM(c1 = 64, c2 = 64)
    out = model(x)

    print(f"input shape: {x.shape}\noutput shape: {out.shape}")


