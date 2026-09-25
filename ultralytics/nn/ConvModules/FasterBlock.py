import torch
import torch.nn as nn

def autopad(k, d = 1, p = None):
    if d > 1:
        k = (k - 1) * d + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

class ConvBnAct(nn.Module):
    def __init__(self, c1, c2,  k, s, p, d, g, act):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, dilation = d,
                              groups = g, padding = autopad(k, d, p), bias = False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        x = self.act(self.bn(self.conv(x)))
        return x

class PartialConv(nn.Module):
    def __init__(self, c1, ratio, kernel_size = 1, groups = 1, dilation = 1, padding = None):
        super().__init__()
        assert ratio >0 and ratio <= 1, "ratio must be in [0.0, 1.0]"
        self.x_conv = max(1, int(ratio * c1))
        self.x_skip = c1 - self.x_conv
        self.conv = nn.Conv2d(self.x_conv, self.x_conv, kernel_size,
                              stride=1,
                              padding=autopad(kernel_size, dilation, padding),
                              dilation=dilation, groups=groups, bias=False)

    def forward(self, x):
        x_conv, x_split = torch.split(x, (self.x_conv, self.x_skip), dim = 1)
        x = self.conv(x_conv)
        return torch.cat([x, x_split], dim = 1)


class FasterBlock(nn.Module):
    def __init__(self, c1, c2, k = 1 , s = 1, p = None, d = 1,g = 1, partial_ratio = 0.25, expend_ratio = 2, short_cut = True):
        super().__init__()
        self.ConBN = ConvBnAct(c1 = c1, c2 = c2,  k = k, s = s, p = p, d = d, g = g, act = False)
        self.PartialConv = PartialConv(c2 , partial_ratio, k)
        self.Conv1 = ConvBnAct(c1 = c2, c2 = c2 * expend_ratio,  k = k, s = s, p = p, d = d, g = g, act = True)
        self.Conv2 = nn.Conv2d(c2 * expend_ratio, c2, kernel_size = k, stride = s, padding = k//2, bias = False)

    def forward(self, x):
        x_shortcut = self.ConBN(x)  #[b, c1, h, w] -> [b, c2, h, w]

        x_partial = self.PartialConv(x_shortcut) #[b, c2, h, w] -> [b, c2, h, w]
        x_partial = self.Conv1(x_partial)   #[b, c2, h, w] -> [b, c2 * expend_ratio, h, w]
        x_partial = self.Conv2(x_partial)   #[b, c2 * expend_ratio, h, w] -> [b, c2, h, w]

        return x_shortcut + x_partial

if __name__ == "__main__":
    x = torch.randn(1, 64, 224, 224)
    model = FasterBlock(c1 = 64, c2 = 128, k = 3)
    out = model(x)

    print(f"input shape: {x.shape}\noutput shape: {out.shape}")