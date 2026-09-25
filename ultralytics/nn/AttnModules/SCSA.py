import torch
import torch.nn as nn
from torch import Tensor
import math
import torch.nn.functional as F

class MS_DWConv1d(nn.Module):
    def __init__(self, channels: int, k: int):
        super().__init__()
        self.dw = nn.Conv1d(channels, channels, kernel_size = k, padding = k // 2, bias = False, groups = channels)

    def forward(self, x: Tensor) -> Tensor:
        return self.dw(x)


class CA_SHSA(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.qkv = nn.Conv2d(channels, 3 * channels, kernel_size = 1, groups = channels, bias = False)

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.size()
        n = h * w
        qkv = self.qkv(x)
        q, k, v = torch.chunk(qkv, 3, dim = 1)
        q = q.view(b, c, n)
        k = k.view(b, c, n)
        v = v.view(b, c, n)

        attn = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(max(1, n))
        attn = F.softmax(attn, dim = 2)     #[b, n, n]

        out = torch.bmm(attn, v).view(b, c, h, w)
        return out

def _safe_gn_groups(C:int, k:int) -> int:
    if C % k == 0:
        return k
    for g in range(k, 0, -1):
        if C % g == 0:
            return g
    return -1

class SCSA(nn.Module):
    def __init__(self, channels: int, n: int = 4, kernels = (3, 5, 7, 9), pool_hw: int = 7):
        super().__init__()
        assert channels > 0
        assert n > 0
        assert channels % n == 0

        self.C = channels
        self.n = n
        self.Cg = channels // n
        if isinstance(kernels, (list, tuple)):
            assert len(kernels) == n
        else:
            raise TypeError("kernels should be a list or tuple")

        self.ms_h = nn.ModuleList([MS_DWConv1d(self.Cg, int(k)) for k in kernels])
        self.ms_w = nn.ModuleList([MS_DWConv1d(self.Cg, int(k)) for k in kernels])

        self.g = _safe_gn_groups(channels, 4)
        self.gn_h = nn.GroupNorm(self.g, channels)
        self.gn_w = nn.GroupNorm(self.g, channels)
        self.sigmoid = nn.Sigmoid()

        self.pool_hw = int(pool_hw)
        self.gn1 = nn.GroupNorm(1, channels)
        self.ca_shsa = CA_SHSA(channels)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.size()
        x_h = x.mean(dim=-1)    #[b,c,h]
        x_w = x.mean(dim=-2)    #[b,c,w]

        x_h_chunks = torch.chunk(x_h, self.n, dim = 1)  # n * [b, c/n, h]
        x_w_chunks = torch.chunk(x_w, self.n, dim = 1)

        h_out = [self.ms_h[i](x_h_chunks[i]) for i in range(self.n)]    #n * [b, c/n, h]
        w_out = [self.ms_w[i](x_w_chunks[i]) for i in range(self.n)]

        h_cat = torch.cat(h_out, dim = 1)  #[b, c, h]
        w_cat = torch.cat(w_out, dim = 1)  #[b, c, w]

        Ah = self.sigmoid(self.gn_h(h_cat))
        Aw = self.sigmoid(self.gn_w(w_cat))

        x_s = Ah.unsqueeze(-1) * Aw.unsqueeze(-2) * x

        p = self.pool_hw
        xp = F.avg_pool2d(x_s, (p,p))
        xp = self.gn1(xp)
        xp = self.ca_shsa(xp)
        mc = torch.sigmoid(self.avg_pool(xp))
        out = mc * x_s

        return out


if __name__ == "__main__":
    input = torch.randn(2, 64, 80, 80)
    model = SCSA(channels=64)
    output = model(input)
    print(f"input_size: {input.shape}\noutput_size: {output.shape}")







