import torch
import torch.nn as nn
import math

__all__ = ["SRU", "CRU", "SCConv"]

def autopad(k, p = None, d = 1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

def make_disvisible_groups(channels, groups):
    groups = min(channels, groups)
    while groups > 1 and channels % groups != 0:
        groups -= 1
    return max(groups, 1)

class ConvBNAct(nn.Module):
    def __init__(self, c1, c2, k=1, p=None, d=1,g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=c1, out_channels=c2,
                              kernel_size=k, stride=1, padding=autopad(k, p, d),
                              groups=g, dilation=d)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class SRU(nn.Module):
    def __init__(self, c1, c2, groups_num=4, gate_threshold=0.5):
        super().__init__()
        self.proj = ConvBNAct(c1=c1, c2=c2, k=1, act=False) if c1 != c2 else nn.Identity()

        gn_groups = make_disvisible_groups(c2, groups_num)
        self.gn = nn.GroupNorm(num_groups=gn_groups, num_channels=c2, affine=True)
        self.sigmoid = nn.Sigmoid()
        self.gate_threshold = gate_threshold

        assert c2 % 2 == 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)    #[b, c2, h, w]

        gn_x = self.gn(x)   #[b, c2, h, w]
        gamma = self.gn.weight  #[c2]
        w_gamma = gamma / (gamma.sum() + 1e-6)  #标准化
        w_gamma = w_gamma.view(1, -1, 1, 1) #[1, c2, 1, 1]

        reweights = self.sigmoid(gn_x * w_gamma)    #[B, C2, H, W]
        info_mask = (reweights >= self.gate_threshold).float()
        noninfo_mask = 1.0 - info_mask

        x_info = x * info_mask
        x_noninfo = x * noninfo_mask

        x11, x12 = torch.chunk(x_info, 2, dim = 1)  #[b, c2 / 2, h, w]
        x21, x22 = torch.chunk(x_noninfo, 2, dim = 1)

        out = torch.cat([x11 + x22, x12 + x21], dim = 1) #[b, c2, h, w]
        return out

class CRU(nn.Module):
    def __init__(self, c1, c2, alpha=0.5,squeeze_ratio=2,
                 group_kernel=3, group_size=2, act=True):
        super().__init__()
        assert alpha >0 and alpha < 1, "alpha must be in (0, 1)"

        self.proj = ConvBNAct(c1=c1, c2=c2, k=1, act=False) if c1 != c2 else nn.Identity()

        self.c_alpha = max(1, int(alpha * c2))
        self.c_beta = (int)((1 - alpha) * c2)
        assert self.c_beta > 0, "c_beta must be > 0"
        self.c_alpha_mid = max(1, int(self.c_alpha // squeeze_ratio))
        self.c_beta_mid = max(1, int(self.c_beta // squeeze_ratio))

        self.cbn_alpha = ConvBNAct(c1=self.c_alpha, c2=self.c_alpha_mid, k=1, act=act)
        self.cnb_beta = ConvBNAct(c1=self.c_beta, c2=self.c_beta_mid, k=1, act=act)

        gwc_groups = math.gcd(self.c_alpha_mid, c2)
        gwc_groups = math.gcd(gwc_groups, group_size)
        gwc_groups = max(gwc_groups, 1)

        self.gwc_alpha = ConvBNAct(c1 = self.c_alpha_mid, c2 = c2,
                            k= group_kernel, g = gwc_groups,act=act)
        self.pwc_alpha = ConvBNAct(c1 = self.c_alpha_mid, c2 = c2, k=1, act=act)

        self.pwc_beta = ConvBNAct(c1 = self.c_beta_mid, c2 = c2 -self.c_beta_mid , k=1, act=act)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)    #[b, c1, h, w] -> [b, c2, h, w]
        x_alpha, x_beta = torch.split(x,[self.c_alpha, self.c_beta], dim=1)

        x_alpha = self.cbn_alpha(x_alpha)
        x_beta = self.cnb_beta(x_beta)

        x_alpha = self.gwc_alpha(x_alpha) + self.pwc_alpha(x_alpha)
        x_beta = torch.cat((self.pwc_beta(x_beta),x_beta), dim = 1)

        x_alpha_attn = self.pool(x_alpha)
        x_beta_attn = self.pool(x_beta)
        x_out = torch.stack([x_alpha_attn, x_beta_attn],dim=1)
        x_out = torch.softmax(x_out, dim=1)
        x_1, x_2 = x_out[:,0], x_out[:,1]

        out = x_alpha * x_1 + x_beta * x_2
        return out

class SCConv(nn.Module):
    def __init__(self, c1,c2,
                 groups_num=4, gate_threshold=0.5,
                 alpha=0.5, squeeze_ratio=2,group_kernel=3, group_size=2, act=True
                 ):
        super().__init__()
        self.sru = SRU(c1, c2, groups_num=group_size, gate_threshold=gate_threshold,
                       )
        self.cru = CRU(c2, c2, alpha=alpha, squeeze_ratio=squeeze_ratio,
                               group_kernel=group_kernel, group_size=group_size, act=act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.sru(x)
        x = self.cru(x)
        return x


if __name__ == "__main__":
    x = torch.randn(1, 64, 224, 224)
    model = SCConv(c1 = 64, c2 = 64)
    out = model(x)

    print(f"input shape: {x.shape}\noutput shape: {out.shape}")
