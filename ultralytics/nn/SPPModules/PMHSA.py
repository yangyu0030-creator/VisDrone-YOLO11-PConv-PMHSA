import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def autopad(k, p = None, d = 1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else tuple(x // 2 for x in k)
    return p

class ConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, k = 1,
                 s = 1, p = None, d = 1, g = 1, act = True):
        super().__init__()
        if p is None:
            p = autopad(k = k, d = d)

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = k, stride = s,
                              padding = p, groups = g, dilation = d, bias = False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace = True) if act else nn.Identity()

    def forward(self, x):
        x = self.act(self.bn(self.conv(x)))
        return x

class DWConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, k = 1, s = 1, p = None, d = 1,  act = True):
        super().__init__()
        assert in_channels == out_channels

        if p is None:
            p = autopad(k)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = k, stride = s,
                              padding = p, groups = in_channels, dilation= d, bias = False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace = True) if act else nn.Identity()

    def forward(self, x):
        x = self.act(self.bn(self.conv(x)))
        return x

class APWD(nn.Module):
    def __init__(self, in_channels, out_channels, dw_k = 3, s = 1):
        super().__init__()
        assert in_channels == out_channels
        self.dwconv = nn.Conv2d(in_channels, out_channels, kernel_size = dw_k, stride = s,
                                padding = dw_k // 2, groups = in_channels, bias = False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace = True)

    def forward(self, x, out_size):
        # 在 CPU 上执行自适应池化，然后再搬回 MPS 设备
        x_pool = F.adaptive_avg_pool2d(x.cpu(), out_size).to(x.device)
        out = x_pool + self.act(self.bn(self.dwconv(x_pool)))
        return out

class PMHSA(nn.Module):
    def __init__(self, c1, c2, dw_k = 3,num_heads = 4, pool_ratios = (1, 2, 3, 6),
                 qkv_bias = True, attn_drop = 0., proj_drop = 0):
        super().__init__()
        assert c2 % num_heads == 0

        self.num_heads = num_heads
        self.dim_head = c2 // num_heads
        self.scale = self.dim_head ** -0.5
        self.pool_ratios = tuple(pool_ratios)

        self.cv_in = ConvBNAct(c1, c2, k = 1, s = 1) if c1 != c2 else nn.Identity()
        self.apdw_branches = nn.ModuleList([
            APWD(c2, c2, dw_k) for _ in self.pool_ratios
        ])

        self.q = nn.Linear(c2, c2, bias = qkv_bias)
        self.kv = nn.Linear(c2, c2 * 2, bias = qkv_bias)
        self.norm = nn.LayerNorm(c2)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(c2, c2, bias = qkv_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        x = self.cv_in(x)
        b, c, h, w = x.shape
        n = h * w

        x_tokens = x.flatten(2).transpose(1, 2) #[b, n, c]
        q = self.q(x_tokens)
        q = q.reshape(b, n, self.num_heads, self.dim_head).permute(0, 2, 1, 3)  # [b, num_heads, n, dim_head]

        pooled_tokens = []
        for ratio, branch in zip(self.pool_ratios, self.apdw_branches):
            hr = max(1, round(h / ratio))
            wr = max(1, round(w / ratio))

            pooled_feat = branch(x, (hr, wr))   #[b, c, hr, wr]
            pooled_feat = pooled_feat.flatten(2) #[b, c, hr * wr]
            pooled_tokens.append(pooled_feat)

        pooled_tokens = torch.cat(pooled_tokens, dim = 2)   #[b, c, sum(hr*wr)]
        pooled_tokens = pooled_tokens.transpose(1, 2)   #[b, sum(hr*wr), c]
        pooled_tokens = self.norm(pooled_tokens)

        kv = self.kv(pooled_tokens) #[b, sum(hr*wr), 2c]
        kv = kv.reshape(b, -1, 2, self.num_heads, self.dim_head)
        kv = kv.permute(2, 0, 3, 1, 4)  #[2, b, nums_heads, length_seq, dim_head]
        k, v = kv[0], kv[1]     #[b, nums_heads, length_seq, dim_head]

        attn = (q @ k.transpose(-2, -1)) * self.scale   #[b, nums_heads, length_seq, length_seq]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v  #[b, nums_heads, length_seq, dim_head]
        out = out.transpose(1, 2).contiguous()  #[b, length_seq, nums_heads, dim_head]
        out = out.reshape(b, n, c)  #[b, length_seq, nums_heads * dim_head = c2]

        out = self.proj(out)
        out = self.proj_drop(out)

        out = out.transpose(1, 2).reshape(b, c, h, w)
        return out

if __name__ == "__main__":
    x = torch.randn(1, 256, 20, 20)
    model = PMHSA(256, 256)
    out = model(x)

    print(f"input shape: {x.shape}\noutput shape: {out.shape}")