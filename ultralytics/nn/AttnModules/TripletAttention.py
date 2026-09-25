import torch
import torch.nn as nn

class TripletAttention(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 7, no_spatial: bool = False):
        super().__init__()
        self.no_spatial = no_spatial
        self.attention_c = AttentionGate(kernel_size)
        self.attention_h = AttentionGate(kernel_size)
        self.attention_w = AttentionGate(kernel_size)

        self.avg = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape

        x_w = x.permute(0, 3, 2, 1).contiguous()     #[b, w, h, c]
        attn_w = self.attention_w(x_w)  #[b, 1, h, c]

        x_h = x.permute(0, 2, 1, 3).contiguous()
        attn_h = self.attention_h(x_h)

        attn_w = attn_w.permute(0, 3, 2, 1).contiguous()
        attn_h = attn_h.permute(0, 2, 1, 3).contiguous()

        attn_c = self.attention_c(x)

        if self.no_spatial:
            return (attn_w * x + attn_h * x) / 2.0
        else:
            return  (attn_c * x + attn_w * x + attn_h * x )/ 3.0


class AttentionGate(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.zpool = Zpool()
        self.conv = BiasConv(2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.zpool(x)
        x = self.conv(x)
        x = self.sigmoid(x)
        return x


class Zpool(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_mean = x.mean(dim=1, keepdim=True)
        x_max = x.max(dim=1, keepdim=True)[0]
        return torch.cat((x_mean, x_max), dim=1)


class BiasConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 1, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, relu: bool = False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU() if relu else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


if __name__ == "__main__":
    input = torch.randn(2, 64, 80, 80)
    model = TripletAttention(channels=64)
    output = model(input)
    print(f"input_size: {input.shape}\noutput_size: {output.shape}")