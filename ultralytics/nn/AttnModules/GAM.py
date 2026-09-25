import torch
import torch.nn as nn

class GAM(nn.Module):
    def __init__(self, channels : int, reduction: int=16, kernel_size : int=7, L : int=1):
        super().__init__()
        assert channels > 0
        hidden = max(L, channels // reduction)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
        )
        self.channels_sigmoid = nn.Sigmoid()

        self.spatial = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(channels),
        )

        self.spatial_sigmoid = nn.Sigmoid()

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        channel_feature = x.permute(0, 2, 3, 1).contiguous().view(b, h * w, c)
        channel_attn = self.channels_sigmoid(self.mlp(channel_feature)).view(b, c, h, w).contiguous()

        spatial_feature = self.spatial_sigmoid(self.spatial(x))

        out = channel_attn * x
        out = out * spatial_feature

        return out


if __name__ == "__main__":
    input = torch.randn(2, 64, 80, 80)
    model = GAM(channels=64)
    output = model(input)
    print(f"input_size: {input.shape}\noutput_size: {output.shape}")
