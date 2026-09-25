import torch
import torch.nn as nn

class CBAM(nn.Module):
    def __init__(self, channels : int, reduction : int =16, kernel_size : int =7):
        super().__init__()
        assert channels > 0
        hidden = max(1, channels // reduction)

        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
        )

        pad = kernel_size // 2
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=pad, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        x = self.sigmoid(avg_out + max_out) * x

        # Spatial attention
        avg_spatial = x.mean(dim=1, keepdim=True)  # (B, 1, H, W)
        max_spatial = x.max(dim=1, keepdim=True)[0]  # (B, 1, H, W)
        concat = torch.cat([avg_spatial, max_spatial], dim=1)  # (B, 2, H, W)
        spatial_weights = self.sigmoid(self.spatial_conv(concat))  # (B, 1, H, W)
        x = spatial_weights * x

        return x

if __name__ == "__main__":
    input = torch.randn(3, 64, 80, 80)
    cbam = CBAM(64, reduction=16)
    output = cbam(input)
    print(f"input shape: {input.shape}\noutput shape: {output.shape}")