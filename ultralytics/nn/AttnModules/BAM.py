import torch
import torch.nn as nn

class BAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        assert channels > 0
        hidden = max(1, channels // reduction)
        dilations = (1, 2, 4)

        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )

        self.spatial_reduce = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )

        block = []
        for d in dilations:
            block.append(
                nn.Sequential(
                    nn.Conv2d(hidden, hidden, kernel_size=3, padding=d, dilation=d, bias=False),
                    nn.BatchNorm2d(hidden),
                    nn.ReLU(inplace=True),
                )
            )

        self.spatial_conv = nn.Sequential(*block)

        self.spatial_out = nn.Conv2d(hidden, 1, kernel_size=1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.sigmoid(self.channel_gate(x) + self.spatial_out(self.spatial_conv(self.spatial_reduce(x))))
        return (out * x) + x


if __name__ == '__main__':
    input = torch.randn(3, 64, 80, 80)
    bam = BAM(64)
    output = bam(input)
    print(f"input shape: {input.shape}\noutput shape: {output.shape}")
