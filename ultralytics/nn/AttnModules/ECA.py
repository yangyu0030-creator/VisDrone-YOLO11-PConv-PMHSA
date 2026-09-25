import torch
import torch.nn as nn
import math

class ECA(nn.Module):
    def __init__(self, channels: int, kernel_size: int = None, gamma: int = 2, b : int = 1):
        super().__init__()
        assert channels > 0

        self.gab = nn.AdaptiveAvgPool2d(1)

        if kernel_size is None:
            t = int(abs((math.log2(channels) / gamma) + b))
            kernel_size = t if t % 2 else t + 1
            kernel_size = min(kernel_size, 3)

        self.conv1 = nn.Conv1d(1, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        temp = x
        x = self.gab(x).squeeze(-1).squeeze(-1) #[b, c, 1, 1] -> [b, c]
        x = self.conv1(x.unsqueeze(1)) #[b, 1, c]
        x = self.sigmoid(x.squeeze(1).unsqueeze(-1).unsqueeze(-1))  #[b, c, 1, 1] <- [b, c]

        return x * temp


if __name__ == "__main__":
    input = torch.randn(2, 64, 80, 80)
    ECA = ECA(channels=64)
    output = ECA(input)
    print(f"input_size: {input.shape}\noutput_size: {output.shape}")