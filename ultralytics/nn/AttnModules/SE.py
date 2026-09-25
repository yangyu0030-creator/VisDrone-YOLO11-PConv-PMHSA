import torch
import torch.nn as nn

"""
input tensor : [b, c, h, w]
output tensor : [b, c, h, w]
"""
class SE(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        assert channels > 0
        hidden = max(channels // reduction, 1)

        #Squeeze
        #[b, c, h, w] -> [b, c, 1, 1]
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        #Excitation : MLP
        #[b, c, 1, 1] -> [b, hidden, 1, 1] -> [b, c, 1, 1]
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1, stride=1, padding=0, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1, stride=1, padding=0, bias=True)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        temp = x
        x = self.avg_pool(x)
        x = self.fc2(self.act(self.fc1(x)))
        x = self.sigmoid(x)

        return x * temp

if __name__ == "__main__":
    input = torch.randn(2, 64, 80, 80)
    se = SE(channels=64, reduction=16)
    output = se(input)
    print(f"input_size: {input.shape}\noutput_size: {output.shape}")