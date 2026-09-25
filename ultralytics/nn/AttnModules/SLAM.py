import torch
import torch.nn as nn

class SLAM(nn.Module):
    def __init__(self, channels :int, reduction : int = 16, kernel_size : int =7):
        super().__init__()
        assert channels > 0
        hidden = max(1, channels // reduction)

        self.conv = nn.Conv2d(1,1,kernel_size = kernel_size,padding=kernel_size//2,bias = False)

        self.proj = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size = 1, bias = False),
            nn.ReLU(inplace = True),
            nn.Conv2d(hidden, channels, kernel_size = 1, bias = False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        x_c = x.mean(dim = 1, keepdim = True)
        x_c = torch.max(x, dim = 1, keepdim = True)[0] + x_c

        x_h = x.mean(dim=2, keepdim=True)
        x_h = torch.max(x, dim=2, keepdim=True)[0] + x_h

        x_w = x.mean(dim=3, keepdim=True)
        x_w = torch.max(x, dim=3, keepdim=True)[0] + x_w

        return self.sigmoid(self.proj(x_h)) * self.sigmoid(self.proj(x_w)) * self.sigmoid(self.conv(x_c))


if __name__ == "__main__":
    input = torch.randn(2, 64, 80, 80)
    model = SLAM(channels=64, reduction=16)
    output = model(input)
    print(f"input_size: {input.shape}\noutput_size: {output.shape}")




