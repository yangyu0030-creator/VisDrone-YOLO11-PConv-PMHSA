import torch
import torch.nn as nn
import torch.nn.functional as F

def _make_divisible_groups(channels, groups):
    if channels % groups == 0:
        return groups
    for g in range(groups, 0, -1):
        if channels % g == 0:
            return g
    return -1

class ELA(nn.Module):
    def __init__(self, channels:int, kernel_size:int = 7, groups:int = 1, use_residual: bool = False):
        super().__init__()
        self.use_residual = use_residual
        self.g = _make_divisible_groups(channels, groups)

        self.conv_h = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2, groups = 1, bias=False)
        self.conv_w = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2, groups = 1, bias=False)

        self.gn = nn.GroupNorm(self.g, channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        x = self.gn(x)
        x_h = x.mean(dim = -1)
        x_w = x.mean(dim = -2)

        x_h = self.conv_h(x_h)
        x_w = self.conv_w(x_w)

        a_h = self.sigmoid(x_h)
        a_w = self.sigmoid(x_w)

        out = a_h.unsqueeze(-1) * a_w.unsqueeze(-2)
        if self.use_residual:
            out = x + out

        return out

if __name__ == "__main__":
    input = torch.randn(2, 64, 80, 80)
    model = ELA(channels=64)
    output = model(input)
    print(f"input_size: {input.shape}\noutput_size: {output.shape}")




