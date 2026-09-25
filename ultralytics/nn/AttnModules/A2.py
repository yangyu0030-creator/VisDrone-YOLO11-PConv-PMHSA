import torch
import torch.nn.functional as F
import torch.nn as nn

class A2(nn.Module):
    def __init__(self, channels: int, reduction: int = 4, L: int = 32):
        super().__init__()
        assert channels > 0
        hidden = max(channels // reduction, L)
        self.hidden = hidden
        self.conv1 = nn.Conv2d(channels, hidden, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(channels, hidden, kernel_size=1, bias=False)
        self.conv3 = nn.Conv2d(channels, hidden, kernel_size=1, bias=False)
        self.proj = nn.Sequential(
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        #tensor :[b, c, h, w] -> [b, hidden, h * w]
        A_attn = self.conv1(x).view(b, self.hidden, h * w)
        B_feat = self.conv2(x).view(b, self.hidden, h * w)
        V_map =  self.conv3(x).view(b, self.hidden, h * w)

        G = torch.bmm(F.softmax(A_attn, dim=-1), B_feat.permute(0, 2, 1))
        Y = torch.bmm(F.softmax(V_map, dim=-1).permute(0, 2, 1), G).view(b, self.hidden, h, w)

        out = self.proj(Y) + x

        return out

if __name__ == "__main__":
    input = torch.randn(3, 64, 80, 80)
    a2 = A2(64)
    output = a2(input)
    print(f"input shape: {input.shape}\noutput shape: {output.shape}")




