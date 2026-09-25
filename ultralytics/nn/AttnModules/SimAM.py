import torch
import torch.nn as nn

class SimAM(nn.Module):
    def __init__(self, channels: int, e_lambda: float = 1e-4):
        super().__init__()
        self.e_lambda = e_lambda
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        mu = x.mean((2, 3), keepdim=True) #[b, c, 1, 1]

        d = (mu - x).pow(2) #[b, c, h, w]
        v = d.mean((2, 3), keepdim=True) #[b, c, 1, 1]

        score = d / 4 * (v + self.e_lambda) + 0.5

        attn = self.sigmoid(score)
        x = x * attn

        return x

if __name__ == "__main__":
    input = torch.randn(2, 64, 80, 80)
    model = SimAM(channels=64)
    output = model(input)
    print(f"input_size: {input.shape}\noutput_size: {output.shape}")
