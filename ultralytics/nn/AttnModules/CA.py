import torch
import torch.nn as nn

class CA(nn.Module):
    def __init__(self, channels: int, reduction: int = 32):
        super().__init__()
        assert channels > 0
        mip = max(8, channels // reduction)

        self.conv1 = nn.Conv2d(channels, mip, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish(inplace=True)

        self.conv_h = nn.Conv2d(mip, channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_w = nn.Conv2d(mip, channels, kernel_size=1, stride=1, padding=0, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x_w = x.mean(2, keepdim=True)  #[b, c, 1, w]
        x_h = x.mean(3, keepdim=True)  #[b, c, h, 1]
        out = torch.cat([x_h, x_w.permute(0, 1, 3, 2)], dim = 2) #[b, c, w+h, 1]

        out = self.act(self.bn1(self.conv1(out)))   #[b, mip, w+h, 1]

        out_h, out_w = torch.split(out, [h, w], 2)  #[b, mip, h, 1] [b, mip, w, 1]
        out_h = self.sigmoid(self.conv_h(out_h))    #[b, c, h, 1]
        out_w = self.sigmoid(self.conv_w(out_w)).permute(0, 1, 3, 2)    #[b, c, 1, w]

        return (out_h * out_w) * x


if __name__ == "__main__":
    input = torch.randn(2, 64, 80, 80)
    model = CA(channels=64)
    output = model(input)
    print(f"input_size: {input.shape}\noutput_size: {output.shape}")


