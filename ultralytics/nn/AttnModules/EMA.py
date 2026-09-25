import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

class EMA(nn.Module):
    def __init__(self, channels: int, groups: int = 32, act: nn.Module = None):
        super().__init__()
        assert channels > 0
        g = min(channels, groups)
        while g > 1 and channels % g != 0:
            g -= 1

        self.groups = max(1, g)
        self.channels_groups = channels // self.groups
        self.conv1x1 = nn.Conv2d(self.channels_groups, self.channels_groups, kernel_size=1)
        self.conv3x3 = nn.Conv2d(self.channels_groups, self.channels_groups, kernel_size=3, padding=1)
        self.gb = nn.GroupNorm(1, self.channels_groups)

        self.act = act if act is not None else nn.SiLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.size()
        # 1. 分组并 reshape
        x = x.view(b * self.groups, c // self.groups, h, w)

        # 2. 提取 X 和 Y (严格按照图片的 Pool 和 Transpose)
        # X: 沿 W 维求均值 -> [bg, cg, h, 1] -> 转置为 [bg, cg, 1, h]
        x_h = x.mean(dim=-1, keepdim=True).permute(0, 1, 3, 2)
        # Y: 沿 H 维求均值 -> [bg, cg, 1, w]
        x_w = x.mean(dim=-2, keepdim=True)

        # 3. 拼接 (图片上的 Concat: 1x1 卷积分支)
        y = torch.cat((x_h, x_w), dim=-1) # [bg, cg, 1, h+w]
        y = self.conv1x1(y)

        # 4. Split (按 H 和 W 拆分)
        y_h, y_w = torch.split(y, (h, w), dim=-1)
        y_h = self.sigmoid(y_h) # [bg, cg, 1, h]
        y_w = self.sigmoid(y_w) # [bg, cg, 1, w]

        # 5. 重塑以广播乘 X
        y_h = y_h.permute(0, 1, 3, 2) # [bg, cg, h, 1]
        y_w = y_w # [bg, cg, 1, w]

        # 6. 得到带权重的 x 并过 GroupNorm
        y = self.gb(y_h * y_w * x) # [bg, cg, h, w]

        # 7. 3x3 分支 (图片上的 Conv3x3)
        z = self.conv3x3(x) # [bg, cg, h, w]

        # 8. 提取通道注意力 (AvgPool) 和空间注意力 (Softmax)
        # 对于 Y 分支
        y_avg = torch.mean(y, dim=(2, 3), keepdim=True).view(b * self.groups, self.channels_groups, 1)
        y_softmax = F.softmax(y.view(b * self.groups, self.channels_groups, h * w), dim=-1)

        # 对于 Z 分支
        z_avg = torch.mean(z, dim=(2, 3), keepdim=True).view(b * self.groups, self.channels_groups, 1)
        z_softmax = F.softmax(z.view(b * self.groups, self.channels_groups, h * w), dim=-1)

        # 9. 跨空间维度的矩阵乘法融合 (图片上的 Mul + Add)
        # (通道注意力 * 空间注意力) + (通道注意力 * 空间注意力)
        out = torch.bmm(y_avg, z_softmax) + torch.bmm(z_avg, y_softmax) # [bg, cg, h*w]

        # 10. 恢复形状并激活，最后乘回原输入 X
        out = out.view(b * self.groups, self.channels_groups, h, w)
        out = self.act(out).view(b, c, h, w)
        return out

if __name__ == "__main__":
    input = torch.randn(2, 64, 80, 80)
    model = EMA(channels=64)
    output = model(input)
    print(f"input_size: {input.shape}\noutput_size: {output.shape}")