import torch
import torch.nn as nn
import torch.nn.functional as F

class SK(nn.Module):
    def __init__(self, channels, kernels=(3,5), reduction=16, groups=1, L=32):
        super().__init__()
        hidden = max(channels // reduction, L)
        self.num_branches = len(kernels)

        # 多分支卷积层
        self.branches = nn.ModuleList()
        for k in kernels:
            p = k // 2
            self.branches.append(
                nn.Sequential(
                    nn.Conv2d(channels, channels, k, padding=p, groups=groups, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                )
            )

        # 注意力权重生成网络（共享全连接层）
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, self.num_branches * channels, 1, bias=False),  # 输出 num_branches * channels
        )

    def forward(self, x):
        # 1. 各分支卷积输出
        branch_outs = [branch(x) for branch in self.branches]  # 每个 (B, C, H, W)

        # 2. 逐元素相加得到融合特征
        fused = torch.stack(branch_outs, dim=0).sum(dim=0)  # (B, C, H, W)

        # 3. 全局平均池化 + 全连接生成注意力向量
        se = self.gap(fused)  # (B, C, 1, 1)
        attn = self.fc(se)   # (B, num_branches * C, 1, 1)
        attn = attn.view(x.size(0), self.num_branches, -1)  # (B, num_branches, C)
        attn = F.softmax(attn, dim=1)  # 在分支维度 softmax，得到每个分支的权重 (B, num_branches, C)

        # 4. 加权求和
        out = 0
        for i, branch_out in enumerate(branch_outs):
            weight = attn[:, i, :, None, None]  # (B, C, 1, 1)  广播到 H,W
            out += weight * branch_out

        return out


if __name__ == "__main__":
    input = torch.randn(2, 64, 80, 80)
    sk = SK(channels=64, reduction=16)
    output = sk(input)
    print(f"input_size: {input.shape}\noutput_size: {output.shape}")