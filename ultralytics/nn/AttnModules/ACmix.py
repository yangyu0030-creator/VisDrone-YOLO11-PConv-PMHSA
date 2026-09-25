import torch
import torch.nn as nn
import torch.nn.functional as F

class ACmix(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, nums_head: int = 4):
        super().__init__()
        assert channels % nums_head == 0, "channels must be divisible by nums_head"
        self.nums_head = nums_head
        self.channels_head = channels // nums_head
        self.kernel_size = kernel_size
        self.pad = kernel_size // 2
        self.k2 = kernel_size * kernel_size

        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size = 1, padding = 0, bias = False)

        self.kernel_gen = nn.Conv2d(channels * 3, nums_head * self.k2,
                                    kernel_size = 1, groups = nums_head, bias = True)

        self.proj = nn.Conv2d(channels, channels, kernel_size = 1, padding = 0, bias = False)
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))



    def forward(self, x: torch.Tensor) -> torch.Tensor:
            B, C, H, W = x.shape
            x = self.qkv(x) #[B, 3C, H, W]
            q, k, v = x.chunk(3, dim = 1)   #【B, C, H, W]

            q_flat = q.view(B, self.nums_head, self.channels_head, H * W)   #[B, nums_haed, channels_head, H * W]
            #[B, nums_head, channels_head, K * K, H * W]
            k_patch = F.unfold(k, kernel_size=self.kernel_size, padding=self.pad).view(B, self.nums_head, self.channels_head, self.k2, H * W)
            v_patch = F.unfold(v, kernel_size=self.kernel_size, padding=self.pad).view(B, self.nums_head, self.channels_head, self.k2, H * W)

            logits = (q_flat.unsqueeze(3) * k_patch).sum(dim = 2)   #[B, nums_head, K * K, H * W]
            attn = F.softmax(logits, dim = 2)   #[B, nums_head, self.k2, H*W]

            attn_out = (attn.unsqueeze(2) * v_patch).sum(dim = 3).view(B, C, H, W)

            #[B, channels * kernel_size * kernel_size, H, W]
            kernel = self.kernel_gen(x)
            kernel = kernel.view(B, self.nums_head, self.k2, H*W)   #[B, nums_head, self.k2, H * w]

            conv_out = (kernel.unsqueeze(2) * v_patch).sum(dim = 3).view(B, C, H, W)

            out = self.alpha * conv_out + self.beta * attn_out
            out = self.proj(out)

            return out

if __name__ == "__main__":
    input = torch.randn(2, 64, 80, 80)
    model = ACmix(channels=64)
    output = model(input)
    print(f"input_size: {input.shape}\noutput_size: {output.shape}")











