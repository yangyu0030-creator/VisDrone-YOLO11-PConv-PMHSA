import torch
import torch.nn as nn

from .SCConv import *
from .C3 import *

class Bottleneck_SCConv(nn.Module):
    """Standard bottleneck."""

    def __init__(
        self, c1: int, c2: int, shortcut: bool = True, g: int = 1, e: float = 0.5
    ):
        """Initialize a standard bottleneck module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            shortcut (bool): Whether to use shortcut connection.
            g (int): Groups for convolutions.
            k (tuple): Kernel sizes for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = SCConv(c_, c2)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply bottleneck with optional shortcut connection."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C3SCConvExp(C3):
    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, e: float = 0.5):
        super().__init__(c1, c2, n = n, e = e)
        self.c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck_SCConv(self.c_, self.c_, shortcut, e=1) for _ in range(n)))

if __name__ == "__main__":
    x = torch.randn(1, 64, 224, 224)
    model = C3SCConvExp(c1 = 64, c2 = 128)
    out = model(x)

    print(f"input shape: {x.shape}\noutput shape: {out.shape}")