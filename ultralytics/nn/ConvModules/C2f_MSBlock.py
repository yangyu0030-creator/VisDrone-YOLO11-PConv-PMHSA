import torch
import torch.nn as nn

try:
    from .C2 import *
    from .MSBlock import *
except:
    from C2 import *
    from MSBlock import *

class C2fMSBlock(C2):
    def __init__(self, c1, c2, n = 1, e = 0.5,
                 branches = 3,k = 3, expansion = 2, shortcut = True, act = True):
        super().__init__(c1, c2)
        self.c_ = int(c2 * e)
        self.m = nn.Sequential(*[MSBlock(self.c_, self.c_, branches, k, expansion, shortcut, act) for _ in range(n)])


if __name__ == "__main__":
    x = torch.randn(1, 64, 224, 224)
    model = C2fMSBlock(c1=64, c2=64)
    out = model(x)

    print(f"input shape: {x.shape}\noutput shape: {out.shape}")