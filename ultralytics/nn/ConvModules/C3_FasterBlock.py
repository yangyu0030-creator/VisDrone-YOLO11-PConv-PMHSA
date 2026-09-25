import torch
import torch.nn as nn

try:
    from .C3 import *
    from .FasterBlock import *
except:
    from C3 import *
    from FasterBlock import *


class C3FasterBlock(C3):
    def __init__(self, c1, c2, n = 3, e = 0.5,
                 k = 1 , s = 1, p = None, d = 1,g = 1, partial_ratio = 0.25, expend_ratio = 2, short_cut = True):
        super().__init__(c1, c2, e = e)
        self.c_ = int(c2 * e)
        self.m = nn.Sequential(*[FasterBlock(self.c_, self.c_, k = k, s = s, p = p, d = d,g = g, partial_ratio = partial_ratio,
                                             expend_ratio = expend_ratio, short_cut = short_cut) for _ in range(n)])

if __name__ == "__main__":
    x = torch.randn(1, 64, 224, 224)
    model = C3FasterBlock(c1 = 64, c2 = 64)
    out = model(x)

    print(f"input shape: {x.shape}\noutput shape: {out.shape}")