import torch
import torch.nn as nn

try:
    from .C2 import *
    from .HGBlock import *
except:
    from C2 import *
    from HGBlock import *

class C2fHGBlock(C2):
    def __init__(self, c1, c2,cm, n = 1, e = 0.5,
                   mk: int = 3, hg_n: int = 6, lightconv: bool = False, shortcut: bool = False, act: Optional[nn.Module] = True):
        super().__init__(c1, c2)
        self.c_ = int(c2 * e)
        self.m = nn.Sequential(*[HGBlockExp(self.c_, cm, self.c_, hg_n, lightconv, shortcut, act) for _ in range(n)])


if __name__ == "__main__":
    x = torch.randn(1, 64, 224, 224)
    model = C2fHGBlock(c1=64, c2=64, cm=32)
    out = model(x)

    print(f"input shape: {x.shape}\noutput shape: {out.shape}")