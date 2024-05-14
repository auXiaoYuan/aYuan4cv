import torch.nn as nn
import torch.nn.functional as F


"""
    常见的激活函数汇总：
        GELU，
"""


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)