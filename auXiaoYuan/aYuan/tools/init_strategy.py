import torch
import math
import warnings
from torch.nn.init import _calculate_fan_in_and_fan_out
"""
    参数初始化理论：Xavier Glorot
        paper: Understanding the difficulty of training deep feedforward neural networks
        url: http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
    参数初始化要求：
        （1）参数不能全部初始化为0，也不能全部初始化同一个值，为什么，请参见“对称失效”；
        （2）最好保证参数初始化的均值为0，正负交错，正负参数大致上数量相等；
        （3）初始化参数不能太大或者是太小，参数太小会导致特征在每层间逐渐缩小而难以产生作用，参数太大会导致数据在逐层间传递时逐渐放大而导致梯度消失发散，不能训练
        （4）如果有可能满足Glorot条件也是不错的: 上面论文中提到的两个条件任一即可
    参数初始化方法：
        xavier_normal_, xavier_uniform_, 
        lecun_normal_, lecun_uniform_,
        kaiming_normal_, kaiming_uniform_
"""


# 截断正态分布初始化 --> 进行参数初始化的时候用的 --> 可以避免产生梯度值
def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


# 截断正态初始化，并且将截断的区间设置为[-2.0, 2.0]
def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


# 用来对神经网络的一层的参数进行方差归一化的函数， lecun_init 和 kaiming_init 会用到
def variance_scaling_(tensor, scale=1.0, mode="fan_in", distribution="trucated_normal"):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if mode == 'fan_in':
        denom = fan_in
    elif mode == 'fan_out':
        denom = fan_out
    elif mode == 'fan_avg':
        denom = (fan_in + fan_out) / 2
    variance = scale / denom
    if distribution == "truncated_normal":
        trunc_normal_(tensor, std=math.sqrt(variance) / .87962566103423978)
    elif distribution == "normal":
        tensor.normal_(std=math.sqrt(variance))
    elif distribution == "uniform":
        bound = math.sqrt(3 * variance)
        tensor.uniform_(-bound, bound)
    else:
        raise ValueError(f"invalid distribution {distribution}")


# Yann LeCun 正太初始化 for tanh
def lecun_normal(tensor):
    variance_scaling_(tensor, mode="fan_in", distribution="truncated_normal")


# He Kaiming 正太初始化 for ReLU
def kaiming_normal(tensor):
    variance_scaling_(tensor, scale=2.0, mode="fan_in", distribution="truncated_normal")


def xavier_normal(tensor):
    torch.nn.init.xavier_normal_(tensor)

if __name__ == '__main__':
    x = torch.ones((5, 6))
    print(x)
    # lecun_normal(x)
    # kaiming_normal(x)
    # xavier_normal(x)
    # print(x)

