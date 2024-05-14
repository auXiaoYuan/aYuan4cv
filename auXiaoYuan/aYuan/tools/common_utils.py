import torch
import logging
from .ssim_torch import ssim
from os.path import join as join_path
import os
import datetime
from fvcore.nn import FlopCountAnalysis

"""
    汇总一些常用工具函数的：
        time2filename,
        generate_logger,
        torch_psnr, torch_ssim,
"""


# 将日期转换成文件名--datetime --> str
def time2filename(dt):
    return f"{dt.year}_{dt.month}_{dt.day}_{dt.hour}_{dt.minute}_{dt.second}"


# 指标的计算：psnr
def torch_psnr(pred, truth):
    pred = (pred*256).round()
    truth = (truth*256).round()
    nC = pred.shape[0]
    psnr = 0
    for i in range(nC):
        mse = torch.mean((pred[i,:,:] - truth[i,:,:]) ** 2)
        psnr += 10 * torch.log10((255*255)/mse)
    return psnr / nC


# 指标的计算：ssim
def torch_ssim(img, ref):  # input [28,256,256]
    return ssim(torch.unsqueeze(img, 0), torch.unsqueeze(ref, 0))

# 生成一个logger来进行记录，
def generate_logger(log_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_file = join_path(log_path, 'log.txt')

    fh = logging.FileHandler(log_file, mode='a')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

# 用来统计参数量和运行所需要的FLOPS
def my_summary(test_model, H = 256, W = 256, C = 28, N = 1, type="bchw",device="cpu"):
    model = test_model.to(device)
    print(model)
    if type=="bchw":
        inputs = torch.randn((N, C, H, W)).to(device)
    elif type == "bhwc":
        inputs = torch.randn((N, H, W, C)).to(device)
    flops = FlopCountAnalysis(model,inputs)
    n_param = sum([p.nelement() for p in model.parameters()])
    print(f'GFLOPs:{flops.total()/(1024*1024*1024)}')
    print(f'Params:{n_param}')


if __name__ == '__main__':
    log_path = r"E:\work_space\2024-5\ABC"
    # a = generate_logger(log_path)
    # a.info("xxxxxxxxx")
