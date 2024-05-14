import torch
import numpy as np
import scipy.io as sio
import os
from os.path import join as join_path

"""
    高光谱重建（压缩感知）里面的一些常用的函数的一个汇总：
        shift, shift_back23
        generate_masks_OF, generate_masks_CASSI, generate_masks, generate_shift_masks
        
"""


# Optical Filter-based HSI System 的 mask生成
def generate_masks_OF(mask_path, batch_size, device="cpu"):
    # 当前给的那个mask文件里面就只有一个变量 mask_3d, 它是256x256x28的double的np.ndarray
    mask_3d = sio.loadmat(join_path(mask_path, "mask_3d.mat"))["mask_3d"]
    mask_3d = np.transpose(mask_3d, (2,0,1)) # np.ndarray:28x256x256
    mask3d = torch.from_numpy(mask_3d) # tensor: 28x256x256
    nC, H, W = mask_3d.shape
    # mask3d --> bs x 28 x 256 x 256 --> cuda --> float
    mask3d_batch = mask3d.expand([batch_size, nC, H, W]).to(device).float()
    return mask3d_batch


# CASSI System 的 mask生成
def generate_masks_CASSI(mask_path, batch_size, device="cpu"):
    # 它的mask是2d的，存在mask.mat里面（不是mask_3d.mat)然后数据存在变量mask里面
    mask = sio.loadmat(join_path(mask_path, "mask.mat"))["mask"] # np.ndarray 256x256
    mask3d = np.tile(np.expand_dims(mask, 0),(28, 1, 1)) # 28x256x256 ndarray
    mask3d = torch.from_numpy(mask3d) # tensor 28x256x256
    nC, H, W = mask3d.shape
    mask3d_batch = mask3d.expand([batch_size, nC, H, W]).to(device).float()
    return mask3d_batch


# 统一的生成mask的API
def generate_masks(mask_path, batch_size, device="cpu", HSI_SYS="OF"):
    if HSI_SYS == "OF":
        return generate_masks_OF(mask_path, batch_size, device)
    elif HSI_SYS == "CASSI":
        return generate_masks_CASSI(mask_path, batch_size, device)
    else:
        raise NotImplementedError(f"针对 #HSI-SYS:{HSI_SYS}# 的mask生成API暂未开发")


# 生成shift的mask, 就是已经有了shift_mask，只需要load就行了
def generate_shift_masks(mask_path, batch_size, device="cpu"):
    # 他的mask是3d的，已经shift过的，存在 mask_3d_shift.mat里面，数据存在同名变量里面
    mask_3d_shift = sio.loadmat(join_path(mask_path, "mask_3d_shift.mat"))["mask_3d_shift"] # np.ndarray:256x310x28
    mask_3d_shift = torch.from_numpy(np.transpose(mask_3d_shift, (2,0,1))) # tensor: 28x256x310
    [nC, H, W] = mask_3d_shift.shape
    Phi_batch = mask_3d_shift.expand([batch_size, nC, H, W]).to(device).float()
    Phi_s_batch = torch.sum(Phi_batch**2, 1)
    Phi_s_batch[Phi_s_batch==0] = 1
    return Phi_batch, Phi_s_batch


# shift转换，就是对应的：HSI或mask 通过色散元件的那个步骤， step是步长
def shift(inputs, step=2, device="cpu"):
    [bs , nC, row, col] = inputs.shape
    # 通过计算输出的形状应该是 bs x nC x row x [(nC-1)step+col]
    output = torch.zeros(bs, nC, row, col+(nC-1)*step).to(device).float()
    # 把每一个通道移动后赋值到对应的位置上去
    for i in range(nC):
        output[:, i, :, i*step:i*step+col] = inputs[:, i, :, :]
    return output


# 转换回去, 从一个bsx256x310 --> bsx28x256x256, 显然不是精确的转换，而且误差很大很大
def shift_back23(inputs, step=2, device="cpu"):
    bs, row, col = inputs.shape
    nC = 28
    output = torch.zeros(bs, nC, row, col-(nC-1) * step).to(device).float()
    for i in range(nC):
        output[:,i,:,:] = inputs[:,:,step*i:step*i+col-(nC-1)*step]
    return output
