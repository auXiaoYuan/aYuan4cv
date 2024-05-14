import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple
from scipy.io import savemat

# ####used for unit test ##########################################################
# import sys
# sys.path.append("..")
# from tools.init_strategy import trunc_normal_, kaiming_normal, lecun_normal
# from tools.activation import GELU
# from tools.common_utils import my_summary
# #### used for unit test ###########################################################

from ..tools.init_strategy import trunc_normal_, kaiming_normal, lecun_normal
from ..tools.activation import GELU


# 这里的shift_back 是可以对下采样后的数据和mask进行shift_back的，因为 使用了不同的 空间尺度， 所以需要改写shift_back, 而且这里的shiftback并没有光谱维度变换
def shift_back(inputs, step=2):  # The shifting step varies with different HSI systems
    [bs, nC, row, col] = inputs.shape
    down_sample = 256 // row
    step = float(step) / float(down_sample * down_sample)
    out_col = row
    for i in range(nC):
        inputs[:, i, :, :out_col] = \
            inputs[:, i, :, int(step * i):int(step * i) + out_col]
    return inputs[:, :, :, :out_col]


#把一个batch的tensor b,h,w,c 按照不同sxs的大小切分成块然后组合起来bx(h/s)x(w/s), s, s, c
def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


# 把拿到的所有的window都重新变换回 图片batch
def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


# transformer里面都要用到的一个LN，这里采用的是前置的LN
# 这里和其他的transformer不太一样：
#       他的是LN -> FFN -> MSA == 一个Transformer block
#       别人的是：LN -> MSA -> FFN == 一个Transformer block
class PreNorm(nn.Module):
    def __init__(self, dim, ffn):
        super().__init__()
        self.ffn = ffn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.ffn(x, *args, **kwargs)


# 前馈神经网络FFN: 网络结构： Conv1x1->GELU->Conv3x3->GELU->Conv1x1 还有形状的变换
class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        out = self.net(x.permute(0, 3, 1, 2))
        return out.permute(0, 2, 3, 1)


# Transformer里面的那个MSA的替换,加了相对位置编码，但是为什么要加载attnmap上面呢？
class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None

        output:(num_windows*B, N, C)
                """
        B_, N, C = x.shape
        # print(x.shape)
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        # q,k,v == (B_, head_num, N, C//heas_num)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        # attn == (B_, head_num, N, N) N == W

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        ################################
        attn = attn + relative_position_bias.unsqueeze(0)
        ################################
        attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        # why need projection ??????????????
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


# swinT的一个block，就是把LN，FFN，W_MSA组合起来， 通过窗口偏移来形成不同的W_MSA
class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # 设置drop_path连接，如果=0.0，就让他变成恒等就行了, 实际上也没用到
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        if self.shift_size > 0:
            # calculate attention mask for SW
            H, W = to_2tuple(self.input_resolution)
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            # print(mask_windows.shape)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        # x: [b,h,w,c]
        B, H, W, C = x.shape
        x = self.norm1(x)

        # cyclic shift, 相当于旋转的操作
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W/SW 但是这里好像并没有用到这个第二个mask参数啊？？？？？？： 没有用到swinT里面的Mask—Attention
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W/SW
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


# 基于CASSI的mask_attention
class MA_CASSI(nn.Module):              # Mask attention for CASSI system
    def __init__(
            self, n_feat):
        super(MA_CASSI, self).__init__()

        self.conv1 = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=True)
        self.conv2 = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=True)
        self.depth_conv = nn.Conv2d(n_feat, n_feat, kernel_size=5, padding=2, bias=True, groups=n_feat)

    def forward(self, mask_shift):
        # x: b,c** stag,h,w
        [bs, nC, row, col] = mask_shift.shape
        mask_shift = self.conv1(mask_shift)
        attn_map = torch.sigmoid(self.depth_conv(self.conv2(mask_shift)))
        res = mask_shift * attn_map
        mask_shift = res + mask_shift
        mask_emb = shift_back(mask_shift)
        return mask_emb


# 基于OF的Mask_Attention
class MA(nn.Module):  # Mask attention for optical filter-based HSI system
    def __init__(
            self, n_feat):
        super(MA, self).__init__()
        self.depth_conv = nn.Conv2d(n_feat, n_feat, kernel_size=5, padding=2, bias=True, groups=n_feat)

    def forward(self, mask_3d):
        attn_map = torch.sigmoid(self.depth_conv(mask_3d))
        res = mask_3d * attn_map
        mask_attn = res + mask_3d
        return mask_attn


# 层次空间注意力机制
class HSA(nn.Module):
    def __init__(
            self,
            dim,
            stage=1, # 当前的层次是多少， 从0开始， 每一层下采样
    ):
        super().__init__()
        self.wa = SwinTransformerBlock(dim=dim, input_resolution=256 // (2 ** stage),
                                       num_heads=2 ** stage, window_size=8,
                                       shift_size=0)
        self.swa = SwinTransformerBlock(dim=dim, input_resolution=256 // (2 ** stage),
                                        num_heads=2 ** stage, window_size=8,
                                        shift_size=4)
        # 最后再来一个LN + FFN
        self.pn = PreNorm(dim, FeedForward(dim=dim))

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out: [b,c,h,w]
        """
        x = x.permute(0, 2, 3, 1)
        x = self.wa(x) + x
        x = self.swa(x) + x
        x = self.pn(x) + x
        out = x.permute(0, 3, 1, 2)
        return out


# 空间光谱Mask注意力机制
class SSM_AB(nn.Module):  # Spatial-spetral-mask attention block
    def __init__(
            self,
            dim,
            dim_head=64,
            heads=8,
            attention_type='full'
    ):
        super().__init__()
        self.num_heads = heads # 多头的头数目
        self.dim_head = dim_head # 每一个光谱通道拿到的特征数目是多少
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.ma = MA(dim)  # Mask Attention (MA)
        self.sa = HSA(dim)  # Hierarchical spatial attention (HSA)
        self.sa_conv = nn.Conv2d(dim, dim, 5, 1, 2, groups=dim, bias=False)
        self.dim = dim
        self.attention_type = attention_type

    def forward(self, x_in, spa_g=None, mask=None):
        """
        x_in: [b,h,w,c]
        mask: [b,h,w,c]
        return out: [b,h,w,c]
        """
        b, h, w, c = x_in.shape

        ## no spatial attention in baseline
        if self.attention_type == 'base':
            x = x_in.reshape(b, h * w, c)
        else:
            x_mid = x_in.permute(0, 3, 1, 2)
            x_mid_out = self.sa(x_mid)
            x_sa_emb = self.sa_conv(spa_g) + spa_g
            if x_sa_emb.shape[3] != x_mid_out.shape[3]:
                x_sa_emb = shift_back(x_sa_emb)
            x = (x_mid_out * x_sa_emb).permute(0, 2, 3, 1).reshape(b, h * w, c)

        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)
        mask_attn = self.ma(mask.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        if b != 0:
            mask_attn = (mask_attn[0, :, :, :]).expand([b, h, w, c])
        q, k, v, mask_attn = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
                                 (q_inp, k_inp, v_inp, mask_attn.flatten(1, 2)))

        ## completed CAB
        if self.attention_type == 'full':
            v = v * mask_attn

        # q: b,heads,hw,c
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attn = (k @ q.transpose(-2, -1))  # A = K^T*Q
        attn = attn * self.rescale
        attn = attn.softmax(dim=-1)
        x = attn @ v  # b,heads,d,hw
        x = x.permute(0, 3, 1, 2)  # Transpose
        x = x.reshape(b, h * w, self.num_heads * self.dim_head)
        out_c = self.proj(x).view(b, h, w, c)
        out_p = self.pos_emb(v_inp.reshape(b, h, w, c).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        out = out_c + out_p
        return out


# 累积空间光谱自注意力机制
class CAB(nn.Module):
    def __init__(
            self,
            dim,
            dim_head=64,
            heads=8,
            num_blocks=1,
            attention_type='full'
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                SSM_AB(dim=dim, dim_head=dim_head, heads=heads, attention_type=attention_type),
                PreNorm(dim, FeedForward(dim=dim))
            ]))

    def forward(self, x, mask):
        """
        x: [b,c,h,w]
        return out: [b,c,h,w]
        """
        x = x.permute(0, 2, 3, 1)
        for (attn, ff) in self.blocks:
            x = attn(x, mask, mask.permute(0, 2, 3, 1)) + x
            x = ff(x) + x
        out = x.permute(0, 3, 1, 2)
        return out


# SPECAT 网络
class SPECAT(nn.Module):
    def __init__(self, dim=28, stage=1, num_blocks=[2, 1], attention_type='full'):
        super(SPECAT, self).__init__()
        self.dim = dim
        self.stage = stage

        # Input projection
        self.embedding = nn.Conv2d(28, self.dim, 3, 1, 1, bias=False)
        self.embedding2 = nn.Conv2d(28, self.dim, 3, 1, 1, bias=False)

        # Encoder
        self.encoder_layers = nn.ModuleList([])
        dim_stage = dim
        for i in range(stage):
            self.encoder_layers.append(nn.ModuleList([
                CAB(
                    dim=dim_stage, num_blocks=num_blocks[i], dim_head=dim, heads=dim_stage // dim,
                    attention_type=attention_type),
                nn.Conv2d(dim_stage, dim_stage * 2, 4, 2, 1, bias=False),
                nn.Conv2d(dim_stage, dim_stage * 2, 4, 2, 1, bias=False)
            ]))
            dim_stage *= 2

        # Bottleneck
        self.bottleneck = CAB(
            dim=dim_stage, dim_head=dim, heads=dim_stage // dim, num_blocks=num_blocks[-1],
            attention_type=attention_type)

        # Decoder
        self.decoder_layers = nn.ModuleList([])
        for i in range(stage):
            self.decoder_layers.append(nn.ModuleList([
                nn.ConvTranspose2d(dim_stage, dim_stage // 2, stride=2, kernel_size=2, padding=0, output_padding=0),
                nn.Conv2d(dim_stage, dim_stage // 2, 1, 1, bias=False),
                CAB(
                    dim=dim_stage // 2, num_blocks=num_blocks[stage - 1 - i], dim_head=dim,
                    heads=(dim_stage // 2) // dim, attention_type=attention_type),
            ]))
            dim_stage //= 2

        # Output projection
        self.mapping = nn.Conv2d(self.dim, 28, 3, 1, 1, bias=False)

        #### activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x, mask=None):
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """

        if mask == None:
            mask = self.lrelu(self.embedding2(x))

        # Embedding
        fea = self.lrelu(self.embedding(x))

        # Encoder
        fea_encoder = []
        masks = []
        for (CAB, FeaDownSample, MaskDownSample) in self.encoder_layers:
            fea = CAB(fea, mask)
            masks.append(mask)
            fea_encoder.append(fea)
            fea = FeaDownSample(fea)
            mask = MaskDownSample(mask)

        # Bottleneck
        fea = self.bottleneck(fea, mask)

        # Decoder
        for i, (FeaUpSample, Fution, CAB) in enumerate(self.decoder_layers):
            fea = FeaUpSample(fea)
            fea = Fution(torch.cat([fea, fea_encoder[self.stage - 1 - i]], dim=1))
            mask = masks[self.stage - 1 - i]
            fea = CAB(fea, mask)

        # Mapping
        out = self.mapping(fea) + x

        return out


#
# if __name__ == '__main__':
#     a = torch.randn((6, 40, 40, 5))
#     # wa = a[1,0:8, 8:16, 0]
#     # w = window_partition(a, 8)
#     # fwa = w[25,:,:,0]
#     # ffwa = w[26,:,:,0]
#     # print(wa.shape,fwa.shape)
#     # print(torch.abs(wa-fwa) < 0.0001)
#     # print(torch.abs(wa-ffwa) < 0.0001)
#     # print(wa)
#     # print(fwa)
#     # print(ffwa)
#     test_LNFFN = PreNorm(28, ffn=FeedForward(28))
#     my_summary(test_LNFFN, H=256, W=256, C=28, N=1, type="bhwc", device="cuda")
#     test_SPECAT = SPECAT(dim=28, stage=1, num_blocks=[2,1], attention_type='full')
#     my_summary(test_SPECAT, H=256, W=256, C=28, N=1, type="bchw", device="cuda")
