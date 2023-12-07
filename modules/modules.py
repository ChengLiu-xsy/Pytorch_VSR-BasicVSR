"""
the code is based on BasicSR,MMagic
https://github.com/XPixelGroup/BasicSR
https://github.com/open-mmlab/mmagic
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.ops.dcn import ModulatedDeformConv, modulated_deform_conv

def flow_warp(x,
              flow,
              interp_mode='biliner',
              padding_mode='zeros',
              align_corners=True):
    """
    warp an image or a feature map with optical flow
    这个方法实现了一个光流变换的功能，对输入图像x的每个像素位置，根据光流场的位移信息，使用插值的方法填充新位置的像素值
    :param x: is Tensor ，size (n, c, h, w)
    :param flow: is Tensor, size(n, h, w, 2),最后一个维度的2表示一个双通道，代表x和y方向上的位移
    :param interp_mode: is str, 表示插值模式：‘最接近’或‘双线性’，默认是‘bilinear’
    :param padding: is str, 填充模式：‘zeros’或‘border’或‘reflection’
    :param align_corners: is bool， 是否对齐角落，默认为True
    :return: 对齐后的图像或特征图，warped image or feature map  （张量）
    """
    assert x.size()[-2:] == flow.size()[1:3]
    _, _, h, w = x.size()
    # create mesh grid
    # 创建grid网格，该网格的坐标表示图像中每个像素的位置。torch.meshgrid 创建了两个网格，分别表示x和y的坐标。然后将这些坐标堆叠在一起，并将其转换为浮点数类型。
    grid_y, grid_x = torch.meshgrid(torch.arange(0, h). torch.arange(0, w))
    grid = torch.stack((grid_x, grid_y), 2).type_as(x)  # h, w, 2
    grid.requires_grad = False
    # 网格grid加上光流场flow得到新的网络fgrid
    g_f = grid + flow
    # sacle g_f to [-1, 1]
    g_f_x = 2.0 * g_f[:, :, :, 0] / max(w - 1, 1) - 1.0
    g_f_y = 2.0 * g_f[:, :, :, 1] / max(h - 1, 1) - 1.0
    g_f = torch.stack((g_f_x, g_f_y), dim = 3)
    output = F.grid_sample(x,
                           g_f,
                           mode=interp_mode,
                           padding_mode=padding_mode,
                           align_corners=align_corners)

    return output

def make_iterblocks(blocks, num_blocks, **kwargs):
    """
    通过堆叠相同的块来制造层
    :param blocks: mm.module class for basic blocks
    :param num_blocks: number of blocks
    :param kwargs:![](C:/Users/ADMINI~1/AppData/Local/Temp/20200814142123910.png)
    :return: nn.Sequential staced blocks in nn.Seqential
    """
    layers = []
    for _ in range(num_blocks):
        layers.append(blocks(**kwargs))
        return nn.Sequential(*layers)

class ResidualBlockNoBN(nn.Module):
    """
    Residual Block without BN
    stage of : Conv-ReLU-Conv
    num_feat (int) 中间特征的通道数， 默认是64
    res_sacle （float) 残差范围，默认0
    """
    def __init__(self, num_feat=64, res_scales=1):
        super().__init__()
        self.res_scale = res_scales
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

        # if res_scale < 1.0, use the default initalization, as in EDSR
        # if res_scale = 1.0, use scaled kaiming_init, as in MSRResNet
        if res_scales == 1.0:
            self.init_weights()

    def init_weights(self):
        """
        Initialize weights for ResidualBlockNoBN
        像kaiming_init这样的初始化方法是用于vgg风格的模块。
        对于有剩余路径的模块，使用较小的std是可行的更好的稳定性和性能。
        我们根据经验使用0.1。
        更多细节见“ESRGAN:增强型超分辨率生成敌对的网络”
        :return: list
        """
        for m in [self.conv1, self.conv2]:
            # 使用nn.init.kaiming_uniform_方法对卷积层的权重进行初始化
            # m.weight 指的是当前卷积层的权重。
            # a=0 表示使用 ReLU 激活函数，这是一种常用的激活函数。
            # mode='fan_in' 指定初始化模式，它表示使用权重初始化时的fan-in模式。
            # nonlinearity='relu' 表示使用 ReLU 激活函数。
            nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
            m.weight.data *= 0.1  # 尝使用0.1帮助加速模型的收敛
            nn.init.constant_(m.bias, 0)  # 将偏置初始化为0

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.relu(x)
        out = self.conv2(x)

        return identity + out * self.res_scale

class ResidualBlocksWithInputConv(nn.Module):
    """
    Residual blocks with a convolution in front.
    前面有一个卷积的残差块
    in_channels(int): Number of input channels of the first conv
    out_channels(int): Number of channels of the residual blocks default64
    num_blocks(int): Number of residual blocks. Default 15
    """
    def __init__(self, in_channels, out_channels=64, num_blocks=15):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks

        layers = []

        # a conv used to match the channels of the residual blocks
        layers.append(nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True))
        layers.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))

        # residual blocks 将残差块按照一定数量连接起来
        layers.append(make_iterblocks(ResidualBlockNoBN, num_blocks, mid_channels=out_channels))

        self.layers = nn.Sequential(*layers)

    def forward(self, feat):
        """
        :args: input feature with shape(n, in_channels, h, w)
        :return: output feature with shape(n, out_channels, h, w)
        """
        return self.layers(feat)

class PixelShuffle(nn.Module):
    """
    Pixel Shuffle Upsample layer
     Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        scale_factor (int): Upsample ratio.
        upsample_kernel (int): Kernel size of Conv layer to expand channels.
    Returns:
        Upsampled feature map.
    """

    def __init__(self, in_channels, out_channels, scale_factor, upsample_kernel):
        super().__init__()
        self.scale_factor = scale_factor
        self.upsample_conv = nn.Conv2d(in_channels,
                                       out_channels * scale_factor * scale_factor,
                                       upsample_kernel,
                                       padding=(upsample_kernel - 1) // 2)
        self.init_weights()

    def init_weighs(self):
        for m in [self.upsample_conv]:
            nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        :param x: (n. c, h, w)
        :return:
        """
        x = self.upsample_conv(x)  # 卷积上采样操作
        x = F.pixel_shuffle(x, self.scale_factor)
        return x


class UPsample(nn.Sequential):
    """
    UPsample module.

    scale(int) : scale factor.support scale: 2^n and 3
    num_feat(int) : channel number of intermediate features
    """
    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))  # 这和上面自定义的PixelShuffle方法都能使用
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))  # 这和上面自定义的PixelShuffle方法都能使用
        else:
            raise ValueError('scale {} is not supported, support scales: 2^n, 3'.format(scale))
        super(UPsample, self).__init__(*m)






























