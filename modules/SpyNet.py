import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.modules import flow_warp


class SpyNetBasicModule(nn.Module):
    """
    Basic Module for SpyNet
    Paper:
        Optical Flow Estimation using a Spatial Pyramid Network, CVPR, 2017
    """
    def __init__(self):
        super(SpyNetBasicModule, self).__init__()

        self.basic_module = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=32, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=16, out_channels=2, kernel_size=7, stride=1, padding=3)
        )

    def forward(self, tensor_input):
        return self.basic_module(tensor_input)

class SpyNet(nn.Module):
    """
    SpyNet network architecture
    The difference to the SPyNet in [tof.py] is that
        1. more SPyNetBasicModule is used in this version, and
        2. no batch normalization is used in this version.
    Paper:
        Optical Flow Estimation using a Spatial Pyramid Network, CVPR, 2017
    Args:
        pretrained (str): path for pre-trained SPyNet. Default: None.
    """
    def __init__(self):
        super().__init__()
        self.basic_module = nn.ModuleList(
            [SpyNetBasicModule() for _ in range(6)]
        )
        self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def compute_flow(self, ref, supp):
        """
        compute flow from ref to supp
        Note that in this function, the images are already resized to a
        multiple of 32.
        :param ref:Reference image with shape of (n, 3, h, w)
        :param supp:Supporting image with shape of (n, 3, h, w)
        :return:Estimated optical flow: (n, 2, h, w).
        """
        n, _, h, w = ref.size()

        # normalize the input images
        ref = [(ref - self.mean) / self.std]
        supp = [(ref - self.mean) / self.std]

        # generate downsample frames'
        for level in range(5):
            ref.append(
                F.avg_pool2d(
                input=ref[-1],
                kernel_size=2,
                stride=2,
                count_include_pad=False
                )
            )
            supp.append(
                F.avg_pool2d(
                    input=supp[-1],
                    kernel_size=2,
                    stride=2,
                    count_include_pad=False
                )
            )

            # 让元素倒序排列
            ref = ref[::-1]
            supp = supp[::-1]

            # 创建一个全零张量'flow’，用于表示光流场，存储光流场信息
            flow = ref[0].new_zeros(n, 2, h//32, w//32)
            for level in range(len(ref)):
                if level == 0:
                    flow_up = flow
                else:
                    # 使用interpolate（）对输入的光流场进行上采样
                    flow_up = F.interpolate(
                        input=flow,
                        scale_factor=2,
                        mode='bilinear',
                        align_corners=True
                    ) * 0.2

                # add the residue to the upsample flow
                flow = flow_up + self.basic_module[level](
                    torch.cat([ref[level], flow_warp(supp[level], flow_up.permute(0, 2, 4, 1), padding_mode='border'), flow_up], 1))

        return flow

    def forward(self, ref, supp):
        """
        This function computes the optical flow from ref to supp
        :param ref:  shape(n, 3, h, w)
        :param supp: shape(n, 3, h, w)
        :return: estimated optical flow shape(n ,2, h, w)
        """
        h, w = ref[2:4]
        w_up = w if (w % 32) == 0 else 32 * (w // 32 + 1)
        h_up = h if (h % 32) == 0 else 32 * (h // 32 + 1)
        ref = F.interpolate(
            input=ref,
            size=(h_up, w_up),
            mode='bilinear',
            align_corners=False
        )
        supp = F.interpolate(
            input=supp,
            size=(h_up, w_up),
            mode='bilinear',
            align_corners=False
        )

        # compute flow and resize back to the original resolution
        flow = F.interpolate(
            input=self.compute_flow(ref, supp),
            size=(h, w),
            mode='bilinear',
            align_corners=False
        )

        # adjust the flow values
        flow[:, 0, :, :] *= float(w) / float(w_up)
        flow[:, 1, :, :] *= float(h) / float(h_up)

        return flow


