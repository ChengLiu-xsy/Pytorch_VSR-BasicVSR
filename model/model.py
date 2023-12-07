import torch
import torch.nn as nn

from modules.SpyNet import SpyNet
from modules.modules import PixelShuffle, ResidualBlocksWithInputConv, flow_warp

class BasicVSR(nn.Module):
    def __init__(self, scale_factor=4, mid_channels=64, num_blocks=30, spynet_pretrained=None):
        super().__init__()
        self.scale_factoe = scale_factor
        self.mid_channels = mid_channels

        # alignment(optical flow network)
        self.spynet = SpyNet(spynet_pretrained)

        # propagation
        self.backward_resblocks = ResidualBlocksWithInputConv(mid_channels + 3, mid_channels, num_blocks)  # 反向
        self.forward_resblocks = ResidualBlocksWithInputConv(mid_channels + 3, mid_channels, num_blocks)  # 前向

        # upsample  or reconstruction
        self.fusion = nn.Conv2d(mid_channels * 2, mid_channels, 1, 1, 0, bias=True)
        self.upsample1 = nn.Conv2d(mid_channels, mid_channels * 4, 3, 1, 1, bias=True)
        self.upsample2 = nn.Conv2d(mid_channels, 64 * 4, 3, 1, 1, bias=True)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 64, 3, 1, 1)

        self.pixel_shuffle = nn.PixelShuffle(2)

        self.img_upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)

        # activation function
        self.lrrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)


    def get_flow(self, lrs):
        """
        compute optical flow using SpyNet for feature warping.

        :param lrs: Input LR images with shape(n, t, c, h, w)
        :return: optical flow (n, t-1, 2, h, w)
        """
        n, t, c, h, w = lrs.size()
        lrs_1 = lrs[:, :-1, :, :, :].reshape(-1, c, h, w)
        lrs_2 = lrs[:, 1:, :, :, :].reshape(-1, c, h, w)

        flows_backward = self.spynet(lrs_1, lrs_2).view(n, t - 1, 2, h, w)
        flows_forward = self.spynet(lrs_2, lrs_1).view(n, t - 1, 2, h, w)

        return flows_forward, flows_backward

    def forward(self, lrs):
        """
        Forward function for BasicVSR

        :param lrs:  Input LR sequence with shape (n, t, c, h, w)

        :return: Tesnsor : Output HR sequence with shape (n, t, c, 4h, 4w).(if scale_factor = 4)

        """
        n, t, c, h, w = lrs.size()
        assert h >= 64 and w >= 64, ('The height and width of inputs should be at least 64, but got {} and {}'.format(h, w))

        # get computed optical flow
        flows_forward, flows_backward = self.get_flow(lrs)

        # backward-time propagation
        outputs = []
        feat_prop = lrs.new_zeros(n, self.mid_channels, h, w)
        for i in range(t-1, -1, -1):
            if i < t - 1:  # no warping required for the last timestep
                flow = flows_backward[:, i:, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))

            feat_prop = torch.cat([lrs[:, i, :, :, :], feat_prop], dim=1)
            feat_prop = self.backward_resblocks(feat_prop)

            outputs.append(feat_prop)
        outputs = outputs[::-1]  # 倒序逆向变成正向

        # forward-time propagation and upsampling
        feat_prop = torch.zeros_like(feat_prop)
        for i in range(0, t - 1):
            lr_curr = lrs[:, i, :, :, :]
            if i > 0:  # no warping required for the first timestep
                if flows_forward is not None:
                    flow = flows_forward[:, i - 1, :, :, :]
                else:
                    flow = flows_forward[:, -i, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0,2, 3, 1))

            feat_prop = torch.cat([lr_curr, feat_prop], dim=1)
            feat_prop = self.forward_resblocks(feat_prop)


            # Upsampling given the backward and forward features
            out = torch.cat([outputs[i], feat_prop], dim=1)
            out = self.lrrelu(self.fusion(out))
            out = self.lrrelu(self.upsample1(out))
            out = self.lrrelu(self.upsample2(out))
            out = self.lrrelu(self.conv_hr(out))
            out = self.conv_last(out)
            base = self.img_upsample(lr_curr)
            out += base
            outputs[i] = out

        return torch.stack(outputs, dim=1)















