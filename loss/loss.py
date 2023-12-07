import torch
import torch.nn as nn
import torch.nn.functional as F


# 计算输入值和目标值之间的L1损失， reduction='none'指定了损失的缩减方式。在这里设置为 'none'，表示不进行缩减，返回每个样本的单独损失值。
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


# mse_loss
def mse_loss(perd, target):
    return F.mse_loss(perd, target, reduction='none')


# charbonnier_loss
def charbonnier_loss(pred, target, sample_wise=False, eps=1e-12):
    return torch.sqrt((pred - target)**2 + eps)


class L1Loss(nn.Module):
    """
    L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['mean', 'none', 'sum']:
            raise ValueError('Unsupported reduction mode {}.'
                             "Supported ones in ['mean', 'none', 'sum']".format(reduction))

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Forward function

        :param pred:  shape (n, c, h, w) tensor
        :param target: shape(n, c, h, w) tensor
        :param weight: (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight + l1_loss(pred, target, weight, reduction = self.reduction)


class MSELoss(nn.Module):
    """MSE (L2) loss.

        Args:
            loss_weight (float): Loss weight for MSE loss. Default: 1.0.
            reduction (str): Specifies the reduction to apply to the output.
                Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        if reduction not in ['mean', 'none', 'sum']:
            raise ValueError('Unsupported reduction mode {}.'
                             "Supported ones in ['mean', 'none', 'sum']".format(reduction))

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Forward function

        :param pred:  shape (n, c, h, w) tensor
        :param target: shape(n, c, h, w) tensor
        :param weight: (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight + mse_loss(pred, target, weight, reduction=self.reduction)


class CharbonnierLoss(nn.Module):
    """Charbonnier loss (one variant of Robust L1Loss, a differentiable variant of L1Loss).

       Described in "Deep Laplacian Pyramid Networks for Fast and Accurate Super-Resolution".

       Args:
           loss_weight (float): Loss weight for L1 loss. Default: 1.0.
           reduction (str): Specifies the reduction to apply to the output.
               Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
          # sample_wise (bool): Whether calculate the loss sample-wise. This
               argument only takes effect when `reduction` is 'mean' and `weight`
               (argument of `forward()`) is not None. It will first reduces loss
               with 'mean' per-sample, and then it means over all the samples.
               Default: False.
           eps (float): A value used to control the curvature near zero.
               Default: 1e-12.
    """

    def __init__(self, loss_weight=1.0, reduction='mean', sample_wise=False, eps=1e-12):
        super(CharbonnierLoss, self).__init__()
        if reduction not in ['mean', 'none', 'sum']:
            raise ValueError('Unsupported reduction mode {}.'
                             "Supported ones in ['mean', 'none', 'sum']".format(reduction))

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.eps = eps
        self.sample_wise = sample_wise

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Forward function

        :param pred:  shape (n, c, h, w) tensor
        :param target: shape(n, c, h, w) tensor
        :param weight: (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * charbonnier_loss(pred,
                                                   target,
                                                   weight,
                                                   eps=self.eps,
                                                   reduction=self.reduction,
                                                   sample_wise=self.sample_wise)











