#
# @Author: Songyang Zhang 
# @Date: 2018-11-16 20:56:08 
# @Last Modified by:   Songyang Zhang 
# @Last Modified time: 2018-11-16 20:56:08 
#

import torch
from torch import nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    r"""Focal Loss
    
    """
    def __init__(self, gamma=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma

    
    def forward(self, input, target):
        if not (target.size() == input.size()):
            raise ValueError('Target size ({}) must be the same as input size ({})'.format(target.size(), input.size()))

        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + \
            ((-max_val).exp() + (-input - max_val).exp()).log()

        invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss

        return loss.sum(dim=1).mean()

class WeightedBinaryClsEntropy(nn.Module):

    """
    weighted binary cls_entropy
    """
    def __init__(self):
        super(WeightedBinaryClsEntropy, self).__init__()

    def forward(self, preds, targets, training=True):

        batch_size = preds.shape[0]

        if training:
            if batch_size == 1:
                loss = F.binary_cross_entropy_with_logits(input=preds, target=targets, size_average=True)

            else:
                targets_num_pos = targets.sum(0)
                targets_num_neg = batch_size-targets_num_pos
                pos_weight = targets_num_neg/(targets_num_pos+1e-6)
                loss = F.binary_cross_entropy_with_logits(input=preds, target=targets, size_average=True, pos_weight=pos_weight)

        else:
            loss = F.binary_cross_entropy_with_logits(input=preds, target=targets, size_average=True)

        return loss


class MeanSquareLoss(nn.Module):
    def __init__(self):
        super(MeanSquareLoss, self).__init__()

    def forward(self, pred, target):

        dist = pred - target
        dist = (dist**2).mean()
        return dist

