#Code forké depuis "huanglianghua" (https://github.com/huanglianghua/siamfc-pytorch)
#Adapté et modifié par Paulin Brissonneau

"""
Cout : entropie croisée sur les pixels des images.
Le coût est équilibré entre instances positives et négatives.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .ops import show_array

class BalancedLoss(nn.Module):

    def __init__(self, neg_weight=1.0):
        super(BalancedLoss, self).__init__()
        self.neg_weight = neg_weight
    
    def forward(self, input, target, viz=False):

        if viz :
            show_array(input[0][0], "input")
            show_array(target[0][0], "target")

        pos_mask = (target == 1)
        neg_mask = (target == 0)
        pos_num = pos_mask.sum().float()
        neg_num = neg_mask.sum().float()
        weight = target.new_zeros(target.size())
        weight[pos_mask] = 1 / pos_num
        weight[neg_mask] = 1 / neg_num * self.neg_weight
        weight /= weight.sum()
        return F.binary_cross_entropy_with_logits(
            input, target, weight, reduction='sum')