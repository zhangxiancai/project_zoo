# -*- coding: utf-8 -*-
"""

"""

import torch
import torch.nn as nn
from data.load_data import CHARS


class FocalCTCLoss(nn.Module):

    def __init__(self, blank, reduction='mean',gamma=0):
        super(FocalCTCLoss, self).__init__()
        self.gamma = gamma
        # self.eps = eps
        self.ce = torch.nn.CTCLoss(blank=blank, reduction=reduction)

    def forward(self, input, target,input_lengths,target_lengths):
        logp = self.ce(input, target,input_lengths,target_lengths)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss
