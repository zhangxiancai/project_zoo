#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 13:14:16 2019
@author: xingyu
"""

import torch.nn as nn
import torch


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2, c2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2, c2, kernel_size=1, bias=False),
            nn.BatchNorm2d(c2),
        )

        self.shortcut = nn.Sequential()
        if c1 != c2:
            self.shortcut = nn.Sequential(
                nn.Conv2d(c1, c2, kernel_size=1, bias=False),
                nn.BatchNorm2d(c2)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))



class LPRNet(nn.Module):
    def __init__(self, class_num, dropout_rate, export=False):
        super(LPRNet, self).__init__()
        self.export = export
        self.class_num = class_num
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, bias=False), # 0
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),  # 2
            nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1), ceil_mode=True),
            Bottleneck(c1=64, c2=64),    # *** 4 ***
            Bottleneck(c1=64, c2=128),    # *** 5 ***
            nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 2), ceil_mode=True),
            Bottleneck(c1=128, c2=128),   # *** 7 ***
            Bottleneck(c1=128, c2=256),   # *** 8 ***
            nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 2), ceil_mode=True),  # 14
            Bottleneck(c1=256, c2=256),   # *** 10 ***
            nn.Dropout(dropout_rate),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1, 5), stride=1),  # 12
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),  # 14
            nn.Dropout(dropout_rate),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(13, 1), stride=1), # 16
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),  # *** 18 ***
        )
        self.container = nn.Sequential(
            Bottleneck(c1=128, c2=128),   # *** 19 ***
            nn.Conv2d(in_channels=128, out_channels=self.class_num, kernel_size=(1,1), stride=(1,1)),
        )

    def forward(self, x):
        x = self.backbone(x) 
        x = self.container(x)
        if self.export:
            return x
        logits = torch.mean(x, dim=2)
        return logits

class LPRNet_Double(nn.Module):
    def __init__(self, class_num, dropout_rate, export=False):
        super(LPRNet_Double, self).__init__()
        self.export = export
        self.class_num = class_num
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, bias=False), # 0 # h-2
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),  # 2
            nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1), ceil_mode=True),# h-2
            Bottleneck(c1=64, c2=64),    # *** 4 ***
            Bottleneck(c1=64, c2=128),    # *** 5 ***
            nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 2), ceil_mode=True),# h-2
            Bottleneck(c1=128, c2=128),   # *** 7 ***
            Bottleneck(c1=128, c2=256),   # *** 8 ***
            nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 2), ceil_mode=True),  # 14 # h-2
            Bottleneck(c1=256, c2=256),   # *** 10 ***
            nn.Dropout(dropout_rate),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1, 5), stride=1),  # 12
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),  # 14
            nn.Dropout(dropout_rate),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(13, 1), stride=1), # 16 # h-13+1
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),  # *** 18 ***
        )
        self.container = nn.Sequential(
            Bottleneck(c1=128, c2=128),   # *** 19 ***
            nn.Conv2d(in_channels=128, out_channels=self.class_num, kernel_size=(1,1), stride=(1,1)),
        )

        self.last_pool=nn.AvgPool2d(kernel_size=(14, 1), stride=(14, 1)) # (h=48-20=28) convert 1*71*28*18 to 1*71*2*18

    def forward(self, x):
        x = self.backbone(x)
        x = self.container(x)
        if self.export:
            return x
        # logits = torch.mean(x, dim=2)
        logits = self.last_pool(x)
        return logits


CHARS = ['京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
     '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
     '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
     '新', '学', '警','挂',
     '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
     'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
     'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
     'W', 'X', 'Y', 'Z', 'I', 'O', '-'
     ]
   
from thop import profile
from thop import clever_format
if __name__ == '__main__':
    input = torch.Tensor(1, 3, 48, 94)
    net = LPRNet_Double(class_num=len(CHARS), dropout_rate=0, export=False)
    net.eval()
    print(net)
    flops, params = profile(net, inputs=(input, ))
    flops, params = clever_format([flops, params], "%.3f")
    print('Flops:', flops, ',Params:' ,params)
    out = net(input)
    print(out.shape)
