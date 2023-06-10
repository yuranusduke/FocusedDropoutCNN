"""
Implementation of Wide ResNet-28
Paper: <Wide Residual Networks>
Reference: https://github.com/weiaicunzai/pytorch-cifar100/blob/master/models/wideresidual.py
We train wrn28 with full precision
"""

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
import sys
import numpy as np
import sys
sys.path.insert(0, '.')
from models.dropout import *

####################################
#         Define utilities         #
####################################
class WideBasic(nn.Module):
    """Define basic block"""

    def __init__(self, in_channels, out_channels, stride = 1):
        """
        Args :
            --in_channels: input channels
            --out_channels: output channels
            --stride: default is 1
        """
        super(WideBasic, self).__init__()
        self.residual = nn.Sequential(
            nn.BatchNorm2d(in_channels), nn.ReLU(inplace = True),
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True),
            nn.Dropout(0.3),
            nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1)
        )

        self.shortcut = nn.Sequential()

        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride = stride)
            )

    def forward(self, x):
        residual = self.residual(x)
        shortcut = self.shortcut(x)
        return residual + shortcut # residual connection

####################################
#       Define WideResNet28        #
####################################
class WideResNet28(nn.Module):
    """Define WideResnet-28"""

    def __init__(self, block, widen_factor = 1, num_classes = 10,
                 drop_type = None, drop_rate = 0.5, block_size = 7):
        """
        Args :
            --block
            --widen_factor
            --num_classes: default is 10
            --drop_type: None, 'd' (dropout), 'sd' (SpatialDropout), 'fd' (FocusedDropout), 'db' (DropBlock)
            --drop_rate: default is 0.5 for all dropout methods
            --block_size: default is 7 for DropBlock
        """
        super(WideResNet28, self).__init__()

        depth = 28
        self.depth = depth
        k = widen_factor
        l = int((depth - 4) / 6)
        self.in_channels = 16
        self.init_conv = nn.Conv2d(3, self.in_channels, 3, 1, padding = 1)
        self.conv2 = self._make_layer(block, 16 * k, l, 1)
        self.conv3 = self._make_layer(block, 32 * k, l, 2)
        self.conv4 = self._make_layer(block, 64 * k, l, 2)
        self.bn = nn.BatchNorm2d(64 * k)
        self.relu = nn.ReLU(inplace = True)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(64 * k, num_classes)

        if drop_type is not None:
            assert drop_type in ('no', 'd', 'sd', 'fd', 'db'), "drop_type must be in one of ('no' (no dropout), 'd' (dropout), 'sd' (SpatialDropout), 'fd' (FocusedDropout), 'db' (DropBlock))!"

        self.drop_type = drop_type
        if drop_type == 'db': # DropBlock
            self.drop_layer = LinearScheduler(
                DropBlock(block_size = block_size, drop_rate = 0., half = False),
                          start_value = 0., stop_value = drop_rate, nr_steps = 5e3)
        elif drop_type == 'd': # Dropout
            self.drop_layer = Dropout(drop_rate = drop_rate)
        elif drop_type == 'sd': # SpatialDropout
            self.drop_layer = SpatialDropout(drop_rate = drop_rate)
        elif drop_type == 'fd': # FocusedDropout
            self.drop_layer = FocusedDropout(half = False)
        else: # no dropout
            self.drop_type = None

        # if self.drop_type is not None:
        #     self.drop_layer = self.drop_layer.half()

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """
        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer
        Return:
            return a resnet layer
        """
        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels

        return nn.Sequential(*layers)

    def forward(self, x, drop = True):
        x = self.init_conv(x)
        x = self.conv2(x)
        if self.drop_type == 'db' and drop: # DropBlock is applied to first two groups' outputs
            x = self.drop_layer(x)

        x = self.conv3(x)
        if self.drop_type == 'db' and drop: # DropBlock is applied to first two groups' outputs
            x = self.drop_layer(x)

        x = self.conv4(x)
        if self.drop_type is not None and self.drop_type != 'db' and drop: # other dropout methods
            x = self.drop_layer(x) # output of penultimate group

        x = self.bn(x)
        x = self.relu(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)

        return x

############## Define different wrn28 models ##############
def wrn28(num_classes = 10,
          drop_type = None, drop_rate = 0.5, block_size = 7):
    """WideResNet28
    Args :
        --num_classes: number of classes, default is 10
        --drop_type: None, 'd' (dropout), 'sd' (SpatialDropout), 'fd' (FocusedDropout), 'db' (DropBlock)
        --drop_rate: default is 0.5 for all dropout methods
        --block_size: default is 7 for DropBlock
    We get wide_factor = 10 and dropout_rate = 0.3 from original paper for CIFAR data sets
    """

    return WideResNet28(block = WideBasic, widen_factor = 10, num_classes = num_classes,
                        drop_type = drop_type, drop_rate = drop_rate, block_size = block_size)


# unit test
if __name__ == '__main__':
    model = wrn28(10, drop_type = 'fd')
    print(model)
    x = torch.rand(1, 3, 32, 32)
    y = model(x)
    print(y.shape)