"""
Implementation of DenseNet100
Paper: <Densely Connected Convolutional Networks>
Official reference: https://github.com/weiaicunzai/pytorch-cifar100/blob/master/models/densenet.py
"""

import torch
import torch.nn as nn
import sys
sys.path.insert(0, '.')
from models.dropout import *

####################################
#         Define utilities         #
####################################
class Bottleneck(nn.Module):
    """Define Bottleneck"""

    def __init__(self, in_channels : int, growth_rate : int):
        """
        Args : 
            --in_channels: input channels
            --growth_rate
        """
        super(Bottleneck, self).__init__()
        inner_channel = 4 * growth_rate

        self.bottle_neck = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels, inner_channel, kernel_size = 1, bias = False),
            nn.BatchNorm2d(inner_channel),
            nn.ReLU(inplace = True),
            nn.Conv2d(inner_channel, growth_rate, kernel_size = 3, padding = 1, bias = False)
        )

    def forward(self, x):
        return torch.cat([x, self.bottle_neck(x)], 1)

# We refer to layers between blocks as transition
# layers, which do convolution and pooling.
class Transition(nn.Module):
    """Define transition module"""

    def __init__(self, in_channels : int, out_channels : int):
        """
        Args : 
            --in_channels: input channels
            --out_channels: output channels
        """
        super(Transition, self).__init__()
        # The transition layers used in our experiments
        # consist of a batch normalization layer and an 1×1
        # convolutional layer followed by a 2×2 average pooling layer
        self.down_sample = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, 1, bias = False),
            nn.AvgPool2d(2, stride = 2)
        )

    def forward(self, x):
        return self.down_sample(x)

####################################
#          Define DenseNet         #
####################################
# DesneNet-BC
# B stands for bottleneck layer(BN-RELU-CONV(1x1)-BN-RELU-CONV(3x3))
# C stands for compression factor(0 <= theta <= 1)
class DenseNet(nn.Module):
    """Define DenseNet"""

    def __init__(self, block, nblocks, growth_rate = 12, reduction = 0.5, num_classes = 10,
                 drop_type = None, drop_rate = 0.5, block_size = 7):
        """
        Args : 
            --block: input block
            --nblocks: number of blocks
            --growth_rate: default is 12
            --reduction: default is 0.5
            --num_classes: number of classes, default is 10
            --drop_type: None, 'd' (dropout), 'sd' (SpatialDropout), 'fd' (FocusedDropout), 'db' (DropBlock)
            --drop_rate: default is 0.5 for all dropout methods
            --block_size: default is 7 for DropBlock
        """
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate

        if drop_type is not None:
            assert drop_type in ('no', 'd', 'sd', 'fd', 'db'), "drop_type must be in one of ('no' (no dropout), 'd' (dropout), 'sd' (SpatialDropout), 'fd' (FocusedDropout), 'db' (DropBlock))!"

        self.drop_type = drop_type
        if drop_type == 'db': # DropBlock
            self.drop_layer = LinearScheduler(
                DropBlock(block_size = block_size, drop_rate = 0.),
                          start_value = 0., stop_value = drop_rate, nr_steps = 5e3)
        elif drop_type == 'd': # Dropout
            self.drop_layer = Dropout(drop_rate = drop_rate)
        elif drop_type == 'sd': # SpatialDropout
            self.drop_layer = SpatialDropout(drop_rate = drop_rate)
        elif drop_type == 'fd': # FocusedDropout
            self.drop_layer = FocusedDropout()
        else: # no dropout
            self.drop_type = None

        if self.drop_type is not None:
            self.drop_layer = self.drop_layer.half()

        # Before entering the first dense block, a convolution
        # with 16 (or twice the growth rate for DenseNet-BC)
        # output channels is performed on the input images.
        inner_channels = 2 * growth_rate

        # For convolutional layers with kernel size 3×3, each
        # side of the inputs is zero-padded by one pixel to keep
        # the feature-map size fixed.
        self.conv1 = nn.Conv2d(3, inner_channels, kernel_size = 3, padding = 1, bias = False)

        self.features = nn.Sequential()

        for index in range(len(nblocks) - 1):
            self.features.add_module("dense_block_layer_{}".format(index), self._make_dense_layers(block, inner_channels, nblocks[index]))
            inner_channels += growth_rate * nblocks[index]

            # If a dense block contains m feature-maps, we let the
            # following transition layer generate θm output feature-
            # maps, where 0 < θ ≤ 1 is referred to as the compression factor.
            out_channels = int(reduction * inner_channels) 
            self.features.add_module("transition_layer_{}".format(index), Transition(inner_channels, out_channels))
            inner_channels = out_channels

        self.features.add_module("dense_block{}".format(len(nblocks) - 1), 
        self._make_dense_layers(block, inner_channels, nblocks[len(nblocks)-1]))
        inner_channels += growth_rate * nblocks[len(nblocks) - 1]
        self.features.add_module('bn', nn.BatchNorm2d(inner_channels))
        self.features.add_module('relu', nn.ReLU(inplace = True))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.linear = nn.Linear(inner_channels, num_classes)

    def _make_dense_layers(self, block, in_channels : int, nblocks):
        """
        Args : 
            --block: block instance
            --in_channels: input channels
            --nblock: number of blocks
        """
        dense_block = nn.Sequential()
        for index in range(nblocks):
            dense_block.add_module('bottle_neck_layer_{}'.format(index), block(in_channels, self.growth_rate)) 
            in_channels += self.growth_rate
        return dense_block

    def forward(self, x, drop = True):
        output = self.conv1(x)
        output = self.features(output)
        # unlike original paper, we add dropout instance only here
        if self.drop_type is not None and drop:
            output = self.drop_layer(output)

        output = self.avgpool(output)
        output = output.view(output.size()[0], -1)
        output = self.linear(output)

        return output

    
############## Define different densenet121 models ##############
def densenet121(num_classes = 10,
			 	drop_type = None, drop_rate = 0.5, block_size = 7):
    """DenseNet121
    Args :
        --num_classes: number of classes, default is 10
        --drop_type: None, 'd' (dropout), 'sd' (SpatialDropout), 'fd' (FocusedDropout), 'db' (DropBlock)
        --drop_rate: default is 0.5 for all dropout methods
        --block_size: default is 7 for DropBlock
    """
    return DenseNet(Bottleneck, [6,12,24,16], growth_rate = 32,
                    num_classes = num_classes,
                    drop_type = drop_type, drop_rate = drop_rate, block_size = block_size)


# unit test
if __name__ == '__main__':
    model = densenet121(10, drop_type = 'no')
    print(model)
    x = torch.rand(1, 3, 32, 32)
    y = model(x)
    print(y.shape)
