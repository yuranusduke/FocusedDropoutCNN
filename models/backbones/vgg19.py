"""
Implementation of VGG19 model
Paper: <Very Deep Convolutional Networks for Large-Scale Image Recognition>
"""

import torch
from torch.nn import functional as F
from torch import nn as nn
import sys
sys.path.insert(0, '.')
from models.dropout import *

####################################
#         Define utilities         #
####################################
def _conv_layer(in_channels : int,
                out_channels : int):
    """Define conv layer
    Args :
        --in_channels: input channels
        --out_channels: output channels
    return :
        --conv layer
    """
    conv_layer = nn.Sequential(
        nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 3, stride = 1, padding = 1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace = True)
    )

    return conv_layer

def vgg_block(in_channels : int,
              out_channels : int,
              repeat : int):
    """Define VGG block
    Args :
        --in_channels: input channels
        --out_channels: output channels
        --repeat
    return :
        --block
    """
    block = [
        _conv_layer(in_channels = in_channels if i == 0 else out_channels,
                    out_channels = out_channels)
        for i in range(repeat)
    ]

    return block

####################################
#          Define VGG19            #
####################################
class VGG19(nn.Module):
    """Define VGG19-style model"""

    def __init__(self, num_classes = 10,
             drop_type = None, drop_rate = 0.5, block_size = 7):
        """
        Args :
            --num_classes: number of classes, default is 10
            --drop_type: None, 'd' (dropout), 'sd' (SpatialDropout), 'fd' (FocusedDropout), 'db' (DropBlock)
            --drop_rate: default is 0.5 for all dropout methods
            --block_size: default is 7 for DropBlock
        """
        super(VGG19, self).__init__()

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

        self.layer1 = nn.Sequential(*vgg_block(in_channels = 3,
                                               out_channels = 64,
                                               repeat = 2))

        self.layer2 = nn.Sequential(*vgg_block(in_channels = 64,
                                               out_channels = 128,
                                               repeat = 2))


        self.layer3 = nn.Sequential(*vgg_block(in_channels = 128,
                                               out_channels = 256,
                                               repeat = 4))


        self.layer4 = nn.Sequential(*vgg_block(in_channels = 256,
                                               out_channels = 512,
                                               repeat = 8))

        self.fc = nn.Sequential(  
            nn.Linear(512, 4096),
            nn.ReLU(inplace = True),
            nn.Dropout(), # here we use original dropout again
            nn.Linear(4096, 4096),
            nn.ReLU(inplace = True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

        self.max_pool = nn.MaxPool2d(kernel_size = 2, stride = 2)


    def forward(self, x, drop = True):
        x1 = self.layer1(x)
        if self.drop_type == 'db' and drop: # DropBlock is applied to first two groups' outputs
            x1 = self.drop_layer(x1)
        x1 = self.max_pool(x1)

        x2 = self.layer2(x1)
        if self.drop_type == 'db' and drop: # DropBlock is applied to first two groups' outputs
            x2 = self.drop_layer(x2)
        x2 = self.max_pool(x2)

        x3 = self.layer3(x2)
        if self.drop_type is not None and self.drop_type != 'db' and drop: # other dropout methods
            x3 = self.drop_layer(x3) # output of penultimate group
        x3 = self.max_pool(x3)

        x4 = self.layer4(x3)
        x4 = self.max_pool(x4)

        x = F.adaptive_avg_pool2d(x4, (1, 1))
        x = x.squeeze(-1).squeeze(-1)
        x = self.fc(x)

        return x

############## Define different vgg19 models ##############
def vgg19(num_classes = 10,
          drop_type = None, drop_rate = 0.5, block_size = 7):
    """VGG19
    Args :
        --num_classes: number of classes, default is 10
        --drop_type: None, 'd' (dropout), 'sd' (SpatialDropout), 'fd' (FocusedDropout), 'db' (DropBlock)
        --drop_rate: default is 0.5 for all dropout methods
        --block_size: default is 7 for DropBlock
    """
    return VGG19(num_classes = num_classes, 
                 drop_type = drop_type, drop_rate = drop_rate, block_size = block_size)


# unit test
if __name__ == '__main__':
    model = vgg19(100, drop_type = 'db')
    print(model)
    x = torch.rand(1, 3, 32, 32)
    y = model(x)
    print(list(model._modules.keys()))
    print(y.shape)
