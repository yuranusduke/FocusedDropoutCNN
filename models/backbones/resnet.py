"""
Implementation of some resnet models for cifar-10/-100 datasets
Paper: <Deep Residual Learning for Image Recognition>
Official Reference: https://github.com/akamaster/pytorch_resnet_cifar10/blob/master/resnet.pytorch_resnet_cifar10

Thank for the original author: Yerlan Idelbayev

name      | layers | params
ResNet20  |    20  | 0.27M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
"""

import torch
from torch import nn as nn
import sys
sys.path.insert(0, '.')
from models.dropout import *
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable

__all__ = ['ResNet', 'resnet20', 'resnet56', 'resnet110']

####################################
#         Define utilities         #
####################################
def _weights_init(m):
    """
    weights initialization
    Args : 
        --m: input layer instance
    """
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    """Define customised layer"""

    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

class BasicBlock(nn.Module):
    """ Define basic block"""

    expansion = 1
    def __init__(self, in_planes : int, planes : int, stride = 1, option = 'A'):
        """
        Args : 
            --in_planes: input channels
            --planes: output channels
            --stride: default is 1
            --option: 'A' or 'B'
        """
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size = 3, stride = stride, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4), 
                                        		  "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size = 1, stride = stride, bias = False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x) # residual connection
        out = F.relu(out)
        return out

####################################
#          Define ResNet           #
####################################
class ResNet(nn.Module):
    """Define resnet"""

    def __init__(self, block, num_blocks, num_classes = 10,
    	         drop_type = None, drop_rate = 0.5, block_size = 7):
        """
        Args : 
            --block: block type
            --num_block: number of blocks
            --num_classes: number of classes, default is 10
            --drop_type: None, 'd' (dropout), 'sd' (SpatialDropout), 'fd' (FocusedDropout), 'db' (DropBlock)
            --drop_rate: default is 0.5 for all dropout methods
            --block_size: default is 7 for DropBlock
        """
        super(ResNet, self).__init__()
        self.in_planes = 16

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

        self.conv1 = nn.Conv2d(3, 16, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride = 1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride = 2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride = 2)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride : int):
        """make resnet layers
        Args:
            --block: block type, basic block or bottle neck block
            --planes: output depth channel number of this layer
            --num_blocks: how many blocks per layer
            --stride: the stride of the first block of this layer
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x, drop = True):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        if self.drop_type == 'db' and drop: # DropBlock is applied to first two groups' outputs
            out = self.drop_layer(out)

        out = self.layer2(out)
        if self.drop_type == 'db' and drop: # DropBlock is applied to first two groups' outputs
            out = self.drop_layer(out)

        out = self.layer3(out)
        if self.drop_type is not None and self.drop_type != 'db' and drop: # other dropout methods
            out = self.drop_layer(out) # output of penultimate group

        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out

############## Define different resnet models ##############
def resnet20(num_classes = 10, drop_type = None, drop_rate = 0.5, block_size = 7):
    """ResNet20
    Args :
        --num_classes: number of classes, default is 10
        --drop_type: None, 'd' (dropout), 'sd' (SpatialDropout), 'fd' (FocusedDropout), 'db' (DropBlock)
        --drop_rate: default is 0.5 for all dropout methods
        --block_size: default is 7 for DropBlock
    """
    return ResNet(BasicBlock, [3, 3, 3], num_classes = num_classes,
                  drop_type = drop_type, drop_rate = drop_rate, block_size = block_size)

def resnet56(num_classes = 10, drop_type = None, drop_rate = 0.5, block_size = 7):
    """ResNet56
    Args :
        --num_classes: number of classes, default is 10
        --drop_type: None, 'd' (dropout), 'sd' (SpatialDropout), 'fd' (FocusedDropout), 'db' (DropBlock)
        --drop_rate: default is 0.5 for all dropout methods
        --block_size: default is 7 for DropBlock
    """
    return ResNet(BasicBlock, [9, 9, 9], 
                  num_classes = num_classes,
                  drop_type = drop_type, drop_rate = drop_rate, block_size = block_size)

def resnet110(num_classes = 10, drop_type = None, drop_rate = 0.5, block_size = 7):
    """ResNet110
    Args :
        --num_classes: number of classes, default is 10
        --drop_type: None, 'd' (dropout), 'sd' (SpatialDropout), 'fd' (FocusedDropout), 'db' (DropBlock)
        --drop_rate: default is 0.5 for all dropout methods
        --block_size: default is 7 for DropBlock
    """
    return ResNet(BasicBlock, [18, 18, 18], 
                  num_classes = num_classes,
                  drop_type = drop_type, drop_rate = drop_rate, block_size = block_size)


# unit test
if __name__ == '__main__':
    model = resnet56(100, drop_type = 'sd')
    print(model)
    model.to('cuda')
    x = torch.rand(1, 3, 32, 32).to('cuda')
    y = model(x)
    print(list(model._modules.keys()))
    print(y.shape)