"""
Simple implementation of DropBlock method
Paper: <Dropblock: A regularization method for convolutional networks>
Reference: https://github.com/miguelvr/dropblock/blob/master/dropblock/dropblock.py
"""

import torch
from torch import nn as nn
import numpy as np
from torch.nn import functional as F

####################################
#         Define DropBlock         #
####################################
class DropBlock(nn.Module):
    """Define drop block module We here implement 2d version"""

    def __init__(self, block_size : int, drop_rate = 0.5):
        """
        Args : 
            --block_size
            --drop_rate: default is 0.5
        """
        super(DropBlock, self).__init__()

        self.drop_rate = drop_rate
        self.block_size = block_size

    def forward(self, x):
        # shape: (bsize, channels, height, width)
        assert x.dim() == 4, \
            "Expected input with 4 dimensions (bsize, channels, height, width)"

        if not self.training or self.drop_rate == 0.:
            return x
        else:
            # get gamma value
            gamma = self.__compute_gamma(x)

            # sample mask
            mask = (torch.rand(x.shape[0], *x.shape[2:]) < gamma).float() # dropped mask

            # place mask on input device
            mask = mask.to(x.device)

            # compute block mask
            block_mask = self.__compute_block_mask(mask)

            # apply block mask
            out = x * block_mask[:, None, :, :]

            # scale output
            out = out * block_mask.numel() / block_mask.sum()

            return out

    def __compute_block_mask(self, mask):
        # obtain mask, we can use max pooling to achieve this
        block_mask = F.max_pool2d(input = mask[:, None, :, :], # do this for all channels in parallel
                                  kernel_size = (self.block_size, self.block_size),
                                  stride = (1, 1),
                                  padding = self.block_size // 2)

        if self.block_size % 2 == 0:
            block_mask = block_mask[:, :, :-1, :-1]

        block_mask = 1 - block_mask.squeeze(1)

        return block_mask

    def __compute_gamma(self, x): # compute gamma, obtained from paper
        return self.drop_rate / (self.block_size ** 2)

####################################
#       Define linear scheduler    #
####################################
class LinearScheduler(nn.Module):
    """Define linear scheduler"""

    def __init__(self, drop_layer, start_value : float, stop_value : float, nr_steps : int):
        """
        Args : 
            --drop_layer: dropout instance
            --start_value: start value of drop rate
            --stop_value: stop value of drop rate
            --nr_steps
        """
        super(LinearScheduler, self).__init__()
        self.drop_layer = drop_layer
        self.i = 0
        self.drop_values = np.linspace(start = start_value, 
                                   stop = stop_value, num = int(nr_steps))

    def forward(self, x):
        return self.drop_layer(x)

    def step(self):
        if self.i < len(self.drop_values):
            self.drop_layer.drop_rate = self.drop_values[self.i]

        self.i += 1
