"""
Simple implementation of Focused Dropout
From paper : <FocusedDropout for Convolutional Neural Network>
"""

import torch
from torch import nn as nn

####################################
#         Define FocusedDropout    #
####################################
class FocusedDropout(nn.Module):
    """Define Focused Dropout module"""

    def __init__(self, low = 0.6, high = 0.9, half = True):
        """
        Args :
            --low: left value in random range, default is 0.6
            --high: right value in random range, default is 0.9
            --half
        """
        super(FocusedDropout, self).__init__()

        self.low = low
        self.high = high
        self.half_ = half
        if self.half_:
            self.avg_pool = nn.AdaptiveAvgPool2d(1).half()
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        assert x.dim() == 4, \
            "Expected input with 4 dimensions (bsize, channels, height, width)"

        if self.training: # training
            # 1. First, we need to do global average pooling
            x_ = self.avg_pool(x) # [m, C, 1, 1]
            x2 = x_.squeeze(-1).squeeze(-1) # [m, C]

            # 2. Find the maximum channel
            max_channel_index = torch.argmax(x2, dim = -1, keepdims = True) # [m, 1]
            max_channel = x.gather(1, max_channel_index[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])) # [m, h, w]
            reshaped_max_channel = max_channel.reshape(max_channel.shape[0], -1) # [m, h * w]

            # 3. Find max pixel
            max_pixel = torch.max(reshaped_max_channel, dim = -1)[0][:, None, None] # [m, 1, 1]
            # 4. Generate mask
            rand = torch.rand(x.shape[0], x.shape[-2], x.shape[-1], dtype = torch.float16 if self.half_ else torch.float32).to(x.device)
            rand = rand * (self.high - self.low) + self.low # [0, 1] -> [low, high]
            back_max_channel = max_channel.reshape(x.shape[0], x.shape[-2], x.shape[-1])
            mask = (back_max_channel > rand * max_pixel) # 1 when max_channel is larger than rand number, similar to dropout # [m, h, w]
            if self.half_:
                mask = mask.to(torch.float16)

            # 5. Multiply mask with x
            x = x * mask.unsqueeze(1)

            return x

        else: # testing
            return x