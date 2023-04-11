"""
We use PyTorch built-in Dropout2d to implement SpatialDropout
Paper: <Efficient Object Localization Using Convolutional Networks>
"""

import torch
from torch import nn as nn

####################################
#       Define SpatialDropout      #
####################################
class SpatialDropout(nn.Module):
	"""Define spatial dropout module"""
	
	def __init__(self, drop_rate = 0.5):
		"""
		Args : 
			--drop_rate: default is 0.5
		"""
		super(SpatialDropout, self).__init__()
		self.dropout = nn.Dropout2d(drop_rate)

	def forward(self, x):
		assert x.dim() == 4, \
            "Expected input with 4 dimensions (bsize, channels, height, width)"
            
		return self.dropout(x)