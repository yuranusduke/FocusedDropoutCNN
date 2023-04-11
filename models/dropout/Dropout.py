"""
Simple traditional Dropout
"""

import torch
from torch import nn as nn

####################################
#          Define Dropout          #
####################################
class Dropout(nn.Module):
	"""Define dropout module"""
	
	def __init__(self, drop_rate = 0.5):
		"""
		Args : 
			--drop_rate: default is 0.5
		"""
		super(Dropout, self).__init__()
		self.dropout = nn.Dropout(drop_rate)

	def forward(self, x):
		assert x.dim() == 4, \
            "Expected input with 4 dimensions (bsize, channels, height, width)"
            
		return self.dropout(x)