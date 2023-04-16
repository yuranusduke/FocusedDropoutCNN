"""
Utility functions
"""

import torch
import torchvision as tv
import yaml
import sys
from models.backbones import *
import os
import tqdm
import math

####################################
#        About configurations      #
####################################
def get_cfg_default(root : str):
	"""Get configurations
	Args :
		--root: root of data
		--level: model level
	return :
		--config dict
	"""
	with open(os.path.join(root, 'config.yaml')) as stream:
		try:
			cfg = yaml.safe_load(stream)
			return cfg
		except yaml.YAMLError as exc:
			print(exc)

def set_device():
	"""
	This function is used to set device of training and testing
	return : 
		--set device
	"""
	is_gpu = torch.cuda.is_available()
	if is_gpu:
		device_count = torch.cuda.device_count()
		if device_count > 1:
			print('Using GPU:1')
			device = torch.device('cuda:1')
		else:
			print('Using GPU:0')
			device = torch.device('cuda')
	else:
		print('Using CPU')
		device = torch.device('cpu')

	return device

####################################
#            About data            #
####################################
def get_dataloader(cfg, mode = 'train'):
	"""
	Get the data set and data loader
	Args : 
		--cfg: configuration yaml
		--mode: 'train'/'test'
	return : 
		--data_loader
	"""
	assert mode in ('train', 'test')

	if mode == 'train': # we follow original paper to include random crop and random flip in training
		transform = \
			tv.transforms.Compose([
				tv.transforms.RandomHorizontalFlip(),
				tv.transforms.RandomCrop(int(cfg['INPUT']['SIZE'][0]), padding = 4),
				tv.transforms.RandomRotation(15),
				tv.transforms.ToTensor(),
				tv.transforms.Normalize(mean = cfg['INPUT']['PIXEL_MEAN'], std = cfg['INPUT']['PIXEL_STD'])
			])
	else: # testing
		transform = \
			tv.transforms.Compose([
				tv.transforms.ToTensor(),
				tv.transforms.Normalize(mean = cfg['INPUT']['PIXEL_MEAN'], std = cfg['INPUT']['PIXEL_STD'])
			])


	if cfg['DATALOADER']['DATANAME'] == 'cifar10': # cifar10
		train_data = tv.datasets.CIFAR10(root = './data',
										 download = True,
										 train = True if mode == 'train' else False,
										 transform = transform)
	else: # cifar100
		train_data = tv.datasets.CIFAR100(root = './data',
										  download = True,
										  train = True if mode == 'train' else False,
										  transform = transform)

	data_loader = torch.utils.data.DataLoader(train_data, shuffle = True if mode == 'train' else False,
										  	  batch_size = cfg['DATALOADER']['BATCH_SIZE'])

	return data_loader

####################################
#          About models            #
####################################
def load_model(cfg, mode = 'train'):
	"""
	Loading model
	Args :
		--cfg
		--mode: 'train'/'test'
	"""
	assert mode in ('train', 'test')

	params = {'num_classes' : cfg['DATALOADER']['NUM_CLASSES'], 'drop_type' : cfg['OPTIM']['DROP_METHOD'],
			  'drop_rate' : cfg['OPTIM']['DROP_RATE'], 'block_size' : cfg['OPTIM']['BLOCK_SIZE']}

	if cfg['MODEL']['BACKBONE'] == 'resnet20': # resnet20
		model = resnet20(**params)
	elif cfg['MODEL']['BACKBONE'] == 'resnet56': # resnet56
		model = resnet56(**params)
	elif cfg['MODEL']['BACKBONE'] == 'resnet110': # resnet110
		model = resnet110(**params)
	elif cfg['MODEL']['BACKBONE'] == 'vgg19': # vgg19
		model = vgg19(**params)
	elif cfg['MODEL']['BACKBONE'] == 'wrn28': # wrn28
		model = wrn28(**params)
	elif cfg['MODEL']['BACKBONE'] == 'densenet100': # densenet100
		model = densenet121(**params)

	if mode == 'test':
		# load pretrained weights for testing
		model_dir = os.path.join(cfg['OUTPUT']['DIR'])
		files = os.listdir(model_dir)
		files = list(filter(lambda x: x.endswith('pth'), files))
		model.load_state_dict(torch.load(os.path.join(model_dir, files[0]), map_location = "cpu")) # first load in cpu

	if cfg['OPTIM']['PREC'] == 'fp16':
		model = model.half()

	return model

############################
#     About Training       #
############################
class AverageMeter(object):
	"""Compute and store the average and current value."""

	def __init__(self, ema = False):
		"""
		Args :
			--ema (bool, optional): apply exponential moving average.
		"""
		self.ema = ema
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n = 1):
		if isinstance(val, torch.Tensor):
			val = val.item()

		self.val = val
		self.sum += val * n
		self.count += n

		if self.ema:
			self.avg = self.avg * 0.9 + self.val * 0.1
		else:
			self.avg = self.sum / self.count

####################################
#          About testing           #
####################################
def cal_acc(model, device, dataloader = None, x = None, y = None, mode = 'train', half = True):
	"""
	Compute accuracy of current model
	Args :
		--model: input model
		--device
		--dataloader: loader of data set
		--x: input batch x
		--y: input batch_y
		--mode: 'train'/'test'
		--verbose
		--half
	"""
	model.eval() # important to disable dropout

	with torch.no_grad():
		if mode == 'train': # train, only one batch
			out = model(x)
			preds = torch.argmax(out, dim = -1)
			acc = float((preds == y).sum()) / x.shape[0]

			return acc

		else: # test
			whole_acc = 0. # we only measure accuracy
			count = 0

			for i, (batch_x, batch_y) in tqdm.tqdm(enumerate(dataloader)):
				sys.stdout.write(f'\r>>Evaluating batch {i + 1} / {len(dataloader)}.')
				sys.stdout.flush()

				if half:
					batch_x = batch_x.half()

				batch_x = batch_x.to(device)
				batch_y = batch_y.to(device)

				out = model(batch_x)
				preds = torch.argmax(out, dim = -1)
				acc = float((preds == batch_y).sum()) / batch_x.shape[0]
				whole_acc += acc
				count += 1

			whole_acc /= count
			print()

			return whole_acc