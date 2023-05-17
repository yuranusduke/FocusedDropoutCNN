"""
Train whole model
"""

from utils import set_device, cal_acc, AverageMeter, get_dataloader, load_model
from engine.train_one_epoch import train_one_epoch
from engine.parse_results import plot_train_stats
import pickle
import torch
import os
from torch import nn as nn

####################################
#          Train function          #
####################################
# Load model
def train(cfg, logger):
	"""
	Train model
	Args : 
		--cfg: configuration
		--logger: logger instance
	"""
	device = set_device()

	logger.info(f"Reading training data set {cfg['DATALOADER']['DATANAME']}")
	dataloader = get_dataloader(cfg = cfg, mode = 'train')
	test_dataloader = get_dataloader(cfg = cfg, mode = 'test') # load test data loader

	logger.info(f"Loading raw model")
	model = load_model(cfg, mode = 'train')
	model.to(device)
	# We now use multiple GPU if possible to do distributed training
	device_count = torch.cuda.device_count()
	if device_count > 1:
		# print(f"Multiple GPUs detected (n_gpus = {device_count}), use all of them!")
		# model = nn.DataParallel(model)
		pass

	# Define loss, optimizer and lr scheduler

	loss = torch.nn.CrossEntropyLoss()
	if cfg['OPTIM']['PREC'] == 'fp16':
		loss = loss.half()
	loss = loss.to(device)
	optimizer = torch.optim.SGD(model.parameters(), lr = cfg['OPTIM']['LR'], momentum = 0.9, weight_decay = cfg['OPTIM']['WEIGHT_DECAY'])
	# optimizer = torch.optim.Adam(model.parameters(), lr = cfg['OPTIM']['LR'],
	# 							 weight_decay = cfg['OPTIM']['WEIGHT_DECAY'])
	# lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, gamma = cfg['OPTIM']['LR_DECAY'], milestones = cfg['OPTIM']['LR_MILESTONES'])
	lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = cfg['OPTIM']['MAX_EPOCH'])

	logger.info(f"\nSaving everything to {cfg['OUTPUT']['DIR']}")
	logger.info(f"Start training...")

	train_losses = []
	val_losses = []
	train_accs = []
	val_accs = []
	for epoch in range(cfg['OPTIM']['MAX_EPOCH']):
		model.train()
		train_loss, train_acc = train_one_epoch(dataloader = dataloader, device = device, model = model, optimizer = optimizer, 
												loss = loss, epoch = epoch, cfg = cfg, logger = logger)
		train_accs.append(train_acc)
		# validation acc
		logger.info(f"Computing epoch {epoch + 1} validation accuracy...")
		half = False
		if cfg['OPTIM']['PREC'] == 'fp16':
			half = True
		val_acc = cal_acc(model = model, device = device, dataloader = test_dataloader, x = None, y = None, mode = 'test',
						  half = half)
		val_accs.append(val_acc)

		train_losses.append(train_loss)
		# validation loss
		logger.info(f"Computing epoch {epoch + 1} validation loss...")
		val_loss = compute_all_loss(model = model, dataloader = test_dataloader, loss = loss, device = device, half = half)
		val_losses.append(val_loss)

		model.train()
		lr_scheduler.step()

	print('Training is done!')

	torch.save(model.state_dict(), os.path.join(cfg['OUTPUT']['DIR'], 'trained_model.pth'))
	# save training statistics into pickle
	res = {'train_loss' : train_losses, 'val_loss' : val_losses, 'train_acc' : train_accs, 'val_acc' : val_accs}
	with open(os.path.join(cfg['OUTPUT']['DIR'], 'stats.pkl'), 'wb') as f:
		pickle.dump(res, f)

	# plot_train_stats(cfg['OUTPUT']['DIR']) # plot training statistics

####################################
#              Utility             #
####################################
def compute_all_loss(model, dataloader, loss, device, half = True):
	"""
	Compute all loss across data set
	Args : 
		--model: model instance
		--dataloader: loader
		--loss: loss function
		--device
		--half
	return : 
		--loss result
	"""
	whole_loss = 0.
	for i, (batch_x, batch_y) in enumerate(dataloader):
		if half:
			batch_x = batch_x.half()
		batch_x = batch_x.to(device)
		batch_y = batch_y.to(device)

		out = model(batch_x)
		batch_loss = loss(out, batch_y)
		whole_loss += batch_loss.item()

	whole_loss /= len(dataloader)

	return whole_loss