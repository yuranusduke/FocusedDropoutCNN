"""
Train model in one epoch
"""

import torch
from utils import cal_acc, AverageMeter
import os
import time
import datetime
import numpy as np

####################################
#         Train one epoch          #
####################################
def train_one_epoch(dataloader, device, model, optimizer, loss, epoch : int, cfg, logger):
	"""
	Train one epoch of the model
	According to paper, we need to only specify only 10% (part_rate) of batches to perform FocusedDropout,
	to perform this, each batch, we generate a binary mask in one number to enable FD. Statistically,
	10% must be achieved since binary mask is a 50% probability, 10% is no problem.
	Args : 
		--dataloader: input data loader
		--device
		--model: model instance
		--optimizer: training optimizer
		--loss: loss instance
		--epoch
		--cfg
		--logger
	return : 
		--epoch_loss and epoch acc
	"""
	model.train()
	end = time.time()
	batch_time = AverageMeter()
	fd_drop_cum = 0.
	epoch_loss, epoch_acc = 0., 0.
	for i, (batch_x, batch_y) in enumerate(dataloader):
		if cfg['OPTIM']['PREC'] == 'fp16':
			batch_x = batch_x.half()
		batch_x = batch_x.to(device)
		batch_y = batch_y.to(device)

		optimizer.zero_grad()
		#with torch.cuda.amp.autocast(): # autocast to avoid data problem, especially in wideresnet28, still don't know why yet
		if cfg['OPTIM']['DROP_METHOD'] != 'no':
			if cfg['OPTIM']['DROP_METHOD'] == 'fd':
				if fd_drop_cum / len(dataloader) <= cfg['OPTIM']['PART_RATE']:
					# generate a binary mask in real number
					enable_fd = np.random.randint(0, 2)
					if enable_fd == 1: # enable fd
						fd_drop_cum += 1
						out = model(batch_x, True)
					else: # disable fd
						out = model(batch_x, False)
				else: # disable fd
					out = model(batch_x, False)

			else:
				out = model(batch_x, True)
		else:
			out = model(batch_x, False)

		batch_loss = loss(out, batch_y)
		batch_loss.backward()
		optimizer.step()

		batch_time.update(time.time() - end)
		batch_acc = cal_acc(model = model, device = device, dataloader = None, 
							x = batch_x, y = batch_y, mode = 'train')
		model.train()

		epoch_loss += batch_loss.item()
		epoch_acc += batch_acc

		if i % cfg['DATALOADER']['BATCH_SIZE'] == 0:
			batch_acc = '{:.2f}%'.format(batch_acc * 100.)
			batch_loss = '{:.4f}'.format(batch_loss)

			nb_remain = 0
			nb_remain += len(dataloader) - i - 1
			nb_remain += (cfg['OPTIM']['MAX_EPOCH'] - epoch - 1) * len(dataloader)
			eta_seconds = batch_time.avg * nb_remain
			eta = str(datetime.timedelta(seconds = int(eta_seconds)))

			model.train()
			logger.info(f"[INFO] {datetime.datetime.now()} Epoch [{epoch + 1}/{cfg['OPTIM']['MAX_EPOCH']}] batch [{i + 1}/{len(dataloader)}] loss {batch_loss} lr {optimizer.param_groups[0]['lr']} acc {batch_acc} eta {eta}")

		end = time.time()

	epoch_loss /= len(dataloader)
	epoch_acc /= len(dataloader)

	return epoch_loss, epoch_acc