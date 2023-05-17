"""
Parse testing results, each model will run three times, record mean and standard deviation
"""

import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
import sys
import cv2
from torch.nn import functional as F
import argparse
import torchvision as tv

import sys
sys.path.insert(0, '.')

import warnings
warnings.filterwarnings("ignore")

####################################
#        Parse test results        #
####################################
def parse_test_results(path):
	"""
	Parse test results we average over three runs and compute standard deviation
	Args : 
		--path, parsing path
	"""
	# parent_path = os.sep.join(path.split(os.sep))
	dirs = os.listdir(path)
	dirs = list(filter(lambda x : x.count('seed'), dirs)) # get all dirs with name "seed" in it

	if len(dirs) < 2:
		print(f'Only {len(dirs)} models, train more (at least 2) models first!')
		return 

	accs = []
	string = f'\nPath to be considered:.\n'

	for subpath in dirs:
		subpath = os.path.join(path, subpath)
		string += subpath + '\n'
		with open(os.path.join(subpath, 'output.log'), 'r') as f:
			last_line = f.readlines()[-1].strip()

		acc = last_line.split(' : ')[1][:-2]
		acc = float(acc)
		accs.append(acc)

	accs = np.array(accs)
	mean = round(np.mean(accs), 2)
	std = round(np.std(accs), 2)

	string += f'Current accs\' mean is {mean}, std is {std} || {mean}±{std}%.\n'
	with open(os.path.join(path, 'test_stats.txt'), 'a+') as f:
		f.write(string)
		f.flush()

	print(string)

####################################
#    Parse training statistics     #
####################################
def plot_train_stats(path):
	"""
	Plot training statistics, we build two subplots, one is for train/val loss and one is for train/val acc
	Args : 
		--path
	"""
	with open(os.path.join(path, 'stats.pkl'), 'rb') as f:
		stats = pickle.load(f)

	f, ax = plt.subplots(1, 2, figsize = (12, 6))
	train_loss, val_loss, train_acc, val_acc = stats['train_loss'], stats['val_loss'], stats['train_acc'], stats['val_acc']

	ax[0].plot(range(len(train_loss)), train_loss, 'b-', label = 'training_loss', linewidth = 2)
	ax[0].plot(range(len(val_loss)), val_loss, 'g--', label = 'val_loss', linewidth = 2)
	ax[0].set_xlabel('Epoch')
	ax[0].set_ylabel('Loss')
	ax[0].grid(True)
	ax[0].set_title('Losses')
	ax[0].legend(loc = 'best')

	ax[1].plot(range(len(train_acc)), train_acc, 'k-', label = 'training_acc', linewidth = 2)
	ax[1].plot(range(len(val_acc)), val_acc, 'r--', label = 'val_acc', linewidth = 2)
	ax[1].set_xlabel('Epoch')
	ax[1].set_ylabel('Accuracy (%)')
	ax[1].grid(True)
	ax[1].set_title('Accs')
	ax[1].legend(loc = 'best')

	plt.savefig(os.path.join(path, 'stats.pdf'))
	plt.close()

####################################
#          Generate CAM map        #
####################################
def returnCAM(feature_conv, weight_softmax, class_idx, img_size):
	""" 
	Generate the class activation maps upsample to image size
	Args : 
		--feature_conv: feature from convolution layer
		--weight_softmax: softmax output weight
		--class_idx: class index
		--img_size
	"""
	size_upsample = (img_size, img_size)
	_, nc, h, w = feature_conv.shape
	output_cam = []
	cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h*w)))
	cam = cam.reshape(h, w)
	cam = cam - np.min(cam)
	cam_img = cam / np.max(cam)
	cam_img = np.uint8(255. * cam_img)
	output_cam.append(cv2.resize(cam_img, size_upsample))

	return output_cam

def get_cam(model, img, cfg, device, layer_name, img_ori):
	"""
	Get CAM map
	Args : 
		--model: model instance
		--features_blobs
		--img
		--cfg
		--device
		--layer_name
		--img_ori: original image
	"""
	# hook
	feature_blobs = []
	def _hook_feature(module, input, output):
		"""
		Hook feature in the network
		Args : 
			--module: 
			--input: input name
			--output
		"""
		feature_blobs.append(output.cpu().data.numpy())

	model._modules.get(layer_name).register_forward_hook(_hook_feature)
	params = list(model.parameters())
	# get weight only from the last layer(linear)
	weight_softmax = np.squeeze(params[-2].cpu().data.numpy())

	logit = model(img).float()
	img = img.squeeze()
	h_x = F.softmax(logit, dim = 1).data.squeeze()
	probs, idx = h_x.sort(0, True)
	CAMs = returnCAM(feature_blobs[0], weight_softmax, [idx[0].item()], img.shape[-1])
	_, height, width = img.shape
	heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (width, height)), cv2.COLORMAP_JET)
	result = heatmap * 0.3 + (img_ori * 255).float().permute(1, 2, 0).cpu().data.numpy() * 0.5 # weighted sum from original paper
	return result

def interpret_cam(dataloader, model, device, cfg, layer_name):
	"""
	Generate CAM map
	CAM paper: <Learning Deep Features for Discriminative Localization>
	Reference: https://github.com/chaeyoung-lee/pytorch-CAM/blob/master/update.py
	Args : 
		--dataloader
		--model
		--device
		--cfg
		--layer_name
	"""
	invTrans = tv.transforms.Normalize(
					   mean = [-m / s for m, s in zip(cfg['INPUT']['PIXEL_MEAN'],
												   cfg['INPUT']['PIXEL_STD'])],
					   std = [1 / s for s in cfg['INPUT']['PIXEL_STD']])
	model.eval()
	path = cfg['OUTPUT']['DIR']
	path = os.path.join(path, 'cam')
	if not os.path.exists(path):
		os.makedirs(path)

	count = 0
	for i, (batch_x, _) in enumerate(dataloader):
		sys.stdout.write(f'\r>>Generating CAM map for batch {i + 1} / {len(dataloader)}.')
		sys.stdout.flush()

		if cfg['OPTIM']['PREC'] == 'fp16':
			batch_x = batch_x.half()
		batch_x = batch_x.to(device)
		# for each batch, we randomly choose 1 images
		indices = np.random.permutation(batch_x.shape[0])[:1]
		x = batch_x[indices]
		for j in range(x.shape[0]):
			f, ax = plt.subplots(1, 2, figsize = (12, 6))
			cam = get_cam(model, x[j : j + 1], cfg, device, layer_name, invTrans(x[j]))
			ax[0].imshow(invTrans(x[j]).float().data.permute(1, 2, 0).cpu().numpy())
			ax[0].axis('off')
			ax[1].imshow(cam.astype('uint8'))
			ax[1].axis('off')
			plt.savefig(os.path.join(path, f'cam_{count + 1}.pdf'))
			plt.close()
			count += 1


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--output_path", type = str, default = "", help = "output path to read statistics, stopping with 'seed subfolder'")

	args = parser.parse_args()
	parse_test_results(args.output_path) # parse results

	dirs = os.listdir(args.output_path)
	dirs = list(filter(lambda x : x.count('seed'), dirs))
	for subfolder in dirs: # plot for each seed
		path = os.path.join(args.output_path, subfolder)
		plot_train_stats(path)