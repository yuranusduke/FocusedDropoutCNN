"""
This simple script is used to generate animated cam from our provided 
trained model, only used for resnet-based models
"""

import torch
import numpy as np
import argparse
# from utils import cal_acc, load_model, set_device, get_dataloader
# from engine.parse_results import interpret_cam
# from main import setup_cfg
import matplotlib.pyplot as plt
from pdf2image import convert_from_path
import os
import sys
import time
from PIL import Image

import warnings
warnings.filterwarnings("ignore")

def generate_anim_cam(args):
	"""
	Generate animated cam
	Args : 
		--args
	"""
	# cfg = setup_cfg(args)

	# Instead of using trained models, here we use collected cam
	#
	# device = set_device()
	#
	# dataloader = get_dataloader(cfg = cfg, mode = 'test')
	#
	# print("Loading trained model...")
	# model = load_model(cfg, mode = 'test')
	# model.to(device)
	# model.eval()
	#
	# layer_name = 'layer3'
	#
	# interpret_cam(dataloader, model, device, cfg, layer_name)

	f, ax = plt.subplots(1, 1, figsize = (30, 14))
	plt.ion()
	# now we can generate animation
	files = os.listdir(os.path.join(args.output_path, 'cam'))
	files = list(filter(lambda x : x.endswith('pdf'), files))
	im = []
	count = 0
	for file in files:
		if count > args.upper_limit:
			print('\nReach upper limit!')
			break
		sys.stdout.write(f'\r>>Writing {file}.')
		sys.stdout.flush()
		file = os.path.join(args.output_path, 'cam', file)
		ax.clear()
		imgs = convert_from_path(file) # read pdf and convert to image
		img = np.array(imgs[0])
		im.append(Image.fromarray(img.astype('uint8')))
		ax.imshow(img)
		ax.axis('off')
		plt.pause(1.5) # used to show you
	# https://stackoverflow.com/questions/72693589/how-to-generate-a-gif-without-saving-importing-image-files
	im[0].save(os.path.join(args.output_path, 'cam', 'im.gif'), save_all = True, append_images = im[1:],
			   optimize = False, duration = 50, loop = 0) # used to save gif
	print()


# unit test
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--output_path", type = str, default = "", help = "output path")

	parser.add_argument("--dataname", type = str, default = "", help = "data set name (cifar10/cifar100)")
	parser.add_argument("--upper_limit", type = int, default = 50, help = "number of images to show in gif")
	parser.add_argument("--backbone", type = str, default = "", help = "backbone (resnet20/resnet56/resnet110/vgg19/wrn28/densent100)")
	parser.add_argument("--drop_method", type = str, default = "no", help = "dropout method ('d' (dropout), 'sd' (SpatialDropout), 'fd' (FocusedDropout), 'db' (DropBlock) )")

	args = parser.parse_args()
	args.output_path = f'checkpoints/{args.dataname}/{args.backbone}/drop_{args.drop_method}/'
	files = os.listdir(args.output_path)
	files = list(filter(lambda x : os.path.isdir(os.path.join(args.output_path, x)), files))
	name = files[0]
	args.output_path = os.path.join(args.output_path, name, 'seed1')
	generate_anim_cam(args)
	# usage:
	# python generate_anim_cam.py --dataname "cifar10" --backbone "resnet20" --drop_method "d"