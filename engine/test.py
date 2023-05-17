"""
Test function
"""

import torch
from utils import cal_acc, load_model, set_device, get_dataloader
from engine.parse_results import interpret_cam
from torch import nn as nn

####################################
#             Evaluate             #
####################################
def evaluate(cfg, logger):
    """This function is used to evaluate model"""
    device = set_device()

    logger.info(f"Reading testing data set {cfg['DATALOADER']['DATANAME']}")
    dataloader = get_dataloader(cfg = cfg, mode = 'test')

    logger.info(f"Loading trained model...")
    model = load_model(cfg, mode = 'test')
    model.to(device)
    model.eval()
    # We now use multiple GPU if possible to do distributed training
    device_count = torch.cuda.device_count()
    if device_count > 1:
        # print(f"Multiple GPUs detected (n_gpus = {device_count}), use all of them!")
        # model = nn.DataParallel(model)
        pass

    half = False
    if cfg['OPTIM']['PREC'] == 'fp16':
        half = True
    acc = cal_acc(model = model, device = device, dataloader = dataloader, x = None, y = None, mode = 'test',
                  half = half)
    logger.info(f"Testing acc : {round(acc * 100., 2)}%.")

    # finally, generate CAM map for only resnet model
    if cfg['MODEL']['BACKBONE'].count('resnet'):
        layer_name = 'layer3'
    elif cfg['MODEL']['BACKBONE'] == 'vgg19' or cfg['MODEL']['BACKBONE'] == 'wrn28':
        # layer_name = 'layer4' # don't generate cam here
        return
    elif cfg['MODEL']['BACKBONE'] == 'densenet100':
        layer_name = 'dense_block3'
        return

    interpret_cam(dataloader, model, device, cfg, layer_name)

    return acc