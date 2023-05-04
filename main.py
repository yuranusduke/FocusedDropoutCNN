"""
This is the main function of all project
We reproduce a published Computer Vision (CV) paper: <FocusedDropout for Convolutional Neural Network>
Where we reproduce some of experiments: CIFAR10/CIFAR100 on 
resnet20/56/110/VGG19/DenseNet100/WRN28, TinyImageNet is not included
due to limited computation. We also consider some CAM visualization
and ablations
"""

import argparse
import torch
from engine.train import train
from engine.test import evaluate
from engine.parse_results import parse_test_results
import os
from utils import get_cfg_default
import sys
import logging

import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)
logger.setLevel(level = logging.DEBUG)
# console
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(level = logging.DEBUG)
logger.addHandler(stream_handler)

####################################
#            Print args            #
####################################
def print_args(args, cfg, logger):
    """Print arguments"""
    logger.info("***************")
    logger.info("** Arguments **")
    logger.info("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        logger.info("{}: {}".format(key, args.__dict__[key]))
    logger.info("************")
    logger.info("** Config **")
    logger.info("************")
    logger.info(cfg)

####################################
#           Set up args            #
####################################
def setup_cfg(args):
    """set up new arguments from users
    Specfically, we set up
        - dataname
        - part_rate
        - weight_decay
        - drop_rate
        - backbone
        - drop_method
    """
    cfg = get_cfg_default(root = './cfg')

    assert args.dataname in ('cifar10', 'cifar100'), "Only cifar10 and cifar100 are supported!"

    cfg['DATALOADER']['DATANAME'] = args.dataname 

    if args.dataname == 'cifar10':
        cfg['DATALOADER']['NUM_CLASSES'] = 10
    else:
        cfg['DATALOADER']['NUM_CLASSES'] = 100

    if args.part_rate:
        cfg['OPTIM']['PART_RATE'] = args.part_rate

    cfg['OPTIM']['WEIGHT_DECAY'] = args.weight_decay

    cfg['OPTIM']['DROP_RATE'] = args.drop_rate

    if args.drop_method != 'no':
        cfg['OPTIM']['DROP_METHOD'] = args.drop_method
    else:
        cfg['OPTIM']['DROP_METHOD'] = 'no'
        cfg['OPTIM']['DROP_RATE'] = 0.

    cfg['MODEL']['BACKBONE'] = args.backbone

    if args.backbone == 'wrn28': # we follow original paper
        cfg['OPTIM']['LR_MILESTONES'] = [60, 80, 120]
        cfg['OPTIM']['PREC'] = 'fp32' # 32 precision for wrn28
        cfg['DATALOADER']['BATCH_SIZE'] = 48 # lower batch size
    else:
        cfg['OPTIM']['LR_MILESTONES'] = [20, 40, 80, 120, 140]
        cfg['OPTIM']['PREC'] = 'fp16' # 16 precision for other models

    cfg['SEED'] = args.seed

    cfg['OUTPUT']['DIR'] = args.output_path

    return cfg

####################################
#          Main function           #
####################################
def main(args):
    """
    Main function
    """
    torch.manual_seed(args.seed)

    cfg = setup_cfg(args)
    if not os.path.exists(cfg['OUTPUT']['DIR']):
        os.makedirs(cfg['OUTPUT']['DIR'])

    file_handler = logging.FileHandler(os.path.join(args.output_path, 'output.log'), 'a+')
    file_handler.setLevel(level = logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info('\n')
    print_args(args, cfg, logger)

    if args.eval_only != 0: # only for testing
        evaluate(cfg, logger)
    else: # both train and evaluate
        train(cfg, logger)
        evaluate(cfg, logger)

# start project!
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_only", type = int, help = "evaluation only")
    parser.add_argument("--seed", type = int, default = 0, help = "random seed")
    parser.add_argument("--output_path", type = str, default = "", help = "output path")

    parser.add_argument("--dataname", type = str, default = "", help = "data set name (cifar10/cifar100)")

    parser.add_argument("--backbone", type = str, default = "", help = "backbone (resnet20/resnet56/resnet110/vgg19/wrn28/densent100)")
    parser.add_argument("--drop_method", type = str, default = "no", help = "dropout method ('d' (dropout), 'sd' (SpatialDropout), 'fd' (FocusedDropout), 'db' (DropBlock) )")
    parser.add_argument("--drop_rate", type = float, default = 0.5, help = "drop rate")
    parser.add_argument("--part_rate", type = float, default = 0.1, help = "participation rate")
    parser.add_argument("--weight_decay", type = float, default = 0., help = "weight decay")

    args = parser.parse_args()
    main(args)