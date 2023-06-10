# Unofficial Implementation of  ***FocusedDropout for Convolutional Neural Network***

![](https://img.shields.io/badge/data%20science-reproducible%20research-blue) ![](https://img.shields.io/badge/deeplearning-computer%20vision-orange) ![](https://img.shields.io/badge/deeplearning-dropout-green) 

### <font color = 'green'> Project for Reproducible Research </font> :art:

Created by: Kunhong Yu (444447)/Islam Islamov (444601)/Leyla Ellazova (444831)

- [Main Contents]()
	- [Background](#background)
		- [Data](#data)
	- [Requirements](#requirements)
	- [Implementation](#implementation)
		- [Code Organization](#code-organization)
		- [Hyper-parameters and Defaults](#hyper-parameters-and-defaults)
		- [Train and Test](#train-and-test)
	- [Results](#results)
	- [Contributions](#contributions)
	- [References](#references)

First of all, thank the authors very much for sharing this excellent paper [***FocusedDropout for Convolutional Neural Network***](https://arxiv.org/abs/2103.15425) with us. This repository contains FoucusedDropout and some basic implementation of experiments in paper. 
Specifically, we reproduce `Table 1`, `Table 2`, `Figure 4` and `Figure 5`. Since authors did not release original code and some important hyper-parameters, we try our best to achieve performance they claim in the paper, but the main motivation is to verify the proposed FocusedDropout is state-of-the-art, not to show exact testing accuracy, we use NVIDIA RTX A5000 to train and test all models, but we can not burden a lot of costs of all experiments, that's why we only reproduce parts of them.

## Background
Dropout is a common-used technique in Neural Network (e.g. CNN, RNN, Transformer, etc.) to prevent co-adaptation and overfitting. In this paper, authors propose a novel dropout technique -- FocusedDropout, which does not drop neurons randomly, but 
focuses more on elements related with classification information.

<p align="center">
	<img src = ./README/fd.png />
</p>

And whole algorithm goes,

<p align="center">
	<img src = ./README/al.png />
</p>


Authors also compare with original [Dropout](https://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf), [SpatialDropout](https://arxiv.org/pdf/1411.4280.pdf), [DropBlock](https://arxiv.org/abs/1810.12890). 
In simple words, Dropout randomly drops neurons in the model and SpatialDropout randomly drops whole feature channel in a feature map and DropBlock drops continuous area in spatial way in a single feature channel, but FocusedDropout drops neurons according to class information. In experiments, authors use [CIFAR10, CIFAR100](https://www.cs.toronto.edu/~kriz/cifar.html) as data sets and [ResNet20/56/110](https://arxiv.org/abs/1512.03385), [VGG19](https://arxiv.org/abs/1409.1556), [WRN28](https://arxiv.org/abs/1605.07146) and [DenseNet100](https://arxiv.org/abs/1608.06993) as backbones. We won't go every detail in these since there is a much broader technique stack in Computer Vision with deep learning, in this repo, we more focus on implementation to reproduce, moreover, we use `fp16` which is half of float32 to define and train models to save memory usage.

### Data
The CIFAR10 dataset consists of 60000 32×32 color images of 10 classes, each with 6000 images including 5000 training images and 1000 test images. The CIFAR100 dataset has the same number of images but 100 classes, which means the training of CIFAR100 is harder than CIFAR10 because of more image types and less training data for each kinds. 

## Requirements

In order to reimplement our results, please first install some packages by typing following command in command line, but before doing this, you should install [anaconda](https://www.anaconda.com/) based on your device information:

```Python
pip install -r requirements.txt 
```

## Implementation

### Code Organization

```bash
├── README.md
├── cfg
│   └── config.yaml
├── checkpoints
├── data
├── engine
│   ├── __init__.py
│   ├── parse_results.py
│   ├── test.py
│   ├── train.py
│   └── train_one_epoch.py
├── generate_anim_cam.py
├── main.py
├── models
│   ├── backbones
│   │   ├── __init__.py
│   │   ├── densenet100.py
│   │   ├── resnet.py
│   │   ├── vgg19.py
│   │   └── wrn28.py
│   └── dropout
│       ├── DropBlock.py
│       ├── Dropout.py
│       ├── FocusedDropout.py
│       ├── SpatialDropout.py
│       └── __init__.py
├── parse.sh
├── parse_full.sh
├── run.sh
├── run_full.sh
└── utils.py
```

Here we display all codes we create in the project. `README.md` is markdown file you are reading now. `config.yaml` is a `yaml` file to build configuration like you will see in the next subsection. `checkpoints` is a folder which saves all trained models and details of training/testing. `data` stores data sets. `engine` stores training and testing functions, where `parse_results.py` is used to parse testing results to command line, `test.py` is used to test models, `train.py` and `train_one_epoch.py` are utilized to train models. `generate_anim_cam.py` generates animated CAM which you will see in the following section. `main.py` is main function of whole project. `models` stores all models' definitions, where `backbones` contains `densenet100`, all kinds of `resnet`, `vgg19` and `wrn28` models, `dropout` subfolder contains all kinds of dropout method, including FocusedDropout. `utils.py` contains utility functions. Last but not least, three more `bash` scripts are provided to run different kinds of experiments which you will gain more intuition in the following sections.	

### Hyper-parameters and Defaults

Some hyper-parameters were not specified in original paper, we define as following from CV research field which are commonly accepted:

```python
OPTIM:
  LR: 0.1 # initial learning rate          
  MAX_EPOCH: 100 # max training epochs
  PART_RATE: 0.1 # participation rate
  WEIGHT_DECAY: 5e-4 # for FocusedDropout, we need to put higher weight decay as paper demonstrated
  DROP_RATE: 0.3 # drop our ratio
  DROP_METHOD: "" # drop method, ('no' (no dropout), 'd' (dropout), 'sd' (SpatialDropout), 'fd' (FocusedDropout), 'db' (DropBlock)
  BLOCK_SIZE: 7 # default # block size in DropBlock

MODEL:
  BACKBONE: "resnet20" # resnet20/resnet56/resnet110/vgg19/densenet100/wrn28
```
In order to reduce training time, we did not follow exact hyper-parameters settings, for example, in original paper, authors used stepped learning rate decay, here, we use [cosine learning rate decay](https://arxiv.org/pdf/1608.03983.pdf) with max learning rate being 0.1, we also set max learning epochs being 100 instead of 300, default weight decay is 5e-4 and drop rate is 0.3 for fair comparison. In DropBlock method, we set block size to be 7. All models' batch size is 128 except 48 in wrn28 since we use `fp32` in wrn28. Due to above settings, we may see different cost curves in `Figure 5` from original paper, `Figure 5` is produced by stepped learning rate decay. As we have mentioned before, we want to know if proposal outperforms than other SOTA methods, ideally, in any reasonable settings, proposal is better than others, so want to reproduce this proof. To our knowledge, one of biggest advantages in FocusedDropout is there are no explicit hyper-parameters, in this repo, what we reproduce for other methods (e.g., Dropout, etc.) are hyper-parameter sensitive, we choose them randomly in a range that is reasonable in CV research.

### Train and Test

We prepare `bash` scripts to run all experiments at your ease, below are some examples:

1. To run cifar10 with resnet20 and no dropout:
```bash
bash run.sh cifar10 resnet20 no 0. 
```

2. To run cifar10 with resnet56 and FocusedDropout:
```bash
bash run.sh cifar10 resnet56 fd 0. 
```

3. To run cifar100 with wrn28 and SpatialDropout and drop rate 0.5:
```bash
bash run.sh cifar100 wrn28 sd 0.5 
```

4. To run cifar100 with vgg19 and DropBlock and drop rate 0.5:
```bash
bash run.sh cifar100 vgg19 db 0.5 
```

5. To run cifar10 with densenet100 and Dropout and drop rate 0.5:
```bash
bash run.sh cifar10 densenet100 d 0.5 
```

In order to let you run all experiments, we also provide a simple `bash` script `run_full.sh`, which demonstrates all experiments we have done for this repo, but this takes too long to run.

We have trained all models but we can not upload pre-trained models since they are very large, but we include all training/testing statistics in the repo, so we can simply parse results using following script as an example:

```bash
bash parse.sh cifar100 wrn28 sd 0.5
```

Or for simplicity, you are highly recommended to run 

```bash 
bash parse_full.sh
```
to parse all results at once.

**NOTE: when running code, there is an error to indicate no corresponding folder, feel free to create one, for example `data`**.


## Results

Please refer to [result page](https://github.com/yuranusduke/FocusedDropoutCNN/tree/results).


## Contributions 

In this part, we dynamically update our contribution in code part for our group with three people. For results' contributions, please refer to [result page](https://github.com/yuranusduke/FocusedDropoutCNN/tree/results).

:heavy_check_mark: 2023/04/07: Kunhong Yu creates original and empty repo to start project

:heavy_check_mark: 2023/04/08: Kunhong Yu adds initial `config/cfg.yaml` file

:heavy_check_mark: 2023/04/09: Kunhong Yu uploads original paper of FocusedDropout

:heavy_check_mark: 2023/04/11: Kunhong Yu adds three dropout methods: Dropout, SpatialDropout and DropBlock

:heavy_check_mark: 2023/04/12: Kunhong Yu adds FocusedDropout implementation

:heavy_check_mark: 2023/04/15: Kunhong Yu adds `utils.py` script 

:heavy_check_mark: 2023/04/16: Kunhong Yu modifies multiple files, including `config.yaml`, and fix all dropout methods' bugs and upload running script `run.sh`

:heavy_check_mark: 2023/04/24: Kunhong Yu finishes `resnet.py` to implement all resnet-based models: ResNet20/56/110.

:heavy_check_mark: 2023/05/04: Kunhong Yu uploads `main.py`, `train_one_epoch.py` and `run_full.sh` files.

:heavy_check_mark: 2023/05/07: Kunhong Yu uploads `generate_anim_cam.py`.

:heavy_check_mark: 2023/05/09: Kunhong Yu creates a new branch `results` to store experimental results.

:heavy_check_mark: 2023/05/13: Kunhong Yu adds `densenet100.py`, and parse shell script.

:heavy_check_mark: 2023/05/17: Kunhong Yu adds few more python files and fixes bugs in code and update README.md.

:heavy_check_mark: 2023/06/10: Islam Islamov adds student id and makes some testing.

:heavy_check_mark: 2023/06/10: Islam Islamov adds `wrn28.py` to backbones folder.

:heavy_check_mark: 2023/06/10: Leyla Ellazova adds student id.

:heavy_check_mark: 2023/06/10: Leyla Ellazova added `vgg19.py` to backbone folder.


## References
- FocusedDropout for Convolutional Neural Network [[ref]](https://arxiv.org/abs/2103.15425)
- Dropout: A Simple Way to Prevent Neural Networks from
Overfitting [[ref]](https://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)
- Efficient Object Localization Using Convolutional Networks [[ref]](https://arxiv.org/pdf/1411.4280.pdf)
- DropBlock: A regularization method for convolutional networks [[ref]](https://arxiv.org/abs/1810.12890)
- Deep Residual Learning for Image Recognition [[ref]](https://arxiv.org/abs/1512.03385)
- Very Deep Convolutional Networks for Large-Scale Image Recognition [[ref]](https://arxiv.org/abs/1409.1556)
- Densely Connected Convolutional Networks [[ref]](https://arxiv.org/abs/1608.06993)
- Wide Residual Networks [[ref]](https://arxiv.org/abs/1605.07146)
- Learning Multiple Layers of Features from Tiny Images [[ref]](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf)
- Learning Deep Features for Discriminative Localization [[ref]](https://arxiv.org/abs/1512.04150)
- SGDR: Stochastic Gradient Descent with Warm Restarts [[ref]](https://arxiv.org/pdf/1608.03983.pdf)


### To cite our work :black_nib:

Also we would appreciate it if receiving a star :star:.

```
@misc{FDI,
  author       = "KunhongYu/Islam Islamov/Leyla Ellazova",
  title        = "Unofficial implementation of FocusedDropout",
  howpublished = "\url{}",
  note         = "Accessed: 2023-xx-xx"
}
```

***<center>Veni，vidi，vici --Caesar</center>***
