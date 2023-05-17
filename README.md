# Unofficial Implementation of  ***FocusedDropout for Convolutional Neural Network***

### <font color = 'green'> Project for Reproducible Research </font>

Created by: Kunhong Yu (444447)/Islam Islamov (444601)/Leyla Ellazova ()

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


## Contribution

In this part, we dynamically update our contribution in our group with three people.

- [x] 2023/04/07: Kunhong Yu creates original and empty repo to start project
- [x] 2023/04/08: Kunhong Yu adds initial `config/cfg.yaml` file
- [x] 2023/04/09: Kunhong Yu uploads original paper of FocusedDropout
- [x] 2023/04/11: Kunhong Yu adds three dropout methods: Dropout, SpatialDropout and DropBlock
- [x] 2023/04/12: Kunhong Yu adds FocusedDropout implementation
- [x] 2023/04/15: Kunhong Yu adds `utils.py` script 
- [x] 2023/04/16: Kunhong Yu modifies multiple files, including `config.yaml`, and fix all dropout methods' bugs and upload running script `run.sh`
- [x] 2023/04/24: Kunhong Yu finishes `resnet.py` to implement all resnet-based models: ResNet20/56/110.
- [x] 2023/05/04: Kunhong Yu uploads `main.py`, `train_one_epoch.py` and `run_full.sh` files.
- [x] 2023/05/07: Kunhong Yu uploads `generate_anim_cam.py`.
- [x] 2023/05/09: Kunhong Yu creates a new branch `results` to store experimental results.
- [x] 2023/05/13: Kunhong Yu adds `densenet100.py`, and parse shell script.
- [x] 2023/05/17: Kunhong Yu adds few more python files and fixes bugs in code.


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


### To cite our work

```
@misc{FDI,
  author       = "KunhongYu/Islam Islamov/Leyla Ellazova",
  title        = "Unofficial implementation of FocusedDropout",
  howpublished = "\url{}",
  note         = "Accessed: 2023-xx-xx"
}
```

***<center>Veni，vidi，vici --Caesar</center>***
