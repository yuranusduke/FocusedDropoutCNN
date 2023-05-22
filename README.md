# Unofficial Implementation of  ***FocusedDropout for Convolutional Neural Network***

![](https://img.shields.io/badge/data%20science-reproducible%20research-blue) ![](https://img.shields.io/badge/deeplearning-pytorch-red) ![](https://img.shields.io/badge/data%20science-XAI-yellowgreen)
## Results Branch

### <font color = 'green'> Project Results for Reproducible Research </font> :bar_chart:

Created by: Kunhong Yu (444447)/Islam Islamov ()/Leyla Ellazova ()

- [Results Contents]()
	- [Results](#results)
		- [Accuracy in Tables](#accuracy-in-tables)
			- [CIFAR10](#cifar10)
			- [CIFAR100](#cifar100)
		- [Stats](#stats)
			- [CIFAR10](#cifar10)
				- [VGG19Net](#vgg19net)
				- [DenseNet](#densenet)
				- [ResNet110](#resnet110)
			- [CIFAR100](#cifar100)
				- [ResNet20](#resnet20)
				- [WRN28](#wrn28)
				- [ResNet56](#resnet56)
		- [CAM](#cam)
			- [CIFAR10](#cifar10)
				- [ResNet56](#resnet56)
				- [ResNet110](#resnet110)
			- [CIFAR100](#cifar100)
				- [ResNet20](#resnet20)
				- [ResNet110](#resnet110)
			- [Animation](#animation)
	- [Contributions](#contributions)


This branch will display all reproduced experiments' results.

For the code part, please refer to [code page](https://github.com/yuranusduke/FocusedDropoutCNN).

## Results

If you refer to use pre-trained **everything**, please download our models [here](https://drive.google.com/file/d/1A4URXtEBpN95B3L6SQQjkOgkox1Afw43/view?usp=share_link), which is about 6GB in `zip` file, unzip it and keep file trees as they are, put them into `checkpoint` folder, in case there is no `checkpoint` folder, feel free to create one.

### Accuracy in Tables
#### CIFAR10
To mimic `Table1` and `Table2` in original paper, we run each experiment either two or three times and report its mean and standard deviation on testing data sets.

| Method              | ResNet20        | ResNet56| ResNet110 | VGGNet19 | DenseNet|WRN28 |
| :-----------------: | :-----------: |:-----------: |:-----------: |:-----------: |:-----------: |:-----------: |
| Baseline       	  |  92.01±0.07%     |      |     |   93.35±0.1%   |      |      |
| Dropout             |  92.08±0.09%    |     |       |   93.24±0.09%  |      |      |
| SpatialDropout      |  91.76±0.15% |      |      |   93.37±0.13%   |     |      |
| DropoutBlock        |  92.01±0.07%  |     |     |   93.35±0.1%   |     |     |
| FocusedDropout      |  <font color = 'red'>**92.94±0.06%**</font> |     |      |    <font color = 'red'>**93.41±0.2%**</font>  |     |      |



## Contributions

In this part, we dynamically update our contribution in results part for our group with three people.

:heavy_check_mark: 2023/05/19: Kunhong Yu adds results on CIFAR10 with VGG19Net for all methods.

:heavy_check_mark: 2023/05/22: Kunhong Yu finishes experiments on CIFAR10 with ResNet20, adds results on CIFAR10 with ResNet20 for all methods.

## To cite our work :black_nib:

Also we would appreciate it if receiving a star :star:.

```
@misc{FDI,
  author       = "KunhongYu/Islam Islamov/Leyla Ellazova",
  title        = "Unofficial implementation of FocusedDropout--results branch",
  howpublished = "\url{}",
  note         = "Accessed: 2023-xx-xx"
}
```

***<center>Veni，vidi，vici --Caesar</center>***