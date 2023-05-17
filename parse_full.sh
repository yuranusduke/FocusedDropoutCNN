#!/bin/bash

# Parse all experiments

# EXP1: cifar10/cifar100 on resnet20/resnet56/resnet110/wrn28/vgg19/densenet100 with no dropout
for dataset in cifar10 cifar100
do
	for model in resnet20 resnet56 vgg19 resnet110 densenet100 wrn28
	do
		bash parse.sh ${dataset} ${model} no 0.
	done
done

# EXP2: cifar10/cifar100 on resnet20/resnet56/resnet110/wrn28/vgg19/densenet100 with dropout rate 0.3
for dataset in cifar10 cifar100
do
	for model in resnet20 resnet56 vgg19 resnet110 densenet100 wrn28
	do
		bash parse.sh ${dataset} ${model} d 0.3
	done
done

# EXP3: cifar10/cifar100 on resnet20/resnet56/resnet110/wrn28/vgg19/densenet100 with SpatialDropout 0.3
for dataset in cifar10 cifar100
do
	for model in resnet20 resnet56 vgg19 resnet110 densenet100 wrn28
	do
		bash parse.sh ${dataset} ${model} sd 0.3
	done
done

# EXP4: cifar10/cifar100 on resnet20/resnet56/resnet110/wrn28/vgg19/densenet100 with FocusedDropout
for dataset in cifar10 cifar100
do
	for model in resnet20 resnet56 vgg19 resnet110 densenet100 wrn28
	do
		bash parse.sh ${dataset} ${model} fd 0.
	done
done

# EXP5: cifar10/cifar100 on resnet20/resnet56/resnet110/wrn28/vgg19/densenet100 with DropBlock 0.3
for dataset in cifar10 cifar100
do
	for model in resnet20 resnet56 vgg19 resnet110 densenet100 wrn28
	do
		bash parse.sh ${dataset} ${model} db 0.3
	done
done

# generate animation
#for dataset in cifar10 cifar100
#do
#	for model in resnet20 resnet56 resnet110
#	do
#    for method in no d sd db fd
#    do
#      echo "Generating for ${dataset} with ${model} and using method ${method}..."
#      python generate_anim_cam.py --dataname ${dataset} --backbone ${model} --drop_method ${method}
#    done
#	done
#done

# zip -r full_fd.zip ../fd -x *.pth