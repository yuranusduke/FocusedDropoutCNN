#!/bin/bash

# Run parsing results

# Usage: bash parse.sh cifar10 resnet20 no 0.5

# custom config
DATA=$1 # data name
BACKBONE=$2 # backbone
DROPMETHOD=$3 # drop method
DROPRATE=$4 # drop rate

DIR=checkpoints/${DATA}/${BACKBONE}/drop_${DROPMETHOD}/rate_${DROPRATE}
if [ ! -d "$DIR" ]; then
    echo "Oops! The results do not exist at ${DIR} (please run this job first at run.sh)"
else
    python ./engine/parse_results.py \
        --output_path ${DIR}
fi
