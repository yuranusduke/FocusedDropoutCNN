#!/bin/bash

# Run training and testing

# Usage: bash run.sh cifar10 resnet20 no 0.

# custom config
DATA=$1 # data name
BACKBONE=$2 # backbone
DROPMETHOD=$3 # drop method
DROPRATE=$4 # drop rate

PARTRATE=0.1 # part rate
WEIGHTDECAY=5e-4 # weight decay
EVALONLY=0 # eval only?

for SEED in 1 2 3 # three runs different seeds
do
  DIR=checkpoints/${DATA}/${BACKBONE}/drop_${DROPMETHOD}/rate_${DROPRATE}/seed${SEED}
  if [ -d "$DIR" ]; then
      if [ "$EVALONLY" == 0 ]; then
          echo "Oops! The results exist at ${DIR} (so skip this job)"
      else
        python main.py \
          --dataname ${DATA} \
          --backbone ${BACKBONE} \
          --drop_method ${DROPMETHOD} \
          --drop_rate ${DROPRATE} \
          --part_rate ${PARTRATE} \
          --output_path ${DIR} \
          --weight_decay ${WEIGHTDECAY} \
          --eval_only ${EVALONLY} \
          --seed ${SEED}
      fi
  else
    python main.py \
          --dataname ${DATA} \
          --backbone ${BACKBONE} \
          --drop_method ${DROPMETHOD} \
          --drop_rate ${DROPRATE} \
          --part_rate ${PARTRATE} \
          --output_path ${DIR} \
          --weight_decay ${WEIGHTDECAY} \
          --eval_only ${EVALONLY} \
          --seed ${SEED}
  fi
done