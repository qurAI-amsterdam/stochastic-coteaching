#!/bin/bash

echo "Bash version ${BASH_VERSION}..."

for a in 64 32 16 8 4 2 1
do
  for b in 64 32 16 8 4 2 1
  do
    CUDA_VISIBLE_DEVICES=0 python train_epochs.py \
    ../output/stocot_wang_a${a}b${b} \
    --network wangresnet \
    --alpha ${a} \
    --beta ${b} \
    --epochs 50 \
    --store_model_every 10 \
    --stocot_delay 10 \
    --stocot_gradual 10 \
    --balance \
    --lr_decay_after 150
  done
done