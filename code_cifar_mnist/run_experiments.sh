#!/bin/bash

echo "Bash version ${BASH_VERSION}..."

for dataset in "mnist" "cifar10" "cifar100"
do
  for noise_type in "symmetric" "symmetric" "pairflip"
  do
    for noise_rate in 20 50 45
    do
      for alpha in 64 32 16 8 4 2 1
      do
        for beta in 64 32 16 8 4 2 1
        do
        CUDA_VISIBLE_DEVICES=5 python train.py \
            -o ../output/${dataset}/${noise_type}_${noise_rate}/stocot/alpha${alpha}_beta${beta} \
            --dataset ${dataset} \
            --noise_type ${noise_type} \
            --noise_rate "$(echo "scale=2; $noise_rate/100" | bc)" \
            --stochastic \
            --stocot_alpha ${alpha} \
            --stocot_beta ${beta} \
            --delay 10 \
            --epochs 200 \
            --num_workers 4
        done
      done
    done
  done
done
