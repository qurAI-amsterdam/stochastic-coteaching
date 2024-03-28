#!/bin/bash

echo "Bash version ${BASH_VERSION}..."

for alpha in 64 32 16 8 4 2 1
do
  for beta in 64 32 16 8 4 2 1
  do
      CUDA_VISIBLE_DEVICES=0 python train.py \
          --output_directory ../output/sunnybrook/noise_20/stocotbeta/alpha${alpha}_beta${beta} \
          --alpha ${alpha} \
          --beta ${beta}
  done
done