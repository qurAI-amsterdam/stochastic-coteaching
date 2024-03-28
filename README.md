# Stochastic co-teaching for training neural networks with unknown levels of label noise

Code for StoCoT: [Stochastic co-teaching for training neural networks with unknown levels of label noise](https://www.nature.com/articles/s41598-023-43864-7).

### Introduction
Stochastic co-teaching (StoCoT) is a method for training supervised classification and segmentation models on image data with high rates of labeling errors (label noise).
This method builds on the co-teaching training paradigm, where two networks select training samples for each other, 
by rejecting a fixed number of samples with the highest loss from a training batch. 
Conversely, in StoCoT, to reject training samples the networks employ a rejection threshold on the estimated posterior probability of the 'ground-truth' class (which is potentially erroneous).
This rejection threshold is randomly sampled from a Beta distribution. 
The added stochasticity of StoCoT results in more robust optimization with respect to varying noise rates, compared to regular co-teaching.
This better enables neural networks to be trained in cases where the rate of labeling inaccuracies is unknown.
Additionally, a rejection rate that has converged at the end of training provides an accurate estimate of the true label noise rate (provided that overfitting has successfully been prevented).

### Training
To run an MNIST, CIFAR10, or CIFAR100 experiment, run:

```python code_cifar_mnist/train.py -o  <output_path> --dataset <dataset> --noise_type <noise_type> --noise_rate <noise_rate> --stochastic --stocot_alpha <alpha> --stocot_beta <beta>```

Fill out the following arguments with the desired parameters:
- <output_path>: Output folder where weights and tensorboard log file is stored.
- \<dataset>: Dataset, either "mnist", "cifar10", or "cifar100".
- <noise_type>: type of simulated label noise, either "symmetric" or "pairflip".
- <noise_rate>: the rate of simulated label noise, choose some number between 0.0 and 0.5.
- \<alpha>: beta-distribution parameter, choose integer between 1 and 64 (recommended: 32).
- \<beta>: beta-distribution parameter, choose integer between 1 and 64 (recommended: 2).

To train an ECG classification network, [download](https://physionet.org/content/ptb-xl/1.0.3/) the PTB dataset, 
put it in the data folder (```stochastic_coteaching/data```), and run:

```
cd ./code_classification_ecg
python train.py --output_directory <output_path> --alpha 32 --beta 2 --network wangresnet --epochs 50 --store_model_every 10 --stocot_delay 10 --stocot_gradual 10 --balance --lr_decay_after 150
```
Replace <output_path> with desired output folder (new folder will be created).

To train a cardiac MRI network, download the Sunnybrook dataset from the [releases](https://github.com/GEJansen/stochastic-coteaching/releases) page of this repository, put it in the data folder 
(```stochastic_coteaching/data```), and run:

```
cd ./code_segmentation_cmr
python train.py  --output_directory <output_path> --alpha 32 --beta 2
```
Replace <output_path> with desired output folder (new folder will be created). 
To view options, run ```python ./code_segmentation_cmr/train.py --help``` or open train.py in your IDE/text editor.

To replicate the experiments from the paper, ```cd``` to the code folder of the desired dataset/experiment and run:

```
bash run_experiments.sh
```
