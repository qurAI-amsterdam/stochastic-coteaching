import data
import argparse
import numpy as np
import os
from os import path
from tqdm import tqdm
import data
import yaml
import torch
import trainers
import SimpleITK as sitk


if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    device = 'cuda'
else:
    torch.set_default_tensor_type(torch.FloatTensor)
    device = 'cpu'

def loadExperimentSettings(fname):
    with open(fname, 'r') as fp:
        args = argparse.Namespace(**yaml.load(fp))
    return args

def save_nifty(image, spacing, fname):
    img = sitk.GetImageFromArray(image)
    img.SetSpacing(spacing[::-1])
    sitk.WriteImage(img, fname)

def parse_args():
    parser = argparse.ArgumentParser(description='Validate a segmentation network')
    parser.add_argument('experiment_directory', type=str)
    parser.add_argument('output_directory', type=str, help='directory for experiment outputs')
    parser.add_argument('-f', '--fold', type=int, default=0)
    return parser.parse_args()

def main():
    # first we obtain the user arguments, set random seeds, make directories, and store the experiment settings.
    args = parse_args()
    os.makedirs(args.output_directory, exist_ok=True)
    experiment_settings = loadExperimentSettings(path.join(args.experiment_directory, 'settings.yaml'))

    model_file = path.join(args.experiment_directory, '100000.model')
    # we create a trainer
    if experiment_settings.heart_as_one_structure:
        n_classes = 2
    else:
        n_classes = 4

    if experiment_settings.three_slice_input:
        n_channels_input = 3
    else:
        n_channels_input = 1

    if experiment_settings.network == 'dcnn':
        trainer = trainers.DCNN2D(learning_rate=experiment_settings.learning_rate,
                                  model_file=model_file,
                                  decay_after=experiment_settings.lr_decay_after,
                                  weight_decay=experiment_settings.weight_decay,
                                  n_classes=n_classes,
                                  n_channels=n_channels_input,
                                  loss=experiment_settings.loss)
        pad = 65
    elif experiment_settings.network == 'drn':
        trainer = trainers.DRN2D(learning_rate=experiment_settings.learning_rate,
                                 model_file=model_file,
                                  decay_after=experiment_settings.lr_decay_after,
                                  weight_decay=experiment_settings.weight_decay,
                                  n_classes=n_classes,
                                  n_channels=n_channels_input,
                                  loss=experiment_settings.loss)
        pad = 0
    elif experiment_settings.network == 'unet':
        trainer = trainers.UNet2D(learning_rate=experiment_settings.learning_rate,
                                  model_file=model_file,
                                  decay_after=experiment_settings.lr_decay_after,
                                  weight_decay=experiment_settings.weight_decay,
                                  n_classes=n_classes,
                                  n_channels=n_channels_input,
                                  loss=experiment_settings.loss)
        pad = 0
    elif experiment_settings.network == 'tiramisu':
        trainer = trainers.Tiramisu2D(learning_rate=experiment_settings.learning_rate,
                                      model_file=model_file,
                                  decay_after=experiment_settings.lr_decay_after,
                                  weight_decay=experiment_settings.weight_decay,
                                  n_classes=n_classes,
                                  n_channels=n_channels_input,
                                  loss=experiment_settings.loss)
        pad = 0





    for sample in data.acdc_validation_fold(experiment_settings.fold):
        image, spacing, reference, id = sample

        if pad > 0:
            image = np.pad(image, ((0,0), (0,0), (pad, pad), (pad, pad)), mode='edge')

        image = image[:, None]
        if experiment_settings.three_slice_input:
            image = np.pad(image, ((1,1), (0, 0), (0, 0), (0, 0)), mode='edge')
            image = np.concatenate((image[:-2], image[1:-1], image[2:]), axis=1)
            print(image.shape)

        image = torch.from_numpy(image)
        segmentation = trainer.predict(image).detach().numpy()
        print(segmentation.shape, segmentation.dtype, segmentation.mean(), segmentation.std())
        fname = path.join(args.output_directory, '{}.nii.gz'.format(id))
        save_nifty(segmentation.astype(np.uint8), spacing, fname)

        print(id)



if __name__ == '__main__':
    main()
