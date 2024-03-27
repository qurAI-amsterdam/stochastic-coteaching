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

from torch.utils.tensorboard import SummaryWriter
import torchvision

import shutil

torch.manual_seed(808)
torch.cuda.manual_seed_all(808)
np.random.seed(808)

if torch.cuda.is_available():
    device = 'cuda'
else:
    torch.set_default_tensor_type(torch.FloatTensor)
    device = 'cpu'


from torch.utils.data.sampler import RandomSampler
from torchvision import transforms, utils

def loadExperimentSettings(fname):
    with open(fname, 'r') as fp:
        args = argparse.Namespace(**yaml.load(fp))
    return args

def saveExperimentSettings(args, fname):
    with open(fname, 'w') as fp:
        yaml.dump(vars(args), fp)

def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentation network')
    parser.add_argument('-f', '--fold', type=int, default=0)
    parser.add_argument('-l', '--learning_rate', type=float, default=0.001)
    parser.add_argument('-w', '--weight_decay', type=float, default=0.0)
    parser.add_argument('--output_directory', type=str, help='directory for experiment outputs')
    parser.add_argument('--loss', type=str, choices=['ce', 'dice'], default='ce')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_iters', type=int, default=50000)
    parser.add_argument('--epochs', type=int, default=1010)
    parser.add_argument('--lr_decay_after', type=int, default=500)
    parser.add_argument('--patch_size', type=int, nargs=2, default=(128, 128))
    parser.add_argument('--number_of_workers', type=int, default=2)
    parser.add_argument('--delay', type=int, default=100)
    parser.add_argument('--num_gradual', type=int, default=100)
    parser.add_argument('--vanilla', action='store_true')
    parser.add_argument('--store_model_every', type=int, default=100)
    parser.add_argument('--alpha', type=int, default=32)
    parser.add_argument('--beta', type=int, default=2)
    parser.add_argument('--store_curves_every', type=int, default=500)
    parser.add_argument('--update_visualizer_every', type=int, default=25)
    parser.add_argument('--num_of_random_dilations', type=int, default=0)
    parser.add_argument('--num_of_random_switch_off', type=int, default=0)
    parser.add_argument('--label_noise', type=float, default=0.2)
    # parser.add_argument('--dilation', type=int, default=4)
    parser.add_argument('--use_label', type=int, default=3)
    parser.add_argument('--exist_ok', action='store_true')

    parser.add_argument('--network', type=str, choices=['dcnn', 'drn', 'unet', 'tiramisu'], default='unet')
    return parser.parse_args()

def main():
    # first we obtain the user arguments, set random seeds, make directories, and store the experiment settings.
    args = parse_args()
    
    rs = np.random.RandomState(808)
    os.makedirs(args.output_directory, exist_ok=args.exist_ok)
    # shutil.copytree('.', args.output_directory + '/code')
    saveExperimentSettings(args, path.join(args.output_directory, 'settings.yaml'))

    print(args)

    n_classes = 3
    n_channels_input = 1

    if args.network == 'dcnn':
        trainer = trainers.DCNN2D(learning_rate=args.learning_rate,
                                  decay_after=args.lr_decay_after,
                                  weight_decay=args.weight_decay,
                                  n_classes=n_classes,
                                  n_channels=n_channels_input,
                                  max_iters=args.max_iters,
                                  tp_gradual=args.num_gradual,
                                  delay=args.delay,
                                  alpha=args.alpha,
                                  beta=args.beta,
                                  loss=args.loss)
        pad = 65
    elif args.network == 'drn':
        trainer = trainers.DRN2D(learning_rate=args.learning_rate,
                                  decay_after=args.lr_decay_after,
                                  weight_decay=args.weight_decay,
                                  n_classes=n_classes,
                                  n_channels=n_channels_input,
                                  max_iters=args.max_iters,
                                  tp_gradual=args.num_gradual,
                                  delay=args.delay,
                                  alpha=args.alpha,
                                  beta=args.beta,
                                  loss=args.loss)
        pad = 0
    elif args.network == 'unet':
        trainer = trainers.UNet2D(learning_rate=args.learning_rate,
                                  decay_after=args.lr_decay_after,
                                  weight_decay=args.weight_decay,
                                  n_classes=n_classes,
                                  n_channels=n_channels_input,
                                  max_iters=args.max_iters,
                                  tp_gradual=args.num_gradual,
                                  delay=args.delay,
                                  alpha=args.alpha,
                                  beta=args.beta,
                                  loss=args.loss)
        pad = 0
    elif args.network == 'tiramisu':
        trainer = trainers.Tiramisu2D(learning_rate=args.learning_rate,
                                  decay_after=args.lr_decay_after,
                                  weight_decay=args.weight_decay,
                                  n_classes=n_classes,
                                  n_channels=n_channels_input,
                                  alpha=args.alpha,
                                  beta=args.beta,
                                  loss=args.loss)
        pad = 0

    # we initialize datasets with augomentations.
    training_augmentations = [
                     data.ConvertLabel(src=3, dst=2, randdst=1),
                     data.RandomPerspective(np.random.RandomState(8210)),
                     data.RandomCrop(args.patch_size, input_padding=pad, rs=rs),
                     data.RandomMirroring(-2, rs=rs),
                     data.RandomMirroring(-1, rs=rs),
                     data.RandomRotation((-2, -1), rs=rs),
                     data.RandomIntensity(rs=np.random.RandomState(89219)),
                     data.ToTensor()]

    validation_augmentations = [
                        data.ConvertLabel(src=3, dst=2, randdst=2),
                        data.ToTensor()]

    to_tensor = data.ToTensor()
    training_set = data.SunnybrookDataset('training',
                                fold=args.fold,
                                preprocessing=data.rescale_intensities,
                                label=args.use_label,
                                transform=transforms.Compose(training_augmentations),
                                num_of_random_dilations=args.num_of_random_dilations,
                                label_noise=args.label_noise,
                                num_of_random_switch_off=args.num_of_random_switch_off)

    # validation_set = data.ACDCDataset('validation',
    validation_set = data.SunnybrookDataset('validation',
                                      fold=args.fold,
                                      preprocessing=data.rescale_intensities,
                                      label=args.use_label,
                                      label_noise=0,
                                      transform=transforms.Compose(validation_augmentations)
                                            )

    print('training_set', len(training_set))
    print('validation_set', len(validation_set))

    # now we create dataloaders

    tra_sampler = RandomSampler(training_set, replacement=True, num_samples=args.batch_size * args.max_iters)
    val_sampler = RandomSampler(validation_set, replacement=True, num_samples=args.batch_size * args.max_iters)

    data_loader_training = torch.utils.data.DataLoader(training_set,
                                                       batch_size=args.batch_size,
                                                       # sampler=tra_sampler,
                                                       shuffle=True,
                                                       # drop_last=True,
                                                       num_workers=args.number_of_workers)

    data_loader_validation = torch.utils.data.DataLoader(validation_set,
                                                         batch_size=args.batch_size,
                                                         # sampler=val_sampler,
                                                         num_workers=args.number_of_workers)

    # and finally we initialize something for visualization in visdom
    summary_writer = SummaryWriter(log_dir=args.output_directory, comment=str(args))

    try:
        for epoch in tqdm(range(args.epochs)):
            if epoch != 0 and not epoch % args.store_model_every:
                trainer.save(args.output_directory, epoch)

            tr_loss_1 = list()
            tr_loss_2 = list()
            tr_acc_1 = list()
            tr_acc_2 = list()
            tr_dice_1 = list()
            tr_dice_2 = list()
            tr_reject_1 = list()
            tr_reject_2 = list()
            for training_batch in data_loader_training:
                loss_1, loss_2, acc_1, acc_2, dsc_1, dsc_2 = trainer.train(training_batch['image'].to(device),
                                                                           training_batch['reference'].to(device))
                tr_loss_1.append(loss_1)
                tr_loss_2.append(loss_2)
                tr_acc_1.append(acc_1)
                tr_acc_2.append(acc_2)
                tr_dice_1.append(dsc_1)
                tr_dice_2.append(dsc_2)
                tr_reject_1.append(trainer.criterion.fraction_reject_1)
                tr_reject_2.append(trainer.criterion.fraction_reject_2)


            val_loss_1 = list()
            val_loss_2 = list()
            val_acc_1 = list()
            val_acc_2 = list()
            val_dice_1 = list()
            val_dice_2 = list()

            for validation_sample in validation_set:
                loss_1, loss_2, acc_1, acc_2, dsc_1, dsc_2 = trainer.evaluate(validation_sample['image'][None], validation_sample['reference'][None])

                val_loss_1.append(loss_1)
                val_loss_2.append(loss_2)
                val_acc_1.append(acc_1)
                val_acc_2.append(acc_2)
                val_dice_1.append(dsc_1)
                val_dice_2.append(dsc_2)

            trainer.step()
            summary_writer.add_scalar('Loss/Training_1', np.mean(tr_loss_1), epoch)
            summary_writer.add_scalar('Loss/Training_2', np.mean(tr_loss_2), epoch)
            summary_writer.add_scalar('Loss/Test_1', np.mean(val_loss_1), epoch)
            summary_writer.add_scalar('Loss/Test_2', np.mean(val_loss_2), epoch)

            summary_writer.add_scalar('Accuracy/Training_1', np.mean(tr_acc_1), epoch)
            summary_writer.add_scalar('Accuracy/Training_2', np.mean(tr_acc_2), epoch)
            summary_writer.add_scalar('Accuracy/Test_1', np.mean(val_acc_1), epoch)
            summary_writer.add_scalar('Accuracy/Test_2', np.mean(val_acc_2), epoch)

            summary_writer.add_scalar('Dice/Training_1', np.mean(tr_dice_1), epoch)
            summary_writer.add_scalar('Dice/Training_2', np.mean(tr_dice_2), epoch)
            summary_writer.add_scalar('Dice/Test_1', np.mean(val_dice_1), epoch)
            summary_writer.add_scalar('Dice/Test_2', np.mean(val_dice_2), epoch)

            summary_writer.add_scalar('Rejected/Training_1', np.mean(tr_reject_1), epoch)
            summary_writer.add_scalar('Rejected/Training_2', np.mean(tr_reject_2), epoch)

            if not epoch % args.update_visualizer_every:

                image = training_batch['image']

                image_cuda = image.cuda()
                reference_cuda = training_batch['reference'].cuda()
                p1 = trainer.criterion.get_probas(trainer.model_1(image_cuda), reference_cuda)
                p2 = trainer.criterion.get_probas(trainer.model_2(image_cuda), reference_cuda)
                mask_1 = trainer.criterion.mask_probas(p1)
                mask_2 = trainer.criterion.mask_probas(p2)
                summary_writer.add_image('training_masks/mask_1',
                                         torchvision.utils.make_grid(
                                             mask_1.cpu().type(torch.FloatTensor).clamp_(0, 1),
                                             pad_value=0), epoch)
                summary_writer.add_image('training_masks/mask_2',
                                         torchvision.utils.make_grid(
                                             mask_2.cpu().type(torch.FloatTensor).clamp_(0, 1),
                                                     pad_value=0), epoch)
                #
                #
                reference = training_batch['reference']
                prediction_1, prediction_2 = trainer.predict(image)
                summary_writer.add_image('training/image', torchvision.utils.make_grid(image ** .5, pad_value=0), epoch)
                summary_writer.add_image('training/prediction_1',
                                          torchvision.utils.make_grid(prediction_1[:, None].type(torch.FloatTensor) / 3,
                                                                     pad_value=1), epoch)
                summary_writer.add_image('training/prediction_2',
                                         torchvision.utils.make_grid(prediction_2[:, None].type(torch.FloatTensor) / 3,
                                                                     pad_value=1), epoch)
                summary_writer.add_image('training/reference',
                                         torchvision.utils.make_grid(reference[:, None].type(torch.FloatTensor) / 3,
                                                                     pad_value=1), epoch)

    except KeyboardInterrupt:
        print('interrupted')

    finally:
        trainer.save(args.output_directory, epoch)

if __name__ == '__main__':
    main()
