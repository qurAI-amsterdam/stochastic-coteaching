import os
import argparse
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import WeightedRandomSampler, RandomSampler, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm
import data
from data import PTBXL
import trainers
from evaluation2020 import evaluate_12ECG_score as evalecg
import shutil
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='Train a network')
    parser.add_argument('--output_directory', type=Path, help='directory for experiment outputs')
    parser.add_argument('--data_set', type=str, choices=['all', 'training', 'validation'], default ='all')
    parser.add_argument('-l', '--learning_rate', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=505)
    parser.add_argument('--lr_decay_after', type=int, default=250)
    parser.add_argument('--store_model_every', type=int, default=250)
    parser.add_argument('--stocot_delay', type=int, default=50)
    parser.add_argument('--stocot_gradual', type=int, default=50)

    parser.add_argument('--alpha', type=int, default=32)
    parser.add_argument('--beta', type=int, default=2)

    parser.add_argument('--weight_decay', type=float, default=0)#0.0005)
    parser.add_argument('--evaluate_every', type=int, default=1)
    parser.add_argument('--label_map', type=Path, default='label_map.txt')
    parser.add_argument('--loss', type=str, choices=['bce', 'mse', 'msew'], default='bce')
    parser.add_argument('--optimizer', type=str, choices=['adam', 'sgd'], default='adam')
    parser.add_argument('--network', type=str, choices=['wangresnet', 'xresnet', 'inception', 'ours'], default='wangresnet')
    parser.add_argument('--dataroot', type=Path, default='../data/ptb-xl-1.0.3')
    parser.add_argument('--sample_size', type=int, default=250)
    parser.add_argument('--balance', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--continue_training', type=Path, default=None)
    return parser.parse_args()

@torch.no_grad()
def evaluate(model_1, model_2, validation_set, num_classes, device='cuda:0'):
    threshold = 0.5
    beta = 2
    # num_classes = 27
    labels = list()
    probabilities = list()

    model_1.eval()
    model_2.eval()

    for idx in range(len(validation_set)):
        smp = validation_set[idx]
        probs_1 = model_1(torch.from_numpy(smp['data'][None]).to(device)).cpu().squeeze().numpy()
        probs_2 = model_2(torch.from_numpy(smp['data'][None]).to(device)).cpu().squeeze().numpy()
        probs = (probs_1 + probs_2) / 2
        probabilities.append(probs)
        labels.append(smp['y'].astype(bool))

    labels = np.array(labels)
    probabilities = np.array(probabilities)
    y_hat = probabilities >= threshold

    auroc, auprc = evalecg.compute_auc(labels, probabilities)
    accuracy = evalecg.compute_accuracy(labels, y_hat)
    f_measure = evalecg.compute_f_measure(labels, y_hat)
    f_beta, g_beta = evalecg.compute_beta_measures(labels, y_hat, beta=2)

    results = dict(auroc=auroc, auprc=auprc, accuracy=accuracy, f_measure=f_measure, f_beta=f_beta, g_beta=g_beta)


    # print(f'AUROC: {auroc:.3f}')
    # print(f'AURPC: {auprc:.3f}')
    # print(f'Accuracy: {accuracy:.3f}')
    # print(f'F1: {f_measure:.3f}')
    # print(f'F1 beta: {f_beta:.3f}')
    # print(f'Jaccard beta: {g_beta:.3f}')

    return results


import math
from itertools import chain
from collections import defaultdict
class ECGClassAlternatingSampler(torch.utils.data.Sampler):
    def __init__(self, data_source, num_samples, num_classes):
        rs = np.random.RandomState(808)
        self.data_source = data_source
        self._num_samples = num_samples

        class_locations = defaultdict(list)
        for idx, smp in tqdm(enumerate(self.data_source), desc='populating ecg sampler'):
            for jdx in np.where(smp['y'])[0]:
                class_locations[jdx].append(idx)

        occurence_per_class = math.ceil(self._num_samples / num_classes)
        self.rand_idcs = (rs.choice(v, occurence_per_class) for v in class_locations.values())

    @property
    def num_samples(self):
        return self._num_samples

    def __iter__(self):
        n = len(self.data_source)
        return chain.from_iterable(zip(*self.rand_idcs))

    def __len__(self):
        return self.num_samples

def load_data(args, balance=False):
    root_dir = args.dataroot
    sample_size = args.sample_size
    RS = np.random.RandomState
    do_nothing = lambda x: x

    random_switch_chans = transforms.RandomChoice([data.RandomlySwitchChannels(RS(808)),
                                                   do_nothing
                                                   ])

    transform = transforms.Compose([
                                    data.Sample(sample_size, rs=np.random.RandomState(8080)),
                                    data.ToTensor()])



    val_transform = transforms.Compose([data.Sample(sample_size, rs=np.random.RandomState(8080)),
                                        data.ToTensor()])

    num_channels = 12
    num_classes = 5



    training_set = PTBXL('train', transform=transform, debug=args.debug, data_dir=args.dataroot)

    val_transform2 = None
    validation_set_noaug = PTBXL('validation', transform=val_transform2, debug=args.debug, data_dir=args.dataroot)

    if balance:
        # Count positive instances for each label
        label_counts = np.zeros(5)
        for sample in training_set:
            label_counts += np.array(sample['y'])

        total_samples = len(training_set)
        label_weights = total_samples / (len(label_counts) * label_counts)

        sample_weights = []

        for sample in training_set:
            weight = np.sum(label_weights * np.array(sample['y']))
            sample_weights.append(weight)

        sample_weights = np.array(sample_weights)

        # Create a WeightedRandomSampler to use balancing weights
        balanced_sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
        data_loader_training = DataLoader(training_set,
                                          batch_size=args.batch_size,
                                          sampler=balanced_sampler,
                                          drop_last=True,
                                          num_workers=0)
    else:
        data_loader_training = DataLoader(training_set,
                                          batch_size=args.batch_size,
                                          # sampler=sampler_training,
                                          drop_last=True,
                                          num_workers=0)

    return data_loader_training, validation_set_noaug, num_channels, num_classes

def main():
    torch.manual_seed(808)
    np.random.seed(808)

    args = parse_args()
    os.makedirs(args.output_directory, exist_ok=False)
    shutil.copytree('.', args.output_directory / 'code')

    summary_writer = SummaryWriter(log_dir=args.output_directory, comment=str(args))


    weights = None

    data_loader_training, validation_set, num_channels, num_classes = load_data(args, args.balance)

    validation_transform = None
    mode = 'batch_train'

    if args.network == 'wangresnet':
        trainer = trainers.CoTeachWangResNetTrainer(max_iters=args.epochs, num_channels=num_channels,
                                                    weight_decay=args.weight_decay,
                                                    num_classes=num_classes, loss=args.loss, weights=weights,
                                                    lr_decay_after=args.lr_decay_after, optimizer=args.optimizer,
                                                    mode=mode, stocot_delay=args.stocot_delay, stocot_gradual=args.stocot_gradual,
                                                    alpha=args.alpha, beta=args.beta,
                                                    steps_per_epoch=len(data_loader_training), one_cycle_lr_scheduler=True)
    elif args.network == 'xresnet':
        trainer = trainers.CoTeachXResNetTrainer(max_iters=args.epochs, num_channels=num_channels,
                                                 weight_decay=args.weight_decay,
                                                 num_classes=num_classes, loss=args.loss, weights=weights,
                                                 lr_decay_after=args.lr_decay_after, optimizer=args.optimizer,
                                                 mode=mode, stocot_delay=args.stocot_delay, stocot_gradual=args.stocot_gradual,
                                                    alpha=args.alpha, beta=args.beta)
    elif args.network == 'inception':
        trainer = trainers.CoTeachInceptionTrainer(max_iters=args.epochs, num_channels=num_channels,
                                                   weight_decay=args.weight_decay,
                                                   num_classes=num_classes, loss=args.loss, weights=weights,
                                                   lr_decay_after=args.lr_decay_after, optimizer=args.optimizer,
                                                   mode=mode, stocot_delay=args.stocot_delay, stocot_gradual=args.stocot_gradual,
                                                    alpha=args.alpha, beta=args.beta)
    elif args.network == 'ours':
        trainer = trainers.CoTeachTrainer(max_iters=args.epochs, num_channels=num_channels,
                                          weight_decay=args.weight_decay,
                                          num_classes=num_classes, loss=args.loss, weights=weights,
                                          lr_decay_after=args.lr_decay_after, optimizer=args.optimizer,
                                          mode=mode, stocot_delay=args.stocot_delay, stocot_gradual=args.stocot_gradual,
                                                    alpha=args.alpha, beta=args.beta)

    if args.continue_training:
        fname = args.continue_training
        print(f'Loading state dicts from {fname}')
        it = int(fname.stem.split('_')[-1])
        state_dict = torch.load(fname)
        trainer.model_1.load_state_dict(state_dict['model1'])
        trainer.model_2.load_state_dict(state_dict['model2'])

        trainer.optimizer_1.load_state_dict(state_dict['optimizer1'])
        trainer.optimizer_2.load_state_dict(state_dict['optimizer2'])

        trainer.scheduler_1.load_state_dict(state_dict['scheduler1'])
        trainer.scheduler_2.load_state_dict(state_dict['scheduler2'])


    try:

#################
        for epoch in tqdm(range(args.epochs)):
            if epoch != 0 and not epoch % args.store_model_every:

                fname = '{}/state_dict_{}.pt'.format(args.output_directory, epoch)
                torch.save({'model1': trainer.model_1.state_dict(),
                            'model2': trainer.model_2.state_dict(),
                            'optimizer1': trainer.optimizer_1.state_dict(),
                            'optimizer2': trainer.optimizer_2.state_dict(),
                            'scheduler1': trainer.scheduler_1.state_dict(),
                            'scheduler2': trainer.scheduler_2.state_dict()},
                           fname)



            tr_loss_1 = list()
            tr_loss_2 = list()
            tr_reject_1 = list()
            tr_reject_2 = list()
            for training_batch in data_loader_training:
                trainer.train(training_batch['data'].cuda(), training_batch['y'].cuda())
                tr_loss_1.append(trainer.current_training_loss[0])
                tr_loss_2.append(trainer.current_training_loss[1])
                tr_reject_1.append(trainer.criterion.fraction_reject_1)
                tr_reject_2.append(trainer.criterion.fraction_reject_2)

            summary_writer.add_scalar('Loss/Training_1', np.mean(tr_loss_1), epoch)
            summary_writer.add_scalar('Loss/Training_2', np.mean(tr_loss_2), epoch)
            summary_writer.add_scalar('Rejected/Training_1', np.mean(tr_reject_1), epoch)
            summary_writer.add_scalar('Rejected/Training_2', np.mean(tr_reject_2), epoch)

            evaluation_results = evaluate(trainer.model_1, trainer.model_2, validation_set, num_classes)

            summary_writer.add_scalar('Evaluation/AUROC', evaluation_results['auroc'], epoch)
            summary_writer.add_scalar('Evaluation/AURPC', evaluation_results['auprc'], epoch)
            summary_writer.add_scalar('Evaluation/Accuracy', evaluation_results['accuracy'], epoch)
            summary_writer.add_scalar('Evaluation/F_measure', evaluation_results['f_measure'], epoch)
            summary_writer.add_scalar('Evaluation/F_beta', evaluation_results['f_beta'], epoch)
            summary_writer.add_scalar('Evaluation/G_beta', evaluation_results['g_beta'], epoch)

            trainer.step()


    except KeyboardInterrupt:
            pass

    finally:
        fname = '{}/state_dict_{}.pt'.format(args.output_directory, epoch)
        torch.save({'model1': trainer.model_1.state_dict(),
                    'model2': trainer.model_2.state_dict(),
                    'optimizer1': trainer.optimizer_1.state_dict(),
                    'optimizer2': trainer.optimizer_2.state_dict(),
                    'scheduler1': trainer.scheduler_1.state_dict(),
                    'scheduler2': trainer.scheduler_2.state_dict()},
                   fname)


if __name__ == '__main__':
    main()