import os
import argparse
import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler, RandomSampler, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm
import trainers
from pathlib import Path
import loss

from data.cifar import CIFAR10, CIFAR100
from data.mnist import MNIST


def parse_args():
    parser = argparse.ArgumentParser(description='Train a network')
    parser.add_argument('-o', '--output_directory', type=Path, default=Path('../output'))
    parser.add_argument('--data_src', type=Path, default=Path('../data'))

    parser.add_argument('--lr', type=float, default=0.001)

    parser.add_argument('--noise_rate', type=float, help='corruption rate, should be less than 1', default=0.45)
    parser.add_argument('--noise_type', type=str, choices=['clean', 'pairflip', 'symmetric'], default='clean')
    parser.add_argument('--delay', type=int, default=10)
    parser.add_argument('--num_gradual', type=int, default=10,
                        help='how many epochs for linear drop rate, can be 5, 10, 15. This parameter is equal to Tk for R(T) in Co-teaching paper.')
    parser.add_argument('--exponent', type=float, default=1,
                        help='exponent of the forget rate, can be 0.5, 1, 2. This parameter is equal to c in Tc for R(T) in Co-teaching paper.')
    parser.add_argument('--stochastic', action='store_true')
    parser.add_argument('--cot_forget_rate', type=float, help='forget rate', default=0.0)
    parser.add_argument('--weight_decay', type=float, help='', default=0)
    parser.add_argument('--batch_size', type=int, help='', default=128)
    parser.add_argument('--stocot_alpha', type=float, help='', default=32)
    parser.add_argument('--stocot_beta', type=float, help='', default=4)

    parser.add_argument('--dataset', type=str, choices=['mnist', 'cifar10', 'cifar100'], default='mnist')
    parser.add_argument('--epochs', type=int, default=203)
    parser.add_argument('--seed', type=int, default=808)
    parser.add_argument('--num_workers', type=int, default=2, help='how many subprocesses to use for data loading')
    parser.add_argument('--exist_ok', action='store_true')



    return parser.parse_args()

def load_data(args):
    if args.dataset == 'mnist':
        num_channels = 1
        num_classes = 10
        DS = MNIST
        transform_train = transforms.ToTensor()
        transform_test = transforms.ToTensor()
    if args.dataset == 'cifar10':
        num_channels = 3
        num_classes = 10

        DS = CIFAR10

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])
    if args.dataset == 'cifar100':
        num_channels = 3
        num_classes = 100
        args.top_bn = False
        DS = CIFAR100
        transform_train = transforms.Compose([
            transforms.ToTensor(),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])

    if args.noise_type == 'none':
        args.noise_type = None

    train_set = DS(root=args.data_src,
                       download=True,
                       train=True,
                       transform=transform_train,
                       noise_type=args.noise_type,
                       noise_rate=args.noise_rate)
    test_set = DS(root=args.data_src,
                      download=True,
                      train=False,
                      transform=transform_test)

    return train_set, test_set, num_channels, num_classes


def main():
    args = parse_args()
    os.makedirs(args.output_directory, exist_ok=args.exist_ok)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    summary_writer = SummaryWriter(log_dir=args.output_directory, comment=str(args))

    batch_size = args.batch_size

    train_set, test_set, num_channels, num_classes = load_data(args)

    train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                               batch_size=batch_size,
                                               num_workers=args.num_workers,
                                               drop_last=True,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                              batch_size=batch_size,
                                              num_workers=args.num_workers,
                                              drop_last=True,
                                              shuffle=False)

    if args.stochastic:
        criterion = loss.StochasticCoTeachingLoss(args.stocot_alpha, args.stocot_beta, args.epochs, args.num_gradual, args.delay)
    else:
        criterion = loss.CoTeachingLoss(args.cot_forget_rate, args.epochs, args.num_gradual)
    validation_criterion = torch.nn.CrossEntropyLoss()

    if args.dataset == 'mnist':
        trainer = trainers.TrainerSmallCNN(criterion=criterion, validation_criterion=validation_criterion,
                                                num_channels=num_channels, num_classes=num_classes,
                                                learning_rate=args.lr)
    else:
        trainer = trainers.TrainerCifarCNN(criterion=criterion, validation_criterion=validation_criterion,
                                           num_channels=num_channels, num_classes=num_classes,
                                           learning_rate=args.lr, weight_decay=args.weight_decay)

    for epoch in tqdm(range(args.epochs), total=args.epochs):
        batches_rejected = 0
        losses1 = list()
        losses2 = list()
        accuracies1 = list()
        accuracies2 = list()
        probas1 = list()
        probas2 = list()
        fractions_rejected1 = list()
        fractions_rejected2 = list()
        for it, train_batch in enumerate(train_loader):
            x, y, indices = train_batch
            try:
                loss1, loss2, accu1, accu2 = trainer.train(x, y, indices)
            except RuntimeError:
                batches_rejected += 1
                continue
            losses1.append(loss1)
            losses2.append(loss2)
            accuracies1.append(accu1)
            accuracies2.append(accu2)
            probas1.append(trainer.criterion.current_probas()[0].cpu().numpy())
            probas2.append(trainer.criterion.current_probas()[1].cpu().numpy())
            fractions_rejected1.append(trainer.criterion.current_fraction_rejected()[0].cpu().numpy())
            fractions_rejected2.append(trainer.criterion.current_fraction_rejected()[1].cpu().numpy())
            # break
        if batches_rejected > 0:
            print(batches_rejected, 'batches rejected in epoch', epoch)
        summary_writer.add_scalar('Loss/Training_1', np.mean(losses1), epoch)
        summary_writer.add_scalar('Loss/Training_2', np.mean(losses2), epoch)
        summary_writer.add_scalar('Accuracy/Training_1', np.mean(accuracies1) * 100, epoch)
        summary_writer.add_scalar('Accuracy/Training_2', np.mean(accuracies2) * 100, epoch)
        summary_writer.add_histogram('Probabilities/Training_1', np.asarray(probas1), epoch)
        summary_writer.add_histogram('Probabilities/Training_2', np.asarray(probas2), epoch)
        summary_writer.add_scalar('Rejected/Training_1', np.mean(fractions_rejected1), epoch)
        summary_writer.add_scalar('Rejected/Training_2', np.mean(fractions_rejected2), epoch)

        losses1 = list()
        losses2 = list()
        accuracies1 = list()
        accuracies2 = list()
        for test_batch  in test_loader:
            x, y, indices = test_batch
            loss1, loss2, accu1, accu2 = trainer.evaluate(x, y)
            losses1.append(loss1)
            losses2.append(loss2)
            accuracies1.append(accu1)
            accuracies2.append(accu2)
            # break
        summary_writer.add_scalar('Loss/Test_1', np.mean(losses1), epoch)
        summary_writer.add_scalar('Loss/Test_2', np.mean(losses2), epoch)
        summary_writer.add_scalar('Accuracy/Test_1', np.mean(accuracies1) * 100, epoch)
        summary_writer.add_scalar('Accuracy/Test_2', np.mean(accuracies2) * 100, epoch)

        trainer.criterion.step()


if __name__ == '__main__':
    main()