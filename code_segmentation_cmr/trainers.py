import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from os import path
use_cuda = True
if torch.cuda.is_available() and use_cuda:
    Tensor = torch.cuda.FloatTensor
else:
    Tensor = torch.FloatTensor
torch.cuda.manual_seed_all(808)

import torchsummary
import losses

from networks import DilatedCNN2D, DRNSeg, drn_d_22, FCDenseNet57, UNet

class AttributeDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

class Cat(nn.Module):
    def __init__(self, dim):
        super(Cat, self).__init__()
        self.dim = dim

    def forward(self, *input):
        cat = torch.cat(input, self.dim)
        return cat

class Trainer(object):
    def __init__(self,
                 n_classes,
                 learning_rate=0.001,
                 decay_after=1000000,
                 weight_decay=0.,
                 alpha=32,
                 beta=2,
                     model_file=None, *args, **kwargs):

        self.criterion = losses.StochasticCoTeachingSegmentationLoss(max_iters=kwargs['max_iters'],
                                                                     delay=kwargs['delay'],
                                                                     tp_gradual=kwargs['tp_gradual'],
                                                                     alpha=alpha,
                                                                     beta=beta)

        self.val_criterion = torch.nn.CrossEntropyLoss()

        self.model_1.cuda()
        self.model_2.cuda()

        self.optimizer_1 = torch.optim.Adam(self.model_1.parameters(), lr=learning_rate,  amsgrad=True,
                                            weight_decay=weight_decay)

        self.optimizer_2 = torch.optim.Adam(self.model_2.parameters(), lr=learning_rate,  amsgrad=True,
                                            weight_decay=weight_decay)

        self.scheduler_1 = torch.optim.lr_scheduler.StepLR(self.optimizer_1, step_size=decay_after)
        self.scheduler_2 = torch.optim.lr_scheduler.StepLR(self.optimizer_2, step_size=decay_after)

        if model_file:
            self.load(model_file)

    @torch.no_grad()
    def predict(self, x, combine=False):
        self.model_1.eval()
        self.model_2.eval()
        # image = torch.unsqueeze(image, 1).cuda()
        x = x.cuda()
        if not combine:
            prediction_1 = self.model_1(x)
            prediction_2 = self.model_2(x)
            _, res_1 = torch.max(prediction_1, 1)
            _, res_2 = torch.max(prediction_2, 1)
            return res_1.cpu(), res_2.cpu()
        else:
            prediction_1 = F.softmax(self.model_1(x), 1)
            prediction_2 = F.softmax(self.model_2(x), 1)

            _, res = torch.max((prediction_1 + prediction_2) / 2, 1)
            return res.cpu()


    @torch.no_grad()
    def accuracy(self, p, y):
        _, y_hat = torch.max(p, dim=1)
        correct = y_hat.eq(y).sum().type(torch.float32)
        return correct / torch.prod(torch.tensor(y.shape))

    @torch.no_grad()
    def dice(self, p, y):
        _, y_hat = torch.max(p, dim=1)
        y_hat = torch.eq(y_hat, 2).type(torch.int)
        y = torch.eq(y, 2).type(torch.int)
        dice = torch.logical_and(y_hat, y).type(torch.float).sum() * 2 / (y.sum() + y_hat.sum())
        return dice

    def train(self, x, y):
        self.model_1.train()
        self.model_2.train()
        self.optimizer_1.zero_grad()
        self.optimizer_2.zero_grad()
        x = x.cuda()
        y = y.cuda()
        pred_1 = self.model_1(x)
        pred_2 = self.model_2(x)

        loss_1, loss_2 = self.criterion(pred_1, pred_2, y)  # , noise_or_not)


        accuracy_1 = self.accuracy(pred_1, y)
        accuracy_2 = self.accuracy(pred_2, y)
        dice_1 = self.dice(pred_1, y)
        dice_2 = self.dice(pred_2, y)

        loss_1.backward()
        loss_2.backward()
        self.optimizer_1.step()
        self.optimizer_2.step()


        return loss_1.item(), loss_2.item(), accuracy_1.item(), accuracy_2.item(), dice_1.item(), dice_2.item()

    @torch.no_grad()
    def evaluate(self, x, y):
        # print(x.shape, y.shape, type(x), type(y))
        x = x.cuda()
        y = y.cuda()
        self.model_1.eval()
        self.model_2.eval()
        pred_1 = self.model_1(x)
        pred_2 = self.model_2(x)
        loss_1 = self.val_criterion(pred_1, y)
        loss_2 = self.val_criterion(pred_2, y)
        accuracy_1 = self.accuracy(pred_1, y)
        accuracy_2 = self.accuracy(pred_2, y)
        dice_1 = self.dice(pred_1, y)
        dice_2 = self.dice(pred_2, y)

        return loss_1.item(), loss_2.item(), accuracy_1.item(), accuracy_2.item(), dice_1.item(), dice_2.item()

    def step(self):
        self.scheduler_1.step()
        self.scheduler_2.step()
        self.criterion.step()

    def load(self, fname):
        state_dict = torch.load(fname)
        self.model_1.load_state_dict(state_dict['model_1'])
        self.model_2.load_state_dict(state_dict['model_2'])
        self.optimizer_1.load_state_dict(state_dict['optimizer_1'])
        self.optimizer_2.load_state_dict(state_dict['optimizer_2'])
        self.scheduler_1.load_state_dict(state_dict['scheduler_1'])
        self.scheduler_2.load_state_dict(state_dict['scheduler_2'])


    def save(self, output_dir, iter):
        fname = path.join(output_dir, '{:0d}.model'.format(iter))
        torch.save({'model_1': self.model_1.state_dict(),
                    'model_2': self.model_2.state_dict(),
                    'optimizer_1': self.optimizer_1.state_dict(),
                    'optimizer_2': self.optimizer_2.state_dict(),
                    'scheduler_1': self.scheduler_1.state_dict(),
                    'scheduler_2': self.scheduler_1.state_dict(),
                    'iteration': iter}, fname)
    

class DCNN2D(Trainer):
    def __init__(self, *args, **kwargs):

        self.model_1 = DilatedCNN2D(n_input=kwargs.get('n_channels'), n_classes=kwargs.get('n_classes'))
        self.model_2 = DilatedCNN2D(n_input=kwargs.get('n_channels'), n_classes=kwargs.get('n_classes'))
        super().__init__(*args, **kwargs)

class DRN2D(Trainer):
    def __init__(self, *args, **kwargs):
        n_classes = kwargs.get('n_classes')
        n_channels = kwargs.get('n_channels')
        self.model_1 = DRNSeg(drn_d_22(input_channels=n_channels, num_classes=n_classes, out_map=False), n_classes)
        self.model_2 = DRNSeg(drn_d_22(input_channels=n_channels, num_classes=n_classes, out_map=False), n_classes)
        super().__init__(*args, **kwargs)

class UNet2D(Trainer):
    def __init__(self, *args, **kwargs):
        n_classes = kwargs.get('n_classes')
        n_channels = kwargs.get('n_channels')
        self.model_1 = UNet(n_channels, n_classes)
        self.model_2 = UNet(n_channels, n_classes)
        super().__init__(*args, **kwargs)


class Tiramisu2D(Trainer):
    def __init__(self, *args, **kwargs):
        n_classes = kwargs.get('n_classes')
        n_channels = kwargs.get('n_channels')
        model = FCDenseNet57(n_channels, kwargs.get('n_classes'))
        super().__init__(model, *args, **kwargs)


