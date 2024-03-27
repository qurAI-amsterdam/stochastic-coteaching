import torch
from torch import nn
import torch.nn.functional as F
from networks import customnet, prednet, vandeleurnet, customnet_dilation, dcnn
from networks import resnet1d
class WeightedMSELoss:
    def __init__(self, weights):
        self.weights = torch.from_numpy(weights).cuda()
    def __call__(self, input, target):
        out = (input - target) ** 2
        out = out * self.weights.expand_as(out)
        loss = out.mean()
        return loss


import torch
import torch.nn as nn
import numpy as np


class CoTeachingLoss:
    def __init__(self, forget_rate, max_iters, tp_gradual, exponent=1, Loss=nn.CrossEntropyLoss):
        rate_schedule = np.ones(max_iters) * forget_rate
        rate_schedule[:tp_gradual] = np.linspace(0, forget_rate ** exponent, tp_gradual)
        self.tp_gradual = tp_gradual
        self.rate_schedule = rate_schedule
        self._it = 0
        self.Loss = Loss(reduction='none')

    def step(self):
        self._it += 1

    def get_probas(self, logits, y):
        probabilities = F.softmax(logits, 1)
        samples, classes = logits.shape
        raveled_indices = y + torch.arange(0, (samples) * classes, classes).cuda()
        return probabilities.take(raveled_indices)

    def mask_losses(self, losses):
        forget_rate = self.rate_schedule[self._it]

        if forget_rate == 0:
            return torch.ones_like(losses)
        elif forget_rate >= 1:
            raise RuntimeError('Forget-rate is equal to or larger than 1')
        mask = torch.zeros_like(losses)
        indices = torch.argsort(losses.view(-1))
        num_to_use = int(len(indices) * (1 - forget_rate))
        take_first_n_indices = indices[:num_to_use]

        mask.view(-1)[take_first_n_indices] = 1
        return mask

    def current_fraction_rejected(self):
        return self.fraction_reject_1, self.fraction_reject_2

    def current_probas(self):
        return self.probas1, self.probas2

    def __call__(self, logits_1, logits_2, y, *args, **kwargs):
        with torch.no_grad():
            loss_1 = self.Loss(logits_1, y)
            loss_2 = self.Loss(logits_2, y)

            self.current_losses = (loss_1, loss_2) # temp\

            mask_1 = self.mask_losses(loss_1)
            mask_2 = self.mask_losses(loss_2)

            self.probas1 = self.get_probas(logits_1, y)
            self.probas2 = self.get_probas(logits_2, y)

            self.fraction_reject_1 = 1. - (mask_1.sum() / len(mask_1))
            self.fraction_reject_2 = 1. - (mask_2.sum() / len(mask_2))

        loss_1_update = torch.sum(self.Loss(logits_1, y) * mask_2) / mask_2.sum()
        loss_2_update = torch.sum(self.Loss(logits_2, y) * mask_1) / mask_1.sum()

        return loss_1_update, loss_2_update


class Trainer:
    def __init__(self, learning_rate=0.001, weight_decay=0, max_iters=None, optimizer='adam', loss='mse', weights=None, lr_decay_after=34000, mode='batch_train'):

        if loss == 'mse':
            output_fn = nn.Sigmoid()
            self.criterion = nn.MSELoss()
        elif loss == 'msew':
            output_fn = nn.Sigmoid()
            self.criterion = WeightedMSELoss(weights)
        elif loss == 'bce':
            self.train_criterion = nn.BCEWithLogitsLoss()
            self.criterion = nn.BCEWithLogitsLoss()
            self.output_fn = nn.Sigmoid()
        else:
            self.criterion = nn.NLLLoss()
            output_fn = nn.LogSoftmax(1)


        self.model.cuda()

        if optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, amsgrad=True,
                                              weight_decay=weight_decay)
        elif optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9,
                                              weight_decay=weight_decay)
        else:
            raise RuntimeError('incorrect optimizer, choose adam or sgd')

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=lr_decay_after)

        self._train_iter = 0
        self.current_training_loss = 0.
        self.current_validation_loss = 0.


    def train(self, x, y):
        self.model.train()
        self.optimizer.zero_grad()
        pred = self.model(x)
        loss = self.train_criterion(pred, y)
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        self._train_iter += 1
        self.current_training_loss = loss.detach().cpu().numpy()

    @torch.no_grad()
    def evaluate(self, x, y):
        self.model.eval()
        pred = self.model(x)
        loss = self.criterion(pred, y)
        self.current_validation_loss = loss.detach().cpu().numpy()

    @torch.no_grad()
    def predict(self, x):
        self.model.eval()
        pred = self.output_fn(self.model(x[None]))
        p, label = torch.max(pred, 1)
        return p.squeeze().detach().cpu().numpy(), label.squeeze().detach().cpu().numpy()

    @torch.no_grad()
    def p(self, x):
        self.model.eval()
        pred = self.output_fn(self.model(x[None]))
        return pred.detach().cpu().squeeze().numpy()

from loss import StochasticCoTeachingMLLoss
class CoTTrainer:
    def __init__(self, learning_rate=0.001, weight_decay=0, max_iters=None, optimizer='adam', loss='mse', weights=None, lr_decay_after=34000, mode='batch_train',
                 stocot_delay=None, stocot_gradual=None, alpha=32, beta=2, steps_per_epoch=None, one_cycle_lr_scheduler=False):

        self.one_cycle_lr_scheduler = one_cycle_lr_scheduler

        self.criterion = StochasticCoTeachingMLLoss(alpha, beta, max_iters, stocot_gradual, stocot_delay, Loss=nn.BCEWithLogitsLoss)

        self.val_criterion = nn.BCEWithLogitsLoss()

        self.output_fn = nn.Sigmoid()

        self.model_1.cuda()
        self.model_2.cuda()


        self.optimizer_1 = torch.optim.Adam(self.model_1.parameters(), lr=learning_rate, amsgrad=True,
                                          weight_decay=weight_decay)

        self.optimizer_2 = torch.optim.Adam(self.model_2.parameters(), lr=learning_rate, amsgrad=True,
                                            weight_decay=weight_decay)

        if self.one_cycle_lr_scheduler:
            # lr scheduler used by Strodthoff et al.
            self.scheduler_1 = torch.optim.lr_scheduler.OneCycleLR(self.optimizer_1, max_lr=1E-2, epochs=max_iters, steps_per_epoch=steps_per_epoch)
            self.scheduler_2 = torch.optim.lr_scheduler.OneCycleLR(self.optimizer_2, max_lr=1E-2, epochs=max_iters, steps_per_epoch=steps_per_epoch)
        else:
            self.scheduler_1 = torch.optim.lr_scheduler.StepLR(self.optimizer_1, step_size=lr_decay_after)
            self.scheduler_2 = torch.optim.lr_scheduler.StepLR(self.optimizer_2, step_size=lr_decay_after)


        self._train_iter = 0
        self.current_training_loss = (0., 0.)
        self.current_validation_loss = (0., 0.)

    def step(self):
        self.criterion.step()

        if not self.one_cycle_lr_scheduler:
            self.scheduler_1.step()
            self.scheduler_2.step()

    def train(self, x, y):
        self.model_1.train()
        self.model_2.train()
        self.optimizer_1.zero_grad()
        self.optimizer_2.zero_grad()
        pred_1 = self.model_1(x)
        pred_2 = self.model_2(x)
        loss_1, loss_2 = self.criterion(pred_1, pred_2, y)
        loss_1.backward()
        loss_2.backward()
        self.optimizer_1.step()
        self.optimizer_2.step()

        if self.one_cycle_lr_scheduler:
            self.scheduler_1.step()
            self.scheduler_2.step()

        self._train_iter += 1
        self.current_training_loss = (loss_1.item(),
                                      loss_2.item())

    @torch.no_grad()
    def evaluate(self, x, y):
        self.model_1.eval()
        self.model_2.eval()
        pred_1 = self.model_1(x)
        pred_2 = self.model_2(x)
        loss_1 = self.val_criterion(pred_1, y)
        loss_2 = self.val_criterion(pred_2, y)
        self.current_validation_loss = (loss_1.item(),
                                        loss_2.item())


    @torch.no_grad()
    def predict(self, x):
        self.model.eval()
        pred = self.output_fn(self.model(x[None]))
        p, label = torch.max(pred, 1)
        return p.squeeze().detach().cpu().numpy(), label.squeeze().detach().cpu().numpy()


    @torch.no_grad()
    def p(self, x):
        self.model.eval()
        pred = self.output_fn(self.model(x[None]))
        return pred.detach().cpu().squeeze().numpy()


class PredNetTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        output_fn = nn.Sigmoid()
        num_channels = kwargs.pop('num_channels')
        num_classes = kwargs.pop('num_classes')
        self.model = prednet.PredNet(num_channels, num_classes, output_fn)
        super().__init__(*args, **kwargs)

class CustomResNetTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        output_fn = nn.Sigmoid()
        num_channels = kwargs.pop('num_channels')
        num_classes = kwargs.pop('num_classes')
        self.model = customnet.resnet_ecg_bob(num_channels=num_channels, num_classes=num_classes, output_fn=output_fn)
        super().__init__(*args, **kwargs)

class CustomDilationResNetTrainer(CoTTrainer):
    def __init__(self, *args, **kwargs):
        output_fn = nn.Sigmoid()
        num_channels = kwargs.pop('num_channels')
        num_classes = kwargs.pop('num_classes')
        self.model = customnet_dilation.resnet_ecg_bob(num_channels=num_channels, num_classes=num_classes, output_fn=output_fn)
        super().__init__(*args, **kwargs)

class WangResNetTrainer(Trainer):
    def __init__(self, *args, **kwargs):

        num_channels = kwargs.pop('num_channels')
        num_classes = kwargs.pop('num_classes')
        # self.model = resnet_wang.ResNetBaseline(in_channels=num_channels, num_pred_classes=num_classes, output_fn=output_fn)
        self.model = resnet1d.resnet_ptbxl_wang(in_channels=num_channels, num_classes=num_classes)
        super().__init__(*args, **kwargs)

class CoTeachWangResNetTrainer(CoTTrainer):
    def __init__(self, *args, **kwargs):
        num_channels = kwargs.pop('num_channels')
        num_classes = kwargs.pop('num_classes')

        self.model_1 = resnet1d.resnet_ptbxl_wang(in_channels=num_channels, num_classes=num_classes)
        self.model_2 = resnet1d.resnet_ptbxl_wang(in_channels=num_channels, num_classes=num_classes)

        super().__init__(*args, **kwargs)

from networks import xresnet1d

class XResNetTrainer(Trainer):
    def __init__(self, *args, **kwargs):

        num_channels = kwargs.pop('num_channels')
        num_classes = kwargs.pop('num_classes')
        self.model = xresnet1d.xresnet101_ptbxl(in_channels=num_channels, num_classes=num_classes)
        super().__init__(*args, **kwargs)

class CoTeachXResNetTrainer(CoTTrainer):
    def __init__(self, *args, **kwargs):
        num_channels = kwargs.pop('num_channels')
        num_classes = kwargs.pop('num_classes')
        self.model_1 = xresnet1d.xresnet101_ptbxl(in_channels=num_channels, num_classes=num_classes)
        self.model_2 = xresnet1d.xresnet101_ptbxl(in_channels=num_channels, num_classes=num_classes)
        super().__init__(*args, **kwargs)


from networks import inception1d
class InceptionTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        num_channels = kwargs.pop('num_channels')
        num_classes = kwargs.pop('num_classes')
        self.model = inception1d.inception1d_ptbxl(in_channels=num_channels, num_classes=num_classes)
        super().__init__(*args, **kwargs)

class CoTeachInceptionTrainer(CoTTrainer):
    def __init__(self, *args, **kwargs):
        num_channels = kwargs.pop('num_channels')
        num_classes = kwargs.pop('num_classes')
        self.model_1 = inception1d.inception1d_ptbxl(in_channels=num_channels, num_classes=num_classes)
        self.model_2 = inception1d.inception1d_ptbxl(in_channels=num_channels, num_classes=num_classes)
        super().__init__(*args, **kwargs)

class CoTeachTrainer(CoTTrainer):
    def __init__(self, *args, **kwargs):
        output_fn = nn.Sigmoid()
        num_channels = kwargs.pop('num_channels')
        num_classes = kwargs.pop('num_classes')
        norm_layer = kwargs.pop('norm_layer', 'batchnorm')

        self.model_1 = customnet.resnet_ecg_bob(num_channels=num_channels, num_classes=num_classes, output_fn=output_fn)
        self.model_2 = customnet.resnet_ecg_bob(num_channels=num_channels, num_classes=num_classes, output_fn=output_fn)
        super().__init__(*args, **kwargs)


class VDLResNetTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        output_fn = nn.Sigmoid()
        num_channels = kwargs.pop('num_channels')
        num_classes = kwargs.pop('num_classes')
        self.model = vandeleurnet.resnet_ecg(num_channels=num_channels, num_classes=num_classes, output_fn=output_fn)
        super().__init__(*args, **kwargs)

class DCNNTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        output_fn = nn.Sigmoid()
        num_channels = kwargs.pop('num_channels')
        num_classes = kwargs.pop('num_classes')
        self.model = dcnn.DCNN(num_channels=num_channels, num_classes=num_classes, output_fn=output_fn)
        super().__init__(*args, **kwargs)