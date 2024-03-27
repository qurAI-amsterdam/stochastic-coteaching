import torch
from torch import nn
import torch.nn.functional as F

import networks

import loss

class Trainer:
    def __init__(self, criterion, validation_criterion, learning_rate=0.001, weight_decay=0):
        self.criterion = criterion
        self.val_criterion = validation_criterion

        self.model_1.cuda()
        self.model_2.cuda()

        self.optimizer_1 = torch.optim.Adam(self.model_1.parameters(), lr=learning_rate,
                                            weight_decay=weight_decay)

        self.optimizer_2 = torch.optim.Adam(self.model_2.parameters(), lr=learning_rate,
                                            weight_decay=weight_decay)

        self.training_output_fn = lambda x: x
        self.testing_output_fn = lambda x: F.softmax(x, dim=1)


    @torch.no_grad()
    def accuracy(self, p, y):
        _, y_hat = torch.max(p, dim=1)
        correct = y_hat.eq(y).sum().type(torch.float32)
        return correct / len(y)

    def train(self, x, y, ind):#, noise_or_not):
        x = x.cuda()
        y = y.cuda()
        self.model_1.train()
        self.model_2.train()
        self.optimizer_1.zero_grad()
        self.optimizer_2.zero_grad()
        pred_1 = self.model_1(x)
        pred_2 = self.model_2(x)
        with torch.no_grad():

            accu_1 = self.accuracy(pred_1, y)

            accu_2 = self.accuracy(pred_2, y)

        loss_1, loss_2 = self.criterion(pred_1, pred_2, y, ind)
        loss_1.backward()
        loss_2.backward()
        self.optimizer_1.step()
        self.optimizer_2.step()

        return loss_1.item(), loss_2.item(), accu_1.item(), accu_2.item()

    @torch.no_grad()
    def evaluate(self, x, y):
        x = x.cuda()
        y = y.cuda()
        self.model_1.eval()
        self.model_2.eval()
        pred_1 = self.model_1(x)
        pred_2 = self.model_2(x)
        loss_1 = self.val_criterion(pred_1, y)
        loss_2 = self.val_criterion(pred_2, y)
        accu_1 = self.accuracy(pred_1, y)
        accu_2 = self.accuracy(pred_2, y)
        return loss_1.item(), loss_2.item(), accu_1.item(), accu_2.item()

class TrainerLargeCNN(Trainer):
    def __init__(self, *args, **kwargs):
        num_channels = kwargs.pop('num_channels')
        num_classes = kwargs.pop('num_classes')
        self.model_1 = networks.CNN(num_channels=num_channels, num_classes=num_classes)
        self.model_2 = networks.CNN(num_channels=num_channels, num_classes=num_classes)
        super().__init__(*args, **kwargs)

class TrainerSmallCNN(Trainer):
    def __init__(self, *args, **kwargs):
        num_channels = kwargs.pop('num_channels')
        num_classes = kwargs.pop('num_classes')
        self.model_1 = networks.smallnet.SmallNet(num_channels=num_channels, num_classes=num_classes)
        self.model_2 = networks.smallnet.SmallNet(num_channels=num_channels, num_classes=num_classes)
        super().__init__(*args, **kwargs)

class TrainerCifarCNN(Trainer):
    def __init__(self, *args, **kwargs):
        num_channels = kwargs.pop('num_channels')
        num_classes = kwargs.pop('num_classes')

        self.model_1 = networks.cotcnn.CNN(num_channels=num_channels, num_classes=num_classes)
        self.model_2 = networks.cotcnn.CNN(num_channels=num_channels, num_classes=num_classes)

        super().__init__(*args, **kwargs)
