import torch
from torch import nn

class PredNet(nn.Module):
    def __init__(self, num_channels, num_classes, output_fn):
        super(PredNet, self).__init__()
        self.model = nn.Sequential(nn.Conv1d(num_channels, 16, 3, 1, 0, bias=False),
                                   nn.MaxPool1d(2),
                                   nn.LeakyReLU(),
                                   nn.BatchNorm1d(16),

                                   nn.Conv1d(16, 16, 3, 1, 0, bias=False),
                                   nn.MaxPool1d(2),
                                   nn.LeakyReLU(),
                                   nn.BatchNorm1d(16),

                                   nn.Conv1d(16, 32, 3, 1, 0, bias=False),
                                   nn.MaxPool1d(2),
                                   nn.LeakyReLU(),
                                   nn.BatchNorm1d(32),

                                   nn.Conv1d(32, 32, 3, 1, 0, bias=False),
                                   nn.MaxPool1d(2),
                                   nn.LeakyReLU(),
                                   nn.BatchNorm1d(32),

                                   nn.Conv1d(32, 64, 3, 1, 0, bias=False),
                                   nn.MaxPool1d(2),
                                   nn.LeakyReLU(),
                                   nn.BatchNorm1d(64),

                                   nn.Conv1d(64, 64, 3, 1, 0, bias=False),
                                   nn.LeakyReLU(),
                                   nn.MaxPool1d(2),
                                   nn.BatchNorm1d(64),

                                   nn.Conv1d(64, 128, 3, 1, 0, bias=False),
                                   nn.LeakyReLU(),
                                   nn.BatchNorm1d(128),


                                   nn.Conv1d(128, 128, 3, 1, 0, bias=False),
                                   nn.LeakyReLU(),
                                   nn.BatchNorm1d(128),
                                   nn.Conv1d(128, 128, 1, 1, 0, bias=False),
                                   nn.LeakyReLU(),
                                   nn.BatchNorm1d(128),
                                   nn.Conv1d(128, 128, 1, 1, 0, bias=False),
                                   nn.LeakyReLU(),
                                   nn.BatchNorm1d(128),
                                   nn.Conv1d(128, num_classes, 1, 1, 0),

                                   )

        self.lsm = output_fn

    def forward(self, x):
        x = self.model(x)
        x, _ = torch.max(x, 2)
        return self.lsm(x)