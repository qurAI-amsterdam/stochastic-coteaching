import torch
from torch import nn
import numpy as np

class PredNet(nn.Module):
    def __init__(self, num_channels, num_classes, output_fn):
        super(PredNet, self).__init__()
        self.model = nn.Sequential(nn.Conv1d(num_channels, 16, 3, 1, 0),
                                   nn.MaxPool1d(2),
                                   nn.LeakyReLU(),
                                   nn.BatchNorm1d(16),

                                   nn.Conv1d(16, 16, 3, 1, 0),
                                   nn.MaxPool1d(2),
                                   nn.LeakyReLU(),
                                   nn.BatchNorm1d(16),

                                   nn.Conv1d(16, 32, 3, 1, 0),
                                   nn.MaxPool1d(2),
                                   nn.LeakyReLU(),
                                   nn.BatchNorm1d(32),

                                   nn.Conv1d(32, 32, 3, 1, 0),
                                   nn.MaxPool1d(2),
                                   nn.LeakyReLU(),
                                   nn.BatchNorm1d(32),

                                   nn.Conv1d(32, 64, 3, 1, 0),
                                   nn.MaxPool1d(2),
                                   nn.LeakyReLU(),
                                   nn.BatchNorm1d(64),

                                   nn.Conv1d(64, 64, 3, 1, 0),
                                   nn.LeakyReLU(),
                                   nn.MaxPool1d(2),
                                   nn.BatchNorm1d(64),

                                   nn.Conv1d(64, 128, 3, 1, 0),
                                   nn.LeakyReLU(),
                                   nn.BatchNorm1d(128),


                                   nn.Conv1d(128, 128, 3, 1, 0),
                                   nn.LeakyReLU(),
                                   nn.BatchNorm1d(128),
                                   nn.Conv1d(128, 128, 1, 1, 0),
                                   nn.LeakyReLU(),
                                   nn.BatchNorm1d(128),
                                   nn.Conv1d(128, 128, 1, 1, 0),
                                   nn.LeakyReLU(),
                                   nn.BatchNorm1d(128),
                                   nn.Conv1d(128, num_classes, 1, 1, 0),

                                   )

        self.lsm = output_fn

    def forward(self, x):
        x = self.model(x)
        x, _ = torch.max(x, 2)
        return self.lsm(x)

class Tester:
    def __init__(self, model_state_dict, num_channels=12, num_classes=9):
        output_fn = nn.Sigmoid()

        self.model = PredNet(num_channels, num_classes, output_fn)
        self.model.cuda()
        self.model.load_state_dict(model_state_dict)
        self.model.eval()
        
    @torch.no_grad()
    def predict(self, x):
        x = x.astype(np.float32)
        x = x.T
        x = torch.from_numpy(x[None]).cuda()
        pred = self.model(x)
        return pred.squeeze().detach().cpu().numpy()
