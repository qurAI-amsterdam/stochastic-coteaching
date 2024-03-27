import torch.nn as nn

class DilatedCNN2D(nn.Module):
    def __init__(self, n_input=1, n_classes=4, n_kernels=32):
        #receptive field [131, 131]
        super().__init__()
        layers = [nn.Conv2d(n_input,   n_kernels, 3, dilation=1),  nn.BatchNorm2d(n_kernels), nn.ELU(),
                  nn.Conv2d(n_kernels, n_kernels, 3, dilation=1),  nn.BatchNorm2d(n_kernels), nn.ELU(),
                  nn.Conv2d(n_kernels, n_kernels, 3, dilation=2),  nn.BatchNorm2d(n_kernels), nn.ELU(),
                  nn.Conv2d(n_kernels, n_kernels, 3, dilation=4),  nn.BatchNorm2d(n_kernels), nn.ELU(),
                  nn.Conv2d(n_kernels, n_kernels, 3, dilation=8),  nn.BatchNorm2d(n_kernels), nn.ELU(),
                  nn.Conv2d(n_kernels, n_kernels, 3, dilation=16), nn.BatchNorm2d(n_kernels), nn.ELU(),
                  nn.Conv2d(n_kernels, n_kernels, 3, dilation=32), nn.BatchNorm2d(n_kernels), nn.ELU(),
                  nn.Conv2d(n_kernels, n_kernels, 3, dilation=1),  nn.BatchNorm2d(n_kernels), nn.ELU(),
                  # nn.Conv2d(n_kernels, n_kernels, 3, dilation=1),  nn.BatchNorm2d(n_kernels), nn.ELU(),
                  nn.Conv2d(n_kernels, n_kernels, 1, dilation=1),  nn.BatchNorm2d(n_kernels), nn.ELU(),
                  nn.Conv2d(n_kernels, n_classes, 1, dilation=1),  nn.LogSoftmax()]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


