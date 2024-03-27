from torch import nn as nn


class CifarCNN(nn.Module):
    """CNN."""

    def __init__(self, num_channels, num_classes):
        """CNN Builder."""
        super().__init__()
        inplace = False

        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                # Conv Layer block 1
                nn.Conv2d(in_channels=num_channels, out_channels=16, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=inplace),
                nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=inplace)),
            nn.Sequential(
                # Conv Layer block 2
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=inplace),
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=inplace)),
            nn.Sequential(
                # Conv Layer block 3
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=inplace),
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=inplace)),
            nn.Sequential(
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=inplace),
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=inplace))])

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc_layer = nn.Sequential(

            nn.Linear(4*128, 512),
            nn.Dropout(),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        """Perform forward."""
        # conv layers
        for l in self.conv_layers:
            initial = x
            x = l(x)
            x[:, :initial.shape[1]] += initial # skip connect
            x = self.pool(x)

        # flatten
        x = x.view(x.size(0), -1)

        # fc layer
        x = self.fc_layer(x)

        return x