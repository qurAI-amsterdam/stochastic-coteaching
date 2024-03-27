import torch
from torch import nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, dilation=1):
        super().__init__()
        AF = nn.LeakyReLU
        self.block = nn.Sequential(nn.Conv1d(in_channels, out_channels,
                                              kernel_size, stride, padding=0,
                                              dilation=dilation, bias=False),
                                    nn.BatchNorm1d(out_channels),
                                    AF(inplace=True))
    def forward(self, x):
        return self.block(x)

class DCNN(nn.Module):
    def __init__(self, num_channels, num_classes, output_fn):
        super().__init__()
        numkernels = 64
        self.conv_layers = nn.Sequential(ConvBlock(num_channels, numkernels, stride=2, dilation=1),
                                         ConvBlock(numkernels, numkernels, stride=2, dilation=1),
                                         ConvBlock(numkernels, numkernels, stride=1, dilation=1),
                                         ConvBlock(numkernels, numkernels, dilation=2),
                                         ConvBlock(numkernels, numkernels, dilation=4),
                                         ConvBlock(numkernels, numkernels, dilation=8),
                                         ConvBlock(numkernels, numkernels, dilation=16),
                                         ConvBlock(numkernels, numkernels, dilation=32),
                                         ConvBlock(numkernels, numkernels, dilation=64),
                                         # ConvBlock(numkernels, numkernels, dilation=128),
                                         # ConvBlock(numkernels, numkernels, dilation=256),
                                         ConvBlock(numkernels, numkernels, kernel_size=1, dilation=1),
                                         ConvBlock(numkernels, numkernels, kernel_size=1, dilation=1),
                                         ConvBlock(numkernels, num_classes, dilation=1),
                                         )

        self.output_fn = output_fn

    def forward(self, x):
        x = self.conv_layers(x)
        x, _ = torch.max(x, 2)
        return self.output_fn(x)



if __name__ == '__main__':
    from torchsummary import summary
    #net = ResNet()
    #from my_resnet1d import ResNet, BasicBlock, Bottleneck

    # net = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=4)
    net = DCNN(1, 4, nn.Sigmoid())


    summary(net.cuda(), (1,2500), 2)
