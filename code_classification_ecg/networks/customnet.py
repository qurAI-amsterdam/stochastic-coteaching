import torch
import torch.nn as nn

AF = nn.LeakyReLU
nonlinearity = 'leaky_relu'

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv5(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """5 convolution with padding"""
    pad = dilation * (5 // 2)
    return nn.Conv1d(in_planes, out_planes, kernel_size=5, stride=stride,
                     padding=pad, groups=groups, bias=False, dilation=dilation)


class BobBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=nn.BatchNorm1d):
        super().__init__()
        self.dropout = nn.Dropout(p=0.3)
        self.normal_block = nn.Sequential(conv5(inplanes, planes, stride=stride),
                                          norm_layer(planes),
                                          AF(inplace=False),
                                          conv5(planes, planes),
                                          norm_layer(planes),
                                          # AF(inplace=True),
                                          )

        # self.dilated_block = nn.Sequential(conv5(inplanes, planes, stride=stride, dilation=dilation),
        #                                    norm_layer(planes),
        #                                    AF(inplace=True),
        #                                    conv5(planes, planes),
        #                                    norm_layer(planes),
        #                                    # AF(inplace=True),
        #                                    )

        self.af = AF(inplace=False)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.normal_block(x)
        # out += self.dilated_block(x)

        out = self.af(out)
        out += identity
        # out = self.dropout(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.af = AF(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.af(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.af(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.af(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, dilation, kernels, num_channels=1, num_classes=4, zero_init_residual=False,
                 groups=1, width_per_group=64, output_fn=nn.LogSoftmax(1),
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        self._norm_layer = norm_layer

        self.output_fn = output_fn

        self.inplanes = kernels[0]
        self.dilation = 1


        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv1d(num_channels, kernels[0], kernel_size=7, stride=2, padding=3,
                               bias=False)

        self.bn1 = norm_layer(kernels[0])
        self.af = AF(inplace=True)


        self.layers = nn.ModuleList()

        for l, d, k in zip(layers, dilation, kernels):
            print(l, d, k)
            self.layers.append(self._make_layer(block, k, l, stride=2,
                                       dilation=d))

        # self.layers = nn.Sequential(*layers)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(kernels[-1] * 1, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity=nonlinearity)
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        # previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.af(x)


        # x = self.layer1(x)
        # # x = self.layer1b(x)
        # x = self.layer2(x)
        # # x = self.layer2b(x)
        # x = self.layer3(x)
        # # x = self.layer3b(x)
        # x = self.layer4(x)
        # # x = self.layer4b(x)
        for l in self.layers:
            x = l(x)

        x = self.avgpool(x)
        # x, _ = torch.max(x, 2)

        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):

        return self.output_fn(self._forward_impl(x))


def resnet_ecg(**kwargs):
    model = ResNet(BobBlock, [2, 2, 2, 2], [1, 2, 4, 8, 16, 32, 64, 128],**kwargs)
    return model

def resnet_ecg32(**kwargs):
    model = ResNet(BobBlock, [3, 4, 6, 3], [1, 2, 4, 8, 16, 32, 64, 128], **kwargs)
    return model

def resnet_ecg_bob(**kwargs):
    model = ResNet(BobBlock, [2, 2, 2, 2, 2, 2, 2, 2], [1, 1, 1, 1, 1, 1, 1, 1], [16, 16,  32, 32, 64, 64, 128, 128], **kwargs)
    return model

def resnet_ecg_bob_small(**kwargs):
    model = ResNet(BobBlock, [2, 2, 2, 2, 2, 2], [1, 1, 1, 1, 1, 1], [16, 16,  32, 32, 64, 64], **kwargs)
    return model


def resnet_ecg_bob2(**kwargs):
    model = ResNet(BobBlock, [2, 2, 2, 2], [2, 8, 32, 128], [16, 32, 64, 128], **kwargs)
    return model

if __name__ == '__main__':
    from torchsummary import summary

    #net = ResNet()
    #from my_resnet1d import ResNet, BasicBlock, Bottleneck

    # net = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=4)
    net = resnet_ecg_bob(num_classes=4)

    # downsample = nn.Sequential(
    #     conv1x1(16, 16, 2),
    # )
    # net = BobBlock(16, 16, stride=2, downsample=downsample, groups=1,
    #              base_width=64, dilation=4, norm_layer=nn.BatchNorm1d)
    summary(net.cuda(), (1,500), 2)

