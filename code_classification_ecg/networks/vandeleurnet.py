import torch
import torch.nn as nn

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


class VanLeurBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=nn.BatchNorm1d):
        super().__init__()
        self.norm_layer = norm_layer(inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.normal_block = nn.Sequential(conv5(inplanes, planes, stride=stride),
                                          norm_layer(planes),
                                          nn.ReLU(inplace=True),
                                          conv5(planes, planes),
                                          nn.Dropout(p=.5, inplace=True)
                                          )

        self.dilated_block = nn.Sequential(conv5(inplanes, planes, stride=stride),
                                           norm_layer(planes),
                                           nn.ReLU(inplace=True),
                                           conv5(planes, planes, dilation=100),
                                           nn.Dropout(p=.5, inplace=True)
                                           )
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        if self.downsample is not None:
            identity = self.downsample(x)
        x = self.relu(self.norm_layer(x))
        out = torch.cat([self.normal_block(x), self.dilated_block(x), self.dilated_block(x)])
        #         out += self.dilated_block(x)
        #         out += identity
        out = self.relu(out)
        return out

class VanDeLeurBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=nn.BatchNorm1d):
        super().__init__()

        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class VanDeLeurResNet(nn.Module):

    def __init__(self, block, layers, num_channels=1, num_classes=4, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None, output_fn=nn.LogSoftmax(1),
                 norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        self._norm_layer = norm_layer

        self.output_fn = output_fn

        self.inplanes = num_channels
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        self.first = nn.Sequential(
            nn.Conv1d(num_channels, num_channels, kernel_size=5, stride=1, padding=2, bias=False),
            norm_layer(num_channels), nn.ReLU(inplace=True),
            nn.Conv1d(num_channels, num_channels, kernel_size=5, stride=1, padding=2, bias=False),
            norm_layer(num_channels), nn.ReLU(inplace=True), nn.Dropout(.3),
            nn.Conv1d(num_channels, num_channels, kernel_size=5, stride=1, padding=2, bias=False))

        self.layer1 = self._make_layer(block, 16, layers[0], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer1b = self._make_layer(block, 16, layers[0], stride=2,
                                        dilate=replace_stride_with_dilation[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer2b = self._make_layer(block, 32, layers[1], stride=2,
                                        dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer3b = self._make_layer(block, 64, layers[2], stride=2,
                                        dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 128, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.layer4b = self._make_layer(block, 128, layers[3], stride=2,
                                        dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.bn = nn.BatchNorm1d(12 * 128)
        self.relu = nn.ReLU(inplace=True)
        self.fcdropout = nn.Dropout(p=0.3)
        self.fc1 = nn.Linear(12 * 128, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
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

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
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
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.first(x)
        x = self.layer1(x)
        x = self.layer1b(x)
        x = self.layer2(x)
        x = self.layer2b(x)
        x = self.layer3(x)
        x = self.layer3b(x)
        x = self.layer4(x)
        x = self.layer4b(x)

        #         x = self.avgpool(x)
        # x, _ = torch.max(x, 2)

        x = torch.flatten(x, 1)
        x = self.bn(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fcdropout(self.fc2(x)))
        x = self.fc3(x)

        return x

    def forward(self, x):

        return self.output_fn(self._forward_impl(x))


def resnet_ecg(**kwargs):
    model = VanDeLeurResNet(VanDeLeurBlock, [2, 2, 2, 2], **kwargs)
    return model
