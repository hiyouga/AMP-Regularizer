import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride, dropout):
        super(BasicBlock, self).__init__()
        self.residual = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * self.expansion)
        )
        self.downsample = nn.AvgPool2d((2,2), stride=(2,2), ceil_mode=True) if stride != 1 else None

    def forward(self, x):
        out = self.residual(x)
        if self.downsample is not None:
            shortcut = self.downsample(x)
            featuremap_size = shortcut.size()[2:4]
        else:
            shortcut = x
            featuremap_size = out.size()[2:4]
        batch_size = out.size()[0]
        residual_channel = out.size()[1]
        shortcut_channel = shortcut.size()[1]
        padding_size = (batch_size, residual_channel - shortcut_channel, featuremap_size[0], featuremap_size[1])
        if residual_channel != shortcut_channel:
            padding = torch.zeros(padding_size, dtype=x.dtype, device=x.device)
            out += torch.cat((shortcut, padding), dim=1)
        else:
            out += shortcut 
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride, dropout):
        super(Bottleneck, self).__init__()
        self.residual = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels * self.expansion)
        )
        self.downsample = nn.AvgPool2d((2,2), stride=(2,2), ceil_mode=True) if stride != 1 else None

    def forward(self, x):
        out = self.residual(x)
        if self.downsample is not None:
            shortcut = self.downsample(x)
            featuremap_size = shortcut.size()[2:4]
        else:
            shortcut = x
            featuremap_size = out.size()[2:4]
        batch_size = out.size()[0]
        residual_channel = out.size()[1]
        shortcut_channel = shortcut.size()[1]
        padding_size = (batch_size, residual_channel - shortcut_channel, featuremap_size[0], featuremap_size[1])
        if residual_channel != shortcut_channel:
            padding = torch.zeros(padding_size, dtype=x.dtype, device=x.device)
            out += torch.cat((shortcut, padding), dim=1)
        else:
            out += shortcut 
        return out


class PyramidNet(nn.Module):

    def __init__(self, depth, alpha, bottleneck, num_classes, dropout):
        super(PyramidNet, self).__init__()
        self.in_channels = 16
        self.out_channels = 16
        if bottleneck:
            n = int((depth - 2) / 9)
            block = Bottleneck
        else:
            n = int((depth - 2) / 6)
            block = BasicBlock
        self.addrate = alpha / (3.0 * n)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.in_channels)
        )
        self.layer1 = self._make_layer(block, n, 1, dropout)
        self.layer2 = self._make_layer(block, n, 2, dropout)
        self.layer3 = self._make_layer(block, n, 2, dropout)
        self.classifier = nn.Sequential(
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(8),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(self.in_channels, num_classes)
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, val=1)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)

    def _make_layer(self, block, block_depth, stride, dropout):
        layers = []
        self.out_channels = self.out_channels + self.addrate
        layers.append(block(self.in_channels, int(round(self.out_channels)), stride, dropout))
        for i in range(1, block_depth):
            temp_channels = self.out_channels + self.addrate
            layers.append(block(int(round(self.out_channels)) * block.expansion, int(round(temp_channels)), 1, dropout))
            self.out_channels = temp_channels
        self.in_channels = int(round(self.out_channels)) * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.classifier(out)
        return out


def pyramidnet110_48(num_classes=10, dropout=0):
    return PyramidNet(110, 48, False, num_classes, dropout)


def pyramidnet110_84(num_classes=10, dropout=0):
    return PyramidNet(110, 84, False, num_classes, dropout)


def pyramidnet110_270(num_classes=10, dropout=0):
    return PyramidNet(110, 270, False, num_classes, dropout)


def pyramidnet164_48(num_classes=10, dropout=0):
    return PyramidNet(164, 48, True, num_classes, dropout)


def pyramidnet164_270(num_classes=10, dropout=0):
    return PyramidNet(164, 270, True, num_classes, dropout)
