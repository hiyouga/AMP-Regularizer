import torch.nn as nn
from .utils import mixup_process


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_channels, out_channels, stride, dropout):
        super(PreActBlock, self).__init__()

        self.blocks = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, self.expansion * out_channels, kernel_size=3, stride=1, padding=1)
        )

        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, padding=0)
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        out = self.blocks(x)
        out = self.residual(out) + self.shortcut(out)
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_channels, out_channels, stride, dropout):
        super(PreActBottleneck, self).__init__()

        self.blocks = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, self.expansion * out_channels, kernel_size=1, stride=1, padding=0)
        )

        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, padding=0)
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        out = self.blocks(x)
        out = self.residual(out) + self.shortcut(out)
        return out


class PreActResNet(nn.Module):

    def __init__(self, block, num_blocks, num_classes, scales, dropout):
        super(PreActResNet, self).__init__()

        self.init_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], 1, dropout)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], 2, dropout)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], 2, dropout)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], 2, dropout)
        self.classifier = nn.Sequential(
            nn.BatchNorm2d(512 * block.expansion),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(4),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(512 * block.expansion, num_classes)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, val=1)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)

        self.original_params = nn.ParameterDict()
        self.perturb_params = nn.ParameterDict()
        self.perturb_modules = nn.ModuleDict({
            'conv1': self.conv1,
            'layer1': self.layer1,
            'layer2': self.layer2,
            'layer3': self.layer3,
            'layer4': self.layer4,
            'classifier': self.classifier
        })
        self.perturb_scale = {
            'conv1': scales[0],
            'layer1': scales[1],
            'layer2': scales[2],
            'layer3': scales[3],
            'layer4': scales[4],
            'classifier': scales[5]
        }

    def _make_layer(self, block, out_channels, num_blocks, stride, dropout):
        strides = [stride] + [1] * (num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.init_channels, out_channels, stride, dropout))
            self.init_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, lamda=None, indices=None):
        if lamda is not None:
            x = mixup_process(x, lamda, indices)
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.classifier(out)
        return out


def preactresnet18(num_classes=10, dropout=0, scales=[1,1,1,1,1,1]):
    return PreActResNet(PreActBlock, [2,2,2,2], num_classes, scales, dropout)


def preactresnet34(num_classes=10, dropout=0, scales=[1,1,1,1,1,1]):
    return PreActResNet(PreActBlock, [3,4,6,3], num_classes, scales, dropout)


def preactresnet50(num_classes=10, dropout=0, scales=[1,1,1,1,1,1]):
    return PreActResNet(PreActBottleneck, [3,4,6,3], num_classes, scales, dropout)


def preactresnet101(num_classes=10, dropout=0, scales=[1,1,1,1,1,1]):
    return PreActResNet(PreActBottleneck, [3,4,23,3], num_classes, scales, dropout)


def preactresnet152(num_classes=10, dropout=0, scales=[1,1,1,1,1,1]):
    return PreActResNet(PreActBottleneck, [3,8,36,3], num_classes, scales, dropout)
