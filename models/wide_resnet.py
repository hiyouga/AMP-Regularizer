import torch.nn as nn
from .utils import mixup_process


class Wide_Basic(nn.Module):

    def __init__(self, in_channels, out_channels, stride, dropout):
        super(Wide_Basic, self).__init__()

        self.residual = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.Dropout(dropout),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        )

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)


class Wide_ResNet(nn.Module):

    def __init__(self, depth, widen_factor, num_classes, scales, dropout):
        super(Wide_ResNet, self).__init__()

        self.in_channels = 16

        assert ((depth-4)%6==0), 'Wide-resnet depth should be 6n+4'
        n = int((depth-4)/6)
        k = widen_factor
        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = nn.Conv2d(3, nStages[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._wide_layer(Wide_Basic, nStages[1], n, 1, dropout)
        self.layer2 = self._wide_layer(Wide_Basic, nStages[2], n, 2, dropout)
        self.layer3 = self._wide_layer(Wide_Basic, nStages[3], n, 2, dropout)
        self.classifier = nn.Sequential(
            nn.BatchNorm2d(nStages[3], momentum=0.9),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(8),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(nStages[3], num_classes)
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
            'classifier': self.classifier
        })
        self.perturb_scale = {
            'conv1': scales[0],
            'layer1': scales[1],
            'layer2': scales[2],
            'layer3': scales[3],
            'classifier': scales[4]
        }

    def _wide_layer(self, block, out_channels, num_blocks, stride, dropout):
        strides = [stride] + [1] * (num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride, dropout))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x, lamda=None, indices=None):
        if lamda is not None:
            x = mixup_process(x, lamda, indices)
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.classifier(out)
        return out


def wrn28_10(num_classes=10, dropout=0, scales=[1,1,1,1,1]):
    return Wide_ResNet(28, 10, num_classes, scales, dropout)


def wrn28_2(num_classes=10, dropout=0, scales=[1,1,1,1,1]):
    return Wide_ResNet(28, 2, num_classes, scales, dropout)
