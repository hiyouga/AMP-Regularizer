import torch.nn as nn
from .utils import mixup_process


cfg = {
    'VGG11': [
        [64, 'M'],
        [128, 'M'],
        [256, 256, 'M'],
        [512, 512, 'M'],
        [512, 512, 'M']
    ],
    'VGG13': [
        [64, 64, 'M'],
        [128, 128, 'M'],
        [256, 256, 'M'],
        [512, 512, 'M'],
        [512, 512, 'M']
    ],
    'VGG16': [
        [64, 64, 'M'],
        [128, 128, 'M'],
        [256, 256, 256, 'M'],
        [512, 512, 512, 'M'],
        [512, 512, 512, 'M']
    ],
    'VGG19': [
        [64, 64, 'M'],
        [128, 128, 'M'],
        [256, 256, 256, 256, 'M'],
        [512, 512, 512, 512, 'M'],
        [512, 512, 512, 512, 'M']
    ]
}


class VGG(nn.Module):
    def __init__(self, vgg_name, num_classes, scales, dropout):
        super(VGG, self).__init__()
        self.init_channels = 3
        self.layer1 = self._make_layers(cfg[vgg_name][0], dropout)
        self.layer2 = self._make_layers(cfg[vgg_name][1], dropout)
        self.layer3 = self._make_layers(cfg[vgg_name][2], dropout)
        self.layer4 = self._make_layers(cfg[vgg_name][3], dropout)
        self.layer5 = self._make_layers(cfg[vgg_name][4], dropout)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(4096, num_classes)
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
            'layer1': self.layer1,
            'layer2': self.layer2,
            'layer3': self.layer3,
            'layer4': self.layer4,
            'layer5': self.layer5,
            'classifier': self.classifier
        })
        self.perturb_scale = {
            'layer1': scales[0],
            'layer2': scales[1],
            'layer3': scales[2],
            'layer4': scales[3],
            'layer5': scales[4],
            'classifier': scales[5]
        }

    def _make_layers(self, cfg, dropout):
        layers = []
        for x in cfg:
            if x == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.append(nn.Conv2d(self.init_channels, x, kernel_size=3, padding=1))
                layers.append(nn.BatchNorm2d(x))
                layers.append(nn.ReLU(inplace=True))
                layers.append(nn.Dropout(dropout))
                self.init_channels = x
        return nn.Sequential(*layers)

    def forward(self, x, lamda=None, indices=None):
        if lamda is not None:
            x = mixup_process(x, lamda, indices)
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.classifier(out)
        return out


def vgg11(num_classes=10, dropout=0, scales=[1,1,1,1,1,1]):
    return VGG('VGG11', num_classes, scales, dropout)


def vgg13(num_classes=10, dropout=0, scales=[1,1,1,1,1,1]):
    return VGG('VGG13', num_classes, scales, dropout)


def vgg16(num_classes=10, dropout=0, scales=[1,1,1,1,1,1]):
    return VGG('VGG16', num_classes, scales, dropout)


def vgg19(num_classes=10, dropout=0, scales=[1,1,1,1,1,1]):
    return VGG('VGG19', num_classes, scales, dropout)
