# Adversarial Model Perturbation (AMP)

Code for reproducing AMP results.

## Requirement

- Python 3.7
- Torch 1.6.0
- TorchVision 0.7.0
- NumPy 1.18.5
- Pillow 6.1.0

## Installation

An easy way to install this code with anaconda environment:

```bash
conda create -n amp python=3.7
conda activate amp
pip install -r requirements.txt
```

## Baseline comparison

### How to run experiments for PreActResNet18 on SVHN

#### ERM

```bash
python main.py --dataset svhn --model preactresnet18 --optimizer sgd --lr 0.1 --method base
```

#### Dropout

```bash
python main.py --dataset svhn --model preactresnet18 --optimizer sgd --lr 0.1 --method base --dropout 0.1
```

#### Label smoothing

```bash
python main.py --dataset svhn --model preactresnet18 --optimizer sgd --lr 0.1 --method base --smoothing 0.2
```

#### Flooding

```bash
python main.py --dataset svhn --model preactresnet18 --optimizer sgd --lr 0.1 --method base --flooding 0.02
```

#### MixUp

```bash
python main.py --dataset svhn --model preactresnet18 --optimizer sgd --lr 0.1 --method mixup --mixup_alpha 1.0 1.0
```

#### Adversarial Training

```bash
python main.py --dataset svhn --model preactresnet18 --optimizer sgd --lr 0.1 --method adv --adv_eps 1
```

#### RMP

```bash
python main.py --dataset svhn --model preactresnet18 --optimizer sgd --lr 0.1 --method rmp --epsilon 0.1
```

#### AMP

```bash
python main.py --dataset svhn --model preactresnet18 --optimizer sgd --lr 0.1 --method amp --epsilon 0.5 --inner_lr 1 --inner_iter 1
```

### How to run experiments for PreActResNet18 on CIFAR-10

#### ERM

```bash
python main.py --dataset cifar10 --model preactresnet18 --optimizer sgd --lr 0.1 --method base
```

#### Dropout

```bash
python main.py --dataset cifar10 --model preactresnet18 --optimizer sgd --lr 0.1 --method base --dropout 0.1
```

#### Label smoothing

```bash
python main.py --dataset cifar10 --model preactresnet18 --optimizer sgd --lr 0.1 --method base --smoothing 0.2
```

#### Flooding

```bash
python main.py --dataset cifar10 --model preactresnet18 --optimizer sgd --lr 0.1 --method base --flooding 0.02
```

#### MixUp

```bash
python main.py --dataset cifar10 --model preactresnet18 --optimizer sgd --lr 0.1 --method mixup --mixup_alpha 1.0 1.0
```

#### Adversarial Training

```bash
python main.py --dataset cifar10 --model preactresnet18 --optimizer sgd --lr 0.1 --method adv --adv_eps 1
```
#### RMP

```bash
python main.py --dataset cifar10 --model preactresnet18 --optimizer sgd --lr 0.1 --method rmp --epsilon 0.1
```

#### AMP

```bash
python main.py --dataset cifar10 --model preactresnet18 --optimizer sgd --lr 0.1 --method amp --epsilon 0.5 --inner_lr 1 --inner_iter 1
```


### How to run experiments for PreActResNet18 on CIFAR-100

#### ERM

```bash
python main.py --dataset cifar100 --model preactresnet18 --optimizer sgd --lr 0.1 --method base
```

#### Dropout

```bash
python main.py --dataset cifar100 --model preactresnet18 --optimizer sgd --lr 0.1 --method base --dropout 0.1
```

#### Label smoothing

```bash
python main.py --dataset cifar100 --model preactresnet18 --optimizer sgd --lr 0.1 --method base --smoothing 0.2
```

#### Flooding

```bash
python main.py --dataset cifar100 --model preactresnet18 --optimizer sgd --lr 0.1 --method base --flooding 0.02
```

#### MixUp

```bash
python main.py --dataset cifar100 --model preactresnet18 --optimizer sgd --lr 0.1 --method mixup --mixup_alpha 1.0 1.0
```

#### Adversarial Training

```bash
python main.py --dataset cifar100 --model preactresnet18 --optimizer sgd --lr 0.1 --method adv --adv_eps 1
```

#### RMP

```bash
python main.py --dataset cifar100 --model preactresnet18 --optimizer sgd --lr 0.1 --method rmp --epsilon 0.1
```

#### AMP

```bash
python main.py --dataset cifar100 --model preactresnet18 --optimizer sgd --lr 0.1 --method amp --epsilon 0.5 --inner_lr 1 --inner_iter 1
```

### How to run experiments for VGG16 on SVHN

#### ERM

```bash
python main.py --dataset svhn --model vgg16 --optimizer sgd --lr 0.1 --method base
```

#### Dropout

```bash
python main.py --dataset svhn --model vgg16 --optimizer sgd --lr 0.1 --method base --dropout 0.1
```

#### Label smoothing

```bash
python main.py --dataset svhn --model vgg16 --optimizer sgd --lr 0.1 --method base --smoothing 0.2
```

#### Flooding

```bash
python main.py --dataset svhn --model vgg16 --optimizer sgd --lr 0.1 --method base --flooding 0.02
```

#### MixUp

```bash
python main.py --dataset svhn --model vgg16 --optimizer sgd --lr 0.1 --method mixup --mixup_alpha 1.0 1.0
```

#### Adversarial Training

```bash
python main.py --dataset svhn --model vgg16 --optimizer sgd --lr 0.1 --method adv --adv_eps 1
```

#### RMP

```bash
python main.py --dataset svhn --model vgg16 --optimizer sgd --lr 0.1 --method rmp --epsilon 0.1
```

#### AMP

```bash
python main.py --dataset svhn --model vgg16 --optimizer sgd --lr 0.1 --method amp --epsilon 0.1 --inner_lr 0.2 --inner_iter 1
```

### How to run experiments for VGG16 on CIFAR-10

#### ERM

```bash
python main.py --dataset cifar10 --model vgg16 --optimizer sgd --lr 0.1 --method base
```

#### Dropout

```bash
python main.py --dataset cifar10 --model vgg16 --optimizer sgd --lr 0.1 --method base --dropout 0.1
```

#### Label smoothing

```bash
python main.py --dataset cifar10 --model vgg16 --optimizer sgd --lr 0.1 --method base --smoothing 0.2
```

#### Flooding

```bash
python main.py --dataset cifar10 --model vgg16 --optimizer sgd --lr 0.1 --method base --flooding 0.02
```

#### MixUp

```bash
python main.py --dataset cifar10 --model vgg16 --optimizer sgd --lr 0.1 --method mixup --mixup_alpha 1.0 1.0
```

#### Adversarial Training

```bash
python main.py --dataset cifar10 --model vgg16 --optimizer sgd --lr 0.1 --method adv --adv_eps 1
```

#### RMP

```bash
python main.py --dataset cifar10 --model vgg16 --optimizer sgd --lr 0.1 --method rmp --epsilon 0.1
```

#### AMP

```bash
python main.py --dataset cifar10 --model vgg16 --optimizer sgd --lr 0.1 --method amp --epsilon 0.1 --inner_lr 0.2 --inner_iter 1
```


### How to run experiments for VGG16 on CIFAR-100

#### ERM

```bash
python main.py --dataset cifar100 --model vgg16 --optimizer sgd --lr 0.1 --method base
```

#### Dropout

```bash
python main.py --dataset cifar100 --model vgg16 --optimizer sgd --lr 0.1 --method base --dropout 0.1
```

#### Label smoothing

```bash
python main.py --dataset cifar100 --model vgg16 --optimizer sgd --lr 0.1 --method base --smoothing 0.2
```

#### Flooding

```bash
python main.py --dataset cifar100 --model vgg16 --optimizer sgd --lr 0.1 --method base --flooding 0.02
```

#### MixUp

```bash
python main.py --dataset cifar100 --model vgg16 --optimizer sgd --lr 0.1 --method mixup --mixup_alpha 1.0 1.0
```

#### Adversarial Training

```bash
python main.py --dataset cifar100 --model vgg16 --optimizer sgd --lr 0.1 --method adv --adv_eps 1
```
#### RMP

```bash
python main.py --dataset cifar100 --model vgg16 --optimizer sgd --lr 0.1 --method rmp --epsilon 0.1
```

#### AMP

```bash
python main.py --dataset cifar100 --model vgg16 --optimizer sgd --lr 0.1 --method amp --epsilon 0.1 --inner_lr 0.2 --inner_iter 1
```

## Use data augmentation

### How to run experiments for WideResNet-28-10 on SVHN with vanilla augmentation

#### ERM

```bash
python main.py --dataset svhn --model wrn28_10 --optimizer sgd --lr 0.1 --method base
```

#### AMP

```bash
python main.py --dataset svhn --model wrn28_10 --optimizer sgd --lr 0.1 --method amp --epsilon 0.5 --inner_lr 1 --inner_iter 1
```

### How to run experiments for WideResNet-28-10 on SVHN with Cutout augmentation

#### ERM

```bash
python main.py --dataset svhn --model wrn28_10 --optimizer sgd --lr 0.1 --method base --cutout
```

#### AMP

```bash
python main.py --dataset svhn --model wrn28_10 --optimizer sgd --lr 0.1 --method amp --epsilon 0.3 --inner_lr 0.5 --inner_iter 1 --cutout
```

### How to run experiments for WideResNet-28-10 on SVHN with AutoAugment augmentation

#### ERM

```bash
python main.py --dataset svhn --model wrn28_10 --optimizer sgd --lr 0.1 --method base --autoaug
```

#### AMP

```bash
python main.py --dataset svhn --model wrn28_10 --optimizer sgd --lr 0.1 --method amp --epsilon 0.1 --inner_lr 0.1 --inner_iter 1 --autoaug
```

### How to run experiments for WideResNet-28-10 on CIFAR-10 with vanilla augmentation

#### ERM

```bash
python main.py --dataset cifar10 --model wrn28_10 --optimizer sgd --lr 0.1 --method base
```

#### AMP

```bash
python main.py --dataset cifar10 --model wrn28_10 --optimizer sgd --lr 0.1 --method amp --epsilon 0.5 --inner_lr 1 --inner_iter 1
```

### How to run experiments for WideResNet-28-10 on CIFAR-10 with Cutout augmentation

#### ERM

```bash
python main.py --dataset cifar10 --model wrn28_10 --optimizer sgd --lr 0.1 --method base --cutout
```

#### AMP

```bash
python main.py --dataset cifar10 --model wrn28_10 --optimizer sgd --lr 0.1 --method amp --epsilon 0.3 --inner_lr 0.5 --inner_iter 1 --cutout
```

### How to run experiments for WideResNet-28-10 on CIFAR-10 with AutoAugment augmentation

#### ERM

```bash
python main.py --dataset cifar10 --model wrn28_10 --optimizer sgd --lr 0.1 --method base --autoaug
```

#### AMP

```bash
python main.py --dataset cifar10 --model wrn28_10 --optimizer sgd --lr 0.1 --method amp --epsilon 0.1 --inner_lr 0.1 --inner_iter 1 --autoaug
```

### How to run experiments for WideResNet-28-10 on CIFAR-100 with vanilla augmentation

#### ERM

```bash
python main.py --dataset cifar100 --model wrn28_10 --optimizer sgd --lr 0.1 --method base
```

#### AMP

```bash
python main.py --dataset cifar100 --model wrn28_10 --optimizer sgd --lr 0.1 --method amp --epsilon 0.5 --inner_lr 1 --inner_iter 1
```

### How to run experiments for WideResNet-28-10 on CIFAR-100 with Cutout augmentation

#### ERM

```bash
python main.py --dataset cifar100 --model wrn28_10 --optimizer sgd --lr 0.1 --method base --cutout
```

#### AMP

```bash
python main.py --dataset cifar100 --model wrn28_10 --optimizer sgd --lr 0.1 --method amp --epsilon 0.3 --inner_lr 0.5 --inner_iter 1 --cutout
```

### How to run experiments for WideResNet-28-10 on CIFAR-100 with AutoAugment augmentation

#### ERM

```bash
python main.py --dataset cifar100 --model wrn28_10 --optimizer sgd --lr 0.1 --method base --autoaug
```

#### AMP

```bash
python main.py --dataset cifar100 --model wrn28_10 --optimizer sgd --lr 0.1 --method amp --epsilon 0.1 --inner_lr 0.1 --inner_iter 1 --autoaug
```

### How to run experiments for PyramidNet-164-270 on SVHN with vanilla augmentation

#### ERM

```bash
python main.py --dataset svhn --model pyramidnet164_270 --optimizer sgd --lr 0.1 --method base
```

#### AMP

```bash
python main.py --dataset svhn --model pyramidnet164_270 --optimizer sgd --lr 0.1 --method amp --epsilon 0.5 --inner_lr 1 --inner_iter 1
```

### How to run experiments for PyramidNet-164-270 on SVHN with Cutout augmentation

#### ERM

```bash
python main.py --dataset svhn --model pyramidnet164_270 --optimizer sgd --lr 0.1 --method base --cutout
```

#### AMP

```bash
python main.py --dataset svhn --model pyramidnet164_270 --optimizer sgd --lr 0.1 --method amp --epsilon 0.3 --inner_lr 0.5 --inner_iter 1 --cutout
```

### How to run experiments for PyramidNet-164-270 on SVHN with AutoAugment augmentation

#### ERM

```bash
python main.py --dataset svhn --model pyramidnet164_270 --optimizer sgd --lr 0.1 --method base --autoaug
```

#### AMP

```bash
python main.py --dataset svhn --model pyramidnet164_270 --optimizer sgd --lr 0.1 --method amp --epsilon 0.1 --inner_lr 0.1 --inner_iter 1 --autoaug
```

### How to run experiments for PyramidNet-164-270 on CIFAR-10 with vanilla augmentation

#### ERM

```bash
python main.py --dataset cifar10 --model pyramidnet164_270 --optimizer sgd --lr 0.1 --method base
```

#### AMP

```bash
python main.py --dataset cifar10 --model pyramidnet164_270 --optimizer sgd --lr 0.1 --method amp --epsilon 0.5 --inner_lr 1 --inner_iter 1
```

### How to run experiments for PyramidNet-164-270 on CIFAR-10 with Cutout augmentation

#### ERM

```bash
python main.py --dataset cifar10 --model pyramidnet164_270 --optimizer sgd --lr 0.1 --method base --cutout
```

#### AMP

```bash
python main.py --dataset cifar10 --model pyramidnet164_270 --optimizer sgd --lr 0.1 --method amp --epsilon 0.3 --inner_lr 0.5 --inner_iter 1 --cutout
```

### How to run experiments for PyramidNet-164-270 on CIFAR-10 with AutoAugment augmentation

#### ERM

```bash
python main.py --dataset cifar10 --model pyramidnet164_270 --optimizer sgd --lr 0.1 --method base --autoaug
```

#### AMP

```bash
python main.py --dataset cifar10 --model pyramidnet164_270 --optimizer sgd --lr 0.1 --method amp --epsilon 0.1 --inner_lr 0.1 --inner_iter 1 --autoaug
```

### How to run experiments for PyramidNet-164-270 on CIFAR-100 with vanilla augmentation

#### ERM

```bash
python main.py --dataset cifar100 --model pyramidnet164_270 --optimizer sgd --lr 0.1 --method base
```

#### AMP

```bash
python main.py --dataset cifar100 --model pyramidnet164_270 --optimizer sgd --lr 0.1 --method amp --epsilon 0.5 --inner_lr 1 --inner_iter 1
```

### How to run experiments for PyramidNet-164-270 on CIFAR-100 with Cutout augmentation

#### ERM

```bash
python main.py --dataset cifar100 --model pyramidnet164_270 --optimizer sgd --lr 0.1 --method base --cutout
```

#### AMP

```bash
python main.py --dataset cifar100 --model pyramidnet164_270 --optimizer sgd --lr 0.1 --method amp --epsilon 0.3 --inner_lr 0.5 --inner_iter 1 --cutout
```

### How to run experiments for PyramidNet-164-270 on CIFAR-100 with AutoAugment augmentation

#### ERM

```bash
python main.py --dataset cifar100 --model pyramidnet164_270 --optimizer sgd --lr 0.1 --method base --autoaug
```

#### AMP

```bash
python main.py --dataset cifar100 --model pyramidnet164_270 --optimizer sgd --lr 0.1 --method amp --epsilon 0.1 --inner_lr 0.1 --inner_iter 1 --autoaug
```