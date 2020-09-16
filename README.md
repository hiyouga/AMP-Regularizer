# Adversarial Model Perturbation (AMP)

Code for reproducing AMP results.

## Requirement

- Python 3.7
- Torch 1.6.0
- TorchVision 0.7.0
- NumPy 1.18.5

## Installation

An easy way to install this code with anaconda environment:

```bash
conda create -n amp python=3.7
conda activate amp
pip install -r requirements.txt
```

## Usage

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

#### RMP

```bash
python main.py --dataset cifar100 --model vgg16 --optimizer sgd --lr 0.1 --method rmp --epsilon 0.1
```

#### AMP

```bash
python main.py --dataset cifar100 --model vgg16 --optimizer sgd --lr 0.1 --method amp --epsilon 0.1 --inner_lr 0.2 --inner_iter 1
```

### How to run experiments for TextCNN on IMDB

#### ERM

```bash
python main.py --dataset imdb --model textcnn --optimizer adam --lr 0.001 --method base
```

#### Dropout

```bash
python main.py --dataset imdb --model textcnn --optimizer adam --lr 0.001 --method base --dropout 0.5
```

#### Label smoothing

```bash
python main.py --dataset imdb --model textcnn --optimizer adam --lr 0.001 --method base --smoothing 0.2
```

#### Flooding

```bash
python main.py --dataset imdb --model textcnn --optimizer adam --lr 0.001 --method base --flooding 0.02
```

#### MixUp

```bash
python main.py --dataset imdb --model textcnn --optimizer adam --lr 0.001 --method mixup --mixup_alpha 1.0 1.0
```

#### RMP

```bash
python main.py --dataset imdb --model textcnn --optimizer adam --lr 0.001 --method rmp --epsilon 0.1
```

#### AMP

```bash
python main.py --dataset imdb --model textcnn --optimizer adam --lr 0.001 --method amp --epsilon 1 --inner_lr 0.1 --inner_iter 1
```

#### AMP+Dropout

```bash
python main.py --dataset imdb --model textcnn --optimizer adam --lr 0.001 --method amp --dropout 0.5 --epsilon 1 --inner_lr 0.1 --inner_iter 1
```

### How to run experiments for TextCNN on Yelp2013

#### ERM

```bash
python main.py --dataset yelp13 --model textcnn --optimizer adam --lr 0.001 --method base
```

#### Dropout

```bash
python main.py --dataset yelp13 --model textcnn --optimizer adam --lr 0.001 --method base --dropout 0.5
```

#### Label smoothing

```bash
python main.py --dataset yelp13 --model textcnn --optimizer adam --lr 0.001 --method base --smoothing 0.2
```

#### Flooding

```bash
python main.py --dataset yelp13 --model textcnn --optimizer adam --lr 0.001 --method base --flooding 0.02
```

#### MixUp

```bash
python main.py --dataset yelp13 --model textcnn --optimizer adam --lr 0.001 --method mixup --mixup_alpha 1.0 1.0
```

#### RMP

```bash
python main.py --dataset yelp13 --model textcnn --optimizer adam --lr 0.001 --method rmp --epsilon 0.1
```

#### AMP

```bash
python main.py --dataset yelp13 --model textcnn --optimizer adam --lr 0.001 --method amp --epsilon 1 --inner_lr 0.1 --inner_iter 1
```

#### AMP+Dropout

```bash
python main.py --dataset yelp13 --model textcnn --optimizer adam --lr 0.001 --method amp --dropout 0.5 --epsilon 1 --inner_lr 0.1 --inner_iter 1
```

### How to run experiments for TextCNN on Yelp2014

#### ERM

```bash
python main.py --dataset yelp14 --model textcnn --optimizer adam --lr 0.001 --method base
```

#### Dropout

```bash
python main.py --dataset yelp14 --model textcnn --optimizer adam --lr 0.001 --method base --dropout 0.5
```

#### Label smoothing

```bash
python main.py --dataset yelp14 --model textcnn --optimizer adam --lr 0.001 --method base --smoothing 0.2
```

#### Flooding

```bash
python main.py --dataset yelp14 --model textcnn --optimizer adam --lr 0.001 --method base --flooding 0.02
```

#### MixUp

```bash
python main.py --dataset yelp14 --model textcnn --optimizer adam --lr 0.001 --method mixup --mixup_alpha 1.0 1.0
```

#### RMP

```bash
python main.py --dataset yelp14 --model textcnn --optimizer adam --lr 0.001 --method rmp --epsilon 0.1
```

#### AMP

```bash
python main.py --dataset yelp14 --model textcnn --optimizer adam --lr 0.001 --method amp --epsilon 1 --inner_lr 0.1 --inner_iter 1
```

#### AMP+Dropout

```bash
python main.py --dataset yelp14 --model textcnn --optimizer adam --lr 0.001 --method amp --dropout 0.5 --epsilon 1 --inner_lr 0.1 --inner_iter 1
```
