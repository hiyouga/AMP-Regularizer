import os
import sys
import torch
import models
import argparse
from trainer import Trainer
from data_utils import load_cv_data, load_nlp_data


class Instructor:

    def __init__(self, args):
        self.args = args
        self._print_args()
        if self.args.dataset in ['imdb', 'yelp13', 'yelp14']:
            self.train_dataloader, self.test_dataloader, tokenizer = load_nlp_data(batch_size=self.args.batch_size, workers=0, dataset=self.args.dataset, data_target_dir=os.path.join(self.args.data_dir, self.args.dataset))
            print(f"=> creating model {self.args.model}")
            model = models.__dict__[self.args.model](num_classes=self.args.num_classes, dropout=self.args.dropout, scales=self.args.scales, tokenizer=tokenizer)
        else:
            self.train_dataloader, self.test_dataloader = load_cv_data(data_aug=(self.args.no_data_aug==False), batch_size=self.args.batch_size, workers=0, dataset=self.args.dataset, data_target_dir=os.path.join(self.args.data_dir, self.args.dataset))
            print(f"=> creating model {self.args.model}")
            model = models.__dict__[self.args.model](num_classes=self.args.num_classes, dropout=self.args.dropout, scales=self.args.scales)
        self.trainer = Trainer(model, self.args)
        self.trainer.model.to(self.args.device)
        if self.args.device.type == 'cuda':
            print(f"=> cuda memory allocated: {torch.cuda.memory_allocated(self.args.device.index)}")

    def _print_args(self):
        print('TRAINING ARGUMENTS:')
        for arg in vars(self.args):
            print(f">>> {arg}: {getattr(self.args, arg)}")

    def _adjust_lr(self, epoch):
        lr = self.args.lr
        for (gamma, step) in zip(self.args.gammas, self.args.schedule):
            if (epoch >= step):
                lr = lr * gamma
            else:
                break
        self.trainer.adjust_lr(lr)

    def _train(self, train_dataloader):
        train_loss, n_top1, n_train = 0, 0, 0
        n_batch = len(train_dataloader)
        self.trainer.model.train()
        for i_batch, (inputs, targets) in enumerate(train_dataloader):
            inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)
            outputs, loss = self.trainer.train(inputs, targets)
            train_loss += loss.item() * targets.size(0)
            n_top1 += (torch.argmax(outputs, -1) == targets).sum().item()
            n_train += targets.size(0)
            ratio = int((i_batch+1)*50/n_batch)
            sys.stdout.write(f"\r[{'>'*ratio}{' '*(50-ratio)}] {i_batch+1}/{n_batch} {(i_batch+1)*100/n_batch:.2f}%")
            sys.stdout.flush()
        print()
        return train_loss / n_train, n_top1 / n_train

    def _validate(self, val_dataloader):
        val_loss, n_top1, n_val = 0, 0, 0
        n_batch = len(val_dataloader)
        self.trainer.model.eval()
        with torch.no_grad():
            for i_batch, (inputs, targets) in enumerate(val_dataloader):
                inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)
                outputs, loss = self.trainer.evaluate(inputs, targets)
                val_loss += loss.item() * targets.size(0)
                n_top1 += (torch.argmax(outputs, -1) == targets).sum().item()
                n_val += targets.size(0)
                ratio = int((i_batch+1)*50/n_batch)
                sys.stdout.write(f"\r[{'>'*ratio}{' '*(50-ratio)}] {i_batch+1}/{n_batch} {(i_batch+1)*100/n_batch:.2f}%")
                sys.stdout.flush()
        print()
        return val_loss / n_val, n_top1 / n_val

    def run(self):
        best_val_loss, best_top1_acc = 0, 0
        for epoch in range(self.args.num_epoch):
            if self.args.optimizer == 'sgd':
                self._adjust_lr(epoch)
            train_loss, train_top1_acc = self._train(self.train_dataloader)
            val_loss, val_top1_acc = self._validate(self.test_dataloader)
            if val_top1_acc > best_top1_acc or (val_top1_acc == best_top1_acc and val_loss < best_val_loss):
                best_top1_acc = val_top1_acc
                best_val_loss = val_loss
            print(f"{epoch+1}/{self.args.num_epoch} - {100*(epoch+1)/self.args.num_epoch:.2f}%")
            print(f"[train] loss: {train_loss:.4f}, acc@1: {train_top1_acc*100:.2f}, err@1: {100-train_top1_acc*100:.2f}")
            print(f"[val] loss: {val_loss:.4f}, acc@1: {val_top1_acc*100:.2f}, err@1: {100-val_top1_acc*100:.2f}")
        print(f"best val loss: {best_val_loss:.4f}, best acc@1: {best_top1_acc*100:.2f}, best err@1: {100-best_top1_acc*100:.2f}")


if __name__ == '__main__':

    model_names = sorted(name for name in models.__dict__ if name.islower() and not name.startswith('__') and callable(models.__dict__[name]))
    method_names = ['base', 'mixup', 'adv', 'textadv', 'rmp', 'amp']
    num_classes = {'svhn': 10, 'cifar10': 10, 'cifar100': 100, 'imdb': 10, 'yelp13': 5, 'yelp14': 5}
    parser = argparse.ArgumentParser(description='Trainer', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, default='cifar10', choices=list(num_classes.keys()), help='Dataset name.')
    parser.add_argument('--data_dir', type=str, default='data', help='Dictionary for dataset.')
    parser.add_argument('--no_data_aug', default=False, action='store_true', help='Disable data augmentation.')
    parser.add_argument('--model', default='preactresnet18', choices=model_names, help='Model architecture.')
    parser.add_argument('--method', type=str, default='base', choices=method_names, help='Training method.')
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adam'], help='Choice of optimizer.')
    parser.add_argument('--num_epoch', type=int, default=200, help='Number of epochs to train.')
    parser.add_argument('--batch_size', type=int, default=50, help='Batch size.')
    parser.add_argument('--lr', type=float, default=0.1, help='Global learning rate.')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
    parser.add_argument('--decay', type=float, default=1e-4, help='Weight decay (L2 penalty).')
    parser.add_argument('--dropout', type=float, default=0, help='Dropout applied to the model.')
    parser.add_argument('--smoothing', type=float, default=0, help='Coefficient of label smoothing regularization.')
    parser.add_argument('--flooding', type=float, default=0, help='Flooding level.')
    parser.add_argument('--mixup_alpha', type=float, nargs=2, default=(0., 0.), help='Alpha parameter for mixup.')
    parser.add_argument('--clip_norm', type=int, default=50, help='Maximum norm of parameter gradient.')
    parser.add_argument('--schedule', type=int, nargs='+', default=[100, 150], help='Decrease learning rate at these epochs.')
    parser.add_argument('--gammas', type=float, nargs='+', default=[0.1, 0.1], help='LR is multiplied by gamma on schedule, number of gammas should be equal to schedule')
    parser.add_argument('--inner_iter', type=int, default=1, help='Inner iteration number.')
    parser.add_argument('--inner_lr', type=float, default=0.5, help='Inner learning rate.')
    parser.add_argument('--epsilon', type=float, default=1, help='Perturbation norm constraint.')
    parser.add_argument('--scales', type=str, default='1,1,1,1,1,1', help='Scales of epsilon applied to different layer.')
    parser.add_argument('--constrain', type=str, default='l2', choices=['l2', 'linf'], help='Norm type for perturbations.')
    parser.add_argument('--adv_ei', type=int, default=1, help='Adversarial training step size.')
    parser.add_argument('--adv_eps', type=int, default=1, help='Adversarial training norm constrain.')
    parser.add_argument('--adv_iter', type=int, default=1, help='Adversarial training iteration number.')
    parser.add_argument('--adv_norm', type=str, default='linf', choices=['l2', 'linf'], help='Adversarial training norm type.')
    parser.add_argument('--device', type=str, default=None, choices=['cpu', 'cuda'], help='Device.')
    args = parser.parse_args()
    args.num_classes = num_classes[args.dataset]
    args.scales = list(map(float, args.scales.split(',')))
    args.device = torch.device(args.device) if args.device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    assert len(args.gammas) == len(args.schedule)
    if not os.path.exists('dats'):
        os.mkdir('dats')
    ins = Instructor(args)
    ins.run()
