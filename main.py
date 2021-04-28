import os
import sys
import torch
import models
import argparse
from amp import AMP
from data_utils import load_data


class Instructor:

    def __init__(self, args):
        for arg in vars(args):
            print(f">>> {arg}: {getattr(args, arg)}")
        print(f"=> creating model {args.model}")
        self.model = models.__dict__[args.model](num_classes=args.num_classes, dropout=args.dropout)
        self.model.to(args.device)
        if args.device.type == 'cuda':
            print(f"=> cuda memory allocated: {torch.cuda.memory_allocated(args.device.index)}")
        self.args = args

    def _train(self, train_dataloader, criterion, optimizer):
        train_loss, n_correct, n_train = 0, 0, 0
        n_batch = len(train_dataloader)
        self.model.train()
        for i_batch, (inputs, targets) in enumerate(train_dataloader):
            inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)
            def closure():
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_norm)
                return outputs, loss
            outputs, loss = optimizer.step(closure)
            train_loss += loss.item() * targets.size(0)
            n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
            n_train += targets.size(0)
            ratio = int((i_batch+1)*50/n_batch)
            sys.stdout.write(f"\r[{'>'*ratio}{' '*(50-ratio)}] {i_batch+1}/{n_batch} {(i_batch+1)*100/n_batch:.2f}%")
            sys.stdout.flush()
        print()
        return train_loss / n_train, n_correct / n_train

    def _test(self, test_dataloader, criterion):
        test_loss, n_correct, n_test = 0, 0, 0
        n_batch = len(test_dataloader)
        self.model.eval()
        with torch.no_grad():
            for i_batch, (inputs, targets) in enumerate(test_dataloader):
                inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item() * targets.size(0)
                n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                n_test += targets.size(0)
                ratio = int((i_batch+1)*50/n_batch)
                sys.stdout.write(f"\r[{'>'*ratio}{' '*(50-ratio)}] {i_batch+1}/{n_batch} {(i_batch+1)*100/n_batch:.2f}%")
                sys.stdout.flush()
        print()
        return test_loss / n_test, n_correct / n_test

    def run(self):
        train_dataloader, test_dataloader = load_data(batch_size=self.args.batch_size,
                                                      workers=0,
                                                      dataset=self.args.dataset,
                                                      data_target_dir=os.path.join(self.args.data_dir, self.args.dataset),
                                                      data_aug=(self.args.no_data_aug==False),
                                                      cutout=self.args.cutout,
                                                      autoaug=self.args.autoaug)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = AMP(params=filter(lambda p: p.requires_grad, self.model.parameters()),
                        epsilon=self.args.epsilon,
                        lr=self.args.lr,
                        inner_lr=self.args.inner_lr,
                        inner_iter=self.args.inner_iter,
                        base_optimizer=torch.optim.SGD,
                        momentum=self.args.momentum,
                        weight_decay=self.args.decay,
                        nesterov=True)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, self.args.milestones, self.args.gamma)
        best_loss, best_acc = 0, 0
        for epoch in range(self.args.num_epoch):
            train_loss, train_acc = self._train(train_dataloader, criterion, optimizer)
            test_loss, test_acc = self._test(test_dataloader, criterion)
            scheduler.step()
            if test_acc > best_acc or (test_acc == best_acc and test_loss < best_loss):
                best_acc = test_acc
                best_loss = test_loss
            print(f"{epoch+1}/{self.args.num_epoch} - {100*(epoch+1)/self.args.num_epoch:.2f}%")
            print(f"[train] loss: {train_loss:.4f}, acc@1: {train_acc*100:.2f}, err@1: {100-train_acc*100:.2f}")
            print(f"[test] loss: {test_loss:.4f}, acc@1: {test_acc*100:.2f}, err@1: {100-test_acc*100:.2f}")
        print(f"best val loss: {best_loss:.4f}, best acc@1: {best_acc*100:.2f}, best err@1: {100-best_acc*100:.2f}")


if __name__ == '__main__':

    model_names = sorted(name for name in models.__dict__ if name.islower() and not name.startswith('__') and callable(models.__dict__[name]))
    num_classes = {'svhn': 10, 'cifar10': 10, 'cifar100': 100}
    parser = argparse.ArgumentParser(description='Trainer', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, default='cifar10', choices=list(num_classes.keys()), help='Dataset name.')
    parser.add_argument('--data_dir', type=str, default='data', help='Dictionary for dataset.')
    parser.add_argument('--no_data_aug', default=False, action='store_true', help='Disable data augmentation.')
    parser.add_argument('--cutout', default=False, action='store_true', help='Enable Cutout augmentation.')
    parser.add_argument('--autoaug', default=False, action='store_true', help='Enable AutoAugment.')
    parser.add_argument('--model', default='preactresnet18', choices=model_names, help='Model architecture.')
    parser.add_argument('--num_epoch', type=int, default=200, help='Number of epochs to train.')
    parser.add_argument('--batch_size', type=int, default=50, help='Batch size.')
    parser.add_argument('--lr', type=float, default=0.1, help='Global learning rate.')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
    parser.add_argument('--decay', type=float, default=1e-4, help='Weight decay (L2 penalty).')
    parser.add_argument('--dropout', type=float, default=0, help='Dropout applied to the model.')
    parser.add_argument('--clip_norm', type=int, default=50, help='Maximum norm of parameter gradient.')
    parser.add_argument('--milestones', type=int, nargs='+', default=[100, 150], help='Decrease learning rate at these epochs.')
    parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule, number of gammas should be equal to schedule')
    parser.add_argument('--inner_iter', type=int, default=1, help='Inner iteration number.')
    parser.add_argument('--inner_lr', type=float, default=1, help='Inner learning rate.')
    parser.add_argument('--epsilon', type=float, default=0.5, help='Perturbation norm ball radius.')
    parser.add_argument('--device', type=str, default=None, choices=['cpu', 'cuda'], help='Device.')
    args = parser.parse_args()
    args.num_classes = num_classes[args.dataset]
    args.device = torch.device(args.device) if args.device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ins = Instructor(args)
    ins.run()
