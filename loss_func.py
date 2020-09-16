import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss(nn.Module):

    def __init__(self, flooding, smoothing, num_classes):
        super(CrossEntropyLoss, self).__init__()
        self.flooding = flooding
        self.smoothing = smoothing
        self.num_classes = num_classes
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, outputs, targets, lamda=None, indices=None):
        output_probs = self.logsoftmax(outputs)
        if lamda is not None: # mixup
            targets_a, targets_b = targets, targets[indices]
            lamda = lamda.view(-1, 1)
            targets_a = F.one_hot(targets_a, self.num_classes).float()
            targets_b = F.one_hot(targets_b, self.num_classes).float()
            reweighted_targets = lamda * targets_a + (1 - lamda) * targets_b
        else:
            if self.smoothing > 0: # label smoothing
                reweighted_targets = torch.zeros_like(outputs)
                reweighted_targets.fill_(self.smoothing / (self.num_classes - 1.0))
                reweighted_targets.scatter_(1, targets.unsqueeze(-1), 1.0 - self.smoothing)
            else: # base
                reweighted_targets = F.one_hot(targets, self.num_classes).float()
        loss = -1 * (output_probs * reweighted_targets).sum(dim=1).mean()
        if self.flooding > 0: # flooding
            loss = (loss - self.flooding).abs() + self.flooding
        return loss
