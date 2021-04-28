import torch
from torch.optim import Optimizer, SGD


class AMP(Optimizer):
    """
    Implements adversarial model perturbation.

    Args:
        params (iterable): iterable of parameters
        lr (float): learning rate for outer optimization
        epsilon (float): perturbation norm ball radius
        inner_lr (float, optional): learning rate for inner optimization (default: 1)
        inner_iter (int, optional): iteration number for inner optimization (default: 1)
        base_optimizer (class, optional): base optimizer class (default: SGD)

    Example:
        >>> optimizer = AMP(model.parameters(), lr=0.1, eps=0.5, momentum=0.9)
        >>> def closure():
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                return outputs, loss
        >>> optimizer.step(closure)
    """

    def __init__(self, params, lr, epsilon, inner_lr=1, inner_iter=1, base_optimizer=SGD, **kwargs):
        if epsilon < 0.0:
            raise ValueError(f"Invalid epsilon: {epsilon}")
        if inner_lr < 0.0:
            raise ValueError(f"Invalid inner lr: {inner_lr}")
        if inner_iter < 0:
            raise ValueError(f"Invalid inner iter: {inner_iter}")
        defaults = dict(lr=lr, epsilon=epsilon, inner_lr=inner_lr, inner_iter=inner_iter, **kwargs)
        super(AMP, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, lr=lr, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def step(self, closure=None):
        if closure is None:
            raise ValueError("Adversarial model perturbation requires closure, but it was not provided")
        closure = torch.enable_grad()(closure)
        self.zero_grad()
        outputs, loss = closure()
        outputs, loss = outputs.detach(), loss.detach()
        for i in range(self.defaults['inner_iter']):
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        if i == 0:
                            self.state[p]['dev'] = 0
                        dev = self.state[p]['dev'] + group['inner_lr'] * p.grad
                        clip_coef = group['epsilon'] / (dev.norm() + 1e-12)
                        dev = clip_coef * dev if clip_coef < 1 else dev
                        p.sub_(self.state[p]['dev']).add_(dev) # update "theta" with "theta+delta"
                        self.state[p]['dev'] = dev
            self.zero_grad()
            closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.sub_(self.state[p]['dev']) # restore "theta" from "theta+delta"
        self.base_optimizer.step()
        return outputs, loss
