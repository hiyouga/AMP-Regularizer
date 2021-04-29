import torch
from torch.optim import Optimizer, SGD


class AMP(Optimizer):
    """
    Implements adversarial model perturbation.

    Args:
        params (iterable): iterable of trainable parameters
        lr (float): learning rate for outer optimization
        epsilon (float): perturbation norm ball radius
        inner_lr (float, optional): learning rate for inner optimization (default: 1)
        inner_iter (int, optional): iteration number for inner optimization (default: 1)
        base_optimizer (class, optional): basic optimizer class (default: SGD)
        **kwargs: keyword arguments passed to the `__init__` method of `base_optimizer`

    Example:
        >>> optimizer = AMP(model.parameters(), lr=0.1, eps=0.5, momentum=0.9)
        >>> for inputs, targets in dataset:
        >>>     def closure():
        >>>         optimizer.zero_grad()
        >>>         outputs = model(inputs)
        >>>         loss = loss_fn(outputs, targets)
        >>>         loss.backward()
        >>>         return outputs, loss
        >>>     outputs, loss = optimizer.step(closure)
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
            raise ValueError('Adversarial model perturbation requires closure, but it was not provided')
        closure = torch.enable_grad()(closure)
        outputs, loss = map(lambda x: x.detach(), closure())
        for i in range(self.defaults['inner_iter']):
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        if i == 0:
                            self.state[p]['dev'] = torch.zeros_like(p.grad)
                        dev = self.state[p]['dev'] + group['inner_lr'] * p.grad
                        clip_coef = group['epsilon'] / (dev.norm() + 1e-12)
                        dev = clip_coef * dev if clip_coef < 1 else dev
                        p.sub_(self.state[p]['dev']).add_(dev) # update "theta" with "theta+delta"
                        self.state[p]['dev'] = dev
            closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.sub_(self.state[p]['dev']) # restore "theta" from "theta+delta"
        self.base_optimizer.step()
        return outputs, loss
