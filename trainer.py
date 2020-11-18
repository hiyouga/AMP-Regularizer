import torch
import torch.nn as nn
import numpy as np
from attacks import pgd
from loss_func import CrossEntropyLoss


class Trainer:

    def __init__(self, model, args):
        _train_functions = {
            'base': self._train_base,
            'mixup': self._train_mixup,
            'adv': self._train_adv,
            'rmp': self._train_rmp,
            'amp': self._train_amp
        }
        _init_perturbation_functions = {
            'base': self._init_perturbation_default,
            'mixup': self._init_perturbation_default,
            'adv': self._init_perturbation_default,
            'rmp': self._init_perturbation_default,
            'amp': self._init_perturbation_default
        }
        _reset_perturbation_functions = {
            'base': None,
            'mixup': None,
            'adv': None,
            'rmp': self._reset_perturbation_rmp,
            'amp': self._reset_perturbation_amp
        }
        _update_params_train_functions = {
            'base': self._update_params_default,
            'mixup': self._update_params_default,
            'adv': self._update_params_default,
            'rmp': self._update_params_train_amp,
            'amp': self._update_params_train_amp
        }
        _update_params_eval_functions = {
            'base': self._update_params_default,
            'mixup': self._update_params_default,
            'adv': self._update_params_default,
            'rmp': self._update_params_default,
            'amp': self._update_params_default
        }
        _update_delta_functions = {
            'base': None,
            'mixup': None,
            'adv': None,
            'rmp': None,
            'amp': self._update_delta_amp
        }
        _update_theta_functions = {
            'base': self._update_theta_default,
            'mixup': self._update_theta_default,
            'adv': self._update_theta_default,
            'rmp': self._update_theta_default,
            'amp': self._update_theta_default,
        }
        _clip_functions = {
            'l2': self._clip_l2,
            'linf': self._clip_linf
        }

        self._train = _train_functions[args.method]
        self._init_perturbation = _init_perturbation_functions[args.method]
        self._reset_perturbation = _reset_perturbation_functions[args.method]
        self._update_params_train = _update_params_train_functions[args.method]
        self._update_params_eval = _update_params_eval_functions[args.method]
        self._update_delta = _update_delta_functions[args.method]
        self._update_theta = _update_theta_functions[args.method]
        self._clip_func = _clip_functions[args.constrain]

        self.model = model
        self.criterion = CrossEntropyLoss(args.flooding, args.smoothing, args.num_classes)
        self._inner_iter = args.inner_iter
        self._epsilon = args.epsilon
        self._mixup_alpha = args.mixup_alpha
        self._clip_grad_norm = args.clip_norm
        self._adv_params = {
            'ei': args.adv_ei / 255.0,
            'eps': args.adv_eps / 255.0,
            'iter': args.adv_iter,
            'norm_type': args.adv_norm
        }
        self._init_perturbation()
        _original_params = filter(lambda p: p.requires_grad, self.model.original_params.parameters())
        _perturb_params = filter(lambda p: p.requires_grad, self.model.perturb_params.parameters())
        self.optimizer = torch.optim.SGD(_original_params,
                                         lr=args.lr,
                                         momentum=args.momentum,
                                         weight_decay=args.decay,
                                         nesterov=True)
        self.nested_optimizer = torch.optim.SGD(_perturb_params,
                                                lr=args.inner_lr,
                                                momentum=0,
                                                weight_decay=0,
                                                nesterov=False)

    def evaluate(self, inputs, targets):
        self._update_params_eval()
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
        return outputs, loss

    def train(self, inputs, targets):
        return self._train(inputs, targets)

    def _train_base(self, inputs, targets):
        self._update_params_train()
        self.optimizer.zero_grad()
        self.model.perturb_modules.zero_grad()
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
        loss.backward()
        self._update_theta()
        self.optimizer.step()
        return outputs, loss

    def _train_mixup(self, inputs, targets):
        self._update_params_train()
        self.optimizer.zero_grad()
        self.model.perturb_modules.zero_grad()
        lamda = self._get_lambda(self._mixup_alpha, batch_size=inputs.size(0))
        lamda = torch.tensor(lamda, dtype=inputs.dtype, device=inputs.device)
        indices = torch.randperm(inputs.size(0), device=inputs.device)
        outputs = self.model(inputs, lamda=lamda, indices=indices)
        loss = self.criterion(outputs, targets, lamda, indices)
        loss.backward()
        self._update_theta()
        self.optimizer.step()
        return outputs, loss

    def _train_adv(self, inputs, targets):
        self._update_params_train()
        self.optimizer.zero_grad()
        self.model.perturb_modules.zero_grad()
        inputs_adv = pgd(self, inputs, targets, self._adv_params, mode='train')
        outputs = self.model(inputs_adv)
        loss = self.criterion(outputs, targets)
        loss.backward()
        self._update_theta()
        self.optimizer.step()
        return outputs, loss

    def _train_rmp(self, inputs, targets):
        self._reset_perturbation()
        self._update_params_train()
        self.optimizer.zero_grad()
        self.model.perturb_modules.zero_grad()
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
        loss.backward()
        self._update_theta()
        self.optimizer.step()
        return outputs, loss

    def _train_amp(self, inputs, targets):
        self._reset_perturbation()
        for i in range(self._inner_iter):
            self._update_params_train()
            self.nested_optimizer.zero_grad()
            self.model.perturb_modules.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self._update_delta()
            self.nested_optimizer.step()
            self._clip_delta()
        self._update_params_train()
        self.optimizer.zero_grad()
        self.model.perturb_modules.zero_grad()
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
        loss.backward()
        self._update_theta()
        self.optimizer.step()
        return outputs, loss

    def adjust_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

    def save_state_dict(self):
        return self.model.state_dict()

    @staticmethod
    def _get_lambda(alpha, batch_size):
        if alpha[0] > 0 and alpha[1] > 0:
            lamda = np.random.beta(alpha[0], alpha[1], size=batch_size)
        else:
            lamda = 1.
        return lamda

    @staticmethod
    def _clip_l2(value, epsilon):
        clip_coef = epsilon / (value.norm() + 1e-6)
        if clip_coef < 1:
            value.mul_(clip_coef)

    @staticmethod
    def _clip_linf(value, epsilon):
        value.clamp_(-epsilon, epsilon)

    @torch.no_grad()
    def _clip_delta(self):
        for name, param in self.model.perturb_modules.named_parameters():
            prefix = name.split('.')[0]
            new_epsilon = self._epsilon * self.model.perturb_scale[prefix]
            name = name.replace('.', '_')
            self._clip_func(self.model.perturb_params[name].data, new_epsilon)

    def _init_perturbation_default(self):
        for name, param in self.model.perturb_modules.named_parameters():
            name = name.replace('.', '_')
            data_value = param.data.clone().detach()
            self.model.original_params[name] = nn.Parameter(data_value)
            self.model.perturb_params[name] = nn.Parameter(torch.zeros_like(data_value))

    def _reset_perturbation_rmp(self):
        for name, param in self.model.perturb_modules.named_parameters():
            prefix = name.split('.')[0]
            new_epsilon = self._epsilon * self.model.perturb_scale[prefix]
            bound = np.sqrt(new_epsilon / np.prod(param.shape)) # l2 norm
            name = name.replace('.', '_')
            torch.nn.init.uniform_(self.model.perturb_params[name], -bound, bound)

    def _reset_perturbation_amp(self):
        for param in self.model.perturb_params.values():
            torch.nn.init.zeros_(param)

    @torch.no_grad()
    def _update_params_default(self):
        for name, param in self.model.perturb_modules.named_parameters():
            name = name.replace('.', '_')
            new_param = self.model.original_params[name]
            setattr(param, 'data', new_param.data)

    @torch.no_grad()
    def _update_params_train_amp(self):
        for name, param in self.model.perturb_modules.named_parameters():
            name = name.replace('.', '_')
            new_param = self.model.original_params[name] + self.model.perturb_params[name]
            setattr(param, 'data', new_param.data)

    @torch.no_grad()
    def _update_delta_amp(self):
        for name, param in self.model.perturb_modules.named_parameters():
            name = name.replace('.', '_')
            grad_value = -1 * param.grad # gradient ascent
            setattr(self.model.perturb_params[name], 'grad', grad_value)

    @torch.no_grad()
    def _update_theta_default(self):
        for name, param in self.model.perturb_modules.named_parameters():
            name = name.replace('.', '_')
            grad_value = param.grad.clone()
            setattr(self.model.original_params[name], 'grad', grad_value)
        torch.nn.utils.clip_grad_norm_(self.model.original_params.parameters(), self._clip_grad_norm)
