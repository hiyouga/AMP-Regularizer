import torch
import numpy as np


def proj(x, x_backup, epsilon, norm_type):
    dev = (x - x_backup).contiguous()
    if norm_type == 'linf':
        dev.clamp_(-epsilon, epsilon)
        x = x_backup + dev
    elif norm_type == 'l2':
        dim = len(x.shape) - 1
        shape = [-1] + [1] * dim
        dev = dev.view(x.size(0), -1)
        dev_norm = dev.norm(dim=1, keepdim=True)
        mask = (dev_norm > epsilon).view(*shape)
        dev = (dev / (dev_norm + 1e-6)) * epsilon
        dev = dev.view(x.shape)
        x = (x_backup + dev) * mask.float() + x * (1 - mask.float())
    return x


def pgd(trainer, inputs, targets, adv_params, mode='eval'):
    ei = adv_params['ei']
    epsilon = adv_params['eps']
    norm_type = adv_params['norm_type']

    if mode == 'train':
        if norm_type == 'linf':
            bound = epsilon
        elif norm_type == 'l2':
            bound = np.sqrt(epsilon / np.prod(inputs[0].shape))
        rand_perturb = torch.zeros_like(inputs).float().uniform_(-bound, bound)
        inputs_adv = (inputs + rand_perturb).detach()
    elif mode == 'eval':
        inputs_adv = inputs.clone().detach()

    is_train = trainer.model.training
    trainer.model.eval()
    with torch.enable_grad():
        for i in range(adv_params['iter']):
            inputs_adv.requires_grad_(True)
            outputs, loss = trainer.evaluate(inputs_adv, targets)
            grads = torch.autograd.grad(loss, inputs_adv, retain_graph=False, only_inputs=True)[0]
            inputs_adv = inputs_adv + ei * torch.sign(grads.data)
            inputs_adv = proj(inputs_adv, inputs, epsilon, norm_type).detach()
    if is_train:
        trainer.model.train()
    else:
        trainer.model.eval()

    return inputs_adv
