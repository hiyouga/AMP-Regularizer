def mixup_process(inputs, lamda, indices):
    dim = len(inputs.shape) - 1
    shape = [-1] + [1] * dim
    lamda = lamda.view(*shape)
    mixed_inputs = lamda * inputs + (1 - lamda) * inputs[indices, :]
    return mixed_inputs
