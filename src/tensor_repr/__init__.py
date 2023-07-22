from collections import OrderedDict

import torch
from cheap_repr import ReprHelper, register_repr


@register_repr(torch.Tensor)
def repr_tensor(tensor: torch.Tensor, helper: ReprHelper):
    properties = get_properties(tensor)
    # 将properties显示为key=value的形式
    properties = ', '.join(f'{key}={value}'
                           for key, value in properties.items())
    rpr = f'torch.Tensor({properties})'
    return rpr


# 获取tensor的属性，比如mean, std, min, max等
def get_properties(tensor: torch.Tensor):
    properties = OrderedDict([
        ('shape', tuple(tensor.shape)),
        ('dtype', str(tensor.dtype)[6:]),
        ('device', tensor.device),
        ('requires_grad', tensor.requires_grad),
        ('grad_fn', tensor.grad_fn),
        ('count_nan', tensor.isnan().sum().item()),
        ('count_inf', tensor.isinf().sum().item()),
    ])

    try:
        properties['min'] = tensor.min().item()
        properties['max'] = tensor.max().item()
        properties['median'] = tensor.median().item()
        properties['mean'] = tensor.mean().item()
        properties['std'] = tensor.std().item()
        properties['25%'] = tensor.quantile(0.25).item()
        properties['75%'] = tensor.quantile(0.75).item()
    except RuntimeError:
        pass

    for key, value in properties.items():
        if isinstance(value, (float, complex)):
            properties[key] = round(value, 5)

    return properties
