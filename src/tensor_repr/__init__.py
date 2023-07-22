import torch
from cheap_repr import register_repr


@register_repr(torch.Tensor)
def repr_tensor(tensor: torch.Tensor, helper):
    rpr = f"""shape={tuple(tensor.shape)} dtype={str(tensor.dtype)[6:]}
device={tensor.device}
requires_grad={tensor.requires_grad} grad_fn={tensor.grad_fn}
count_nan={tensor.isnan().sum().item()} count_inf={tensor.isinf().sum().item()}""" # noqa

    tensor_min, tensor_max = tensor.min().item(), tensor.max().item()

    if tensor.dtype.is_floating_point:

        # 如果是浮点数，就将其四舍五入到小数点后5位
        tensor_max, tensor_min = round(tensor_max, 5), round(tensor_min, 5)

        # 如果是浮点数，还要计算一些统计量
        rpr += f''' mean={round(tensor.mean().item(), 5)}
std={round(tensor.std().item(), 5)}
25%={round(tensor.quantile(0.25).item(), 5)}
median={round(tensor.median().item(), 5)}
75%={round(tensor.quantile(0.75).item(), 5)}'''

    # 获取tensor的最大值和最小值
    rpr += f' min={tensor_min} max={tensor_max}'

    # 处理一下字符串格式
    rpr = rpr.replace('\n', ' ').replace(' ', ', ')
    rpr = f'torch.Tensor({rpr})'
    return rpr
