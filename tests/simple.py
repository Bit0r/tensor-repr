import snoop
import tensor_repr
import torch

del tensor_repr


@snoop
def myfunc(mask, x):
    y = torch.zeros(6, device='cuda')
    y.masked_scatter_(mask, x)
    return y


mask = torch.tensor([0, 1, 0, 1, 1, 0], device='cuda', dtype=torch.bool)
source = torch.tensor([1.0, 2.0, 3.0], device='cuda')
y = myfunc(mask, source)

print(repr(y))
