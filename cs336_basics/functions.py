import torch

# def cross_entropy(inputs: torch.Tensor, targets: torch.Tensor):
#     b = inputs.shape[0]
#     o, _ = inputs.max(dim=-1, keepdim=True)
#     x = inputs - o  # sub max for numeric stability
#     # here we still keep the exp and log, but they should be able to be converted to cum multi
#     # however, there might be numeric stability concern using multi instead of addition
#     s = x.exp().sum(dim=-1, keepdim=True).log()
#     y = x[torch.arange(b), targets].view(-1, 1) - s
#     return -y.mean()

def cross_entropy(inputs: torch.Tensor, targets: torch.Tensor):
    # reshape the inputs to collapse all the dimensions except the last one
    inputs = inputs.view(-1, inputs.shape[-1])
    o, _ = inputs.max(dim=-1, keepdim=True)
    x = inputs - o  # sub max for numeric stability
    # here we still keep the exp and log, but they should be able to be converted to cum multi
    # however, there might be numeric stability concern using multi instead of addition
    s = x.exp().sum(dim=-1, keepdim=True).log()
    t = targets.flatten()
    y = x[torch.arange(t.shape[0]), t].view(-1, 1) - s
    return -y.mean()