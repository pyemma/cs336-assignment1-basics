import math

import torch


class AdamW(torch.optim.Optimizer):

    def __init__(self, params, lr, weight_decay, betas, eps):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "betas": betas,
            "eps": eps,
        }
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                t = state.get("t", 0) + 1
                if "m" not in state:
                    state["m"] = torch.zeros_like(p.data)
                m = state.get("m")
                if "v" not in state:
                    state["v"] = torch.zeros_like(p.data)
                v = state.get("v")
                g = p.grad.data
                m = beta1 * m + (1 - beta1) * g
                v = beta2 * v + (1 - beta2) * (g ** 2)
                lr_t = lr * math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)
                p.data = p.data - lr_t * m / (v.sqrt() + eps)
                p.data = p.data - lr * weight_decay * p.data
                # stateful optimizer, store the m and v
                state["m"] = m
                state["v"] = v
                state["t"] = t
        return loss


def learning_rate_schedule(t, alpha_max, alpha_min, t_warmup, t_cosine):
    if t < t_warmup:
        return t / t_warmup * alpha_max
    elif t >= t_warmup and t < t_cosine:
        return alpha_min + (1 + math.cos((t - t_warmup) / (t_cosine - t_warmup) * math.pi)) * (alpha_max - alpha_min) / 2
    else:
        return alpha_min


def gradient_clipping(params, max_l2_norm):
    """
        This is computed over all gradients, not each gradients individually
    """
    l2_norm = 0.0
    for param in params:
        if param.grad is None:
            continue
        l2_norm += (param.grad.data ** 2).sum()
    if l2_norm.sqrt() > max_l2_norm:
        ratio =  max_l2_norm / (l2_norm.sqrt() + 1e-6)
        for param in params:
            if param.grad is None:
                continue
            param.grad.data = param.grad.data * ratio