import torch


def save_checkpoint(model, optimizer, iteration, out):
    obj = {}
    obj['model_state'] = model.state_dict()
    obj['optimizer_state'] = optimizer.state_dict()
    obj['iteration'] = iteration
    torch.save(obj, out)


def load_checkpoint(src, model, optimizer):
    obj = torch.load(src)
    model.load_state_dict(obj['model_state'])
    optimizer.load_state_dict(obj['optimizer_state'])
    return obj['iteration']