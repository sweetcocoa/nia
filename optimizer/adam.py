from torch.optim import Adam


def get_optimizer(*pargs, **kwargs):
    return Adam(*pargs, **kwargs)
