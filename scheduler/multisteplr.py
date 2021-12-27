import torch.optim as optim


def get_scheduler():
    scheduler_kwargs = dict(milestones=[x * 5 for x in range(1000)], gamma=0.99)
    scheduler = optim.lr_scheduler.MultiStepLR
    # scheduler = "MultiStepLR"
    return scheduler, scheduler_kwargs
