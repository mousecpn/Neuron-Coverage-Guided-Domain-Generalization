import torch.optim as optim
import math
def get_optim_and_scheduler(network, epochs, lr, nesterov=False):
    params = network.parameters()
    optimizer = optim.SGD(params, weight_decay=.0005, momentum=.9, nesterov=nesterov, lr=lr)
    # optimizer = optim.Adam(params, lr=lr)
    step_size = int(epochs * .4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size)
    print("Step size: %d" % step_size)
    return optimizer, scheduler

def adjust_learning_rate(optimizer, epoch, epochs, lr):
    """Decay the learning rate based on schedule"""
    lr *= 0.5 * (1. + math.cos(math.pi * epoch / epochs))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr