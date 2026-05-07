import numpy as np
def op_copy(optimizer):
    for pg in optimizer.param_groups:
        pg['lr0'] = pg['lr']
    return optimizer
def target_model_lr_scheduler(optimizer, iter_num: int, max_iter: int, gamma: float = 10, power: float = 0.75):
    decay = (1.0 + gamma * iter_num / max_iter) ** (-power)
    for pg in optimizer.param_groups:
        pg['lr'] = pg['lr0'] * decay
    return optimizer
def target_model_flat_cosine_scheduler(
    optimizer,
    iter_num: int,
    max_iter: int,
    decay_start_ratio: float = 0.3,
    lr_floor_ratio: float = 0.0,
):
    start_ratio = min(max(decay_start_ratio, 0.0), 0.999999)
    floor_ratio = min(max(lr_floor_ratio, 0.0), 1.0)
    progress = iter_num / max(max_iter, 1)
    if progress <= start_ratio:
        cos_decay = 1.0
    else:
        tail_progress = (progress - start_ratio) / max(1.0 - start_ratio, 1e-12)
        cos_decay = 0.5 * (1.0 + np.cos(np.pi * tail_progress))
    decay = floor_ratio + (1.0 - floor_ratio) * cos_decay
    for pg in optimizer.param_groups:
        pg['lr'] = pg['lr0'] * decay
    return optimizer
def peer_model_lr_scheduler(
    optimizer,
    iter_num: int,
    max_iter: int,
    lr_floor: float = 0.0,
):
    cos_decay = 0.5 * (1.0 + np.cos(np.pi * iter_num / max_iter))
    for pg in optimizer.param_groups:
        group_floor = min(lr_floor, pg['lr0'])
        pg['lr'] = max(pg['lr0'] * cos_decay, group_floor)
    return optimizer