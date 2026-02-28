import os
import math
import numpy as np
import random
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR


from clip import clip

def get_transform(cfg):
    return clip._transform(cfg.input_size[0])


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cal_MTIL_metrics(acc_list):
    acc_list = np.array(acc_list)
    acc_list *= 100
    avg = acc_list.mean(axis=0)
    last = np.array(acc_list[-1, :])
    transfer = np.array([np.mean([acc_list[j, i] for j in range(i)]) for i in range(1, acc_list.shape[1])])
    g = lambda x: np.around(x.mean(), decimals=1) if len(x) > 0 else -1
    f = lambda x: [np.around(i, decimals=1) for i in x]
    return {"transfer": {"transfer": f(transfer)}, "avg": {"avg": f(avg)}, "last": {"last": f(last)}, 
            "results_mean": {"transfer": g(transfer), "avg": g(avg), "last": g(last)}}

class AverageMeter:
    """Compute and store the average and current value.

    Examples::
        >>> # 1. Initialize a meter to record loss
        >>> losses = AverageMeter()
        >>> # 2. Update meter after every mini-batch update
        >>> losses.update(loss_value, batch_size)
    """

    def __init__(self, ema=False):
        """
        Args:
            ema (bool, optional): apply exponential moving average.
        """
        self.ema = ema
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if isinstance(val, torch.Tensor):
            val = val.item()

        self.val = val
        self.sum += val * n
        self.count += n

        if self.ema:
            self.avg = val if self.count == n else self.avg * 0.9 + self.val * 0.1
        else:
            self.avg = self.sum / self.count


def low_rank_approx(weight, alpha, device=None):
    original_device = weight.device
    original_dtype = weight.dtype
    
    if device is not None:
        compute_device = torch.device(f'cuda:{device}' if isinstance(device, int) else device)
        weight = weight.to(device=compute_device, dtype=torch.float32)
    else:
        weight = weight.to(dtype=torch.float32)

    U, S, Vh = torch.linalg.svd(weight, full_matrices=False)

    total_sum = S.sum()
    cumsum = S.cumsum(dim=0)
    r = (cumsum / total_sum >= alpha).nonzero(as_tuple=True)[0][0].item() + 1

    print(r, end="")

    U_r = U[:, :r]
    S_r = S[:r]
    Vh_r = Vh[:r, :]
    weight_r = torch.matmul(U_r, torch.matmul(torch.diag(S_r), Vh_r))

    return weight_r.to(device=original_device, dtype=original_dtype)
    
    