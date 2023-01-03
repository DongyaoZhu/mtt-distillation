import torch
import numpy as np
from copy import deepcopy
from typing import TYPE_CHECKING

import torch
from tqdm import tqdm
import wandb
import numpy as np
from torch.autograd import Variable

import pickle
from label_smooth import LabelSmoothingCrossEntropy

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def train(
    model: "nn.Module",
    mask: "Masking",
    train_loader: "DataLoader",
    optimizer: "optim",
    lr_scheduler: "lr_scheduler",
    global_step: int,
    epoch: int,
    device: torch.device,
    warmup_epochs,
    label_smoothing: float = 0.0,
    log_interval: int = 100,
    use_wandb: bool = False,
    masking_apply_when: str = "epoch_end",
    masking_interval: int = 1,
    masking_end_when: int = -1,
    masking_print_FLOPs: bool = False,
) -> "Union[float,int]":
    assert masking_apply_when in ["step_end", "epoch_end"]
    model.train()
    _mask_update_counter = 0
    # _loss_collector = SmoothenValue()
    pbar = tqdm(total=len(train_loader), dynamic_ncols=True)
    smooth_CE = LabelSmoothingCrossEntropy(label_smoothing)

    alpha = 1.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        data, targets_a, targets_b, lam = mixup_data(data, target, alpha, use_cuda=True)
        data, targets_a, targets_b = map(Variable, (data, targets_a, targets_b))

        optimizer.zero_grad()

        output = model(data)
        loss = mixup_criterion(smooth_CE, output, targets_a, targets_b, lam)
        #loss = smooth_CE(output, target)
        loss.backward()
        # L2 Regularization

        # Exp avg collection
        # _loss_collector.add_value(loss.item())

        # Mask the gradient step
        stepper = mask if mask else optimizer
        if (
            mask
            and masking_apply_when == "step_end"
            and global_step < masking_end_when
            and ((global_step + 1) % masking_interval) == 0
        ):
            mask.update_connections()
            _mask_update_counter += 1
        else:
            stepper.step()

        # Lr scheduler
        lr_scheduler.step()
        pbar.update(1)
        global_step += 1

    return global_step
    # return _loss_collector.smooth, global_step