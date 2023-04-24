from typing import Optional, Union

import torch
from torch import nn

from base.train import BaseTrainer


class Trainer(BaseTrainer):
    def __init__(self, model: nn.Module, loss_fn, data, base_lr: float, steps, lr_decay_ratio, log_dir: str, n_exp: int,
                 save_iter: int = 300, clip_grad_value: Optional[float] = None,
                 max_epochs: Optional[int] = 1000, patience: Optional[int] = 1000,
                 device: Optional[Union[torch.device, str]] = None, **args):
        super().__init__(model, loss_fn, data, base_lr, steps, lr_decay_ratio, log_dir, n_exp, save_iter,
                         clip_grad_value, max_epochs, patience, device)
