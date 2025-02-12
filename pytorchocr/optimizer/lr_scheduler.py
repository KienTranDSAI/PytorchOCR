# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import torch
from torch.optim.lr_scheduler import _LRScheduler


class CyclicalCosineDecay(_LRScheduler):
    def __init__(self, optimizer, T_max, cycle=1, eta_min=0.0, last_epoch=-1, verbose=False):
        """
        Cyclical cosine learning rate decay
        Args:
            optimizer (Optimizer): Wrapped optimizer.
            T_max (int): Maximum number of iterations.
            cycle (int): Number of cycles
            eta_min (float): Minimum learning rate. Default: 0.
            last_epoch (int): The index of last epoch. Default: -1.
            verbose (bool): If ``True``, prints a message to stdout for each update. Default: ``False``.
        """
        self.T_max = T_max
        self.cycle = cycle
        self.eta_min = eta_min
        super(CyclicalCosineDecay, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch == 0:
            return [group['lr'] for group in self.optimizer.param_groups]
        
        relative_epoch = self.last_epoch % self.cycle
        lr = [self.eta_min + 0.5 * (base_lr - self.eta_min) * 
              (1 + math.cos(math.pi * relative_epoch / self.cycle))
              for base_lr in self.base_lrs]
        return lr


class OneCycleDecay(_LRScheduler):
    def __init__(self, optimizer, max_lr, epochs, steps_per_epoch, pct_start=0.3,
                 anneal_strategy='cos', div_factor=25.0, final_div_factor=1e4,
                 three_phase=False, last_epoch=-1, verbose=False):
        """
        OneCycle learning rate scheduler
        Args:
            optimizer (Optimizer): Wrapped optimizer.
            max_lr (float or list): Upper learning rate boundaries.
            epochs (int): Number of epochs to train for.
            steps_per_epoch (int): Number of steps per epoch.
            pct_start (float): Percentage of cycle spent increasing lr. Default: 0.3
            anneal_strategy (str): {'cos', 'linear'} Specifies the annealing strategy. Default: 'cos'
            div_factor (float): Determines the initial learning rate. Default: 25.0
            final_div_factor (float): Determines the minimum learning rate. Default: 1e4
            three_phase (bool): If True, use a third phase to annihilate the learning rate. Default: False
            last_epoch (int): The index of last epoch. Default: -1
            verbose (bool): If True, prints a message to stdout for each update. Default: False
        """
        self.total_steps = epochs * steps_per_epoch
        self.max_lr = max_lr
        self.pct_start = pct_start
        self.anneal_strategy = anneal_strategy
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor
        self.three_phase = three_phase
        
        super(OneCycleDecay, self).__init__(optimizer, last_epoch, verbose)
        
        # Initialize step sizes and learning rates
        self.step_size_up = int(self.total_steps * self.pct_start)
        self.step_size_down = self.total_steps - self.step_size_up
        
        # Calculate initial and final learning rates
        self.initial_lr = self.max_lr / self.div_factor
        self.min_lr = self.initial_lr / self.final_div_factor

    def _annealing_cos(self, start, end, pct):
        "Cosine anneal from `start` to `end` as pct goes from 0.0 to 1.0."
        cos_out = math.cos(math.pi * pct) + 1
        return end + (start - end) / 2.0 * cos_out

    def _annealing_linear(self, start, end, pct):
        "Linearly anneal from `start` to `end` as pct goes from 0.0 to 1.0."
        return (end - start) * pct + start

    def get_lr(self):
        if self.last_epoch >= self.total_steps:
            return [self.min_lr for _ in self.base_lrs]
            
        step_num = self.last_epoch
        
        if step_num <= self.step_size_up:
            # Increasing phase
            pct = step_num / self.step_size_up
            if self.anneal_strategy == 'cos':
                lr = [self._annealing_cos(self.initial_lr, self.max_lr, pct) 
                      for _ in self.base_lrs]
            else:
                lr = [self._annealing_linear(self.initial_lr, self.max_lr, pct) 
                      for _ in self.base_lrs]
        else:
            # Decreasing phase
            pct = (step_num - self.step_size_up) / self.step_size_down
            if self.anneal_strategy == 'cos':
                lr = [self._annealing_cos(self.max_lr, self.min_lr, pct) 
                      for _ in self.base_lrs]
            else:
                lr = [self._annealing_linear(self.max_lr, self.min_lr, pct) 
                      for _ in self.base_lrs]
                
        return lr


class TwoStepCosineDecay(_LRScheduler):
    def __init__(self, optimizer, T_max1, T_max2, eta_min=0, last_epoch=-1, verbose=False):
        """
        Two-step cosine learning rate decay
        Args:
            optimizer (Optimizer): Wrapped optimizer.
            T_max1 (int): First maximum number of iterations.
            T_max2 (int): Second maximum number of iterations.
            eta_min (float): Minimum learning rate. Default: 0.
            last_epoch (int): The index of last epoch. Default: -1.
            verbose (bool): If True, prints a message to stdout for each update. Default: False.
        """
        self.T_max1 = T_max1
        self.T_max2 = T_max2
        self.eta_min = eta_min
        super(TwoStepCosineDecay, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch <= self.T_max1:
            # First cosine decay phase
            return [self.eta_min + (base_lr - self.eta_min) *
                    (1 + math.cos(math.pi * self.last_epoch / self.T_max1)) / 2
                    for base_lr in self.base_lrs]
        else:
            # Second cosine decay phase
            return [self.eta_min + (base_lr - self.eta_min) *
                    (1 + math.cos(math.pi * self.last_epoch / self.T_max2)) / 2
                    for base_lr in self.base_lrs]
