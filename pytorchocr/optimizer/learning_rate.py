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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, StepLR, MultiStepLR
from .lr_scheduler import CyclicalCosineDecay, OneCycleDecay, TwoStepCosineDecay


class Linear(object):
    """
    Linear learning rate decay
    Args:
        lr (float): The initial learning rate. It is a python float number.
        epochs(int): The decay step size. It determines the decay cycle.
        end_lr(float, optional): The minimum final learning rate. Default: 0.0001.
        power(float, optional): Power of polynomial. Default: 1.0.
        last_epoch (int, optional):  The index of last epoch. Can be set to restart training. Default: -1, means initial learning rate.
    """

    def __init__(
        self,
        learning_rate,
        epochs,
        step_each_epoch,
        end_lr=0.0,
        power=1.0,
        warmup_epoch=0,
        last_epoch=-1,
        **kwargs,
    ):
        super(Linear, self).__init__()
        self.learning_rate = learning_rate
        self.epochs = epochs * step_each_epoch
        self.end_lr = end_lr
        self.power = power
        self.last_epoch = last_epoch
        self.warmup_epoch = round(warmup_epoch * step_each_epoch)

    def __call__(self):
        # Return initial learning rate for optimizer creation
        if callable(self.learning_rate):
            return self.learning_rate()
        return self.learning_rate

    def get_lr_lambda(self):
        def lr_lambda(current_step):
            if current_step < self.warmup_epoch:
                return float(current_step) / float(max(1, self.warmup_epoch))
            else:
                return (1.0 - self.end_lr) * ((1.0 - float(current_step) / float(self.epochs)) ** self.power) + self.end_lr
        return lr_lambda


class Cosine(object):
    """
    Cosine learning rate decay
    lr = 0.05 * (math.cos(epoch * (math.pi / epochs)) + 1)
    Args:
        lr(float): initial learning rate
        step_each_epoch(int): steps each epoch
        epochs(int): total training epochs
        last_epoch (int, optional):  The index of last epoch. Can be set to restart training. Default: -1, means initial learning rate.
    """

    def __init__(
        self,
        learning_rate,
        step_each_epoch,
        epochs,
        warmup_epoch=0,
        last_epoch=-1,
        **kwargs,
    ):
        super(Cosine, self).__init__()
        self.learning_rate = learning_rate
        self.T_max = step_each_epoch * epochs
        self.last_epoch = last_epoch
        self.warmup_epoch = round(warmup_epoch * step_each_epoch)

    def __call__(self):
        def lr_lambda(current_step):
            if current_step < self.warmup_epoch:
                return float(current_step) / float(max(1, self.warmup_epoch))
            else:
                progress = float(current_step - self.warmup_epoch) / float(max(1, self.T_max - self.warmup_epoch))
                return 0.5 * (1.0 + math.cos(math.pi * progress))
                
        return lr_lambda


class LinearWarmupCosine(object):
    """
    LinearWarmupCosine learning rate decay
    Args:
        learning_rate(float): initial learning rate
        step_each_epoch(int): steps each epoch
        epochs(int): total training epochs
        start_lr (float): Initial learning rate of warm up.
        min_lr (float): Minimum learning rate in CosineAnnealingDecay.
        last_epoch (int, optional):  The index of last epoch. Can be set to restart training. Default: -1, means initial learning rate.
    """

    def __init__(
        self,
        learning_rate,
        step_each_epoch,
        epochs,
        warmup_steps=5000,
        start_lr=1e-5,
        min_lr=1e-8,
        last_epoch=-1,
        **kwargs,
    ):
        super(LinearWarmupCosine, self).__init__()
        self.learning_rate = float(learning_rate)
        self.T_max = step_each_epoch * epochs
        self.last_epoch = last_epoch
        self.warmup_steps = warmup_steps
        self.start_lr = float(start_lr)
        self.min_lr = float(min_lr)

    def __call__(self):
        learning_rate = CosineAnnealingLR(
            optimizer=None,
            T_max=self.T_max,
            eta_min=self.min_lr,
            last_epoch=self.last_epoch,
        )
        if self.warmup_steps > 0:
            learning_rate = LambdaLR(
                optimizer=None,
                lr_lambda=lambda current_step: self.start_lr + (self.learning_rate - self.start_lr) * (current_step / self.warmup_steps),
                last_epoch=self.last_epoch,
            )
        return learning_rate


class Step(object):
    """
    Piecewise learning rate decay
    Args:
        step_each_epoch(int): steps each epoch
        learning_rate (float): The initial learning rate. It is a python float number.
        step_size (int): the interval to update.
        gamma (float, optional): The Ratio that the learning rate will be reduced. ``new_lr = origin_lr * gamma`` .
            It should be less than 1.0. Default: 0.1.
        last_epoch (int, optional):  The index of last epoch. Can be set to restart training. Default: -1, means initial learning rate.
    """

    def __init__(
        self,
        learning_rate,
        step_size,
        step_each_epoch,
        gamma=0.1,
        warmup_epoch=0,
        last_epoch=-1,
        **kwargs,
    ):
        super(Step, self).__init__()
        self.step_size = step_each_epoch * step_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.last_epoch = last_epoch
        self.warmup_epoch = round(warmup_epoch * step_each_epoch)

    def __call__(self):
        def lr_lambda(current_step):
            if current_step < self.warmup_epoch:
                return float(current_step) / float(max(1, self.warmup_epoch))
            else:
                return self.gamma ** (current_step // self.step_size)
                
        return lr_lambda


class Piecewise(object):
    """
    Piecewise learning rate decay
    Args:
        boundaries(list): A list of steps numbers. The type of element in the list is python int.
        values(list): A list of learning rate values that will be picked during different epoch boundaries.
            The type of element in the list is python float.
        last_epoch (int, optional):  The index of last epoch. Can be set to restart training. Default: -1, means initial learning rate.
    """

    def __init__(
        self,
        step_each_epoch,
        decay_epochs,
        values,
        warmup_epoch=0,
        last_epoch=-1,
        **kwargs,
    ):
        super(Piecewise, self).__init__()
        self.boundaries = [step_each_epoch * e for e in decay_epochs]
        self.values = values
        self.last_epoch = last_epoch
        self.warmup_epoch = round(warmup_epoch * step_each_epoch)

    def __call__(self):
        learning_rate = LambdaLR(
            optimizer=None,
            lr_lambda=lambda current_step: self.values[bisect.bisect_right(self.boundaries, current_step) - 1],
            last_epoch=self.last_epoch,
        )
        if self.warmup_epoch > 0:
            learning_rate = LambdaLR(
                optimizer=None,
                lr_lambda=lambda current_step: self.values[0] + (self.values[0] - self.values[1]) * (current_step / self.warmup_epoch),
                last_epoch=self.last_epoch,
            )
        return learning_rate


class CyclicalCosine(object):
    """
    Cyclical cosine learning rate decay
    Args:
        learning_rate(float): initial learning rate
        step_each_epoch(int): steps each epoch
        epochs(int): total training epochs
        cycle(int): period of the cosine learning rate
        last_epoch (int, optional):  The index of last epoch. Can be set to restart training. Default: -1, means initial learning rate.
    """

    def __init__(
        self,
        learning_rate,
        step_each_epoch,
        epochs,
        cycle,
        warmup_epoch=0,
        last_epoch=-1,
        **kwargs,
    ):
        super(CyclicalCosine, self).__init__()
        self.learning_rate = learning_rate
        self.T_max = step_each_epoch * epochs
        self.last_epoch = last_epoch
        self.warmup_epoch = round(warmup_epoch * step_each_epoch)
        self.cycle = round(cycle * step_each_epoch)

    def __call__(self):
        learning_rate = CyclicalCosineDecay(
            learning_rate=self.learning_rate,
            T_max=self.T_max,
            cycle=self.cycle,
            last_epoch=self.last_epoch,
        )
        if self.warmup_epoch > 0:
            learning_rate = LambdaLR(
                optimizer=None,
                lr_lambda=lambda current_step: self.learning_rate + (self.learning_rate - learning_rate) * (current_step / self.warmup_epoch),
                last_epoch=self.last_epoch,
            )
        return learning_rate


class OneCycle(object):
    """
    One Cycle learning rate decay
    Args:
        max_lr(float): Upper learning rate boundaries
        epochs(int): total training epochs
        step_each_epoch(int): steps each epoch
        anneal_strategy(str): {‘cos’, ‘linear’} Specifies the annealing strategy: “cos” for cosine annealing, “linear” for linear annealing.
            Default: ‘cos’
        three_phase(bool): If True, use a third phase of the schedule to annihilate the learning rate according to ‘final_div_factor’
            instead of modifying the second phase (the first two phases will be symmetrical about the step indicated by ‘pct_start’).
        last_epoch (int, optional):  The index of last epoch. Can be set to restart training. Default: -1, means initial learning rate.
    """

    def __init__(
        self,
        max_lr,
        epochs,
        step_each_epoch,
        anneal_strategy="cos",
        three_phase=False,
        warmup_epoch=0,
        last_epoch=-1,
        **kwargs,
    ):
        super(OneCycle, self).__init__()
        self.max_lr = max_lr
        self.epochs = epochs
        self.steps_per_epoch = step_each_epoch
        self.anneal_strategy = anneal_strategy
        self.three_phase = three_phase
        self.last_epoch = last_epoch
        self.warmup_epoch = round(warmup_epoch * step_each_epoch)

    def __call__(self):
        learning_rate = OneCycleDecay(
            max_lr=self.max_lr,
            epochs=self.epochs,
            steps_per_epoch=self.steps_per_epoch,
            anneal_strategy=self.anneal_strategy,
            three_phase=self.three_phase,
            last_epoch=self.last_epoch,
        )
        if self.warmup_epoch > 0:
            learning_rate = LambdaLR(
                optimizer=None,
                lr_lambda=lambda current_step: self.max_lr + (self.max_lr - learning_rate) * (current_step / self.warmup_epoch),
                last_epoch=self.last_epoch,
            )
        return learning_rate


class Const(object):
    """
    Const learning rate decay
    Args:
        learning_rate(float): initial learning rate
        step_each_epoch(int): steps each epoch
        last_epoch (int, optional):  The index of last epoch. Can be set to restart training. Default: -1, means initial learning rate.
    """

    def __init__(
        self, learning_rate, step_each_epoch, warmup_epoch=0, last_epoch=-1, **kwargs
    ):
        super(Const, self).__init__()
        self.learning_rate = learning_rate
        self.last_epoch = last_epoch
        self.warmup_epoch = round(warmup_epoch * step_each_epoch)

    def __call__(self):
        def lr_lambda(current_step):
            if current_step < self.warmup_epoch:
                return float(current_step) / float(max(1, self.warmup_epoch))
            return 1.0
            
        return lr_lambda


class DecayLearningRate(object):
    """
    DecayLearningRate learning rate decay
    new_lr = (lr - end_lr) * (1 - epoch/decay_steps)**power + end_lr
    Args:
        learning_rate(float): initial learning rate
        step_each_epoch(int): steps each epoch
        epochs(int): total training epochs
        factor(float): Power of polynomial, should greater than 0.0 to get learning rate decay. Default: 0.9
        end_lr(float): The minimum final learning rate. Default: 0.0.
    """

    def __init__(
        self, learning_rate, step_each_epoch, epochs, factor=0.9, end_lr=0, **kwargs
    ):
        super(DecayLearningRate, self).__init__()
        self.learning_rate = learning_rate
        self.epochs = epochs + 1
        self.factor = factor
        self.end_lr = 0
        self.decay_steps = step_each_epoch * epochs

    def __call__(self):
        def lr_lambda(current_step):
            if current_step < self.decay_steps:
                return (1.0 - self.end_lr) * ((1.0 - float(current_step) / float(self.decay_steps)) ** self.factor) + self.end_lr
            return self.end_lr
            
        return lr_lambda


class MultiStepDecay(object):
    """
    Piecewise learning rate decay
    Args:
        step_each_epoch(int): steps each epoch
        learning_rate (float): The initial learning rate. It is a python float number.
        step_size (int): the interval to update.
        gamma (float, optional): The Ratio that the learning rate will be reduced. ``new_lr = origin_lr * gamma`` .
            It should be less than 1.0. Default: 0.1.
        last_epoch (int, optional):  The index of last epoch. Can be set to restart training. Default: -1, means initial learning rate.
    """

    def __init__(
        self,
        learning_rate,
        milestones,
        step_each_epoch,
        gamma,
        warmup_epoch=0,
        last_epoch=-1,
        **kwargs,
    ):
        super(MultiStepDecay, self).__init__()
        self.milestones = [step_each_epoch * e for e in milestones]
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.last_epoch = last_epoch
        self.warmup_epoch = round(warmup_epoch * step_each_epoch)

    def __call__(self):
        learning_rate = MultiStepLR(
            optimizer=None,
            milestones=self.milestones,
            gamma=self.gamma,
            last_epoch=self.last_epoch,
        )
        if self.warmup_epoch > 0:
            learning_rate = LambdaLR(
                optimizer=None,
                lr_lambda=lambda current_step: self.learning_rate * (self.gamma ** (current_step // self.warmup_epoch)),
                last_epoch=self.last_epoch,
            )
        return learning_rate


class TwoStepCosine(object):
    """
    Cosine learning rate decay
    lr = 0.05 * (math.cos(epoch * (math.pi / epochs)) + 1)
    Args:
        lr(float): initial learning rate
        step_each_epoch(int): steps each epoch
        epochs(int): total training epochs
        last_epoch (int, optional):  The index of last epoch. Can be set to restart training. Default: -1, means initial learning rate.
    """

    def __init__(
        self,
        learning_rate,
        step_each_epoch,
        epochs,
        warmup_epoch=0,
        last_epoch=-1,
        **kwargs,
    ):
        super(TwoStepCosine, self).__init__()
        self.learning_rate = learning_rate
        self.T_max1 = step_each_epoch * 200
        self.T_max2 = step_each_epoch * epochs
        self.last_epoch = last_epoch
        self.warmup_epoch = round(warmup_epoch * step_each_epoch)

    def __call__(self):
        learning_rate = TwoStepCosineDecay(
            learning_rate=self.learning_rate,
            T_max1=self.T_max1,
            T_max2=self.T_max2,
            last_epoch=self.last_epoch,
        )
        if self.warmup_epoch > 0:
            learning_rate = LambdaLR(
                optimizer=None,
                lr_lambda=lambda current_step: self.learning_rate + (self.learning_rate - learning_rate) * (current_step / self.warmup_epoch),
                last_epoch=self.last_epoch,
            )
        return learning_rate
