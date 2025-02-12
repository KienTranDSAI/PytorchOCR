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

import torch.optim as optim


class Momentum(object):
    """
    Simple Momentum optimizer with velocity state.
    Args:
        learning_rate (float|Variable) - The learning rate used to update parameters.
        momentum (float) - Momentum factor.
        weight_decay (float, optional) - The weight decay coefficient.
    """
    def __init__(self, learning_rate, momentum, weight_decay=None, grad_clip=None, **args):
        super(Momentum, self).__init__()
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip

    def __call__(self, model):
        train_params = [param for param in model.parameters() if param.requires_grad]
        optimizer = optim.SGD(
            train_params,
            lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay if self.weight_decay is not None else 0.0
        )
        return optimizer


class Adam(object):
    def __init__(self,
                 learning_rate=0.001,
                 beta1=0.9,
                 beta2=0.999,
                 epsilon=1e-08,
                 parameter_list=None,
                 weight_decay=None,
                 grad_clip=None,
                 name=None,
                 lazy_mode=False,
                 **kwargs):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.parameter_list = parameter_list
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self.group_lr = kwargs.get('group_lr', False)
        self.training_step = kwargs.get('training_step', None)

    def __call__(self, model):
        if self.group_lr:
            if self.training_step == "LF_2":
                if hasattr(model, 'module'):  # For DataParallel
                    mlm = model.module.head.MLM_VRM.MLM.parameters()
                    pre_mlm_pp = model.module.head.MLM_VRM.Prediction.pp_share.parameters()
                    pre_mlm_w = model.module.head.MLM_VRM.Prediction.w_share.parameters()
                else:
                    mlm = model.head.MLM_VRM.MLM.parameters()
                    pre_mlm_pp = model.head.MLM_VRM.Prediction.pp_share.parameters()
                    pre_mlm_w = model.head.MLM_VRM.Prediction.w_share.parameters()

                total = []
                for param in mlm:
                    total.append(id(param))
                for param in pre_mlm_pp:
                    total.append(id(param))
                for param in pre_mlm_w:
                    total.append(id(param))

                group_base_params = [param for param in model.parameters() if id(param) in total]
                group_small_params = [param for param in model.parameters() if id(param) not in total]
                train_params = [
                    {'params': group_base_params},
                    {'params': group_small_params, 'lr': self.learning_rate * 0.1}
                ]
            else:
                print("group lr currently only support VisionLAN in LF_2 training step")
                train_params = [param for param in model.parameters() if param.requires_grad]
        else:
            train_params = [param for param in model.parameters() if param.requires_grad]

        # Get initial learning rate value if it's a callable
        initial_lr = self.learning_rate() if callable(self.learning_rate) else self.learning_rate

        optimizer = optim.Adam(
            train_params,
            lr=initial_lr,  # Use the initial learning rate value
            betas=(self.beta1, self.beta2),
            eps=self.epsilon,
            weight_decay=self.weight_decay if self.weight_decay is not None else 0.0
        )
        return optimizer


class RMSProp(object):
    """
    Root Mean Squared Propagation (RMSProp) optimizer.
    """
    def __init__(self,
                 learning_rate,
                 momentum=0.0,
                 rho=0.95,
                 epsilon=1e-6,
                 weight_decay=None,
                 grad_clip=None,
                 **args):
        super(RMSProp, self).__init__()
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.rho = rho
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip

    def __call__(self, model):
        train_params = [param for param in model.parameters() if param.requires_grad]
        optimizer = optim.RMSprop(
            train_params,
            lr=self.learning_rate,
            momentum=self.momentum,
            alpha=self.rho,
            eps=self.epsilon,
            weight_decay=self.weight_decay if self.weight_decay is not None else 0.0
        )
        return optimizer


class AdamW(object):
    def __init__(self,
                 learning_rate=0.001,
                 beta1=0.9,
                 beta2=0.999,
                 epsilon=1e-8,
                 weight_decay=0.01,
                 multi_precision=False,
                 grad_clip=None,
                 no_weight_decay_name=None,
                 one_dim_param_no_weight_decay=False,
                 **args):
        super().__init__()
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self.no_weight_decay_name_list = no_weight_decay_name.split() if no_weight_decay_name else []
        self.one_dim_param_no_weight_decay = one_dim_param_no_weight_decay

    def __call__(self, model):
        # Filter parameters that don't require weight decay
        decay_parameters = []
        no_decay_parameters = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            if (len(param.shape) == 1 and self.one_dim_param_no_weight_decay) or \
               any(nd in name for nd in self.no_weight_decay_name_list):
                no_decay_parameters.append(param)
            else:
                decay_parameters.append(param)

        optimizer_grouped_parameters = [
            {'params': decay_parameters, 'weight_decay': self.weight_decay},
            {'params': no_decay_parameters, 'weight_decay': 0.0}
        ]

        optimizer = optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.learning_rate,
            betas=(self.beta1, self.beta2),
            eps=self.epsilon
        )
        return optimizer
