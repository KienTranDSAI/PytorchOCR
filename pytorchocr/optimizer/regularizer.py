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


class L1Decay(object):
    """
    L1 Weight Decay Regularization
    Args:
        factor (float): regularization coefficient. Default: 0.0
    """
    def __init__(self, factor=0.0):
        super(L1Decay, self).__init__()
        self.factor = factor

    def __call__(self):
        return self.factor


class L2Decay(object):
    """
    L2 Weight Decay Regularization
    Args:
        factor (float): regularization coefficient. Default: 0.0
    """
    def __init__(self, factor=0.0):
        super(L2Decay, self).__init__()
        self.factor = float(factor)

    def __call__(self):
        return self.factor
