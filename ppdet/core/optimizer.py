#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
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

import paddle.fluid.optimizer as optimizer
import paddle.fluid.layers as layers
import paddle.fluid.regularizer as regularizer

__all__ = ['OptimizerBuilder']


class OptimizerBuilder():
    def __init__(self, cfg):
        self.cfg = cfg
        self._build_optimizer()

    def get_optimizer(self):
        return self.optimizer

    def get_bn_regularizer(self):
        return self.bn_regularizer
    
    def _build_optimizer(self):
        self._build_regularizer()
        self._build_learning_rate()

        opt_params = dict()
        for k, v in self.cfg.items():
            if k == 'TYPE':
                opt_func = getattr(optimizer, v)
            elif not isinstance(v, dict):
                opt_params.update({k.lower(): v})

        # parse warmup
        warmup_cfg = self.cfg.LR_WARMUP
        if self.cfg.LR_WARMUP.WARMUP:
            warmup_steps = warmup_cfg.WARMUP_STEPS
            end_lr = warmup_cfg.WARMUP_END_LR
            start_lr = (warmup_cfg.WARMUP_INIT_FACTOR) * \
                       warmup_cfg.WARMUP_END_LR
            self.learning_rate = layers.linear_lr_warmup(
                                    learning_rate=self.learning_rate,
                                    warmup_steps=warmup_steps,
                                    start_lr=start_lr,
                                    end_lr=end_lr)

        self.optimizer = opt_func(regularization=self.regularizer,
                                  learning_rate=self.learning_rate,
                                  **opt_params)

    def _build_regularizer(self):
        reg_cfg = self.cfg.WEIGHT_DECAY
        reg_func = getattr(regularizer, reg_cfg.TYPE+"Decay")
        self.regularizer = reg_func(reg_cfg.FACTOR)
        self.bn_regularizer = reg_func(reg_cfg.BN_FACTOR)

    def _build_learning_rate(self):
        policy, params = self._parse_lr_decay_cfg()
        decay_func = getattr(layers, policy)
        self.learning_rate = decay_func(**params)
    
    def _parse_lr_decay_cfg(self):
        lr_cfg = self.cfg.LR_DECAY
        policy = lr_cfg.POLICY
        # parse decay params
        params = dict()
        for k, v in lr_cfg.items():
            if k != 'POLICY':
                params.update({k.lower(): v})
        return policy, params

if __name__ == "__main__":
    from config import load_cfg

    def test_opt_with_cfg(cfg_file):
        cfg = load_cfg(cfg_file)
        ob = OptimizerBuilder(cfg.OPTIMIZER)
        assert ob.get_optimizer() is not None
        assert ob.get_bn_regularizer() is not None

    test_opt_with_cfg('./config/faster-rcnn_ResNet-50.yml')
    test_opt_with_cfg('./config/yolov3_darknet53_syncbn.yml')
