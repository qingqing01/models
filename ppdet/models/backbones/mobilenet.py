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

import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.framework import Variable
from paddle.fluid.regularizer import L2Decay

from ..registry import Backbones
from .base import BackboneBase

__all__ = ['MobileNetV1Backbone']


class MobileNet(object):
    def __init__(self, scale, bn_decay=True):
        """
        Args:
            scale (float): the scale of groups number/ filter number
            bn_decay (bool): whether perform L2Decay in batch_norm
        """
        self.scale = scale
        self.bn_decay = bn_decay

    def _conv_norm(self,
                   input,
                   filter_size,
                   num_filters,
                   stride,
                   padding,
                   channels=None,
                   num_groups=1,
                   act='relu',
                   use_cudnn=True,
                   name=None):
        parameter_attr = ParamAttr(
            learning_rate=0.1,
            initializer=fluid.initializer.MSRA(),
            name=name + "_weights")
        conv = fluid.layers.conv2d(
            input=input,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            groups=num_groups,
            act=None,
            use_cudnn=use_cudnn,
            param_attr=parameter_attr,
            bias_attr=False)

        bn_name = name + "_bn"
        bn_decay = float(self.bn_decay)
        bn_param_attr = ParamAttr(regularizer=L2Decay(bn_decay),
                                  name=bn_name + '_scale')
        bn_bias_attr = ParamAttr(regularizer=L2Decay(bn_decay),
                                 name=bn_name + '_offset')
        return fluid.layers.batch_norm(
            input=conv,
            act=act,
            param_attr=bn_param_attr,
            bias_attr=bn_bias_attr,
            moving_mean_name=bn_name + '_mean',
            moving_variance_name=bn_name + '_variance')

    def depthwise_separable(self,
                            input,
                            num_filters1,
                            num_filters2,
                            num_groups,
                            stride,
                            scale,
                            name=None):
        depthwise_conv = self._conv_norm(
            input=input,
            filter_size=3,
            num_filters=int(num_filters1 * scale),
            stride=stride,
            padding=1,
            num_groups=int(num_groups * scale),
            use_cudnn=False,
            name=name + "_dw")

        pointwise_conv = self._conv_norm(
            input=depthwise_conv,
            filter_size=1,
            num_filters=int(num_filters2 * scale),
            stride=1,
            padding=0,
            name=name + "_sep")
        return pointwise_conv

    def get_backone(self, input):
        """
        Args:
            input (Variable): input variable.
        """
        blocks = []
        # input 1/1
        out = self._conv_norm(input, 3, int(32 * self.scale), 2, 1, 3,
                              name="conv1")
        # 1/2
        out = self.depthwise_separable(out, 32, 64, 32, 1, self.scale, 
                                       name="conv2_1")
        out = self.depthwise_separable(out, 64, 128, 64, 2, self.scale, 
                                       name="conv2_2")
        # 1/4
        out = self.depthwise_separable(out, 128, 128, 128, 1, self.scale, 
                                       name="conv3_1")
        out = self.depthwise_separable(out, 128, 256, 128, 2, self.scale, 
                                       name="conv3_2")
        # 1/8
        out = self.depthwise_separable(out, 256, 256, 256, 1, self.scale, 
                                       name="conv4_1")
        blocks.append(out)
        out = self.depthwise_separable(out, 256, 512, 256, 2, self.scale, 
                                       name="conv4_2")
        # 1/16
        for i in range(5):
            out = self.depthwise_separable(out, 512, 512, 512, 1,
                                           self.scale, 
                                           name="conv5_" + str(i + 1))
        blocks.append(out)

        out = self.depthwise_separable(out, 512, 1024, 512, 2, self.scale, 
                                       name="conv5_6")
        # 1/32
        out = self.depthwise_separable(out, 1024, 1024, 1024, 1, self.scale,
                                       name="conv6")
        blocks.append(out)
        return blocks


@Backbones.register
class MobileNetV1Backbone(BackboneBase):
    def __init__(self, cfg):
        super(MobileNetV1Backbone, self).__init__(cfg)
        self.scale = cfg.MODEL.CONV_GROUP_SCALE
        self.bn_decay = getattr(cfg.OPTIMIZER.WEIGHT_DECAY, 
                                'BN_DECAY', True)

    def __call__(self, input):
        """
        Get the backbone of MobileNetV1.
        Args:
            input (Variable): input variable.
            scale (float): the scale of groups number/ filter number
        Returns:
            The three feature maps of MobileNet.
        """
        model = MobileNet(scale=self.scale, bn_decay=self.bn_decay)
        return model.get_backone(input)
