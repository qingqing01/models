# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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


class FetchName(object):
    """
        fetch the backbones layer names
        Args:
            model_type (str): model type, 'ResNet', 'ResNeXt' or 'SEResNeXt'
            variant (str): ResNet variant, supports 'a', 'b', 'c', 'd' currently
        """

    def __init__(self, model_type, variant):
        super(FetchName, self).__init__()
        self._model_type = model_type
        self._variant = variant

    def fetch_conv_norm(self, name):
        if name == "conv1":
            bn_name = "bn_" + name
        else:
            bn_name = "bn" + name[3:]
        # the naming rule is same as pretrained weight
        if self._model_type == 'SEResNeXt':
            bn_name = name + "_bn"
        return bn_name

    def fetch_shortcut(self, name):
        if self._model_type == 'SEResNeXt':
            name = 'conv' + name + '_prj'
        return name

    def fetch_bottleneck(self, name):
        if self._model_type == 'SEResNeXt':
            conv_name1 = 'conv' + name + '_x1'
            conv_name2 = 'conv' + name + '_x2'
            conv_name3 = 'conv' + name + '_x3'
            shortcut_name = name
        else:
            conv_name1 = name + "_branch2a"
            conv_name2 = name + "_branch2b"
            conv_name3 = name + "_branch2c"
            shortcut_name = name + "_branch1"
        return conv_name1, conv_name2, conv_name3, shortcut_name

    def fetch_layer_warp(self, stage_num, count, i):
        name = 'res' + str(stage_num)
        if count > 10 and stage_num == 4:
            if i == 0:
                conv_name = name + "a"
            else:
                conv_name = name + "b" + str(i)
        else:
            conv_name = name + chr(ord("a") + i)
        if self._model_type == 'SEResNeXt':
            conv_name = str(stage_num + 2) + '_' + str(i + 1)
        return conv_name

    def fetch_c1_stage(self, out_chan):
        if self._variant in ['c', 'd']:
            conv_def = [
                [out_chan / 2, 3, 2, "conv1_1"],
                [out_chan / 2, 3, 1, "conv1_2"],
                [out_chan, 3, 1, "conv1_3"],
            ]
        else:
            conv1_name = "conv1"
            # the naming rule is same as pretrained weight
            if self._model_type == 'ResNext':
                conv1_name = "res_conv1"
            conv_def = [[out_chan, 7, 2, conv1_name]]
        return conv_def
