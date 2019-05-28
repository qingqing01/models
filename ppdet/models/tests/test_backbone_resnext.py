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

import numpy as np
import unittest

import paddle.fluid as fluid
import paddle.compat as cpt

from ppdet.core.config import load_cfg, merge_cfg
from ppdet.models.backbones.resnext import ResNeXt, ResNeXt101C4Backbone


def bottleneck_names(name, bn_affine, short_conv=True):
    """
    The naming rules are same as them in
    https://github.com/PaddlePaddle/models/blob/develop/PaddleCV/image_classification/models/resnet.py
    """

    def conv_norm(name):
        pnames = [name + '_weights']
        pnames += ['bn' + name[3:] + '_scale']
        pnames += ['bn' + name[3:] + '_offset']
        if not bn_affine:
            pnames += ['bn' + name[3:] + '_mean']
            pnames += ['bn' + name[3:] + '_variance']
        return pnames

    names = conv_norm(name + "_branch2a")
    names.extend(conv_norm(name + "_branch2b"))
    names.extend(conv_norm(name + "_branch2c"))
    if short_conv:
        names.extend(conv_norm(name + "_branch1"))
    return names


class TestResNeXt(unittest.TestCase):
    def setUp(self):
        self.dshape = [3, 224, 224]
        res_cfg = {50: [3, 4, 6, 3], 101: [3, 4, 23, 3], 152: [3, 8, 36, 3]}

        self.layers = 101
        self.depth = res_cfg[self.layers]
        self.fixbn = True
        cfg_file = 'configs/faster-rcnn_ResNeXt101-C4_1x.yml'
        self.cfg = load_cfg(cfg_file)

    def get_C1ToC4_params(self, bn_affine):
        """
        The naming rules are same as them in
        https://github.com/PaddlePaddle/models/blob/develop/PaddleCV/image_classification/models/resnet.py

        Args:
            bn_affine (bool): meaning use affine_channel

        Return:
            list[string]: all parameter names
        """
        param_names = ['conv1_weights', 'bn_conv1_scale', 'bn_conv1_offset']
        if not bn_affine:
            param_names += ['bn_conv1_mean', 'bn_conv1_variance']

        for b in range(len(self.depth) - 1):
            for i in range(self.depth[b]):
                if self.layers in [101, 152] and b == 2:
                    if i == 0:
                        name = "res" + str(b + 2) + "a"
                    else:
                        name = "res" + str(b + 2) + "b" + str(i)
                else:
                    name = "res" + str(b + 2) + chr(97 + i)
                param_names.extend(bottleneck_names(name, bn_affine, i == 0))
        return param_names

    def compare_C1ToC4(self, bn_affine):
        prog = fluid.Program()
        startup_prog = fluid.Program()
        with fluid.program_guard(prog, startup_prog):
            data = fluid.layers.data(
                name='input', shape=self.dshape, dtype='float32')
            backbone = ResNeXt101C4Backbone(self.cfg)
            feat = backbone(data)
            # actual names
            parameters = prog.global_block().all_parameters()
            actual_pnames = [cpt.to_bytes(p.name) for p in parameters]
        # expected names
        expect_pnames = self.get_C1ToC4_params(bn_affine)

        actual_pnames.sort()
        expect_pnames.sort()

        self.assertEqual(len(actual_pnames), len(expect_pnames))
        # check parameter names
        for a, e in zip(actual_pnames, expect_pnames):
            self.assertTrue(a == e, "Parameter names have diff: \n" +
                            " Actual: " + str(a) + "\n Expect: " + str(e))
        # check learning rate of batch_norm
        for p in parameters:
            if 'bn' in p.name and ('scale' in p.name or 'offset' in p.name):
                self.assertTrue(p.optimize_attr['learning_rate'] == 0.)
                self.assertTrue(p.stop_gradient)

    def test_C1ToC4_bn(self):
        merge_cfg({
            'AFFINE_CHANNEL': False,
            'FREEZE_BN': True,
            'FREEZE_AT': 2
        }, self.cfg.MODEL)
        assert not self.cfg.MODEL.AFFINE_CHANNEL
        self.compare_C1ToC4(False)

    def test_C1ToC4_affine(self):
        merge_cfg({
            'AFFINE_CHANNEL': True,
            'FREEZE_BN': True,
            'FREEZE_AT': 2
        }, self.cfg.MODEL)
        assert self.cfg.MODEL.AFFINE_CHANNEL
        self.compare_C1ToC4(True)


if __name__ == '__main__':
    unittest.main()
