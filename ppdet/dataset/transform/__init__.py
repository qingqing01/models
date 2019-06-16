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

import copy
from . import operator
from . import arrange_sample
from . import transformer
from . import post_map
from .parallel_map import ParallelMappedDataset
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

__all__ = ['build', 'map', 'batch', 'batch_mapper']


def build(ops, context=None):
    """ Build a mapper for operators in 'ops'

    Args:
        ops (list of operator.BaseOperator or list of op dict): 
            configs for oprators, eg:
            [{'name': 'DecodeImage', 'params': {'to_rgb': True}}, {xxx}]
        context (dict): a context object for mapper

    Returns:
        a mapper function which accept one argument 'sample' and
        return the processed result
    """
    new_ops = []
    for _dict in ops:
        new_dict = {}
        for i, j in _dict.items():
            new_dict[i.lower()] = j
        new_ops.append(new_dict)
    ops = new_ops
    op_funcs = []
    op_repr = []
    for op in ops:
        if type(op) is dict and 'op' in op:
            op_func = getattr(operator.BaseOperator, op['op'])
            params = copy.deepcopy(op)
            del params['op']
            o = op_func(**params)
        elif not isinstance(op, operator.BaseOperator):
            op_func = getattr(operator.BaseOperator, op['name'])
            params = {} if 'params' not in op else op['params']
            o = op_func(**params)
        else:
            assert isinstance(
                op, operator.BaseOperator), 'invalid operator when build ops'
            o = op
        op_funcs.append(o)
        op_repr.append('{%s}' % str(o))
    op_repr = '[%s]' % ','.join(op_repr)

    def _mapper(sample):
        ctx = {} if context is None else copy.deepcopy(context)
        for f in op_funcs:
            try:
                out = f(sample, ctx)
                sample = out
            except Exception as e:
                logger.warn('failed to map operator[%s] with exception[%s]' \
                    % (f, str(e)))
        return out

    _mapper.ops = op_repr
    return _mapper


def map(ds, mapper, worker_args=None):
    """ apply 'mapper' to 'ds'
    Args:
        ds (instance of Dataset): dataset to be mapped
        mapper (function): action to be executed for every data sample
        worker_args (dict): configs for concurrent mapper
    Returns:
        a mapped dataset
    """
    if worker_args is not None:
        return ParallelMappedDataset(ds, mapper, worker_args)
    else:
        return transformer.MappedDataset(ds, mapper)


def batch(ds, batchsize, drop_last=False):
    """ Batch data samples to batches
    Args:
        batchsize (int): number of samples for a batch
        drop_last (bool): drop last few samples if not enough for a batch
        
    Returns:
        a batched dataset
    """
    return transformer.BatchedDataset(ds, batchsize, drop_last=drop_last)


def batch_map(ds, config):
    """ Post process the batches.
    Args:
        ds (instance of Dataset): dataset to be mapped
        mapper (function): action to be executed for every batch
    Returns:
        a batched dataset which is processed
    """
    mapper = post_map.build(**config)
    return transformer.MappedDataset(ds, mapper)


for nm in operator.registered_ops:
    op = getattr(operator.BaseOperator, nm)
    locals()[nm] = op

__all__ += operator.registered_ops
