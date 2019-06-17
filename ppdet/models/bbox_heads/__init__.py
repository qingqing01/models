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

from . import bbox_head
from .bbox_head import *

from . import mask_head
from .mask_head import *

from . import cascade_head
from .cascade_head import CascadeHead, FC6FC7

__all__ = bbox_head.__all__
__all__ += mask_head.__all__
__all__ += cascade_head.__all__
