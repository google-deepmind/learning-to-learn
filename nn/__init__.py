# Copyright 2016 Google Inc.
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
# ==============================================================================
"""This python module contains Neural Network Modules for TensorFlow."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nn.base import AbstractModule
from nn.base import Error
from nn.base import IncompatibleShapeError
from nn.base import Module
from nn.base import NotConnectedError
from nn.base import NotSupportedError
from nn.base import ParentNotBuiltError
from nn.base import Transposable
from nn.base import UnderspecifiedError
from nn.basic import BatchFlatten
from nn.basic import BatchReshape
from nn.basic import Linear
from nn.basic_rnn import DeepRNN
from nn.batch_norm import BatchNorm
from nn.conv import Conv2D
from nn.conv import Conv2DTranspose
from nn.conv import SAME
from nn.conv import VALID
from nn.convnet import ConvNet2D
from nn.gated_rnn import LSTM
from nn.mlp import MLP
from nn.rnn_core import RNNCore
from nn.rnn_core import TrainableInitialState
from nn.sequential import Sequential
from nn.util import get_variables_in_module
