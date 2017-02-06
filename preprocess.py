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
"""Learning 2 Learn preprocessing modules."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import nn


class Clamp(nn.AbstractModule):

  def __init__(self, min_value=None, max_value=None, name="clamp"):
    super(Clamp, self).__init__(name=name)
    self._min = min_value
    self._max = max_value

  def _build(self, inputs):
    output = inputs
    if self._min is not None:
      output = tf.maximum(output, self._min)
    if self._max is not None:
      output = tf.minimum(output, self._max)
    return output


class LogAndSign(nn.AbstractModule):
  """Log and sign preprocessing.

  As described in https://arxiv.org/pdf/1606.04474v1.pdf (Appendix A).
  """

  def __init__(self, k, name="preprocess_log"):
    super(LogAndSign, self).__init__(name=name)
    self._k = k

  def _build(self, gradients):
    """Connects the LogAndSign module into the graph.

    Args:
      gradients: `Tensor` of gradients with shape `[d_1, ..., d_n]`.

    Returns:
      `Tensor` with shape `[d_1, ..., d_n-1, 2 * d_n]`. The first `d_n` elements
      along the nth dimension correspond to the log output and the remaining
      `d_n` elements to the sign output.
    """
    eps = np.finfo(gradients.dtype.as_numpy_dtype).eps
    ndims = gradients.get_shape().ndims

    log = tf.log(tf.abs(gradients) + eps)
    clamped_log = Clamp(min_value=-1.0)(log / self._k)  # pylint: disable=not-callable
    sign = Clamp(min_value=-1.0, max_value=1.0)(gradients * np.exp(self._k))  # pylint: disable=not-callable

    return tf.concat([clamped_log, sign], ndims - 1)
