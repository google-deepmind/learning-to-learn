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
"""Basic Modules for TensorFlow nn.

Modules defining the simplest building blocks for Neural Networks.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numbers


import numpy as np
import tensorflow as tf

from nn import base
from nn import util


def create_linear_initializer(input_size):
  """Returns a default initializer for weights or bias of a linear module."""
  stddev = 1 / math.sqrt(input_size)
  return tf.truncated_normal_initializer(stddev=stddev)


class Linear(base.AbstractModule, base.Transposable):
  """Linear module, optionally including bias."""

  def __init__(self,
               output_size,
               use_bias=True,
               initializers=None,
               partitioners=None,
               name="linear"):
    """Constructs a Linear module.

    Args:
      output_size: Output dimensionality. `output_size` can be either an integer
          or a callable. In the latter case, since the function invocation is
          deferred to graph construction time, the user must only ensure that
          output_size can be called, returning an integer, when build is called.
      use_bias: Whether to include bias parameters. Default `True`.
      initializers: Optional dict containing initializers to initialize the
          weights (with key 'w') or biases (with key 'b'). The default
          initializers are truncated normal initializers, which are commonly
          used when the inputs are zero centered (see
          https://arxiv.org/pdf/1502.03167v3.pdf).
      partitioners: Optional dict containing partitioners to partition
          weights (with key 'w') or biases (with key 'b'). As a default, no
          partitioners are used.
      name: Name of the module.

    Raises:
      KeyError: If an initializer is provided for a key other than 'w' or 'b' if
          `use_bias` is `True`..
      TypeError: If a provided initializer is not a callable function.
    """
    super(Linear, self).__init__(name=name)
    self._output_size = output_size
    self._use_bias = use_bias
    self._input_shape = None
    self._w = None
    self._b = None
    self.possible_keys = self.get_possible_initializer_keys(use_bias=use_bias)
    self._initializers = util.check_initializers(
        initializers, self.possible_keys)
    self._partitioners = util.check_partitioners(
        partitioners, self.possible_keys)

  @classmethod
  def get_possible_initializer_keys(cls, use_bias=True):
    return {"w", "b"} if use_bias else {"w"}

  def _build(self, inputs):
    """Connects the Linear module into the graph, with input Tensor `inputs`.

    If this is not the first time the module has been connected to the graph,
    the Tensor provided here must have the same final dimension, in order for
    the existing variables to be the correct size for the multiplication. The
    batch size may differ for each connection.

    Args:
      inputs: A 2D Tensor of size [batch_size, input_size].

    Returns:
      A 2D Tensor of size [batch_size, output_size].

    Raises:
      base.IncompatibleShapeError: If the input is not a 2-D `Tensor` with
          the size of the second dimension specified.
      base.IncompatibleShapeError: If reconnecting an already connected module
          into the graph, and the shape of the input is not compatible with
          previous inputs.
    """
    input_shape = tuple(inputs.get_shape().as_list())

    if len(input_shape) != 2:
      raise base.IncompatibleShapeError(
          "{}: rank of shape must be 2 not: {}".format(
              self.name, len(input_shape)))

    if input_shape[1] is None:
      raise base.IncompatibleShapeError(
          "{}: Input size must be specified at module build time".format(
              self.name))

    if self._input_shape is not None and input_shape[1] != self._input_shape[1]:
      raise base.IncompatibleShapeError(
          "{}: Input shape must be [batch_size, {}] not: [batch_size, {}]"
          .format(self.name, self._input_shape[1], input_shape[1]))

    self._input_shape = input_shape

    if "w" not in self._initializers:
      self._initializers["w"] = create_linear_initializer(self._input_shape[1])

    if "b" not in self._initializers and self._use_bias:
      self._initializers["b"] = create_linear_initializer(self._input_shape[1])

    weight_shape = (self._input_shape[1], self.output_size)
    dtype = inputs.dtype
    self._w = tf.get_variable("w",
                              shape=weight_shape,
                              dtype=dtype,
                              initializer=self._initializers["w"],
                              partitioner=self._partitioners.get("w", None))
    outputs = tf.matmul(inputs, self._w)

    if self._use_bias:
      bias_shape = (self.output_size,)
      self._b = tf.get_variable("b",
                                shape=bias_shape,
                                dtype=dtype,
                                initializer=self._initializers["b"],
                                partitioner=self._partitioners.get("b", None))
      outputs += self._b

    return outputs

  @property
  def w(self):
    """Returns the Variable containing the weight matrix.

    Returns:
      Variable object containing the weights, from the most recent __call__.

    Raises:
      base.NotConnectedError: If the module has not been connected to the
          graph yet, meaning the variables do not exist.
    """
    self._ensure_is_connected()
    return self._w

  @property
  def b(self):
    """Returns the Variable containing the bias.

    Returns:
      Variable object containing the bias, from the most recent __call__.

    Raises:
      base.NotConnectedError: If the module has not been connected to the
          graph yet, meaning the variables do not exist.
      AttributeError: If the module does not use bias.
    """
    self._ensure_is_connected()
    if not self._use_bias:
      raise AttributeError(
          "No bias Variable in Linear Module when `use_bias=False`.")
    return self._b

  @property
  def output_size(self):
    """Returns the module output size."""
    if callable(self._output_size):
      self._output_size = self._output_size()
    return self._output_size

  @property
  def has_bias(self):
    """Returns `True` if bias Variable is present in the module."""
    return self._use_bias

  # Implements Transposable interface.
  @property
  def input_shape(self):
    """Returns shape of input `Tensor` passed at last call to `_build`."""
    self._ensure_is_connected()
    return self._input_shape

  # Implements Transposable interface
  def transpose(self, name=None):
    """Returns transposed `Linear` module.

    Args:
      name: Optional string assigning name of transpose module. The default name
          is constructed by appending "_transpose" to `self.name`.

    Returns:
      Transposed `Linear` module.
    """
    if name is None:
      name = self.name + "_transpose"
    return Linear(output_size=lambda: self.input_shape[1],
                  use_bias=self._use_bias,
                  initializers=self._initializers,
                  name=name)


class BatchReshape(base.AbstractModule, base.Transposable):
  """Reshapes input Tensor, preserving the batch dimension."""

  def __init__(self, shape, name="batch_reshape"):
    """Constructs a BatchReshape module.

    Args:
      shape: Shape to reshape the input Tensor to while preserving its
          batch size; `shape` can be either a tuple/list, or a callable that
          returns the actual shape. The callable does not need to be ready to
          return something meaningful at construction time, but it will be
          required to be able to do so when the module is connected to the
          graph. When the special value -1 appears in `shape` the corresponding
          size is automatically inferred. Note that -1 can only appear once in
          `shape`. To flatten all non-batch dimensions, the nn.BatchFlatten
          module can also be used.
      name: Name of the module.
    """
    super(BatchReshape, self).__init__(name=name)

    self._input_shape = None
    self._shape = shape

    if not callable(self._shape):
      self._shape = tuple(self._shape)

  def _infer_shape(self, dimensions):
    """Replaces the -1 wildcard in the output shape vector.

    This function infers the correct output shape given the input dimensions.

    Args:
      dimensions: List of input non-batch dimensions.

    Returns:
      Tuple of non-batch output dimensions.
    """
    # Size of input
    n = np.prod(dimensions)
    # Size of output where defined
    m = np.prod(abs(np.array(self._shape)))
    # Replace wildcard
    v = np.array(self._shape)
    v[v == -1] = n // m
    return tuple(v)

  def _build(self, inputs):
    """Connects the module into the graph, with input Tensor `inputs`.

    Args:
      inputs: A Tensor of shape [batch_size] + input_shape.

    Returns:
      A Tensor of shape [batch_size] + output_shape, with output_shape as
         defined in constructor.

    Raises:
      ValueError: If output shape is incompatible with input shape; or if
          shape array contains non numeric entries; or if shape array contains
          more than 1 wildcard -1.
    """
    if callable(self._shape):
      self._shape = tuple(self._shape())

    if not all([isinstance(x, numbers.Integral) and (x > 0 or x == -1)
                for x in self._shape]):
      raise ValueError("Input array shape can contain positive integral "
                       "numbers only, and the wildcard -1 used once")

    if self._shape.count(-1) > 1:
      raise ValueError("Wildcard -1 can appear only once in shape")

    self._input_shape = inputs.get_shape()[1:].as_list()
    if self._shape.count(-1) > 0:
      shape = (-1,) + self._infer_shape(self._input_shape)
    else:
      shape = (-1,) + self._shape

    if np.prod(self._input_shape) != np.prod(shape[1:]):
      raise ValueError("Output shape is incompatible with input shape")
    return tf.reshape(inputs, shape)

  @property
  def input_shape(self):
    self._ensure_is_connected()
    return self._input_shape

  # Implements Transposable interface.
  def transpose(self, name=None):
    """Returns transpose batch reshape."""
    if name is None:
      name = self.name + "_transpose"
    return BatchReshape(shape=lambda: self.input_shape, name=name)


class BatchFlatten(BatchReshape):
  """Flattens the input Tensor, preserving the batch dimension."""

  def __init__(self, name="batch_flatten"):
    """Constructs a BatchFlatten module.

    Args:
      name: Name of the module.
    """
    super(BatchFlatten, self).__init__(name=name, shape=(-1,))
