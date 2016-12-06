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
"""Implementation of convolutional nn modules.

Classes defining convolutional operations, inheriting from `nn.Module`, with
easy weight sharing.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import numbers


import numpy as np
import tensorflow as tf

from nn import base
from nn import util


SAME = "SAME"
VALID = "VALID"
ALLOWED_PADDINGS = {SAME, VALID}


def _fill_shape(x, n):
  """Idempotentally converts an integer to a tuple of integers of a given size.

  This is used to allow shorthand notation for various configuration parameters.
  A user can provide either, for example, `2` or `[2, 2]` as a kernel shape, and
  this function returns `(2, 2)` in both cases. Passing `[1, 2]` will return
  `(1, 2)`.

  Args:
    x: An integer or an iterable of integers
    n: An integer, the size of the desired output list

  Returns:
    If `x` is an integer, a tuple of size `n` containing `n` copies of `x`.
    If `x` is an iterable of integers of size `n`, it returns `tuple(x)`.

  Raises:
    TypeError: If n is not a positive integer;
      or if x is neither integer nor an iterable of size n.
  """
  if not isinstance(n, numbers.Integral) or n < 1:
    raise TypeError("n must be a positive integer")

  if isinstance(x, numbers.Integral):
    return (x,) * n
  elif (isinstance(x, collections.Iterable) and len(x) == n and
        all(isinstance(v, numbers.Integral) for v in x)):
    return tuple(x)
  else:
    raise TypeError("x is {}, must be either an integer "
                    "or an iterable of integers of size {}".format(x, n))


def _fill_and_verify_kernel_shape(x, n):
  """Expands x if necessary into a `n`-D kernel shape and reports errors."""
  try:
    return _fill_shape(x, n)
  except TypeError as e:
    raise base.IncompatibleShapeError("Invalid kernel shape: {}".format(e))


def _verify_padding(padding):
  """Verifies that the provided padding is supported. Returns padding."""
  if padding not in ALLOWED_PADDINGS:
    raise ValueError(
        "Padding must be member of '{}', not {}".format(
            ALLOWED_PADDINGS, padding))
  return padding


def _fill_and_one_pad_stride(stride, n):
  """Expands the provided stride to size n and pads it with 1s."""
  try:
    return (1,) + _fill_shape(stride, n) + (1,)
  except TypeError:
    raise base.IncompatibleShapeError(
        "stride is {} ({}), must be either an integer or an iterable of "
        "integers of size {}".format(stride, type(stride), n))


def create_weight_initializer(fan_in_shape):
  """Returns a default initializer for the weights of a convolutional module."""
  stddev = 1 / math.sqrt(np.prod(fan_in_shape))
  return tf.truncated_normal_initializer(stddev=stddev)


def create_bias_initializer(bias_shape):
  """Returns a default initializer for the biases of a convolutional module."""
  stddev = 1 / math.sqrt(np.prod(bias_shape))
  return tf.truncated_normal_initializer(stddev=stddev)


class Conv2D(base.AbstractModule, base.Transposable):
  """Spatial convolution and dilated convolution module, including bias.

  This acts as a light wrapper around the TensorFlow ops `tf.nn.conv2d` and
  `tf.nn.atrous_conv2d`, abstracting away variable creation and sharing.

  The current implementation of `tf.nn.atrous_conv2d` does not easily permit for
  strides > 1 when performing dilated convolution (see b/29893301). Therefore,
  strides > 1 are currently disabled if the rate is set > 1.
  """

  def __init__(self, output_channels, kernel_shape, stride=1, rate=1,
               padding=SAME, use_bias=True, initializers=None, mask=None,
               name="conv_2d"):
    """Constructs a Conv2D module.

    See the following documentation for an explanation of VALID versus SAME
    padding modes:
    https://www.tensorflow.org/versions/r0.8/api_docs/python/nn.html#convolution

    Args:
      output_channels: Number of output channels. `output_channels` can be
          either a number or a callable. In the latter case, since the function
          invocation is deferred to graph construction time, the user must only
          ensure that output_channels can be called, returning an integer,
          when `_build` is called.
      kernel_shape: List of kernel sizes, or integer that is used to define
          kernel size in all dimensions.
      stride: List of kernel strides, or integer that is used to define
          stride in all dimensions.
      rate: A positive integer, `rate=1` corresponds to standard 2D convolution,
          `rate > 1` corresponds to dilated convolution.
      padding: Padding algorithm, either `nn.SAME` or `nn.VALID`.
      use_bias: Whether to include bias parameters. Default `True`.
      initializers: Optional dict containing ops to initialize the filters (with
          key 'w') or biases (with key 'b'). The default initializers are
          truncated normal initializers, which are commonly used when the inputs
          are zero centered (see https://arxiv.org/pdf/1502.03167v3.pdf).
      mask: Optional 2D or 4D array, tuple or numpy array containing values to
          multiply the weights by component-wise.
      name: Name of the module.

    Raises:
      base.IncompatibleShapeError: If the given kernel shape is not an integer;
          or if the given kernel shape is not a sequence of two integers.
      base.IncompatibleShapeError: If the given stride is not an integer; or if
          the given stride is not a sequence of two or four integers.
      base.IncompatibleShapeError: If a mask is given and its rank is neither 2
          nor 4.
      base.NotSupportedError: If the given dilation rate is not a positive
          integer.
      base.NotSupportedError: If rate > 1 and the stride in any dimension is
          > 1.
      ValueError: If the given padding is not `nn.VALID` or `nn.SAME`.
      KeyError: If initializers contains any keys other than 'w' or 'b'.
      TypeError: If any of the given initializers are not callable.
      TypeError: If mask is given and is not an array, tuple or a numpy array.
    """
    super(Conv2D, self).__init__(name=name)

    self._output_channels = output_channels
    self._input_shape = None
    self._kernel_shape = _fill_and_verify_kernel_shape(kernel_shape, 2)
    try:
      self._stride = (1,) + _fill_shape(stride, 2) + (1,)
    except TypeError as e:
      # We want to support passing native strides akin to [1, m, n, 1].
      if len(stride) == 4:
        self._stride = tuple(stride)
      else:
        raise base.IncompatibleShapeError("Invalid stride: {}".format(e))

    if not isinstance(rate, numbers.Integral) or rate < 1:
      raise base.NotSupportedError(
          "Rate, {}, must be integer >= 1".format(rate))
    elif any(x > 1 for x in self._stride) and rate > 1:
      raise base.NotSupportedError(
          "Cannot have stride > 1 with rate > 1")
    else:
      self._rate = rate

    self._padding = _verify_padding(padding)
    self._use_bias = use_bias
    self.possible_keys = self.get_possible_initializer_keys(use_bias=use_bias)
    self._initializers = util.check_initializers(
        initializers, self.possible_keys)

    if mask is not None:
      if not isinstance(mask, (list, tuple, np.ndarray)):
        raise TypeError("Invalid type for mask: {}".format(type(mask)))
      self._mask = np.asanyarray(mask)
      mask_rank = mask.ndim
      if mask_rank != 2 and mask_rank != 4:
        raise base.IncompatibleShapeError(
            "Invalid mask rank: {}".format(mask_rank))
    else:
      self._mask = None

  @classmethod
  def get_possible_initializer_keys(cls, use_bias=True):
    return {"w", "b"} if use_bias else {"w"}

  def _build(self, inputs):
    """Connects the Conv2D module into the graph, with input Tensor `inputs`.

    If this is not the first time the module has been connected to the graph,
    the input Tensor provided here must have the same final 3 dimensions, in
    order for the existing variables to be the correct size for the
    multiplication. The batch size may differ for each connection.

    Args:
      inputs: A 4D Tensor of shape [batch_size, input_height, input_width,
          input_channels].

    Returns:
      A 4D Tensor of shape [batch_size, output_height, output_width,
          output_channels].

    Raises:
      ValueError: If connecting the module into the graph any time after the
          first time and the inferred size of the input does not match previous
          invocations.
      base.IncompatibleShapeError: If the input tensor has the wrong number
          of dimensions.
      base.IncompatibleShapeError: If a mask is present and its shape is
          incompatible with the shape of the weights.
      base.UnderspecifiedError: If the input tensor has an unknown
          `input_channels`.
      base.UnderspecifiedError: If rate > 1 is used with an input tensor with
          unknown `input_width` or `input_height`.
      TypeError: If input Tensor dtype is not `tf.float32`.
    """
    # Handle input whose shape is unknown during graph creation.
    self._input_shape = tuple(inputs.get_shape().as_list())

    if len(self._input_shape) != 4:
      raise base.IncompatibleShapeError(
          "Input Tensor must have shape (batch_size, input_height, input_"
          "width, input_channels)")

    if self._input_shape[3] is None:
      raise base.UnderSpecifiedError(
          "Number of input channels must be known at module build time")
    else:
      input_channels = self._input_shape[3]

    if inputs.dtype != tf.float32:
      raise TypeError(
          "Input must have dtype tf.float32, but dtype was {}".format(
              inputs.dtype))

    weight_shape = (
        self._kernel_shape[0],
        self._kernel_shape[1],
        input_channels,
        self.output_channels)

    bias_shape = (self.output_channels,)

    if "w" not in self._initializers:
      self._initializers["w"] = create_weight_initializer(weight_shape[:3])

    if "b" not in self._initializers and self._use_bias:
      self._initializers["b"] = create_bias_initializer(bias_shape)

    self._w = tf.get_variable("w",
                              shape=weight_shape,
                              initializer=self._initializers["w"])

    w = self._w

    if self._mask is not None:
      mask_rank = self._mask.ndim
      mask_shape = self._mask.shape
      if mask_rank == 2:
        if mask_shape != self._kernel_shape:
          raise base.IncompatibleShapeError(
              "Invalid mask shape: {}".format(mask_shape))
        mask = np.reshape(self._mask, self._kernel_shape + (1, 1))
      elif mask_rank == 4:
        if mask_shape != tuple(weight_shape):
          raise base.IncompatibleShapeError(
              "Invalid mask shape: {}".format(mask_shape))
        mask = self._mask
      mask_tensor, = tf.py_func(lambda: mask, [], [w.dtype], stateful=False)
      mask_tensor.set_shape(weight_shape)
      w *= mask

    if self._rate > 1:
      if any(x is None for x in self._input_shape[1:-1]):
        raise base.UnderspecifiedError(
            "Can't use atrous convolutions with unknown input_width or "
            "input_height at graph build time")
      outputs = tf.nn.atrous_conv2d(inputs,
                                    w,
                                    rate=self._rate,
                                    padding=self._padding)
    else:
      outputs = tf.nn.conv2d(inputs,
                             w,
                             strides=self._stride,
                             padding=self._padding)

    if self._use_bias:
      self._b = tf.get_variable("b",
                                shape=bias_shape,
                                initializer=self._initializers["b"])
      outputs += self._b

    return outputs

  @property
  def output_channels(self):
    """Returns the number of output channels."""
    if callable(self._output_channels):
      self._output_channels = self._output_channels()
    return self._output_channels

  @property
  def kernel_shape(self):
    """Returns the kernel shape."""
    return self._kernel_shape

  @property
  def stride(self):
    """Returns the stride."""
    return self._stride

  @property
  def rate(self):
    """Returns the dilation rate."""
    return self._rate

  @property
  def padding(self):
    """Returns the padding algorithm."""
    return self._padding

  @property
  def w(self):
    """Returns the Variable containing the weight matrix."""
    self._ensure_is_connected()
    return self._w

  @property
  def b(self):
    """Returns the Variable containing the bias.

    Returns:
      Variable object containing the bias, from the most recent __call__.

    Raises:
      base.NotConnectedError: If the module has not been connected to the graph
          yet, meaning the variables do not exist.
      AttributeError: If the module does not use bias.
    """
    self._ensure_is_connected()
    if not self._use_bias:
      raise AttributeError(
          "No bias Variable in Conv2D Module when `use_bias=False`.")
    return self._b

  @property
  def has_bias(self):
    """Returns `True` if bias Variable is present in the module."""
    return self._use_bias

  @property
  def initializers(self):
    """Returns the initializers dictionary."""
    return self._initializers

  # Implements Transposable interface.
  @property
  def input_shape(self):
    """Returns the input shape."""
    self._ensure_is_connected()
    return self._input_shape

  # Implements Transposable interface.
  def transpose(self, name=None):
    """Returns matching `Conv2DTranspose` module.

    Args:
      name: Optional string assigning name of transpose module. The default name
        is constructed by appending "_transpose" to `self.name`.

    Returns:
      `Conv2DTranspose` module.

    Raises:
     base.NotSupportedError: If `rate > 1`.
    """
    if self._rate > 1:
      raise base.NotSupportedError(
          "Cannot transpose a dilated convolution module.")

    if name is None:
      name = self.name + "_transpose"
    return Conv2DTranspose(output_channels=lambda: self.input_shape[-1],
                           output_shape=lambda: self.input_shape[1:3],
                           kernel_shape=self.kernel_shape,
                           stride=self.stride,
                           padding=self.padding,
                           use_bias=self._use_bias,
                           initializers=self.initializers,
                           name=name)


class Conv2DTranspose(base.AbstractModule, base.Transposable):
  """Spatial transposed / reverse / up 2D convolution module, including bias.

  This acts as a light wrapper around the TensorFlow op `tf.nn.conv2d_transpose`
  abstracting away variable creation and sharing.
  """

  def __init__(self, output_channels, output_shape, kernel_shape, stride=1,
               padding=SAME, use_bias=True, initializers=None,
               name="conv_2d_transpose"):
    """Constructs a `Conv2DTranspose module`.

    See the following documentation for an explanation of VALID versus SAME
    padding modes:
    https://www.tensorflow.org/versions/r0.8/api_docs/python/nn.html#convolution

    Args:
      output_channels: Number of output channels.
          Can be either a number or a callable. In the latter case, since the
          function invocation is deferred to graph construction time, the user
          must only ensure `output_channels` can be called, returning an
          integer, when build is called.
      output_shape: Output shape of transpose convolution.
          Can be either an iterable of integers or a callable. In the latter
          case, since the function invocation is deferred to graph construction
          time, the user must only ensure that `output_shape` can be called,
          returning an iterable of format `(out_height, out_width)` when
          `_build` is called. Note that `output_shape` defines the size of
          output signal domain, as opposed to the shape of the output `Tensor`.
      kernel_shape: List of kernel sizes, must be length 2.
      stride: List of kernel strides.
      padding: Padding algorithm, either `nn.SAME` or `nn.VALID`.
      use_bias: Whether to include bias parameters. Default `True`.
      initializers: Optional dict containing ops to initialize the filters (with
          key 'w') or biases (with key 'b').
      name: Name of the module.

    Raises:
      base.IncompatibleShapeError: If the given kernel shape is neither an
          integer nor a sequence of two integers.
      base.IncompatibleShapeError: If the given stride is neither an integer nor
          a sequence of two or four integers.
      ValueError: If the given padding is not `nn.VALID` or `nn.SAME`.
      KeyError: If `initializers` contains any keys other than 'w' or 'b'.
      TypeError: If any of the given initializers are not callable.
    """
    super(Conv2DTranspose, self).__init__(name)

    self._output_channels = output_channels
    if callable(output_shape):
      self._output_shape = output_shape
    else:
      self._output_shape = tuple(output_shape)
    self._input_shape = None

    self._kernel_shape = _fill_and_verify_kernel_shape(kernel_shape, 2)
    # We want to support passing native strides akin to [1, m, n, 1].
    if isinstance(stride, collections.Iterable) and len(stride) == 4:
      if not stride[0] == stride[3] == 1:
        raise base.IncompatibleShapeError(
            "Invalid stride: First and last element must be 1.")
      self._stride = tuple(stride)
    else:
      self._stride = _fill_and_one_pad_stride(stride, 2)

    self._padding = _verify_padding(padding)
    self._use_bias = use_bias
    self.possible_keys = self.get_possible_initializer_keys(use_bias=use_bias)
    self._initializers = util.check_initializers(
        initializers, self.possible_keys)

  @classmethod
  def get_possible_initializer_keys(cls, use_bias=True):
    return {"w", "b"} if use_bias else {"w"}

  def _build(self, inputs):
    """Connects the Conv2DTranspose module into the graph.

    If this is not the first time the module has been connected to the graph,
    the input Tensor provided here must have the same final 3 dimensions, in
    order for the existing variables to be the correct size for the
    multiplication. The batch size may differ for each connection.

    Args:
      inputs: A 4D Tensor of shape [batch_size, input_height, input_width,
          input_channels].

    Returns:
      A 4D Tensor of shape [batch_size, output_height, output_width,
          output_channels].

    Raises:
      ValueError: If connecting the module into the graph any time after the
          first time and the inferred size of the input does not match previous
          invocations.
      base.IncompatibleShapeError: If the input tensor has the wrong number of
          dimensions; or if the input tensor has an unknown `input_channels`; or
          or if `output_shape` is an iterable and is not in the format
          `(out_height, out_width)`.
      TypeError: If input Tensor dtype is not `tf.float32`.
    """
    # Handle input whose shape is unknown during graph creation.
    self._input_shape = tuple(inputs.get_shape().as_list())

    if len(self._input_shape) != 4:
      raise base.IncompatibleShapeError(
          "Input Tensor must have shape (batch_size, input_height, "
          "input_width, input_channels)")

    if self._input_shape[3] is None:
      raise base.IncompatibleShapeError(
          "Number of input channels must be known at module build time")
    input_channels = self._input_shape[3]

    if inputs.dtype != tf.float32:
      raise TypeError("Input must have dtype tf.float32, but dtype was " +
                      inputs.dtype)

    if len(self.output_shape) != 2:
      raise base.IncompatibleShapeError("Output shape must be specified as "
                                        "(output_height, output_width)")

    weight_shape = (self._kernel_shape[0], self._kernel_shape[1],
                    self.output_channels, input_channels)

    bias_shape = (self.output_channels,)

    if "w" not in self._initializers:
      fan_in_shape = weight_shape[:2] + (weight_shape[3],)
      self._initializers["w"] = create_weight_initializer(fan_in_shape)

    if "b" not in self._initializers and self._use_bias:
      self._initializers["b"] = create_bias_initializer(bias_shape)

    self._w = tf.get_variable("w",
                              shape=weight_shape,
                              initializer=self._initializers["w"])

    # Use tensorflow shape op to manipulate inputs shape, so that unknown batch
    # size - which can happen when using input placeholders - is handled
    # correcly.
    batch_size = tf.expand_dims(tf.shape(inputs)[0], 0)
    conv_output_shape = tf.convert_to_tensor(
        tuple(self.output_shape) + (self.output_channels,))
    output_shape = tf.concat(0, [batch_size, conv_output_shape])

    outputs = tf.nn.conv2d_transpose(inputs,
                                     self._w,
                                     output_shape,
                                     strides=self._stride,
                                     padding=self._padding)

    if self._use_bias:
      self._b = tf.get_variable("b",
                                shape=bias_shape,
                                initializer=self._initializers["b"])
      outputs += self._b

    # Recover output tensor shape value and pass it to set_shape in order to
    # enable shape inference.
    batch_size_value = inputs.get_shape()[0]
    output_shape_value = ((batch_size_value,) + self.output_shape +
                          (self.output_channels,))
    outputs.set_shape(output_shape_value)

    return outputs

  @property
  def output_channels(self):
    """Returns the number of output channels."""
    if callable(self._output_channels):
      self._output_channels = self._output_channels()
    return self._output_channels

  @property
  def kernel_shape(self):
    """Returns the kernel shape."""
    return self._kernel_shape

  @property
  def stride(self):
    """Returns the stride."""
    return self._stride

  @property
  def output_shape(self):
    """Returns the output shape."""
    if callable(self._output_shape):
      self._output_shape = tuple(self._output_shape())
    return self._output_shape

  @property
  def padding(self):
    """Returns the padding algorithm."""
    return self._padding

  @property
  def w(self):
    """Returns the Variable containing the weight matrix."""
    self._ensure_is_connected()
    return self._w

  @property
  def b(self):
    """Returns the Variable containing the bias.

    Returns:
      Variable object containing the bias, from the most recent __call__.

    Raises:
      base.NotConnectedError: If the module has not been connected to the graph
          yet, meaning the variables do not exist.
      AttributeError: If the module does not use bias.
    """
    self._ensure_is_connected()
    if not self._use_bias:
      raise AttributeError(
          "No bias Variable in Conv2DTranspose Module when `use_bias=False`.")
    return self._b

  @property
  def has_bias(self):
    """Returns `True` if bias Variable is present in the module."""
    return self._use_bias

  @property
  def initializers(self):
    """Returns the initializers dictionary."""
    return self._initializers

  # Implements Transposable interface.
  @property
  def input_shape(self):
    """Returns the input shape."""
    self._ensure_is_connected()
    return self._input_shape

  # Implements Transposable interface.
  def transpose(self, name=None):
    """Returns matching `Conv2D` module.

    Args:
      name: Optional string assigning name of transpose module. The default name
          is constructed by appending "_transpose" to `self.name`.

    Returns:
      `Conv2D` module.
    """
    if name is None:
      name = self.name + "_transpose"
    return Conv2D(output_channels=lambda: self.input_shape[-1],
                  kernel_shape=self.kernel_shape,
                  stride=self.stride,
                  padding=self.padding,
                  use_bias=self._use_bias,
                  initializers=self.initializers,
                  name=name)
