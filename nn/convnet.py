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
"""A minimal interface convolutional networks module."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import tensorflow as tf

from nn import base
from nn import batch_norm
from nn import conv
from nn import util


def _replicate_elements(input_iterable, num_times):
  """Replicates entry in `input_iterable` if `input_iterable` is of length 1."""
  if len(input_iterable) == 1:
    return (input_iterable[0],) * num_times
  return tuple(input_iterable)


class ConvNet2D(base.AbstractModule, base.Transposable):
  """A 2D Convolutional Network module."""

  POSSIBLE_INITIALIZER_KEYS = {"w", "b"}

  def __init__(self,
               output_channels,
               kernel_shapes,
               strides,
               paddings,
               activation=tf.nn.relu,
               activate_final=False,
               initializers=None,
               use_batch_norm=False,
               use_bias=True,
               batch_norm_config=None,
               name="conv_net_2d"):
    """Constructs a `ConvNet2D` module.

    By default, neither batch normalization nor activation are applied to the
    output of the final layer.

    Args:
      output_channels: Iterable of output channels, as defined in
        `conv.Conv2D`. Output channels can be defined either as number or via a
        callable. In the latter case, since the function invocation is deferred
        to graph construction time, the user must only ensure that entries can
        be called when build is called. Each entry in the iterable defines
        properties in the corresponding convolutional layer.
      kernel_shapes: Iterable of kernel sizes as defined in `conv.Conv2D`; if
        the list contains one element only, the same kernel shape is used in
        each layer of the network.
      strides: Iterable of kernel strides as defined in `conv.Conv2D`; if the
        list contains one element only, the same stride is used in each layer of
        the network.
      paddings: Iterable of padding options, either `nn.SAME` or
        `nn.VALID`; if the Iterable contains one element only, the same padding
        is used in each layer of the network.
      activation: An activation op.
      activate_final: Boolean determining if the activation and batch
        normalization, if turned on, are applied to the final layer.
      initializers: Optional dict containing ops to initialize the filters of
        the whole network (with key 'w') or biases (with key 'b').
      use_batch_norm: Boolean determining if batch normalization is applied
        after convolution.
      use_bias: Whether to include bias parameters in the convolutional layers.
        Default `True`.
      batch_norm_config: Optional mapping of additional configuration for the
        `nn.BatchNorm` modules.
      name: Name of the module.

    Raises:
      TypeError: If `output_channels` is not iterable; or if `kernel_shapes` is
        not iterable; or `strides` is not iterable; or `paddings` is not
        iterable; or if `activation` is not callable; or `batch_norm_config` is
        not a mappable (e.g. `dict`).
      ValueError: If `output_channels` is empty; or if `kernel_shapes` has not
        length 1 or `len(output_channels)`; or if `strides` has not
        length 1 or `len(output_channels)`; or if `paddings` has not
        length 1 or `len(output_channels)`.
      Error: If initializers contains any keys other than 'w' or 'b'.
    """
    if not isinstance(output_channels, collections.Iterable):
      raise TypeError("output_channels must be iterable")
    output_channels = tuple(output_channels)

    if not isinstance(kernel_shapes, collections.Iterable):
      raise TypeError("kernel_shapes must be iterable")
    kernel_shapes = tuple(kernel_shapes)

    if not isinstance(strides, collections.Iterable):
      raise TypeError("strides must be iterable")
    strides = tuple(strides)

    if not isinstance(paddings, collections.Iterable):
      raise TypeError("paddings must be iterable")
    paddings = tuple(paddings)

    super(ConvNet2D, self).__init__(name)

    if not output_channels:
      raise ValueError("output_channels must not be empty")
    self._output_channels = tuple(output_channels)
    self._num_layers = len(self._output_channels)

    self._input_shape = None

    self._initializers = util.check_initializers(
        initializers, self.POSSIBLE_INITIALIZER_KEYS)

    if not callable(activation):
      raise TypeError("Input 'activation' must be callable")
    self._activation = activation
    self._activate_final = activate_final

    self._kernel_shapes = _replicate_elements(kernel_shapes, self._num_layers)
    if len(self._kernel_shapes) != self._num_layers:
      raise ValueError(
          "kernel_shapes must be of length 1 or len(output_channels)")

    self._strides = _replicate_elements(strides, self._num_layers)
    if len(self._strides) != self._num_layers:
      raise ValueError(
          """strides must be of length 1 or len(output_channels)""")

    self._paddings = _replicate_elements(paddings, self._num_layers)
    if len(self._paddings) != self._num_layers:
      raise ValueError(
          """paddings must be of length 1 or len(output_channels)""")

    self._use_batch_norm = use_batch_norm

    if batch_norm_config is not None:
      if not isinstance(batch_norm_config, collections.Mapping):
        raise TypeError("`batch_norm_config` must be a mapping, e.g. `dict`.")
      self._batch_norm_config = batch_norm_config
    else:
      self._batch_norm_config = {}

    self._use_bias = use_bias
    self._instantiate_layers()

  def _instantiate_layers(self):
    """Instantiates all the convolutional modules used in the network."""

    with tf.variable_scope(self._template.var_scope):
      self._layers = tuple(conv.Conv2D(name="conv_2d_{}".format(i),
                                       output_channels=self._output_channels[i],
                                       kernel_shape=self._kernel_shapes[i],
                                       stride=self._strides[i],
                                       padding=self._paddings[i],
                                       use_bias=self._use_bias,
                                       initializers=self._initializers)
                           for i in xrange(self._num_layers))

  def _build(self, inputs, is_training=True, test_local_stats=True):
    """Assembles the `ConvNet2D` and connects it to the graph.

    Args:
      inputs: A 4D Tensor of shape `[batch_size, input_height, input_width,
        input_channels]`.
      is_training: Boolean to indicate to `nn.BatchNorm` if we are
        currently training. By default `True`.
      test_local_stats: Boolean to indicate to `nn.BatchNorm` if batch
        normalization should  use local batch statistics at test time.
        By default `True`.

    Returns:
      A 4D Tensor of shape `[batch_size, output_height, output_width,
        output_channels[-1]]`.
    """
    self._input_shape = tuple(inputs.get_shape().as_list())
    net = inputs

    final_index = len(self._layers) - 1
    for i, layer in enumerate(self._layers):
      net = layer(net)

      if i != final_index or self._activate_final:
        if self._use_batch_norm:
          bn = batch_norm.BatchNorm(name="batch_norm_{}".format(i),
                                    **self._batch_norm_config)
          net = bn(net,
                   is_training=is_training,
                   test_local_stats=test_local_stats)

        net = self._activation(net)

    return net

  @property
  def layers(self):
    """Returns a tuple containing the convolutional layers of the network."""
    return self._layers

  @property
  def strides(self):
    return self._strides

  @property
  def paddings(self):
    return self._paddings

  @property
  def kernel_shapes(self):
    return self._kernel_shapes

  @property
  def output_channels(self):
    return tuple([l() if callable(l) else l for l in self._output_channels])

  @property
  def use_bias(self):
    return self._use_bias

  @property
  def use_batch_norm(self):
    return self._use_batch_norm

  @property
  def activate_final(self):
    return self._activate_final

  # Implements Transposable interface.
  @property
  def input_shape(self):
    """Returns shape of input `Tensor` passed at last call to `_build`."""
    self._ensure_is_connected()
    return self._input_shape

  # Implements Transposable interface.
  def transpose(self, name=None, output_channels=None):
    """Returns transposed conv net.

    Args:
      name: Optional string specifiying the name of the transposed module. The
        default name is constructed by appending "_transpose" to `self.name`.
      output_channels: Optional iterable of numbers of output channels.

    Returns:
      Matching `ConvNetTranspose2D` module.

    Raises:
      ValueError: If output_channels is specified and its length does not match
        the number of layers.
    """
    if name is None:
      name = self.name + "_transpose"

    if output_channels is None:
      output_channels = []
      for layer in reversed(self._layers):
        output_channels.append(lambda l=layer: l.input_shape[-1])

    elif len(output_channels) != len(self._layers):
      raise ValueError("Iterable output_channels length must match the"
                       "number of layers ({}), but is {} instead.".format(
                           len(self._layers), len(output_channels)))

    output_shapes = []
    for layer in reversed(self._layers):
      output_shapes.append(lambda l=layer: l.input_shape[1:-1])

    return ConvNet2DTranspose(name=name,
                              output_channels=output_channels,
                              output_shapes=output_shapes,
                              kernel_shapes=reversed(self.kernel_shapes),
                              strides=reversed(self.strides),
                              paddings=reversed(self.paddings),
                              activation=self._activation,
                              activate_final=self._activate_final,
                              initializers=self._initializers,
                              use_batch_norm=self._use_batch_norm,
                              use_bias=self._use_bias,
                              batch_norm_config=self._batch_norm_config)


class ConvNet2DTranspose(ConvNet2D):
  """A 2D Transpose-Convolutional Network module."""

  def __init__(self,
               output_channels,
               output_shapes,
               kernel_shapes,
               strides,
               paddings,
               activation=tf.nn.relu,
               activate_final=False,
               initializers=None,
               use_batch_norm=False,
               use_bias=True,
               batch_norm_config=None,
               name="conv_net_2d_transpose"):
    """Constructs a `ConvNetTranspose2D` module.

    `output_{shapes,channels}` can be defined either as iterable of
    {iterables,integers} or via a callable. In the latter case, since the
    function invocation is deferred to graph construction time, the user
    must only ensure that entries can be called returning meaningful values when
    build is called. Each entry in the iterable defines properties in the
    corresponding convolutional layer.

    By default, neither batch normalization nor activation are applied to the
    output of the final layer.

    Args:
      output_channels: Iterable of numbers of output channels.
      output_shapes: Iterable of output shapes as defined in
        `conv.conv2DTranpose`; if the iterable contains one element only, the
        same shape is used in each layer of the network.
      kernel_shapes: Iterable of kernel sizes as defined in `conv.Conv2D`; if
        the list contains one element only, the same kernel shape is used in
        each layer of the network.
      strides: Iterable of kernel strides as defined in `conv.Conv2D`; if the
        list contains one element only, the same stride is used in each layer of
        the network.
      paddings: Iterable of padding options, either `nn.SAME` or
        `nn.VALID`; if the Iterable contains one element only, the same padding
        is used in each layer of the network.
      activation: An activation op.
      activate_final: Boolean determining if the activation and batch
        normalization, if turned on, are applied to the final layer.
      initializers: Optional dict containing ops to initialize the filters of
        the whole network (with key 'w') or biases (with key 'b').
      use_batch_norm: Boolean determining if batch normalization is applied
        after convolution.
      use_bias: Whether to include bias parameters in the convolutional layers.
        Default `True`.
      batch_norm_config: Optional mapping of additional configuration for the
        `nn.BatchNorm` modules.
      name: Name of the module.

    Raises:
      TypeError: If `output_channels` is not iterable; or if `output_channels`
        is not iterable; or if `kernel_shapes` is not iterable; or `strides` is
        not iterable; or `paddings` is not iterable; or if `activation` is not
        callable.
      ValueError: If `output_channels` is empty; or if `kernel_shapes` has not
        length 1 or `len(output_channels)`; or if `strides` has not
        length 1 or `len(output_channels)`; or if `paddings` has not
        length 1 or `len(output_channels)`.
      Error: If initializers contains any keys other than 'w' or 'b'.
    """
    if not isinstance(output_channels, collections.Iterable):
      raise TypeError("output_channels must be iterable")
    output_channels = tuple(output_channels)
    num_layers = len(output_channels)

    if not isinstance(output_shapes, collections.Iterable):
      raise TypeError("output_shapes must be iterable")
    output_shapes = tuple(output_shapes)

    self._output_shapes = _replicate_elements(output_shapes, num_layers)
    if len(self._output_shapes) != num_layers:
      raise ValueError(
          "output_shapes must be of length 1 or len(output_channels)")

    super(ConvNet2DTranspose, self).__init__(
        output_channels,
        kernel_shapes,
        strides,
        paddings,
        activation=activation,
        activate_final=activate_final,
        initializers=initializers,
        use_batch_norm=use_batch_norm,
        use_bias=use_bias,
        batch_norm_config=batch_norm_config,
        name=name)

  def _instantiate_layers(self):
    """Instantiates all the convolutional modules used in the network."""

    with tf.variable_scope(self._template.var_scope):
      self._layers = tuple(
          conv.Conv2DTranspose(name="conv_2d_transpose_{}".format(i),
                               output_channels=self._output_channels[i],
                               output_shape=self._output_shapes[i],
                               kernel_shape=self._kernel_shapes[i],
                               stride=self._strides[i],
                               padding=self._paddings[i],
                               initializers=self._initializers,
                               use_bias=self._use_bias)
          for i in xrange(self._num_layers))

  @property
  def output_shapes(self):
    return tuple([l() if callable(l) else l for l in self._output_shapes])

  # Implement Transposable interface.
  def transpose(self, name=None, output_channels=None):
    """Returns transposed conv net.

    Args:
      name: Optional string specifiying the name of the transposed module. The
        default name is constructed by appending "_transpose" to `self.name`.
      output_channels: Optional iterable of numbers of output channels.

    Returns:
      Matching `ConvNetTranspose2D` module.

    Raises:
      ValueError: If output_channels is specified and its length does not match
        the number of layers.
    """
    if name is None:
      name = self.name + "_transpose"

    if output_channels is None:
      output_channels = []
      for layer in reversed(self._layers):
        output_channels.append(lambda l=layer: l.input_shape[-1])

    elif len(output_channels) != len(self._layers):
      raise ValueError("Iterable output_channels length must match the"
                       "number of layers ({}), but is {} instead.".format(
                           len(self._layers), len(output_channels)))

    return ConvNet2D(name=name,
                     output_channels=output_channels,
                     kernel_shapes=reversed(self.kernel_shapes),
                     strides=reversed(self.strides),
                     paddings=reversed(self.paddings),
                     activation=self._activation,
                     activate_final=self._activate_final,
                     initializers=self._initializers,
                     use_batch_norm=self._use_batch_norm,
                     use_bias=self._use_bias,
                     batch_norm_config=self._batch_norm_config)
