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
"""A minimal interface mlp module."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import tensorflow as tf

from nn import base
from nn import basic
from nn import util


class MLP(base.AbstractModule, base.Transposable):
  """A Multi-Layer perceptron module."""

  def __init__(self,
               output_sizes,
               activation=tf.nn.relu,
               activate_final=False,
               initializers=None,
               use_bias=True,
               name="mlp"):
    """Constructs an MLP module.

    Args:
      output_sizes: An iterable of output dimensionalities as defined in
        `basic.Linear`. Output size can be defined either as number or via a
        callable. In the latter case, since the function invocation is deferred
        to graph construction time, the user must only ensure that entries can
        be called when build is called. Each entry in the iterable defines
        properties in the corresponding linear layer.
      activation: An activation op. The activation is applied to intermediate
        layers, and optionally to the output of the final layer.
      activate_final: Boolean determining if the activation is applied to
        the output of the final layer. Default `False`.
      initializers: Optional dict containing ops to initialize the linear
        layers' weights (with key 'w') or biases (with key 'b').
      use_bias: Whether to include bias parameters in the linear layers.
        Default `True`.
      name: Name of the module.

    Raises:
      Error: If initializers contains any keys other than 'w' or 'b'.
      ValueError: If output_sizes is empty.
      TypeError: If `activation` is not callable; or if `output_sizes` is not
        iterable.
    """
    super(MLP, self).__init__(name=name)

    if not isinstance(output_sizes, collections.Iterable):
      raise TypeError("output_sizes must be iterable")
    output_sizes = tuple(output_sizes)
    if not output_sizes:
      raise ValueError("output_sizes must not be empty")
    self._output_sizes = output_sizes
    self._num_layers = len(self._output_sizes)
    self._input_shape = None

    self.possible_keys = self.get_possible_initializer_keys(use_bias=use_bias)
    self._initializers = util.check_initializers(initializers,
                                                 self.possible_keys)
    if not callable(activation):
      raise TypeError("Input 'activation' must be callable")
    self._activation = activation
    self._activate_final = activate_final

    self._use_bias = use_bias
    self._instantiate_layers()

  def _instantiate_layers(self):
    """Instantiates all the linear modules used in the network.

    Layers are instantiated in the constructor, as opposed to the build
    function, because MLP implements the Transposable interface, and the
    transpose function can be called before the module is actually connected
    to the graph and build is called.

    Notice that this is safe since layers in the transposed module are
    instantiated using a lambda returning input_size of the mlp layers, and
    this doesn't have to return sensible values until the original module is
    connected to the graph.
    """

    with tf.variable_scope(self._template.var_scope):
      self._layers = [basic.Linear(self._output_sizes[i],
                                   name="linear_{}".format(i),
                                   initializers=self._initializers,
                                   use_bias=self.use_bias)
                      for i in xrange(self._num_layers)]

  @classmethod
  def get_possible_initializer_keys(cls, use_bias=True):
    return basic.Linear.get_possible_initializer_keys(use_bias=use_bias)

  def _build(self, inputs):
    """Assembles the `MLP` and connects it to the graph.

    Args:
      inputs: A 2D Tensor of size `[batch_size, input_size]`.

    Returns:
      A 2D Tensor of size `[batch_size, output_sizes[-1]]`.
    """
    self._input_shape = tuple(inputs.get_shape().as_list())
    net = inputs

    final_index = self._num_layers - 1
    for layer_id in xrange(self._num_layers):
      net = self._layers[layer_id](net)

      if final_index != layer_id or self._activate_final:
        net = self._activation(net)

    return net

  @property
  def layers(self):
    """Returns a tuple containing the linear layers of the `MLP`."""
    return self._layers

  @property
  def output_sizes(self):
    return tuple([l() if callable(l) else l for l in self._output_sizes])

  @property
  def use_bias(self):
    return self._use_bias

  @property
  def activate_final(self):
    return self._activate_final

  # Implements Transposable interface
  @property
  def input_shape(self):
    """Returns shape of input `Tensor` passed at last call to `build`."""
    self._ensure_is_connected()
    return self._input_shape

  # Implements Transposable interface
  def transpose(self, name=None):
    """Returns transposed `MLP`.

    Args:
      name: Optional string specifiying the name of the transposed module. The
        default name is constructed by appending "_transpose" to `self.name`.

    Returns:
      Matching transposed `MLP` module.
    """
    if name is None:
      name = self.name + "_transpose"
    output_sizes = [lambda l=layer: l.input_shape[1] for layer in self._layers]
    output_sizes.reverse()
    return MLP(name=name,
               output_sizes=output_sizes,
               activation=self._activation,
               activate_final=self._activate_final,
               initializers=self._initializers,
               use_bias=self._use_bias)
