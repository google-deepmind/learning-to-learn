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
"""Learning 2 Learn meta-optimizer networks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import collections
import sys

import dill as pickle
import numpy as np
import six
import tensorflow as tf

import nn
import preprocess


def factory(net, net_options=(), net_path=None):
  """Network factory."""

  net_class = getattr(sys.modules[__name__], net)
  net_options = dict(net_options)

  if net_path:
    with open(net_path, "rb") as f:
      net_options["initializer"] = pickle.load(f)

  return net_class(**net_options)


def save(network, sess, filename=None):
  """Save the variables contained by a network to disk."""
  to_save = collections.defaultdict(dict)
  variables = nn.get_variables_in_module(network)

  for v in variables:
    split = v.name.split(":")[0].split("/")
    module_name = split[-2]
    variable_name = split[-1]
    to_save[module_name][variable_name] = v.eval(sess)

  if filename:
    with open(filename, "wb") as f:
      pickle.dump(to_save, f)

  return to_save


@six.add_metaclass(abc.ABCMeta)
class Network(nn.RNNCore):
  """Base class for meta-optimizer networks."""

  @abc.abstractmethod
  def initial_state_for_inputs(self, inputs, **kwargs):
    """Initial state given inputs."""
    pass


def _convert_to_initializer(initializer):
  """Returns a TensorFlow initializer.

  * Corresponding TensorFlow initializer when the argument is a string (e.g.
  "zeros" -> `tf.zeros_initializer`).
  * `tf.constant_initializer` when the argument is a `numpy` `array`.
  * Identity when the argument is a TensorFlow initializer.

  Args:
    initializer: `string`, `numpy` `array` or TensorFlow initializer.

  Returns:
    TensorFlow initializer.
  """

  if isinstance(initializer, str):
    return getattr(tf, initializer + "_initializer")(dtype=tf.float32)
  elif isinstance(initializer, np.ndarray):
    return tf.constant_initializer(initializer)
  else:
    return initializer


def _get_initializers(initializers, fields):
  """Produces a nn initialization `dict` (see Linear docs for a example).

  Grabs initializers for relevant fields if the first argument is a `dict` or
  reuses the same initializer for all fields otherwise. All initializers are
  processed using `_convert_to_initializer`.

  Args:
    initializers: Initializer or <variable, initializer> dictionary.
    fields: Fields nn is expecting for module initialization.

  Returns:
    nn initialization dictionary.
  """

  result = {}
  for f in fields:
    if isinstance(initializers, dict):
      if f in initializers:
        # Variable-specific initializer.
        result[f] = _convert_to_initializer(initializers[f])
    else:
      # Common initiliazer for all variables.
      result[f] = _convert_to_initializer(initializers)

  return result


def _get_layer_initializers(initializers, layer_name, fields):
  """Produces a nn initialization dictionary for a layer.

  Calls `_get_initializers using initializers[layer_name]` if `layer_name` is a
  valid key or using initializers otherwise (reuses initializers between
  layers).

  Args:
    initializers: Initializer, <variable, initializer> dictionary,
        <layer, initializer> dictionary.
    layer_name: Layer name.
    fields: Fields nn is expecting for module initialization.

  Returns:
    nn initialization dictionary.
  """

  # No initializers specified.
  if initializers is None:
    return None

  # Layer-specific initializer.
  if isinstance(initializers, dict) and layer_name in initializers:
    return _get_initializers(initializers[layer_name], fields)

  return _get_initializers(initializers, fields)


class StandardDeepLSTM(Network):
  """LSTM layers with a Linear layer on top."""

  def __init__(self, output_size, layers, preprocess_name="identity",
               preprocess_options=None, scale=1.0, initializer=None,
               name="deep_lstm"):
    """Creates an instance of `StandardDeepLSTM`.

    Args:
      output_size: Output sizes of the final linear layer.
      layers: Output sizes of LSTM layers.
      preprocess_name: Gradient preprocessing class name (in `l2l.preprocess` or
          tf modules). Default is `tf.identity`.
      preprocess_options: Gradient preprocessing options.
      scale: Gradient scaling (default is 1.0).
      initializer: Variable initializer for linear layer. See `nn.Linear` and
          `nn.LSTM` docs for more info. This parameter can be a string (e.g.
          "zeros" will be converted to tf.zeros_initializer).
      name: Module name.
    """
    super(StandardDeepLSTM, self).__init__(name)

    self._output_size = output_size
    self._scale = scale

    if hasattr(preprocess, preprocess_name):
      preprocess_class = getattr(preprocess, preprocess_name)
      self._preprocess = preprocess_class(**preprocess_options)
    else:
      self._preprocess = getattr(tf, preprocess_name)

    with tf.variable_scope(self._template.variable_scope):
      self._cores = []
      for i, size in enumerate(layers, start=1):
        name = "lstm_{}".format(i)
        init = _get_layer_initializers(initializer, name,
                                       ("w_gates", "b_gates"))
        self._cores.append(nn.LSTM(size, name=name, initializers=init))
      self._rnn = nn.DeepRNN(self._cores, skip_connections=False,
                             name="deep_rnn")

      init = _get_layer_initializers(initializer, "linear", ("w", "b"))
      self._linear = nn.Linear(output_size, name="linear", initializers=init)

  def _build(self, inputs, prev_state):
    """Connects the `StandardDeepLSTM` module into the graph.

    Args:
      inputs: 2D `Tensor` ([batch_size, input_size]).
      prev_state: `DeepRNN` state.

    Returns:
      `Tensor` shaped as `inputs`.
    """
    # Adds preprocessing dimension and preprocess.
    inputs = self._preprocess(tf.expand_dims(inputs, -1))
    # Incorporates preprocessing into data dimension.
    inputs = tf.reshape(inputs, [inputs.get_shape().as_list()[0], -1])
    output, next_state = self._rnn(inputs, prev_state)
    return self._linear(output) * self._scale, next_state

  def initial_state_for_inputs(self, inputs, **kwargs):
    batch_size = inputs.get_shape().as_list()[0]
    return self._rnn.initial_state(batch_size, **kwargs)


class CoordinateWiseDeepLSTM(StandardDeepLSTM):
  """Coordinate-wise `DeepLSTM`."""

  def __init__(self, name="cw_deep_lstm", **kwargs):
    """Creates an instance of `CoordinateWiseDeepLSTM`.

    Args:
      name: Module name.
      **kwargs: Additional `DeepLSTM` args.
    """
    super(CoordinateWiseDeepLSTM, self).__init__(1, name=name, **kwargs)

  def _reshape_inputs(self, inputs):
    return tf.reshape(inputs, [-1, 1])

  def _build(self, inputs, prev_state):
    """Connects the CoordinateWiseDeepLSTM module into the graph.

    Args:
      inputs: Arbitrarily shaped `Tensor`.
      prev_state: `DeepRNN` state.

    Returns:
      `Tensor` shaped as `inputs`.
    """
    input_shape = inputs.get_shape().as_list()
    reshaped_inputs = self._reshape_inputs(inputs)

    build_fn = super(CoordinateWiseDeepLSTM, self)._build
    output, next_state = build_fn(reshaped_inputs, prev_state)

    # Recover original shape.
    return tf.reshape(output, input_shape), next_state

  def initial_state_for_inputs(self, inputs, **kwargs):
    reshaped_inputs = self._reshape_inputs(inputs)
    return super(CoordinateWiseDeepLSTM, self).initial_state_for_inputs(
        reshaped_inputs, **kwargs)


class KernelDeepLSTM(StandardDeepLSTM):
  """`DeepLSTM` for convolutional filters.

  The inputs are assumed to be shaped as convolutional filters with an extra
  preprocessing dimension ([kernel_w, kernel_h, n_input_channels,
  n_output_channels]).
  """

  def __init__(self, kernel_shape, name="kernel_deep_lstm", **kwargs):
    """Creates an instance of `KernelDeepLSTM`.

    Args:
      kernel_shape: Kernel shape (2D `tuple`).
      name: Module name.
      **kwargs: Additional `DeepLSTM` args.
    """
    self._kernel_shape = kernel_shape
    output_size = np.prod(kernel_shape)
    super(KernelDeepLSTM, self).__init__(output_size, name=name, **kwargs)

  def _reshape_inputs(self, inputs):
    transposed_inputs = tf.transpose(inputs, perm=[2, 3, 0, 1])
    return tf.reshape(transposed_inputs, [-1] + self._kernel_shape)

  def _build(self, inputs, prev_state):
    """Connects the KernelDeepLSTM module into the graph.

    Args:
      inputs: 4D `Tensor` (convolutional filter).
      prev_state: `DeepRNN` state.

    Returns:
      `Tensor` shaped as `inputs`.
    """
    input_shape = inputs.get_shape().as_list()
    reshaped_inputs = self._reshape_inputs(inputs)

    build_fn = super(KernelDeepLSTM, self)._build
    output, next_state = build_fn(reshaped_inputs, prev_state)
    transposed_output = tf.transpose(output, [1, 0])

    # Recover original shape.
    return tf.reshape(transposed_output, input_shape), next_state

  def initial_state_for_inputs(self, inputs, **kwargs):
    """Batch size given inputs."""
    reshaped_inputs = self._reshape_inputs(inputs)
    return super(KernelDeepLSTM, self).initial_state_for_inputs(
        reshaped_inputs, **kwargs)


class Sgd(Network):
  """Identity network which acts like SGD."""

  def __init__(self, learning_rate=0.001, name="sgd"):
    """Creates an instance of the Identity optimizer network.

    Args:
      learning_rate: constant learning rate to use.
      name: Module name.
    """
    super(Sgd, self).__init__(name)
    self._learning_rate = learning_rate

  def _build(self, inputs, _):
    return -self._learning_rate * inputs, []

  def initial_state_for_inputs(self, inputs, **kwargs):
    return []


def _update_adam_estimate(estimate, value, b):
  return (b * estimate) + ((1 - b) * value)


def _debias_adam_estimate(estimate, b, t):
  return estimate / (1 - tf.pow(b, t))


class Adam(Network):
  """Adam algorithm (https://arxiv.org/pdf/1412.6980v8.pdf)."""

  def __init__(self, learning_rate=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8,
               name="adam"):
    """Creates an instance of Adam."""
    super(Adam, self).__init__(name=name)
    self._learning_rate = learning_rate
    self._beta1 = beta1
    self._beta2 = beta2
    self._epsilon = epsilon

  def _build(self, g, prev_state):
    """Connects the Adam module into the graph."""
    b1 = self._beta1
    b2 = self._beta2

    g_shape = g.get_shape().as_list()
    g = tf.reshape(g, (-1, 1))

    t, m, v = prev_state

    t_next = t + 1

    m_next = _update_adam_estimate(m, g, b1)
    m_hat = _debias_adam_estimate(m_next, b1, t_next)

    v_next = _update_adam_estimate(v, tf.square(g), b2)
    v_hat = _debias_adam_estimate(v_next, b2, t_next)

    update = -self._learning_rate * m_hat / (tf.sqrt(v_hat) + self._epsilon)
    return tf.reshape(update, g_shape), (t_next, m_next, v_next)

  def initial_state_for_inputs(self, inputs, dtype=tf.float32, **kwargs):
    batch_size = int(np.prod(inputs.get_shape().as_list()))
    t = tf.zeros((), dtype=dtype)
    m = tf.zeros((batch_size, 1), dtype=dtype)
    v = tf.zeros((batch_size, 1), dtype=dtype)
    return (t, m, v)
