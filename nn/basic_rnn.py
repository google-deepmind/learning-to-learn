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
"""Basic RNN Cores for TensorFlow nn.

This file contains the definitions of the simplest building blocks for Recurrent
Neural Networks.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import tensorflow as tf

from tensorflow.python.framework import tensor_shape
from tensorflow.python.util import nest

from nn import rnn_core


def _get_flat_core_sizes(cores):
  """Obtains the list flattened output sizes of a list of cores.

  Args:
    cores: list of cores to get the shapes from.

  Returns:
    List of lists that, for each core, contains the list of its output
      dimensions.
  """
  core_sizes_lists = []
  for core in cores:
    flat_output_size = nest.flatten(core.output_size)
    core_sizes_lists.append([tensor_shape.as_shape(size).as_list()
                             for size in flat_output_size])
  return core_sizes_lists


class DeepRNN(rnn_core.RNNCore):
  """RNN core which passes data through a number of internal modules or ops.

  This module is constructed by passing an iterable of externally constructed
  modules or ops. The DeepRNN takes `(input, prev_state)` as input and passes
  the input through each internal module in the order they were presented,
  using elements from `prev_state` as necessary for internal recurrent cores.
  The output is `(output, next_state)` in common with other RNN cores.
  By default, skip connections from the input to all internal modules and from
  each intermediate output to the final output are used.

  E.g.:

  ```python
  lin = nn.Linear(hidden_size=128)
  tanh = tf.tanh
  lstm = nn.LSTM(hidden_size=256)
  deep_rnn = nn.DeepRNN([lin, tanh, lstm])
  output, next_state = deep_rnn(input, prev_state)
  ```

  The computation set up inside the DeepRNN has the same effect as:

  ```python
  lin_output = lin(input)
  tanh_output = tanh(tf.concat(1, [input, lin_output]))
  lstm_output, lstm_next_state = lstm(
      tf.concat(1, [input, tanh_output]), prev_state[0])

  next_state = (lstm_next_state,)
  output = tf.concat(1, [lin_output, tanh_output, lstm_output])
  ```

  Every internal module receives the preceding module's output and the entire
  core's input. The output is created by concatenating each internal module's
  output. In the case of internal recurrent elements, corresponding elements
  of the state are used such that `state[i]` is passed to the `i`'th internal
  recurrent element. Note that the state of a `DeepRNN` is always a tuple, which
  will contain the same number of elements as there are internal recurrent
  cores. If no internal modules are recurrent, the state of the DeepRNN as a
  whole is the empty tuple. Wrapping non-recurrent modules into a DeepRNN can
  be useful to produce something API compatible with a "real" recurrent module,
  simplifying code that handles the cores.

  Without skip connections the previous example would become the following
  (note the only difference is the addition of `skip_connections=False`):

  ```python
  # ... declare other modules as above
  deep_rnn = nn.DeepRNN([lin, tanh, lstm], skip_connections=False)
  output, next_state = deep_rnn(input, prev_state)
  ```

  which is equivalent to:

  ```python
  lin_output = lin(input)
  tanh_output = tanh(lin_output)
  lstm_output, lstm_next_state = lstm(tanh_output, prev_state[0])

  next_state = (lstm_next_state,)
  output = lstm_output
  ```
  """

  def __init__(self, cores, skip_connections=True, name="deep_rnn"):
    """Construct a Deep RNN core.

    Args:
      cores: iterable of modules or ops.
      skip_connections: a boolean that indicates whether to use skip
        connections. This means that the input is fed to all the layers, after
        being concatenated with the output of the previous layer. The output
        of the module will be the concatenation of all the outputs of the
        internal modules.
      name: name of the module.

    Raises:
      ValueError: if `cores` is not an iterable.
    """
    super(DeepRNN, self).__init__(name=name)

    if not isinstance(cores, collections.Iterable):
      raise ValueError("Cores should be an iterable object.")
    self._cores = tuple(cores)
    self._skip_connections = skip_connections

    if self._skip_connections:
      self._check_cores_output_sizes()

    self._is_recurrent_list = [isinstance(core, rnn_core.RNNCore)
                               for core in self._cores]
    self._num_recurrent = sum(self._is_recurrent_list)

  def _check_cores_output_sizes(self):
    """Checks the output_sizes of the cores of the DeepRNN module.

    Raises:
      ValueError: if the outputs of the cores cannot be concatenated along their
        first dimension.
    """
    for core_sizes in zip(*tuple(_get_flat_core_sizes(self._cores))):
      first_core_list = core_sizes[0][1:]
      for i, core_list in enumerate(core_sizes[1:]):
        if core_list[1:] != first_core_list:
          raise ValueError("The outputs of the provided cores are not able "
                           "to be concatenated along the first feature "
                           "dimension. Core 0 has size %s, whereas Core %d "
                           "has size %s" % (first_core_list, i, core_list))

  def _build(self, inputs, prev_state):
    """Connects the DeepRNN module into the graph.

    If this is not the first time the module has been connected to the graph,
    the Tensors provided as input_ and state must have the same final
    dimension, in order for the existing variables to be the correct size for
    their corresponding multiplications. The batch size may differ for each
    connection.

    Args:
      inputs: a nested tuple of Tensors of arbitrary dimensionality, with at
        least an initial batch dimension.
      prev_state: a tuple of `prev_state`s that corresponds to the state
        of each one of the cores of the `DeepCore`.

    Returns:
      output: a nested tuple of Tensors of arbitrary dimensionality, with at
        least an initial batch dimension.
      next_state: a tuple of `next_state`s that corresponds to the updated state
        of each one of the cores of the `DeepCore`.

    Raises:
      ValueError: if connecting the module into the graph any time after the
        first time, and the inferred size of the inputs does not match previous
        invocations. This may happen if one connects a module any time after the
        first time that does not have the configuration of skip connections as
        the first time.
    """
    current_input = inputs
    next_states = []
    outputs = []
    recurrent_idx = 0
    for i, core in enumerate(self._cores):
      if self._skip_connections and i > 0:
        flat_input = (nest.flatten(inputs), nest.flatten(current_input))
        flat_input = [tf.concat(1, input_) for input_ in zip(*flat_input)]
        current_input = nest.pack_sequence_as(structure=inputs,
                                              flat_sequence=flat_input)

      # Determine if this core in the stack is recurrent or not and call
      # accordingly.
      if self._is_recurrent_list[i]:
        current_input, next_state = core(current_input,
                                         prev_state[recurrent_idx])
        next_states.append(next_state)
        recurrent_idx += 1
      else:
        current_input = core(current_input)

      if self._skip_connections:
        outputs.append(current_input)

    if self._skip_connections:
      flat_outputs = tuple(nest.flatten(output) for output in outputs)
      flat_outputs = [tf.concat(1, output) for output in zip(*flat_outputs)]
      output = nest.pack_sequence_as(structure=outputs[0],
                                     flat_sequence=flat_outputs)
    else:
      output = current_input

    return output, tuple(next_states)

  def initial_state(self, batch_size, dtype=tf.float32, trainable=False,
                    trainable_initializers=None):
    """Builds the default start state for a DeepRNN.

    Args:
      batch_size: An int, float or scalar Tensor representing the batch size.
      dtype: The data type to use for the state.
      trainable: Boolean that indicates whether to learn the initial state.
      trainable_initializers: An initializer function or nested structure of
          functions with same structure as the `state_size` property of the
          core, to be used as initializers of the initial state variable.

    Returns:
      A tensor or nested tuple of tensors with same structure and shape as the
      `state_size` property of the core.

    Raises:
      ValueError: if the number of passed initializers is not the same as the
          number of recurrent cores.
    """
    initial_state = []
    if trainable_initializers is None:
      trainable_initializers = [None] * self._num_recurrent

    num_initializers = len(trainable_initializers)

    if num_initializers != self._num_recurrent:
      raise ValueError("The number of initializers and recurrent cores should "
                       "be the same. Received %d initializers for %d specified "
                       "recurrent cores."
                       % (num_initializers, self._num_recurrent))

    recurrent_idx = 0
    for is_recurrent, core in zip(self._is_recurrent_list, self._cores):
      if is_recurrent:
        with tf.variable_scope("%s-rec_core%d" % (self.name, recurrent_idx)):
          core_initial_state = core.initial_state(
              batch_size, dtype=dtype, trainable=trainable,
              trainable_initializers=trainable_initializers[recurrent_idx])
        initial_state.append(core_initial_state)
        recurrent_idx += 1
    return tuple(initial_state)

  @property
  def state_size(self):
    sizes = []
    for is_recurrent, core in zip(self._is_recurrent_list, self._cores):
      if is_recurrent:
        sizes.append(core.state_size)
    return tuple(sizes)

  @property
  def output_size(self):
    if self._skip_connections:
      output_size = []
      for core_sizes in zip(*tuple(_get_flat_core_sizes(self._cores))):
        added_core_size = core_sizes[0]
        added_core_size[0] = sum([size[0] for size in core_sizes])
        output_size.append(tf.TensorShape(added_core_size))
      return nest.pack_sequence_as(structure=self._cores[0].output_size,
                                   flat_sequence=output_size)
    else:
      return self._cores[-1].output_size
