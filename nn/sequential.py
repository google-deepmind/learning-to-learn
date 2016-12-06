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
"""Sequential Module for TensorFlow nn.

A Module that wraps a list of other modules and ops, connecting the output of
each to the input of the next.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nn import base


class Sequential(base.AbstractModule):
  """Builds a module out of a sequence of callables."""

  def __init__(self, layers, name="sequential"):
    """Constructs a Sequential module.

    This feeds the output of each layer into the next and returns the output
    of the final layer.

    If a layer returns a tuple, it is assumed that this must be unpacked into
    the argument list of the next layer. If it is not a tuple, it is simply
    passed through to the next layer unchanged.

    Args:
      layers: Iterable of callables to stack together, which can be modules
          or ops.
      name: Name of the module.

    Raises:
      TypeError: If `layers` is None or contains any non-callable items.
    """
    super(Sequential, self).__init__(name=name)

    # Store a copy of the iterable in a tuple to ensure users cannot modify the
    # iterable later, and protect against iterables which can only be read once.
    self._layers = tuple(layers)

    is_not_callable = [(i, mod) for i, mod in enumerate(self._layers)
                       if not callable(mod)]

    if is_not_callable:
      raise TypeError("Items {} not callable with types: {}".format(
          ", ".join(str(i) for i, _ in is_not_callable),
          ", ".join(type(layer).__name__ for _, layer in is_not_callable)))

  def _build(self, *args):
    """Connects the Sequential module into the graph.

    Args:
      *args: A tuple of inputs, to be unpacked as the arguments to the first
          layer.

    Returns:
      The output value of the last layer.
    """
    net = args

    for layer in self._layers:
      if isinstance(net, tuple):
        net = layer(*net)
      else:
        net = layer(net)

    return net

  @property
  def layers(self):
    return self._layers
