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

"""Utility functions for dealing with nn Modules."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re

import tensorflow as tf


def get_variables_in_scope(scope, collection=tf.GraphKeys.TRAINABLE_VARIABLES):
  """Returns a tuple `tf.Variable`s in a scope for a given collection.

  Args:
    scope: `tf.VariableScope` instance to retrieve variables from.
    collection: Collection to restrict query to. By default this is
        `tf.Graphkeys.TRAINABLE_VARIABLES`, which doesn't include non-trainable
        variables such as moving averages.

  Returns:
    A tuple of `tf.Variable` objects.
  """
  # Escape the name in case it contains any "." characters. Add a closing slash
  # so we will not search any scopes that have this scope name as a prefix.
  scope_name = re.escape(scope.name) + "/"

  return tuple(tf.get_collection(collection, scope_name))


def get_variables_in_module(module,
                            collection=tf.GraphKeys.TRAINABLE_VARIABLES):
  """Returns tuple of `tf.Variable`s declared inside an `nn.Module`.

  Note that this operates by searching the variable scope a module contains,
  and so does not know about any modules which were constructed elsewhere but
  used inside this module.

  Args:
    module: `nn.Module` instance to query the scope of.
    collection: Collection to restrict query to. By default this is
      `tf.Graphkeys.TRAINABLE_VARIABLES`, which doesn't include non-trainable
      variables such as moving averages.

  Returns:
    A tuple of `tf.Variable` objects.

  Raises:
    NotConnectedError: If the module is not connected to the Graph.
  """
  return get_variables_in_scope(module.variable_scope, collection=collection)


def check_initializers(initializers, keys):
  """Checks the given initializers.

  This checks that `initializers` is a dictionary that only contains keys in
  `keys`, and furthermore the entries in `initializers` are functions or
  further dictionaries (the latter used, for example, in passing initializers
  to modules inside modules) which must satisfy the same constraints.

  Args:
    initializers: Dictionary of initializers (allowing nested dictionaries) or
      None.
    keys: Iterable of valid keys for `initializers`.

  Returns:
    Copy of checked dictionary of initializers.

  Raises:
    KeyError: If an initializer is provided for a key not in `keys`.
    TypeError: If a provided initializer is not a callable function, or if the
      dict of initializers is not in fact a dict.
  """
  if initializers is None:
    return {}

  keys = set(keys)

  # If the user is creating modules that nests other modules, then it is
  # possible that they might not nest the initializer dictionaries correctly. If
  # that is the case, then we might find that initializers is not a dict here.
  # We raise a helpful exception in this case.
  if not issubclass(type(initializers), dict):
    raise TypeError("A dict of initializers was expected, but not "
                    "given. You should double-check that you've nested the "
                    "initializers for any sub-modules correctly.")

  if not set(initializers) <= keys:
    extra_keys = set(initializers) - keys
    raise KeyError(
        "Invalid initializer keys {}, initializers can only "
        "be provided for {}".format(
            ", ".join("'{}'".format(key) for key in extra_keys),
            ", ".join("'{}'".format(key) for key in keys)))

  def check_nested_callables(dictionary):
    for key, entry in dictionary.items():
      if isinstance(entry, dict):
        check_nested_callables(entry)
      elif not callable(entry):
        raise TypeError(
            "Initializer for '{}' is not a callable function or dictionary"
            .format(key))

  check_nested_callables(initializers)

  return dict(initializers)


def check_partitioners(partitioners, keys):
  """Checks the given partitioners.

  This checks that `partitioners` is a dictionary that only contains keys in
  `keys`, and furthermore the entries in `partitioners` are functions or
  further dictionaries (the latter used, for example, in passing partitioners
  to modules inside modules) which must satisfy the same constraints.

  Args:
    partitioners: Dictionary of partitioners (allowing nested dictionaries) or
        None.
    keys: Iterable of valid keys for `partitioners`.

  Returns:
    Checked dictionary of partitioners.

  Raises:
    KeyError: If an partitioner is provided for a key not in `keys`.
    TypeError: If a provided partitioner is not a callable function.
  """
  if partitioners is None:
    return {}

  keys = set(keys)

  if not set(partitioners) <= keys:
    extra_keys = set(partitioners) - keys
    raise KeyError(
        "Invalid partitioner keys {}, partitioners can only "
        "be provided for {}".format(
            ", ".join("'{}'".format(key) for key in extra_keys),
            ", ".join("'{}'".format(key) for key in keys)))

  def check_nested_callables(dictionary):
    for key, entry in dictionary.items():
      if isinstance(entry, dict):
        check_nested_callables(entry)
      elif not callable(entry):
        raise TypeError(
            "Partitioner for '{}' is not a callable function or dictionary"
            .format(key))

  check_nested_callables(partitioners)

  return partitioners
