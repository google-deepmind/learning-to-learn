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
"""Tests for L2L networks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nose_parameterized import parameterized
import numpy as np
import tensorflow as tf

import networks
import nn


class CoordinateWiseDeepLSTMTest(tf.test.TestCase):
  """Tests CoordinateWiseDeepLSTM network."""

  def testShape(self):
    shape = [10, 5]
    gradients = tf.random_normal(shape)
    net = networks.CoordinateWiseDeepLSTM(layers=(1, 1))
    state = net.initial_state_for_inputs(gradients)
    update, _ = net(gradients, state)
    self.assertEqual(update.get_shape().as_list(), shape)

  def testTrainable(self):
    """Tests the network contains trainable variables."""
    shape = [10, 5]
    gradients = tf.random_normal(shape)
    net = networks.CoordinateWiseDeepLSTM(layers=(1,))
    state = net.initial_state_for_inputs(gradients)
    net(gradients, state)
    # Weights and biases for two layers.
    variables = nn.get_variables_in_module(net)
    self.assertEqual(len(variables), 4)

  @parameterized.expand([
      ["zeros"],
      [{"w": "zeros", "b": "zeros", "bad": "bad"}],
      [{"w": tf.zeros_initializer, "b": np.array([0])}],
      [{"linear": {"w": tf.zeros_initializer, "b": "zeros"}}]
  ])
  def testResults(self, initializer):
    """Tests zero updates when last layer is initialized to zero."""
    shape = [10]
    gradients = tf.random_normal(shape)
    net = networks.CoordinateWiseDeepLSTM(layers=(1, 1),
                                          initializer=initializer)
    state = net.initial_state_for_inputs(gradients)
    update, _ = net(gradients, state)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      update_np = sess.run(update)
      self.assertAllEqual(update_np, np.zeros(shape))


class KernelDeepLSTMTest(tf.test.TestCase):
  """Tests KernelDeepLSTMTest network."""

  def testShape(self):
    kernel_shape = [5, 5]
    shape = kernel_shape + [2, 2]  # The input has to be 4-dimensional.
    gradients = tf.random_normal(shape)
    net = networks.KernelDeepLSTM(layers=(1, 1), kernel_shape=kernel_shape)
    state = net.initial_state_for_inputs(gradients)
    update, _ = net(gradients, state)
    self.assertEqual(update.get_shape().as_list(), shape)

  def testTrainable(self):
    """Tests the network contains trainable variables."""
    kernel_shape = [5, 5]
    shape = kernel_shape + [2, 2]  # The input has to be 4-dimensional.
    gradients = tf.random_normal(shape)
    net = networks.KernelDeepLSTM(layers=(1,), kernel_shape=kernel_shape)
    state = net.initial_state_for_inputs(gradients)
    net(gradients, state)
    # Weights and biases for two layers.
    variables = nn.get_variables_in_module(net)
    self.assertEqual(len(variables), 4)

  @parameterized.expand([
      ["zeros"],
      [{"w": "zeros", "b": "zeros", "bad": "bad"}],
      [{"w": tf.zeros_initializer, "b": np.array([0])}],
      [{"linear": {"w": tf.zeros_initializer, "b": "zeros"}}]
  ])
  def testResults(self, initializer):
    """Tests zero updates when last layer is initialized to zero."""
    kernel_shape = [5, 5]
    shape = kernel_shape + [2, 2]  # The input has to be 4-dimensional.
    gradients = tf.random_normal(shape)
    net = networks.KernelDeepLSTM(layers=(1, 1),
                                  kernel_shape=kernel_shape,
                                  initializer=initializer)
    state = net.initial_state_for_inputs(gradients)
    update, _ = net(gradients, state)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      update_np = sess.run(update)
      self.assertAllEqual(update_np, np.zeros(shape))


class SgdTest(tf.test.TestCase):
  """Tests Sgd network."""

  def testShape(self):
    shape = [10, 5]
    gradients = tf.random_normal(shape)
    net = networks.Sgd()
    state = net.initial_state_for_inputs(gradients)
    update, _ = net(gradients, state)
    self.assertEqual(update.get_shape().as_list(), shape)

  def testNonTrainable(self):
    """Tests the network doesn't contain trainable variables."""
    shape = [10, 5]
    gradients = tf.random_normal(shape)
    net = networks.Sgd()
    state = net.initial_state_for_inputs(gradients)
    net(gradients, state)
    variables = nn.get_variables_in_module(net)
    self.assertEqual(len(variables), 0)

  def testResults(self):
    """Tests network produces zero updates with learning rate equal to zero."""
    shape = [10]
    learning_rate = 0.01
    gradients = tf.random_normal(shape)
    net = networks.Sgd(learning_rate=learning_rate)
    state = net.initial_state_for_inputs(gradients)
    update, _ = net(gradients, state)

    with self.test_session() as sess:
      gradients_np, update_np = sess.run([gradients, update])
      self.assertAllEqual(update_np, -learning_rate * gradients_np)


class AdamTest(tf.test.TestCase):
  """Tests Adam network."""

  def testShape(self):
    shape = [10, 5]
    gradients = tf.random_normal(shape)
    net = networks.Adam()
    state = net.initial_state_for_inputs(gradients)
    update, _ = net(gradients, state)
    self.assertEqual(update.get_shape().as_list(), shape)

  def testNonTrainable(self):
    """Tests the network doesn't contain trainable variables."""
    shape = [10, 5]
    gradients = tf.random_normal(shape)
    net = networks.Adam()
    state = net.initial_state_for_inputs(gradients)
    net(gradients, state)
    variables = nn.get_variables_in_module(net)
    self.assertEqual(len(variables), 0)

  def testZeroLearningRate(self):
    """Tests network produces zero updates with learning rate equal to zero."""
    shape = [10]
    gradients = tf.random_normal(shape)
    net = networks.Adam(learning_rate=0)
    state = net.initial_state_for_inputs(gradients)
    update, _ = net(gradients, state)

    with self.test_session() as sess:
      update_np = sess.run(update)
      self.assertAllEqual(update_np, np.zeros(shape))


if __name__ == "__main__":
  tf.test.main()
