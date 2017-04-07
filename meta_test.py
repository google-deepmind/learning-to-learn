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
"""Tests for L2L meta-optimizer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile

from nose_parameterized import parameterized
import numpy as np
from six.moves import xrange
import sonnet as snt
import tensorflow as tf

import meta
import problems


def train(sess, minimize_ops, num_epochs, num_unrolls):
  """L2L training."""
  step, update, reset, loss_last, x_last = minimize_ops

  for _ in xrange(num_epochs):
    sess.run(reset)
    for _ in xrange(num_unrolls):
      cost, final_x, unused_1, unused_2 = sess.run([loss_last, x_last,
                                                    update, step])

  return cost, final_x


class L2LTest(tf.test.TestCase):
  """Tests L2L meta-optimizer."""

  def testResults(self):
    """Tests reproducibility of Torch results."""
    problem = problems.simple()
    optimizer = meta.MetaOptimizer(net=dict(
        net="CoordinateWiseDeepLSTM",
        net_options={
            "layers": (),
            "initializer": "zeros"
        }))
    minimize_ops = optimizer.meta_minimize(problem, 5)
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      cost, final_x = train(sess, minimize_ops, 1, 2)

    # Torch results
    torch_cost = 0.7325327
    torch_final_x = 0.8559

    self.assertAlmostEqual(cost, torch_cost, places=4)
    self.assertAlmostEqual(final_x[0], torch_final_x, places=4)

  @parameterized.expand([
      # Shared optimizer.
      (
          None,
          {
              "net": {
                  "net": "CoordinateWiseDeepLSTM",
                  "net_options": {"layers": (1, 1,)}
              }
          }
      ),
      # Explicit sharing.
      (
          [("net", ["x_0", "x_1"])],
          {
              "net": {
                  "net": "CoordinateWiseDeepLSTM",
                  "net_options": {"layers": (1,)}
              }
          }
      ),
      # Different optimizers.
      (
          [("net1", ["x_0"]), ("net2", ["x_1"])],
          {
              "net1": {
                  "net": "CoordinateWiseDeepLSTM",
                  "net_options": {"layers": (1,)}
              },
              "net2": {"net": "Adam"}
          }
      ),
      # Different optimizers for the same variable.
      (
          [("net1", ["x_0"]), ("net2", ["x_0"])],
          {
              "net1": {
                  "net": "CoordinateWiseDeepLSTM",
                  "net_options": {"layers": (1,)}
              },
              "net2": {
                  "net": "CoordinateWiseDeepLSTM",
                  "net_options": {"layers": (1,)}
              }
          }
      ),
  ])
  def testMultiOptimizer(self, net_assignments, net_config):
    """Tests different variable->net mappings in multi-optimizer problem."""
    problem = problems.simple_multi_optimizer(num_dims=2)
    optimizer = meta.MetaOptimizer(**net_config)
    minimize_ops = optimizer.meta_minimize(problem, 3,
                                           net_assignments=net_assignments)
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      train(sess, minimize_ops, 1, 2)

  def testSecondDerivatives(self):
    """Tests second derivatives for simple problem."""
    problem = problems.simple()
    optimizer = meta.MetaOptimizer(net=dict(
        net="CoordinateWiseDeepLSTM",
        net_options={"layers": ()}))
    minimize_ops = optimizer.meta_minimize(problem, 3,
                                           second_derivatives=True)
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      train(sess, minimize_ops, 1, 2)

  def testConvolutional(self):
    """Tests L2L applied to problem with convolutions."""
    kernel_shape = 4
    def convolutional_problem():
      conv = snt.Conv2D(output_channels=1,
                        kernel_shape=kernel_shape,
                        stride=1,
                        name="conv")
      output = conv(tf.random_normal((100, 100, 3, 10)))
      return tf.reduce_sum(output)

    net_config = {
        "conv": {
            "net": "KernelDeepLSTM",
            "net_options": {
                "kernel_shape": [kernel_shape] * 2,
                "layers": (5,)
            },
        },
    }
    optimizer = meta.MetaOptimizer(**net_config)
    minimize_ops = optimizer.meta_minimize(
        convolutional_problem, 3,
        net_assignments=[("conv", ["conv/w"])]
    )
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      train(sess, minimize_ops, 1, 2)

  def testWhileLoopProblem(self):
    """Tests L2L applied to problem with while loop."""
    def while_loop_problem():
      x = tf.get_variable("x", shape=[], initializer=tf.ones_initializer())

      # Strange way of squaring the variable.
      _, x_squared = tf.while_loop(
          cond=lambda t, _: t < 1,
          body=lambda t, x: (t + 1, x * x),
          loop_vars=(0, x),
          name="loop")
      return x_squared

    optimizer = meta.MetaOptimizer(net=dict(
        net="CoordinateWiseDeepLSTM",
        net_options={"layers": ()}))
    minimize_ops = optimizer.meta_minimize(while_loop_problem, 3)
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      train(sess, minimize_ops, 1, 2)

  def testSaveAndLoad(self):
    """Tests saving and loading a meta-optimizer."""
    layers = (2, 3)
    net_options = {"layers": layers, "initializer": "zeros"}
    num_unrolls = 2
    num_epochs = 1

    problem = problems.simple()

    # Original optimizer.
    with tf.Graph().as_default() as g1:
      optimizer = meta.MetaOptimizer(net=dict(
          net="CoordinateWiseDeepLSTM",
          net_options=net_options))
      minimize_ops = optimizer.meta_minimize(problem, 3)

    with self.test_session(graph=g1) as sess:
      sess.run(tf.global_variables_initializer())
      train(sess, minimize_ops, 1, 2)

      # Save optimizer.
      tmp_dir = tempfile.mkdtemp()
      save_result = optimizer.save(sess, path=tmp_dir)
      net_path = next(iter(save_result))

      # Retrain original optimizer.
      cost, x = train(sess, minimize_ops, num_unrolls, num_epochs)

    # Load optimizer and retrain in a new session.
    with tf.Graph().as_default() as g2:
      optimizer = meta.MetaOptimizer(net=dict(
          net="CoordinateWiseDeepLSTM",
          net_options=net_options,
          net_path=net_path))
      minimize_ops = optimizer.meta_minimize(problem, 3)

    with self.test_session(graph=g2) as sess:
      sess.run(tf.global_variables_initializer())
      cost_loaded, x_loaded = train(sess, minimize_ops, num_unrolls, num_epochs)

    # The last cost should be the same.
    self.assertAlmostEqual(cost, cost_loaded, places=3)
    self.assertAlmostEqual(x[0], x_loaded[0], places=3)

    # Cleanup.
    os.remove(net_path)
    os.rmdir(tmp_dir)


if __name__ == "__main__":
  tf.test.main()
