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
"""Tests for L2L problems."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange
import tensorflow as tf

from nose_parameterized import parameterized

import problems


class SimpleTest(tf.test.TestCase):
  """Tests simple problem."""

  def testShape(self):
    problem = problems.simple()
    f = problem()
    self.assertEqual(f.get_shape().as_list(), [])

  def testVariables(self):
    problem = problems.simple()
    problem()
    variables = tf.trainable_variables()
    self.assertEqual(len(variables), 1)
    self.assertEqual(variables[0].get_shape().as_list(), [])

  @parameterized.expand([(-1,), (0,), (1,), (10,)])
  def testValues(self, value):
    problem = problems.simple()
    f = problem()

    with self.test_session() as sess:
      output = sess.run(f, feed_dict={"x:0": value})
      self.assertEqual(output, value**2)


class SimpleMultiOptimizerTest(tf.test.TestCase):
  """Tests multi-optimizer simple problem."""

  def testShape(self):
    num_dims = 3
    problem = problems.simple_multi_optimizer(num_dims=num_dims)
    f = problem()
    self.assertEqual(f.get_shape().as_list(), [])

  def testVariables(self):
    num_dims = 3
    problem = problems.simple_multi_optimizer(num_dims=num_dims)
    problem()
    variables = tf.trainable_variables()
    self.assertEqual(len(variables), num_dims)
    for v in variables:
      self.assertEqual(v.get_shape().as_list(), [])

  @parameterized.expand([(-1,), (0,), (1,), (10,)])
  def testValues(self, value):
    problem = problems.simple_multi_optimizer(num_dims=1)
    f = problem()

    with self.test_session() as sess:
      output = sess.run(f, feed_dict={"x_0:0": value})
      self.assertEqual(output, value**2)


class QuadraticTest(tf.test.TestCase):
  """Tests Quadratic problem."""

  def testShape(self):
    problem = problems.quadratic()
    f = problem()
    self.assertEqual(f.get_shape().as_list(), [])

  def testVariables(self):
    batch_size = 5
    num_dims = 3
    problem = problems.quadratic(batch_size=batch_size, num_dims=num_dims)
    problem()
    variables = tf.trainable_variables()
    self.assertEqual(len(variables), 1)
    self.assertEqual(variables[0].get_shape().as_list(), [batch_size, num_dims])

  @parameterized.expand([(-1,), (0,), (1,), (10,)])
  def testValues(self, value):
    problem = problems.quadratic(batch_size=1, num_dims=1)
    f = problem()

    w = 2.0
    y = 3.0

    with self.test_session() as sess:
      output = sess.run(f, feed_dict={"x:0": [[value]],
                                      "w:0": [[[w]]],
                                      "y:0": [[y]]})
      self.assertEqual(output, ((w * value) - y)**2)


class EnsembleTest(tf.test.TestCase):
  """Tests Ensemble problem."""

  def testShape(self):
    num_dims = 3
    problem_defs = [{"name": "simple", "options": {}} for _ in xrange(num_dims)]
    ensemble = problems.ensemble(problem_defs)
    f = ensemble()
    self.assertEqual(f.get_shape().as_list(), [])

  def testVariables(self):
    num_dims = 3
    problem_defs = [{"name": "simple", "options": {}} for _ in xrange(num_dims)]
    ensemble = problems.ensemble(problem_defs)
    ensemble()
    variables = tf.trainable_variables()
    self.assertEqual(len(variables), num_dims)
    for v in variables:
      self.assertEqual(v.get_shape().as_list(), [])

  @parameterized.expand([(-1,), (0,), (1,), (10,)])
  def testValues(self, value):
    num_dims = 1
    weight = 0.5
    problem_defs = [{"name": "simple", "options": {}} for _ in xrange(num_dims)]
    ensemble = problems.ensemble(problem_defs, weights=[weight])
    f = ensemble()

    with self.test_session() as sess:
      output = sess.run(f, feed_dict={"problem_0/x:0": value})
      self.assertEqual(output, weight * value**2)


if __name__ == "__main__":
  tf.test.main()
