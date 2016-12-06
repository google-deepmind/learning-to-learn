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
"""Tests for L2L preprocessors."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import preprocess


class ClampTest(tf.test.TestCase):
  """Tests Clamp module."""

  def testShape(self):
    shape = [2, 3]
    inputs = tf.random_normal(shape)
    clamp = preprocess.Clamp(min_value=-1.0, max_value=1.0)
    output = clamp(inputs)
    self.assertEqual(output.get_shape().as_list(), shape)

  def testMin(self):
    shape = [100]
    inputs = tf.random_normal(shape)
    clamp = preprocess.Clamp(min_value=0.0)
    output = clamp(inputs)

    with self.test_session() as sess:
      output_np = sess.run(output)
      self.assertTrue(np.all(np.greater_equal(output_np, np.zeros(shape))))

  def testMax(self):
    shape = [100]
    inputs = tf.random_normal(shape)
    clamp = preprocess.Clamp(max_value=0.0)
    output = clamp(inputs)

    with self.test_session() as sess:
      output_np = sess.run(output)
      self.assertTrue(np.all(np.less_equal(output_np, np.zeros(shape))))

  def testMinAndMax(self):
    shape = [100]
    inputs = tf.random_normal(shape)
    clamp = preprocess.Clamp(min_value=0.0, max_value=0.0)
    output = clamp(inputs)

    with self.test_session() as sess:
      output_np = sess.run(output)
      self.assertAllEqual(output_np, np.zeros(shape))


class LogAndSignTest(tf.test.TestCase):
  """Tests LogAndSign module."""

  def testShape(self):
    shape = [2, 3]
    inputs = tf.random_normal(shape)
    module = preprocess.LogAndSign(k=1)
    output = module(inputs)
    self.assertEqual(output.get_shape().as_list(), shape[:-1] + [shape[-1] * 2])

  def testLogWithOnes(self):
    shape = [1]
    inputs = tf.ones(shape)
    module = preprocess.LogAndSign(k=10)
    output = module(inputs)

    with self.test_session() as sess:
      output_np = sess.run(output)
      log_np = output_np[0]
      self.assertAlmostEqual(log_np, 0.0)

  def testSign(self):
    shape = [2, 1]
    inputs = tf.random_normal(shape)
    module = preprocess.LogAndSign(k=1)
    output = module(inputs)

    with self.test_session() as sess:
      inputs_np, output_np = sess.run([inputs, output])
      sign_np = output_np[:, 1:]
      self.assertAllEqual(np.sign(sign_np), np.sign(inputs_np))


if __name__ == "__main__":
  tf.test.main()
