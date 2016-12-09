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
"""Batch normalization module for nn.

This contains the module BatchNorm, which performs batch normalization on
its inputs. It has an optional post-normalization scale and offset, and it
maintains moving averages of the statistics for use at test time.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.contrib.layers.python.layers import utils
from tensorflow.python.training import moving_averages
from nn import base
from nn import util


class BatchNorm(base.AbstractModule):
  """Batch normalization module, including optional affine transformation.

  This module maintains exponential moving averages of the mean and
  variance, used for calculating more accurate shifted statistics at training
  time and optionally used to normalize at test time.

  In order to update the moving averages, the user must run the
  ops in the tf.GraphKeys.UPDATE_OPS TensorFlow collection. For example:

      bn = BatchNorm()
      train_net = bn(train_inputs, is_training=True)
      test_net = bn(test_inputs, is_training=False, test_local_stats=False)

      ...

      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      with tf.control_dependencies(update_ops):
        train_op = tf.group(train_op)

  Then, whenever `train_op` is run so also are the moving average update ops.

  At training time, batch statistics (mean, variance) are not shared between
  separate connections. The moving averages are shared between separate
  connections. At both training and test time, the optional affine
  transformations are shared between separate connections.

  Local batch statistics are used by default at test time, but the moving
  averages can be used by specifying a flag when connecting. One often wants
  to use local batch statistics at test time to track the progress while the
  model is trained as it would ensure that moving average updates do not affect
  the training curves. Once the training is finished, it's often advantageous
  to use moving average statistics, since it would make evaluation agnostic to
  the batch size, and might even lead to small improvements over the local
  batch statistics.
  """

  GAMMA = "gamma"
  BETA = "beta"
  POSSIBLE_INITIALIZER_KEYS = {GAMMA, BETA}

  def __init__(self, reduction_indices=None, offset=True, scale=False,
               decay_rate=0.999, eps=1e-3, initializers=None,
               use_legacy_moving_second_moment=False,
               name="batch_norm"):
    """Constructs a BatchNorm module.

    By default reduces over all input tensor dimensions apart from the final
    dimension. This has the effect of treating pixels in 1D/2D/3D images as
    additional elements of the minibatch.

    If this is not the desired behaviour, the user can specify the tensor
    indices to reduce over with `reduction_indices`.

    Args:
      reduction_indices: Optional indices of dimensions to reduce over.
      offset: Optional boolean to specify whether or not to apply a trained
        component-wise bias after the batch normalization and scaling.
      scale: Optional boolean to specify whether or not to apply a trained
        component-wise scale after the batch normalization.
      decay_rate: Decay rate of the exponential moving averages of the mean
        and variance.
      eps: Small number to avoid dividing by zero when diving by the standard
        deviation.
      initializers: Optional dict containing ops to initialize the weights of
        the affine transform (`gamma` and `beta`).
      use_legacy_moving_second_moment: Keep a moving second moment, rather than
        the moving variance. This is deprecated, but is kept for backwards
        compatability with old checkpoints. By default `False`.
      name: Name of the module.

    Raises:
      base.Error: If initializers contains any keys other
          than `gamma` or `beta`.
      ValueError: If `use_legacy_moving_second_moment` is not `True`.
    """
    super(BatchNorm, self).__init__(name)

    self._reduction_indices = reduction_indices
    self._offset = offset
    self._scale = scale
    self._decay_rate = decay_rate
    self._eps = eps
    self._use_legacy_moving_second_moment = use_legacy_moving_second_moment

    self._initializers = util.check_initializers(
        initializers, self.POSSIBLE_INITIALIZER_KEYS)

  def _set_default_initializer(self, var_name):
    """Sets up a default initializer for a variable if one doesn't exist.

    For the offset (beta), a zeros initializer is used by default.
    For the scale (gamma), a ones initializer is used by default.

    Args:
      var_name: name of variable as a string.
    """
    if var_name not in self._initializers:
      if var_name == self.GAMMA:
        self._initializers[self.GAMMA] = tf.ones_initializer()
      elif var_name == self.BETA:
        self._initializers[self.BETA] = tf.zeros_initializer

  def _build_statistics_variance(self, input_batch,
                                 reduction_indices, use_batch_stats):
    """Builds the statistics part of the graph when using moving variance.

    Args:
      input_batch: Input batch Tensor.
      reduction_indices: Indices of `input_batch` to reduce over.
      use_batch_stats: Boolean to indicate if batch statistics should be
        calculated, otherwise moving averages are returned.

    Returns:
      Tuple of (mean, variance).
    """
    # Set up our moving statistics. When connecting in parallel, this is shared.
    self._moving_mean = tf.get_variable(
        "moving_mean",
        shape=self._mean_shape,
        collections=[tf.GraphKeys.MOVING_AVERAGE_VARIABLES,
                     tf.GraphKeys.VARIABLES],
        initializer=tf.zeros_initializer,
        trainable=False)

    self._moving_variance = tf.get_variable(
        "moving_variance",
        shape=self._mean_shape,
        collections=[tf.GraphKeys.MOVING_AVERAGE_VARIABLES,
                     tf.GraphKeys.VARIABLES],
        initializer=tf.ones_initializer(),
        trainable=False)

    def build_batch_stats():
      """Builds the batch statistics calculation ops."""

      # We use the moving mean as an estimate of the mean in order to perform
      # a more numerically stable calculation of the batch mean.
      # Copy for better stability.
      shift = tf.add(self._moving_mean, 0)
      counts, shifted_sum_x, shifted_sum_x2, _ = tf.nn.sufficient_statistics(
          input_batch,
          reduction_indices,
          keep_dims=True,
          shift=shift,
          name="batch_norm_ss")

      mean, variance = tf.nn.normalize_moments(counts,
                                               shifted_sum_x,
                                               shifted_sum_x2,
                                               shift,
                                               name="normalize_moments")

      return mean, variance

    def build_moving_stats():
      return (
          tf.identity(self._moving_mean),
          tf.identity(self._moving_variance),
      )

    mean, variance = utils.smart_cond(
        use_batch_stats,
        build_batch_stats,
        build_moving_stats,
    )

    return mean, variance

  def _build_statistics_second_moment(self, input_batch,
                                      reduction_indices, use_batch_stats):
    """Builds the statistics part of the graph when using moving second moment.

    Args:
      input_batch: Input batch Tensor.
      reduction_indices: Indices of `input_batch` to reduce over.
      use_batch_stats: Boolean to indicate if batch statistics should be
        calculated, otherwise moving averages are returned.

    Returns:
      Tuple of (mean, variance, second_moment).
    """
    # Set up our moving statistics. When connecting in parallel, this is shared.
    self._moving_mean = tf.get_variable(
        "moving_mean",
        shape=self._mean_shape,
        collections=[tf.GraphKeys.MOVING_AVERAGE_VARIABLES,
                     tf.GraphKeys.VARIABLES],
        initializer=tf.zeros_initializer,
        trainable=False)

    self._moving_second_moment = tf.get_variable(
        "moving_second_moment",
        shape=self._mean_shape,
        collections=[tf.GraphKeys.MOVING_AVERAGE_VARIABLES,
                     tf.GraphKeys.VARIABLES],
        initializer=tf.ones_initializer(),
        trainable=False)

    self._moving_variance = tf.sub(self._moving_second_moment,
                                   tf.square(self._moving_mean),
                                   name="moving_variance")

    def build_batch_stats():
      """Builds the batch statistics calculation ops."""

      # Copy for better stability.
      # We use the moving mean as an estimate of the mean in order to perform
      # a more numerically stable calculation of the batch mean.
      shift = tf.add(self._moving_mean, 0)
      counts, shifted_sum_x, shifted_sum_x2, _ = tf.nn.sufficient_statistics(
          input_batch,
          reduction_indices,
          keep_dims=True,
          shift=shift,
          name="batch_norm_ss")

      mean, variance = tf.nn.normalize_moments(counts,
                                               shifted_sum_x,
                                               shifted_sum_x2,
                                               shift,
                                               name="normalize_moments")
      second_moment = variance + tf.square(mean)

      return mean, variance, second_moment

    def build_moving_stats():
      return (
          tf.identity(self._moving_mean),
          tf.identity(self._moving_variance),
          tf.identity(self._moving_second_moment),
      )

    mean, variance, second_moment = utils.smart_cond(
        use_batch_stats,
        build_batch_stats,
        build_moving_stats,
    )

    return mean, variance, second_moment

  def _build_update_ops_variance(self, mean, variance, is_training):
    """Builds the moving average update ops when using moving variance.

    Args:
      mean: The mean value to update with.
      variance: The variance value to update with.
      is_training: Boolean Tensor to indicate if we're currently in
        training mode.
    """

    def build_update_ops():
      """Builds the exponential moving average update ops."""

      update_mean_op = moving_averages.assign_moving_average(
          variable=self._moving_mean,
          value=mean,
          decay=self._decay_rate,
          name="update_moving_mean").op

      update_variance_op = moving_averages.assign_moving_average(
          variable=self._moving_variance,
          value=variance,
          decay=self._decay_rate,
          name="update_moving_variance").op

      return update_mean_op, update_variance_op

    def build_no_ops():
      return (tf.no_op(), tf.no_op())

    # Only make the ops if we know that `is_training=True`, or the value of
    # `is_training` is unknown.
    is_training_const = utils.constant_value(is_training)
    if is_training_const is None or is_training_const:
      update_mean_op, update_variance_op = utils.smart_cond(
          is_training,
          build_update_ops,
          build_no_ops,
      )

      # Every new connection creates a new op which adds its contribution
      # to the running average when ran.
      tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_mean_op)
      tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_variance_op)

  def _build_update_ops_second_moment(self, mean, second_moment, is_training):
    """Builds the moving average update ops when using the moving second moment.

    Args:
      mean: The mean value to update with.
      second_moment: The second_moment value to update with.
      is_training: Boolean Tensor to indicate if we're currently in
        training mode.
    """

    def build_update_ops():
      """Builds the exponential moving average update ops."""

      update_mean_op = moving_averages.assign_moving_average(
          variable=self._moving_mean,
          value=mean,
          decay=self._decay_rate,
          name="update_moving_mean").op

      update_second_moment_op = moving_averages.assign_moving_average(
          variable=self._moving_second_moment,
          value=second_moment,
          decay=self._decay_rate,
          name="update_moving_second_moment").op

      return update_mean_op, update_second_moment_op

    def build_no_ops():
      return (tf.no_op(), tf.no_op())

    # Only make the ops if we know that `is_training=True`, or the value of
    # `is_training` is unknown.
    is_training_const = utils.constant_value(is_training)
    if is_training_const is None or is_training_const:
      update_mean_op, update_second_moment_op = utils.smart_cond(
          is_training,
          build_update_ops,
          build_no_ops,
      )

      # Every new connection creates a new op which adds its contribution
      # to the running average when ran.
      tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_mean_op)
      tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_second_moment_op)

  def _build(self, input_batch, is_training=True, test_local_stats=True):
    """Connects the BatchNorm module into the graph.

    Args:
      input_batch: A Tensor of arbitrary dimension. By default, the final
        dimension is not reduced over when computing the minibatch statistics.
      is_training: A boolean to indicate if the module should be connected in
        training mode, meaning the moving averages are updated. By default
        `True`. Can be a Tensor.
      test_local_stats: A boolean to indicate if local batch statistics should
        be used when `is_training=False`. If not, moving averages are used.
        By default `True`. Can be a Tensor.

    Returns:
      A tensor with the same shape as `input_batch`.

    Raises:
      base.IncompatibleShapeError: If `reduction_indices` is not valid for the
        input shape or has negative entries.
      base.NotSupportedError: If `input_batch` has data type of `tf.float16`.
    """
    input_shape = input_batch.get_shape()

    if self._reduction_indices is not None:
      if len(self._reduction_indices) > len(input_shape):
        raise base.IncompatibleShapeError(
            "Too many reduction indices specified.")

      if max(self._reduction_indices) >= len(input_shape):
        raise base.IncompatibleShapeError(
            "Reduction index too large for input shape.")

      if min(self._reduction_indices) < 0:
        raise base.IncompatibleShapeError(
            "Reduction indeces must be non-negative.")

      reduction_indices = self._reduction_indices
    else:
      # Reduce over all dimensions except the last.
      reduction_indices = range(len(input_shape))[:-1]

    if input_batch.dtype == tf.float16:
      raise base.NotSupportedError(
          "BatchNorm does not support `tf.float16`, insufficient "
          "precision for calculating sufficient statistics.")

    self._mean_shape = input_batch.get_shape().as_list()
    for index in reduction_indices:
      self._mean_shape[index] = 1

    use_batch_stats = is_training | test_local_stats

    # Use the legacy moving second moment if the flag is set.
    if self._use_legacy_moving_second_moment:
      tf.logging.warning(
          "nn.BatchNorm `use_legacy_second_moment=True` is deprecated.")

      mean, variance, second_moment = self._build_statistics_second_moment(
          input_batch,
          reduction_indices,
          use_batch_stats)

      self._build_update_ops_second_moment(mean, second_moment, is_training)
    else:
      mean, variance = self._build_statistics_variance(
          input_batch,
          reduction_indices,
          use_batch_stats)

      self._build_update_ops_variance(mean, variance, is_training)

    # Set up optional scale and offset factors.
    if self._offset:
      self._set_default_initializer(self.BETA)
      self._beta = tf.get_variable(
          self.BETA,
          shape=self._mean_shape,
          initializer=self._initializers[self.BETA])
    else:
      self._beta = None

    if self._scale:
      self._set_default_initializer(self.GAMMA)
      self._gamma = tf.get_variable(
          self.GAMMA,
          shape=self._mean_shape,
          initializer=self._initializers[self.GAMMA])
    else:
      self._gamma = None

    out = tf.nn.batch_normalization(
        input_batch,
        mean,
        variance,
        self._beta,
        self._gamma,
        self._eps,
        name="batch_norm")

    return out

  @property
  def moving_mean(self):
    self._ensure_is_connected()
    return self._moving_mean

  @property
  def moving_second_moment(self):
    self._ensure_is_connected()
    return self._moving_second_moment

  @property
  def moving_variance(self):
    self._ensure_is_connected()
    return self._moving_variance

  @property
  def beta(self):
    self._ensure_is_connected()

    if self._beta is None:
      raise base.Error(
          "Batch normalization doesn't have an offset, so no beta")
    else:
      return self._beta

  @property
  def gamma(self):
    self._ensure_is_connected()

    if self._gamma is None:
      raise base.Error(
          "Batch normalization doesn't have a scale, so no gamma")
    else:
      return self._gamma
