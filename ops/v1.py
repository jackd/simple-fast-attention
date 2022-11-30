# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

"""Implementation of multiheaded FAVOR-attention & FAVOR-self-attention layers.

Prefix Sum Tensorflow implementation by Valerii Likhosherstov.
"""
import typing as tp
import tensorflow as tf


def causal_numerator(qs: tf.Tensor, ks: tf.Tensor, vs: tf.Tensor):
    """Computes not-normalized FAVOR causal attention A_{masked}V.

    Args:
      qs: query_prime tensor of the shape [L,B,H,M].
      ks: key_prime tensor of the shape [L,B,H,M].
      vs: value tensor of the shape [L,B,H,D].

    Returns:
      Not-normalized FAVOR causal attention A_{masked}V.
    """
    # rhs = tf.einsum('lbhm,lbhd->lbhdm', ks, vs)
    rhs = tf.expand_dims(ks, axis=-2) * tf.expand_dims(vs, axis=-1)  # [L,B,H,D,M]
    rhs = tf.cumsum(rhs, axis=0)
    # return tf.einsum('lbhm,lbhdm->lbhd', qs, rhs)
    return tf.linalg.matvec(rhs, qs)


def causal_denominator(qs, ks):
    """Computes FAVOR normalizer in causal attention.

    Args:
      qs: query_prime tensor of the shape [L,B,H,M].
      ks: key_prime tensor of the shape [L,B,H,M].

    Returns:
      FAVOR normalizer in causal attention.
    """
    rhs = tf.cumsum(ks, axis=0)
    return tf.einsum("lbhm,lbhm->lbh", qs, rhs)
    # return tf.reduce_sum(qs * rhs, axis=-1)
