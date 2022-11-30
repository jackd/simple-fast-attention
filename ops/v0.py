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
import tensorflow as tf


@tf.custom_gradient
def causal_numerator(qs, ks, vs):
    """Computes not-normalized FAVOR causal attention A_{masked}V.

    Args:
      qs: query_prime tensor of the shape [L,B,H,M].
      ks: key_prime tensor of the shape [L,B,H,M].
      vs: value tensor of the shape [L,B,H,D].

    Returns:
      Not-normalized FAVOR causal attention A_{masked}V.
    """

    result = []
    sums = tf.zeros_like(tf.einsum("ijk,ijl->ijkl", ks[0], vs[0]))

    for index in range(qs.shape[0]):
        sums = sums + tf.einsum("ijk,ijl->ijkl", ks[index], vs[index])
        result.append(tf.einsum("ijkl,ijk->ijl", sums, qs[index])[None, Ellipsis])

    result = tf.concat(result, axis=0)

    def grad(res_grad):

        grads = tf.zeros_like(tf.einsum("ijk,ijl->ijkl", ks[0], vs[0]))

        gr_sums = sums

        q_grads = []
        k_grads = []
        v_grads = []

        for index in range(qs.shape[0] - 1, -1, -1):

            q_grads.append(
                tf.einsum("ijkl,ijl->ijk", gr_sums, res_grad[index])[None, Ellipsis]
            )
            grads = grads + tf.einsum("ijk,ijl->ijkl", qs[index], res_grad[index])
            k_grads.append(tf.einsum("ijkl,ijl->ijk", grads, vs[index])[None, Ellipsis])
            v_grads.append(tf.einsum("ijkl,ijk->ijl", grads, ks[index])[None, Ellipsis])
            gr_sums = gr_sums - tf.einsum("ijk,ijl->ijkl", ks[index], vs[index])

        q_grads = tf.concat(q_grads[::-1], axis=0)
        k_grads = tf.concat(k_grads[::-1], axis=0)
        v_grads = tf.concat(v_grads[::-1], axis=0)

        return q_grads, k_grads, v_grads

    return result, grad


@tf.custom_gradient
def causal_denominator(qs, ks):
    """Computes FAVOR normalizer in causal attention.

    Args:
      qs: query_prime tensor of the shape [L,B,H,M].
      ks: key_prime tensor of the shape [L,B,H,M].

    Returns:
      FAVOR normalizer in causal attention.
    """

    result = []
    sums = tf.zeros_like(ks[0])

    for index in range(qs.shape[0]):
        sums = sums + ks[index]
        result.append(tf.reduce_sum(qs[index] * sums, axis=2)[None, Ellipsis])

    result = tf.concat(result, axis=0)

    def grad(res_grad):

        k_grad = tf.zeros_like(ks[0])

        gr_sums = sums

        q_grads = []
        k_grads = []

        for index in range(qs.shape[0] - 1, -1, -1):

            q_grads.append(
                tf.einsum("ijk,ij->ijk", gr_sums, res_grad[index])[None, Ellipsis]
            )
            k_grad = k_grad + tf.einsum("ijk,ij->ijk", qs[index], res_grad[index])
            k_grads.append(k_grad[None, Ellipsis])
            gr_sums = gr_sums - ks[index]

        q_grads = tf.concat(q_grads[::-1], axis=0)
        k_grads = tf.concat(k_grads[::-1], axis=0)

        return q_grads, k_grads

    return result, grad
