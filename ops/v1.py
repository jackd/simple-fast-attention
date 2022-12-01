import tensorflow as tf


def causal_numerator(qs: tf.Tensor, ks: tf.Tensor, vs: tf.Tensor) -> tf.Tensor:
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


def causal_denominator(qs: tf.Tensor, ks: tf.Tensor) -> tf.Tensor:
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
