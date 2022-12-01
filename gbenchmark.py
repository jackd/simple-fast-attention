from time import time

import google_benchmark as gbm
import tensorflow as tf
from absl import flags

from ops import v0, v1

flags.DEFINE_integer("seed", 0, "Random seed")
flags.DEFINE_integer("l", 256, "Sequence length")
flags.DEFINE_integer("b", 2, "Batch size")
flags.DEFINE_integer("d", 64, "Number of features")
flags.DEFINE_integer("m", 128, "Random projection dimension")
flags.DEFINE_integer("h", 16, "???")


def get_args():
    FLAGS = flags.FLAGS
    tf.random.set_seed(FLAGS.seed)
    l = FLAGS.l
    b = FLAGS.b
    d = FLAGS.d
    m = FLAGS.m
    h = FLAGS.h
    return (
        tf.random.normal((l, b, h, m)),
        tf.random.normal((l, b, h, m)),
        tf.random.normal((l, b, h, d)),
    )


def v0_forward(qs, ks, vs):
    qs, ks, vs = get_args()
    return v0.causal_numerator(qs, ks, vs), v0.causal_denominator(qs, ks)


def v1_forward(qs, ks, vs):
    qs, ks, vs = get_args()
    return v1.causal_numerator(qs, ks, vs), v1.causal_denominator(qs, ks)


def v0_backward(qs, ks, vs):
    with tf.GradientTape() as tape:
        tape.watch((qs, ks, vs))
        loss = tf.reduce_sum(v0.causal_numerator(qs, ks, vs)) + tf.reduce_sum(
            v0.causal_denominator(qs, ks)
        )
    return tape.gradient(loss, (qs, ks, vs))


def v1_backward(qs, ks, vs):
    with tf.GradientTape() as tape:
        tape.watch((qs, ks, vs))
        loss = tf.reduce_sum(v1.causal_numerator(qs, ks, vs)) + tf.reduce_sum(
            v1.causal_denominator(qs, ks)
        )
    return tape.gradient(loss, (qs, ks, vs))


def register(jit: bool = False, device: str = "cpu"):
    def ret_fn(fn):
        name = f"{fn.__name__}-{device}"
        if jit:
            name = f"{name}-jit"

        @gbm.register(name=name)
        def registered(state):
            with tf.device(device):
                args = get_args()

                @tf.function(jit_compile=jit)
                def call(args):
                    fn(*args)

                t = time()
                call(args)
                dt = time() - t
                print(f"Warmup time for {name}: {dt}")
                while state:
                    call(args)

        return fn

    return ret_fn


r = register(jit=True, device="cpu")

fns = (v0_forward, v1_forward, v0_backward, v1_backward)

for jit in (False, True):
    for device in ("cpu", "gpu"):
        for fn in fns:
            register(jit=jit, device=device)(fn)


if __name__ == "__main__":
    gbm.main()
