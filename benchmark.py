import tensorflow as tf
from absl import flags
from tfbm import Benchmark, benchmark

from ops import v0, v1

flags.DEFINE_integer("seed", 0, "Random seed")
flags.DEFINE_integer("l", 256, "Sequence length")
flags.DEFINE_integer("b", 2, "Batch size")
flags.DEFINE_integer("d", 64, "Number of features")
flags.DEFINE_integer("m", 128, "Random projection dimension")
flags.DEFINE_integer("h", 16, "???")


def get_args():
    FLAGS = flags.FLAGS
    rng = tf.random.Generator.from_seed(FLAGS.seed)
    l = FLAGS.l
    b = FLAGS.b
    d = FLAGS.d
    m = FLAGS.m
    h = FLAGS.h
    return (
        rng.normal((l, b, h, m)),
        rng.normal((l, b, h, m)),
        rng.normal((l, b, h, d)),
    )


SPEC = (
    benchmark(device="cpu", xla_jit=False),
    benchmark(device="cpu", xla_jit=True),
    benchmark(device="gpu", xla_jit=False),
    benchmark(device="gpu", xla_jit=True),
)


class ForwardBenchmark(Benchmark):
    BENCHMARK_SPEC = SPEC

    @benchmark()
    def v0_forward(self):
        qs, ks, vs = get_args()
        return v0.causal_numerator(qs, ks, vs), v0.causal_denominator(qs, ks)

    @benchmark()
    def v1_forward(self):
        qs, ks, vs = get_args()
        return v1.causal_numerator(qs, ks, vs), v1.causal_denominator(qs, ks)


class BackwardBenchmark(Benchmark):
    BENCHMARK_SPEC = SPEC

    @benchmark()
    def v0_backward(self):
        qs, ks, vs = get_args()
        with tf.GradientTape() as tape:
            tape.watch((qs, ks, vs))
            loss = tf.reduce_sum(v0.causal_numerator(qs, ks, vs)) + tf.reduce_sum(
                v0.causal_denominator(qs, ks)
            )
        return tape.gradient(loss, (qs, ks, vs))

    @benchmark()
    def v1_backward(self):
        qs, ks, vs = get_args()
        with tf.GradientTape() as tape:
            tape.watch((qs, ks, vs))
            loss = tf.reduce_sum(v1.causal_numerator(qs, ks, vs)) + tf.reduce_sum(
                v1.causal_denominator(qs, ks)
            )
        return tape.gradient(loss, (qs, ks, vs))


if __name__ == "__main__":
    from tfbm.cli import main

    main()
