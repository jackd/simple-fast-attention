import tensorflow as tf
from ops import v0, v1


class OpsTest(tf.test.TestCase):
    def test_numerator_consistent(
        self, l: int = 5, b: int = 3, h: int = 7, d: int = 2, m: int = 11, seed: int = 0
    ):
        rng = tf.random.Generator.from_seed(seed)
        qs = rng.normal((l, b, h, m))
        ks = rng.normal((l, b, h, m))
        vs = rng.normal((l, b, h, d))
        args = qs, ks, vs

        with tf.GradientTape() as tape:
            tape.watch(args)
            expected = v0.causal_numerator(*args)
            loss = tf.reduce_sum(expected)
        expected_grads = tape.gradient(loss, args)

        with tf.GradientTape() as tape:
            tape.watch(args)
            actual = v1.causal_numerator(*args)
            loss = tf.reduce_sum(actual)
        actual_grads = tape.gradient(loss, args)

        self.assertAllClose(actual, expected)
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            self.assertAllClose(actual_grad, expected_grad)

    def test_denominator_consistent(
        self, l: int = 5, b: int = 3, h: int = 7, d: int = 2, m: int = 11, seed: int = 0
    ):
        rng = tf.random.Generator.from_seed(seed)
        qs = rng.normal((l, b, h, m))
        ks = rng.normal((l, b, h, m))
        args = qs, ks

        with tf.GradientTape() as tape:
            tape.watch(args)
            expected = v0.causal_denominator(*args)
            loss = tf.reduce_sum(expected)
        expected_grads = tape.gradient(loss, args)

        with tf.GradientTape() as tape:
            tape.watch(args)
            actual = v1.causal_denominator(*args)
            loss = tf.reduce_sum(actual)
        actual_grads = tape.gradient(loss, args)

        self.assertAllClose(actual, expected)
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            self.assertAllClose(actual_grad, expected_grad)


if __name__ == "__main__":
    tf.test.main()
