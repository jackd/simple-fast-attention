# Fast Attention: Causal Implementation Experiments

Comparing implementations of [fast causal attention](https://ai.googleblog.com/2020/10/rethinking-attention-with-performers.html).

Having looked at google-research's [fast attention tensorflow implementation](https://github.com/google-research/google-research/blob/master/performer/fast_attention/tensorflow/fast_attention.py), I was left scratching my head about the causal attention implementation. This gist investigates a simpler implementation.

## TL;DR

- [ops/v1.py](./ops/v1.py) provide significantly simpler implementations that use neither loops over tensors nor custom gradients.
- the implementations are much shorter (3 and 2 lines vs. 25 and 22 of the original), making it much easier to reason about
- jit-compiling these operations is significantly faster than the originals ([ops/v0.py](./ops/v0.py))
- results are the same between the two implementations
- computation time using `google-benchmark` is indistinguishable

## Theory

The [google-ai blog post](https://ai.googleblog.com/2020/10/rethinking-attention-with-performers.html) provides a visualisation of causal attention.

![Causal Attention](https://1.bp.blogspot.com/-kJKZ5veuREk/X5IcGdqtbCI/AAAAAAAAGtM/PWmo0lHnhSUQ5nabOwhKIN9rh6pYxFItQCLcBGAsYHQ/s1238/image4.gif)

It's not immediately apparent to me what's going on here, and looking at the code (originally [here](https://github.com/google-research/google-research/blob/master/performer/fast_attention/tensorflow/fast_attention.py) but with relevant part included [here](./ops/v0.py) for convenience)

My implementation (v1) takes a different approach.

We consider the task is to compute the noncausal numerator $N$, where

$N = \left[(Q K^T) \circ L\right] V$

where $Q$, $K$ and $V$ are the query, key and value matrices used in non-causal fast attention, $L$ is a lower triangular matrix with values of $1$ on and below the diagonal and $\circ$ is the _Hadamard product_ (elementwise product). Noting that $Q$ and $K$ are low-rank (that's the whole point of performers), we can use the following handy dandy property of Hadamard products ([Property 1](http://pi.math.cornell.edu/~ajt/presentations/HadamardProduct.pdf)):

$\left[A \circ \sum_j u_j v_j^T\right]x = \sum_j D(u_j) A D(v_j) x$

where $D(z)$ is the diagonal matrix with diagonal values $z$. This means we can express our fast causal attention output as

$N = \sum_m D(q_m) L D(k_m) V.$

Note it is neither efficient nor necessary to compute any of the new matrices above. $D(k_m) Z$ is just the scaling of rows of $Z$ by $k_m$, while $L Z$ is the cumulative sum of $Z$ on the leading dimension. This results in a significantly simpler tensorflow implementation without the need to implement custom gradients or use python loops.

The implementation looks slighty different to the maths above because we compute $D(k_m) V$ simultaneously for all $m$ and then perform the sum over $m$ using `tf.linalg.matvec`.

```python
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
```

That's a 3-line implementation, as opposed to the 25 used in the [original](./ops/v0.py#L23-71).

### Denominator

The noncausal denominator function is conceptually the same as the numerator except using the ones vector for $V$. Since the first operation involves scaling $V$, we can skip this entirely and just use the keys $ks$:

```python
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
```

That's 2 lines compared to 22 in [the original](./ops/v0.py?#L74-L118).

## Benchmarks

The following benchmarks were run on my fairly old laptop with a 1050-Ti. Note that I trust the timings from `google-benchmark` a lot more than I do from `tfbm` / `tf.test.Benchmark`, but maybe I'm just misinterpretting them.

### google-benchmark

[gbenchmark.py](./gbenchmark.py) uses [google-benchmark](https://pypi.org/project/google-benchmark/). The output is a lot simpler compared to `tf.test.Benchmark` which means I trust these timings a lot more.

Take-aways:

- There isn't much difference between v0 and v1 implementations in terms of computation time
- v1 implementations warm-up significantly faster
- jit compilation significantly reduces forward time on cpu, but is negligible on gpu

```bash
python gbenchmark.py
```

```txt
--------------------------------------------------------------
Benchmark                    Time             CPU   Iterations
--------------------------------------------------------------
v0_forward-cpu         5403096 ns       364764 ns         1000
v1_forward-cpu         5419832 ns       365650 ns         1000
v0_backward-cpu         268558 ns       238634 ns         2896
v1_backward-cpu         267089 ns       235842 ns         2937
v0_forward-gpu          288531 ns       241580 ns         2874
v1_forward-gpu          285695 ns       238078 ns         2908
v0_backward-gpu         268220 ns       237309 ns         2869
v1_backward-gpu         268324 ns       240429 ns         2751
v0_forward-cpu-jit      299143 ns       271613 ns         2516
v1_forward-cpu-jit      291873 ns       269618 ns         2538
v0_backward-cpu-jit     303150 ns       275359 ns         2483
v1_backward-cpu-jit     303948 ns       276806 ns         2482
v0_forward-gpu-jit      278147 ns       277842 ns         2450
v1_forward-gpu-jit      276128 ns       275956 ns         2523
v0_backward-gpu-jit     256809 ns       256798 ns         2706
v1_backward-gpu-jit     252543 ns       252537 ns         2769

Warmup time for v0_forward-cpu: 6.56445574760437
Warmup time for v1_forward-cpu: 0.1015627384185791
Warmup time for v0_backward-cpu: 22.0670325756073
Warmup time for v1_backward-cpu: 0.08140373229980469
Warmup time for v0_forward-gpu: 6.233572244644165
Warmup time for v1_forward-gpu: 0.028412342071533203
Warmup time for v0_backward-gpu: 22.226712226867676
Warmup time for v1_backward-gpu: 0.051419734954833984
Warmup time for v0_forward-cpu-jit: 6.481787443161011
Warmup time for v1_forward-cpu-jit: 0.05790424346923828
Warmup time for v0_backward-cpu-jit: 24.72081184387207
Warmup time for v1_backward-cpu-jit: 0.09151363372802734
Warmup time for v0_forward-gpu-jit: 8.328083515167236
Warmup time for v1_forward-gpu-jit: 0.08592033386230469
Warmup time for v0_backward-gpu-jit: 24.7033634185791
Warmup time for v1_backward-gpu-jit: 0.12377095222473145
```

### tfbm

[benchmark.py](./benchmark.py) requires [tfbm](https://github.com/jackd/tfbm), a wrapper around [tf.test.Benchmark](https://www.tensorflow.org/api_docs/python/tf/test/Benchmark?hl=en) implementations.

```txt
Results for cls=Forward
Uniform results:
+--------+---------+-------+
| run_id |     cls | iters |
+--------+---------+-------+
|    NOW | Forward |    10 |
+--------+---------+-------+
Varied results:
+--------------------+----------------+--------+------------------------+-----------------+--------------------------+---------+
|               test | wall_time (us) | device | max_mem_GPU_0_bfc (Mb) | max_mem_cpu (b) | max_mem_gpu_host_bfc (b) | xla_jit |
+--------------------+----------------+--------+------------------------+-----------------+--------------------------+---------+
| v1_forward_xla_gpu |        218.511 |    gpu |                    --- |         192.000 |                   49.000 |    True |
+--------------------+----------------+--------+------------------------+-----------------+--------------------------+---------+
| v0_forward_xla_gpu |        260.711 |    gpu |                    --- |         192.000 |                   49.000 |    True |
+--------------------+----------------+--------+------------------------+-----------------+--------------------------+---------+
| v1_forward_xla_cpu |        272.274 |    cpu |                    --- |         192.000 |                   49.000 |    True |
+--------------------+----------------+--------+------------------------+-----------------+--------------------------+---------+
| v0_forward_xla_cpu |        284.910 |    cpu |                    --- |         192.000 |                   49.000 |    True |
+--------------------+----------------+--------+------------------------+-----------------+--------------------------+---------+
|     v0_forward_gpu |      20148.039 |    gpu |               2563.000 |         192.000 |                 1088.000 |   False |
+--------------------+----------------+--------+------------------------+-----------------+--------------------------+---------+
|     v1_forward_gpu |      58524.966 |    gpu |                516.000 |         192.000 |                   64.000 |   False |
+--------------------+----------------+--------+------------------------+-----------------+--------------------------+---------+
|     v0_forward_cpu |     100526.690 |    cpu |                    --- |  1776322300.000 |                      --- |   False |
+--------------------+----------------+--------+------------------------+-----------------+--------------------------+---------+
|     v1_forward_cpu |     233323.097 |    cpu |                    --- |   541065220.000 |                      --- |   False |
+--------------------+----------------+--------+------------------------+-----------------+--------------------------+---------+
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Results for cls=Backward
Uniform results:
+--------+----------+-------+
| run_id |      cls | iters |
+--------+----------+-------+
|    NOW | Backward |    10 |
+--------+----------+-------+
Varied results:
+---------------------+----------------+--------+------------------------+-----------------+--------------------------+---------+
|                test | wall_time (us) | device | max_mem_GPU_0_bfc (Mb) | max_mem_cpu (b) | max_mem_gpu_host_bfc (b) | xla_jit |
+---------------------+----------------+--------+------------------------+-----------------+--------------------------+---------+
| v0_backward_xla_gpu |        225.782 |    gpu |                    --- |         192.000 |                   49.000 |    True |
+---------------------+----------------+--------+------------------------+-----------------+--------------------------+---------+
| v1_backward_xla_gpu |        231.147 |    gpu |                    --- |         192.000 |                   49.000 |    True |
+---------------------+----------------+--------+------------------------+-----------------+--------------------------+---------+
| v1_backward_xla_cpu |        239.730 |    cpu |                    --- |         192.000 |                   49.000 |    True |
+---------------------+----------------+--------+------------------------+-----------------+--------------------------+---------+
|     v0_backward_gpu |      47311.544 |    gpu |               2641.594 |         192.000 |                 1084.000 |   False |
+---------------------+----------------+--------+------------------------+-----------------+--------------------------+---------+
| v0_backward_xla_cpu |      77256.083 |    cpu |                    --- |   288358548.000 |                   49.000 |    True |
+---------------------+----------------+--------+------------------------+-----------------+--------------------------+---------+
|     v1_backward_gpu |     110480.189 |    gpu |                788.031 |         192.000 |                   68.000 |   False |
+---------------------+----------------+--------+------------------------+-----------------+--------------------------+---------+
|     v0_backward_cpu |     260773.897 |    cpu |                    --- |  1936089700.000 |                      --- |   False |
+---------------------+----------------+--------+------------------------+-----------------+--------------------------+---------+
|     v1_backward_cpu |     524332.523 |    cpu |                    --- |  1092648988.000 |                      --- |   False |
+---------------------+----------------+--------+------------------------+-----------------+--------------------------+---------+
```
