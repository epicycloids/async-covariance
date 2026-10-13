# Measurements

Release build, Ryzen 7 7735U, Rust 1.92, Python 3.12.12, NumPy 2.5.4,
SciPy 1.18.1. BLAS used one thread; native batch scans used up to eight CPU
threads. The build uses portable x86-64 code, without native CPU flags or fast
math. These are local synthetic workloads, not latency guarantees.

## Moment engine

Median of three complete runs, including Python validation and native input
copies. Selected graphs have average off-diagonal degree eight. Observations
are globally ordered with randomly staggered per-asset timestamps. Revisions
replace existing historical prices; the scalar measurement includes one Python
call per replacement.

| Assets | Points | Off-diagonal pairs | Batch, ms | Serial batch, ms | Online build, ms | 512 revisions, ms | Snapshot, ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 2,048 | 120 | 1.45 | 1.31 | 3.52 | 5.48 | 0.15 |
| 64 | 8,192 | 2,016 | 5.26 | 19.06 | 51.34 | 23.93 | 0.67 |
| 256 | 16,384 | 32,640 | 29.54 | 179.01 | 835.46 | 125.74 | 10.34 |
| 256 | 32,768 | 1,024 | 5.87 | 19.02 | 86.44 | 12.34 | 0.42 |
| 1,024 | 65,536 | 4,096 | 18.16 | 34.71 | 212.54 | 12.09 | 0.86 |
| 4,096 | 131,072 | 16,384 | 34.44 | 85.19 | 551.83 | 8.61 | 3.58 |

The 4,096-asset selected graph ingests about 238,000 points/s. Its 512 scalar
revisions take 9.53 ms, or about 53,700/s. A separate 1,024-asset case with
131,072 supplied points, a 16-unit window and 32-unit half-life builds online
in 873 ms, retaining 17,408 points and 15,360 increments. Batch fitting takes
12.39 ms and 512 revisions take 7.82 ms. Expiry work is included.

Peak resident memory over the engine benchmark process was 156 MiB. Full-graph
cost grows with the number of requested pairs; a selected graph changes both
work and the information available to the factor model. Serial and parallel
batch outputs are bitwise identical. Batch and online outputs agree within
floating-point tolerance after the revision round trip.

## Factor fitting

Single-run component timings for 12 sweeps of the same weighted sparse loss,
with seeded random initialization. The graph contains a connected ring with
triangles and additional random edges. This comparison includes fitting and
initialization, but excludes moment estimation.

| Assets | Rank | Native fit, ms | Python reference, ms | Process peak, MiB |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 4 | 10.0 | 599.6 | 72 |
| 1,024 | 8 | 47.8 | 1,255.4 | 101 |

Default-start objective histories agree to numerical precision, and covariance
solve residuals are about `3e-15`. Optional spectral starts cost 19.7 and
113.6 ms on these fixtures. They can follow different local paths between
backends. Fixed sweeps do not certify convergence or identification; the loss
is nonconvex. The same script times dense spectral projections separately;
those solve a different problem and are not a speed baseline for sparse ALS.

## Statistical error

One hundred independent paths per case, seed 20261012. Errors below are RMSE
plus/minus a Monte Carlo standard error, not an estimator confidence interval.
The three-asset asynchronous cases use exogenous uniform schedules on a
one-day horizon. Both methods use the same path in each replication.

| Increment counts by asset | HY Frobenius RMSE | Overlap Frobenius RMSE |
| --- | ---: | ---: |
| 512 / 512 / 512 | 0.299 ± 0.010 | 0.217 ± 0.007 |
| 2,048 / 128 / 16 | 0.918 ± 0.043 | 0.626 ± 0.024 |
| 512 / 512 / 512, supplied drift `[2, -1, .5]` | 0.304 ± 0.011 | 0.232 ± 0.007 |

Overlap weighting performs better in these cases. The table does not establish
the best estimator for other correlations, schedules, noise or changing rates.

The point-noise cases have two synchronously sampled assets, covariance rate
0.6 and point-error variance 0.05 per asset. The grid is half-open `[0, 1)`.
Known noise variance is supplied for diagonal correction. HY and overlap
coincide on this equal-duration grid. The table reports cross-covariance error.

| Increments per asset | Clean RMSE | Noisy RMSE | Noisy, coarsened to 64 time bins |
| ---: | ---: | ---: | ---: |
| 128 | 0.108 ± 0.007 | 1.562 ± 0.109 | 1.137 ± 0.075 |
| 512 | 0.056 ± 0.004 | 2.885 ± 0.227 | 1.291 ± 0.101 |
| 2,048 | 0.031 ± 0.002 | 5.555 ± 0.358 | 1.189 ± 0.088 |
| 8,192 | 0.015 ± 0.001 | 11.755 ± 0.762 | 0.967 ± 0.067 |

More clean observations improve precision; more noisy observations can destroy
it. Fixed-bin thinning reduces this example's error but discards information.
It bounds points per bin, not minimum time spacing, and is not a noise-robust
infill estimator. Pre-averaged HY and spectral LMM are the next statistical
comparators discussed in [the source review](literature.md).

## Reproduce

Install the package and run from the repository root:

```sh
OPENBLAS_NUM_THREADS=1 python benchmarks/engine.py --output benchmarks/local/engine.json
OPENBLAS_NUM_THREADS=1 python benchmarks/projection.py --assets 512 --rank 4 --output benchmarks/local/projection512.json
OPENBLAS_NUM_THREADS=1 python benchmarks/projection.py --assets 1024 --rank 8 --output benchmarks/local/projection1024.json
OPENBLAS_NUM_THREADS=1 python benchmarks/quality.py --reps 100 --output benchmarks/local/quality.json
```

The checked reports in [benchmarks/results](../benchmarks/results) contain
parameters, timings, loss histories, pairwise biases and Monte Carlo errors.
Linux memory measurements use the process's `VmHWM`, avoiding a launcher peak
inherited through `ru_maxrss`.
