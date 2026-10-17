# Benchmarks and Empirical Results

All measurements below were recorded on an AMD Ryzen 7 7735U with a release
build of `async-covariance` (Rust 1.92, `lto = "thin"`, `codegen-units = 1`,
portable x86-64 code generation without architecture-specific SIMD flags or
`fast-math`), Python 3.12.12, NumPy 2.5.4, and SciPy 1.18.1. BLAS was pinned to
a single thread (`OPENBLAS_NUM_THREADS=1`), while parallel native batch scans
used up to eight CPU threads. Peak resident memory is read from Linux `VmHWM` in
`/proc/self/status` so that parent-process memory is not inherited through
`ru_maxrss`.

---

## 1. Moment Engine Throughput and Latency (`benchmarks/engine.py`)

Each fixture generates $`N`$ assets with $`m`$ staggered asynchronous
observations per asset ($`N \times m`$ total points), ordered globally by time.
Selected-graph cases use a ring-lattice graph of degree 8 ($`4N`$ off-diagonal
pairs), whereas full-graph cases track all $`\binom{N}{2}`$ off-diagonal pairs.

Reported times are the median of three end-to-end runs, including Python
validation and defensive NumPy buffer copies before releasing the GIL.
Historical revision timings measure replacing 512 existing prices inside the
retained history in two ways:

- **512 batched revisions (`revision_seconds`)**: a single vectorized
  `engine.update(t[revise], a[revise], p[revise] + 0.001)` call.
- **512 scalar revisions (`scalar_revision_seconds`)**: 512 individual Python
  `engine.update(t[i], a[i], p[i])` calls restoring the original prices.

| Assets | Points | Off-diagonal pairs | Batch, ms | Serial batch, ms | Online build, ms | 512 batched revisions, ms | 512 scalar revisions, ms | Snapshot, ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 2,048 | 120 | 1.45 | 1.31 | 3.52 | 5.48 | 6.10 | 0.15 |
| 64 | 8,192 | 2,016 | 5.26 | 19.06 | 51.34 | 23.93 | 25.90 | 0.67 |
| 256 | 16,384 | 32,640 | 29.54 | 179.01 | 835.46 | 125.74 | 115.33 | 10.34 |
| 256 | 32,768 | 1,024 | 5.87 | 19.02 | 86.44 | 12.34 | 8.74 | 0.42 |
| 1,024 | 65,536 | 4,096 | 18.16 | 34.71 | 212.54 | 12.09 | 6.65 | 0.86 |
| 4,096 | 131,072 | 16,384 | 34.44 | 85.19 | 551.83 | 8.61 | 9.53 | 3.58 |

Key observations from `benchmarks/results/engine.json`:

- **Streaming and revision throughput**: On the 4,096-asset selected graph,
  online ingestion processes about 238,000 points/s (551.83 ms for 131,072
  points). Updating 512 historical observations takes 8.61 ms (about 59,500
  revisions/s) in a single batched array call, and 9.53 ms (about 53,700
  revisions/s) across 512 individual scalar Python calls.
- **Sliding window and decay**: A separate 1,024-asset selected-graph fixture
  with 131,072 points, `window=16`, and `half_life=32` builds online in
  873.30 ms (including lazy interval and point expiration), retaining 17,408
  points and 15,360 active increments; parallel batch fitting takes 12.39 ms
  (20.78 ms serial), 512 batched revisions take 7.82 ms (9.07 ms scalar), and
  snapshot evaluation takes 0.99 ms.
- **Memory and numerical parity**: Peak resident memory across the entire engine
  benchmark suite was 155.9 MiB. Parallel and serial batch scans produce
  bitwise-identical outputs, and online snapshots after applying and reverting
  the 512 historical revisions match the direct batch scan to floating-point
  tolerance (`rtol=1e-9, atol=1e-12`).

---

## 2. Factor Projection Performance (`benchmarks/projection.py`)

This benchmark compares 12 Gauss–Seidel sweeps of weighted sparse alternating
least squares (`fit_factor`) between the compiled Rust solver
(`backend="native"`) and the NumPy/SciPy reference solver (`backend="python"`).
Each fixture constructs a connected graph containing a ring with triangles plus
randomly sampled edges ($`5{,}018`$ observed off-diagonal pairs at $`N = 512`$,
and $`10{,}130`$ observed pairs at $`N = 1{,}024`$). Timings cover initialization
and alternating row updates, excluding pairwise moment estimation.

| Assets ($`N`$) | Rank ($`K`$) | Initialization | Native fit, ms | Python reference, ms | Process peak, MiB |
| ---: | ---: | :--- | ---: | ---: | ---: |
| 512 | 4 | `"random"` (default) | 10.0 | 599.6 | 72 |
| 512 | 4 | `"spectral"` (`eigsh`) | 19.7 | 605.1 | 72 |
| 1,024 | 8 | `"random"` (default) | 47.8 | 1,255.4 | 101 |
| 1,024 | 8 | `"spectral"` (`eigsh`) | 113.6 | 1,573.7 | 101 |

- **Solver parity and conditioning**: From the default seeded random start,
  native and Python objective trajectories agree to machine precision, and
  Woodbury linear solve relative residuals
  $`\| \Sigma_{\mathrm{model}} \Sigma_{\mathrm{model}}^{-1} x - x \|_2 / \|x\|_2`$
  are $`1.6 \times 10^{-15}`$ ($`N = 512`$) and $`3.0 \times 10^{-15}`$
  ($`N = 1{,}024`$). Under spectral initialization on the nonconvex $`N = 1{,}024`$
  fixture, floating-point differences between cyclic Jacobi and LAPACK eigensolvers
  can steer boundary-constrained rows along slightly different local paths while
  maintaining monotonically nonincreasing loss at every sweep.
- **Dense spectral projection (`spectral_factor`)**: Recorded in the same JSON
  files as a separate dense baseline; at $`N = 512, K = 4`$ it runs in 17.5 ms
  (`eigsh`) and 28.0 ms (`eigh`), and at $`N = 1{,}024, K = 8`$ in 104.3 ms
  (`eigsh`) and 196.3 ms (`eigh`). Dense spectral projection operates on a fully
  populated correlation matrix rather than weighting observed graph edges.

---

## 3. Statistical Error and Microstructure Noise (`benchmarks/quality.py`)

The statistical suite simulates 100 independent sample paths per configuration
(`seed = 20261012`) on a one-day horizon ($`T = 1`$) with ground-truth
covariance rate

```math
\Sigma = \begin{bmatrix}
1.0 & 0.6 & 0.25 \\
0.6 & 1.5 & -0.1 \\
0.25 & -0.1 & 0.8
\end{bmatrix}.
```

Reported values are the root-mean-square error (RMSE) across replications
$`\pm`$ one delta-method Monte Carlo standard error (which measures finite
simulation uncertainty across the 100 replications, not an estimator confidence
interval). Both `"hy"` and `"overlap"` are evaluated on the exact same sample
paths.

### Asynchronous schedules without microstructure noise

Observations for each of the three assets are drawn independently and uniformly
on $`[0, 1]`$ with fixed endpoints $`0`$ and $`1`$:

| Increment counts by asset | `"hy"` Frobenius RMSE | `"overlap"` Frobenius RMSE |
| :--- | ---: | ---: |
| 512 / 512 / 512 (`drift = [0, 0, 0]`) | $`0.299 \pm 0.010`$ | $`0.217 \pm 0.007`$ |
| 2,048 / 128 / 16 (`drift = [0, 0, 0]`) | $`0.918 \pm 0.043`$ | $`0.626 \pm 0.024`$ |
| 512 / 512 / 512 (`drift = [2, -1, 0.5]`) | $`0.304 \pm 0.011`$ | $`0.232 \pm 0.007`$ |

In these noiseless constant-rate experiments, duration-normalized overlap
weighting (`"overlap"`) achieves lower Frobenius RMSE than `"hy"`, particularly
when observation frequencies differ strongly across assets (2,048 / 128 / 16).
Relative efficiency under other correlation structures, endogenous arrival
times, microstructure noise, or time-varying volatility depends on the regime.

### Additive point noise under infill sampling

The noisy infill cases simulate the first two assets ($`\Sigma_{00} = 1.0`$,
$`\Sigma_{11} = 1.5`$, $`\Sigma_{01} = 0.6`$) on a synchronous uniform grid over
$`[0, 1)`$ with independent Gaussian point errors of variance
$`\tau_0^2 = \tau_1^2 = 0.05`$. Known `noise_variance` is supplied for exact
diagonal bias correction. Because the grid is synchronous and uniform, `"hy"`
and `"overlap"` are identical. The table below reports the RMSE of the
off-diagonal cross-covariance estimate $`\widehat{\Sigma}_{01}`$:

| Increments per asset ($`n`$) | Clean $`\widehat{\Sigma}_{01}`$ RMSE | Noisy $`\widehat{\Sigma}_{01}`$ RMSE | Noisy, thinned to 64 bins (`coarsen`) |
| ---: | ---: | ---: | ---: |
| 128 | $`0.108 \pm 0.007`$ | $`1.562 \pm 0.109`$ | $`1.137 \pm 0.075`$ |
| 512 | $`0.056 \pm 0.004`$ | $`2.885 \pm 0.227`$ | $`1.291 \pm 0.101`$ |
| 2,048 | $`0.031 \pm 0.002`$ | $`5.555 \pm 0.358`$ | $`1.189 \pm 0.088`$ |
| 8,192 | $`0.015 \pm 0.001`$ | $`11.755 \pm 0.762`$ | $`0.967 \pm 0.067`$ |

While clean error decays as $`O(n^{-1/2})`$, the raw noisy cross-covariance
RMSE grows as $`O(n^{1/2})`$, matching the theoretical asymptotic standard
deviation $`\sqrt{6 n}\,\tau_0 \tau_1 / T \approx 0.1225 \sqrt{n}`$ derived in
[Statistical Model and Literature](literature.md#4-microstructure-noise-and-sampling-frequency)
($`1.39, 2.77, 5.54, 11.09`$ at $`n = 128, 512, 2048, 8192`$). Thinning to 64
fixed time bins via `coarsen` caps the number of noisy increments and stabilizes
RMSE around $`1.0\text{--}1.3`$, trading sample size against noise amplification.

---

## 4. Reproducing the Benchmarks

From an installed checkout at the repository root:

```sh
OPENBLAS_NUM_THREADS=1 python benchmarks/engine.py --output benchmarks/local/engine.json
OPENBLAS_NUM_THREADS=1 python benchmarks/projection.py --assets 512 --rank 4 --output benchmarks/local/projection512.json
OPENBLAS_NUM_THREADS=1 python benchmarks/projection.py --assets 1024 --rank 8 --output benchmarks/local/projection1024.json
python benchmarks/quality.py --reps 100 --output benchmarks/local/quality.json
```

Reference outputs are checked into [`benchmarks/results/`](../benchmarks/results).
