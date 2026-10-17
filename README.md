# async-covariance

`async-covariance` estimates continuous-time covariance rates from irregularly
spaced, asynchronous log-price series and fits positive-definite
factor-plus-diagonal covariance models to the resulting pairwise estimates.

The library separates covariance estimation into two stages:

1. **Pairwise moment accumulation**: From asynchronous `(time, asset, log_price)`
   observations, a Rust engine constructs per-asset price increments, finds
   overlapping time intervals across requested asset pairs, and accumulates
   sufficient statistics for two covariance-rate estimators: the
   Hayashi–Yoshida overlap estimator (`method="hy"`) and the duration-normalized
   overlap regression estimator (`method="overlap"`). Immutable batch scans
   (`fit`) and streaming state (`OnlineCovariance`, supporting out-of-order
   insertions, historical price revisions, deletions, exponential decay, and
   sliding windows) produce identical sufficient statistics.
2. **Positive-definite factor projection**: Raw pairwise covariance-rate
   estimates can be incomplete (when pairs have no overlapping increments) or
   indefinite. The factor solver fits a positive-definite model
   $`\Sigma_{\mathrm{model}} = B B^\top + \mathrm{diag}(d)`$ to the observed
   off-diagonal pairs by constrained weighted alternating least squares
   (`MomentSnapshot.factor`, `fit_factor`) or dense spectral truncation
   (`spectral_factor`), returning an immutable `FactorCovariance` operator.

## Installation

Requires Python 3.10+, NumPy ($`\ge 1.24`$), and SciPy ($`\ge 1.10`$).
Prebuilt wheels for Linux x86-64 (`glibc 2.35+`) are available on
[GitHub Releases](https://github.com/epicycloids/async-covariance/releases).
To build from source with Rust 1.83+:

```sh
python -m pip install .
```

## Quick Start

### Batch estimation and factor modeling

```python
from async_covariance import fit

# Timestamps share a single time unit; prices are natural logarithms.
times = [0.0, 0.2, 1.0, 1.2, 2.0, 2.2]
assets = [0, 1, 0, 1, 0, 1]
log_prices = [0.0, 0.0, 0.1, 0.2, 0.15, 0.16]

# 1. Estimate pairwise covariance rates (log-return covariance per unit time).
raw = fit(times, assets, log_prices, method="hy")
dense_raw = raw.dense()  # (2, 2) array; unobserved pairs are NaN

# 2. Fit a positive-definite rank-1 factor model to the observed moments.
fit_result = raw.factor(rank=1)
model = fit_result.model

# 3. Apply the fitted operator without allocating an N x N matrix.
#    matvec is O(NK); solve caches a rank-K Woodbury factor in O(NK^2 + K^3)
#    on first call and solves each right-hand side in O(NK + K^2).
y = model.matvec([1.0, -1.0])
x = model.solve([1.0, -1.0])

# 4. Materialize dense N x N matrices only when needed.
cov_matrix = model.covariance()
moment_horizon = model.second_moment(horizon=5.0, drift=[0.01, 0.02])
```

### Streaming updates, revisions, and sliding windows

```python
from async_covariance import OnlineCovariance

# Retain whole increments starting within 100 time units of the watermark,
# weighting overlaps with an exponential half-life of 10 time units.
online = OnlineCovariance(n_assets=2, half_life=10.0, window=100.0)

online.update(times, assets, log_prices)
online.update(1.0, 0, 0.11)  # replace the log price of asset 0 at t = 1.0
online.remove(1, 1.2)        # delete asset 1's point at t = 1.2 and join adjacent increments
online.advance(3.0)          # advance the event-time watermark and expire old increments

# Evaluate pairwise estimates at any supplied drift and point-noise variance
# from the retained sufficient statistics without replaying observations.
snapshot = online.snapshot(
    method="overlap",
    drift=[0.0, 0.0],
    noise_variance=[0.0, 0.0],
)
```

## Usage Notes

- **Observations and timestamps**: Each point is keyed by `(asset, time)` with
  nonnegative integer `asset` indices. Use relative timestamps so interval
  durations retain `float64` precision; estimated drift and covariance rates are
  expressed per unit of `time`. Duplicate keys replace their previous values
  (in `fit`, after stable sorting by time; in `OnlineCovariance.update`, in
  supplied arrival order).
- **Estimators (`"hy"` and `"overlap"`)**: `"hy"` divides weighted return
  products by weighted overlap duration. `"overlap"` weights each product by
  overlap divided by the product of the two increment durations. Both target a
  constant Brownian covariance rate under exogenous sampling; see
  [Statistical Model and Literature](docs/literature.md) for their assumptions
  and finite-sample properties.
- **Windows and decay**: `half_life` decays each overlap from its midpoint.
  `window` retains whole increments starting at or after `watermark - window`
  and rejects older edits; omit `window` to retain full history and allow
  arbitrary backfills.
- **Drift and microstructure noise**: `fit` centers increments at `drift` before
  accumulating. When streaming series have a large trend relative to volatility,
  pass a baseline `reference_drift` to `OnlineCovariance` at construction and
  pass the target `drift` to `snapshot`. Supplying known `noise_variance`
  corrects diagonal bias under independent additive point errors. Even with
  diagonal bias removed, the variance of raw high-frequency return products
  grows with sampling frequency under noisy infill sampling. `estimate_noise`
  computes the unclipped adjacent-return autocovariance moment (unbiased under
  exogenous sampling, independent Brownian increments, correct drift centering,
  and independent point errors), and `coarsen` thins batch observations to the
  last actual point per fixed time bucket.
- **Sparse graphs and factor projection**: Passing `edges=[(a, b), ...]`
  restricts off-diagonal work to selected pairs while tracking all diagonals.
  Assets without active increments have `NaN` in `MomentSnapshot.variance`;
  factor fitting requires positive finite marginal variances for all $`N`$
  assets (either observed or supplied via `variance=`). Unobserved off-diagonal
  pairs receive zero loss weight. Default fitting weights are decay-weighted
  overlap coverage (`weighted_coverage`), a support heuristic rather than
  inverse estimation variance. `FactorFit.diagnostics` reports convergence metadata
  and structural graph obstructions (such as disconnected components, bipartite
  components, or row degrees below `rank`).

## Verification and Benchmarks

```sh
python -m pip install '.[test]'
cargo test --locked --lib
OPENBLAS_NUM_THREADS=1 python -m pytest -q
OPENBLAS_NUM_THREADS=1 python benchmarks/engine.py
OPENBLAS_NUM_THREADS=1 python benchmarks/projection.py
python benchmarks/quality.py --reps 100
```

## Documentation

- [**Statistical Model and Literature (`docs/literature.md`)**](docs/literature.md):
  Continuous-time observation model, derivations of the `"hy"` and `"overlap"`
  estimators, Gaussian product variance, microstructure noise under infill
  sampling, factor completion on sparse graphs, and primary references.
- [**Architecture and Implementation (`docs/design.md`)**](docs/design.md):
  Event-state representation, interval treap and neighbor indexing, window
  expiration invariants, 14-term sufficient-statistic algebra, trust-region
  alternating least squares, and Python–Rust concurrency.
- [**Benchmarks and Empirical Results (`docs/benchmarks.md`)**](docs/benchmarks.md):
  Throughput, revision latency, memory footprint, factor-solver timings, and
  Monte Carlo statistical error measurements.
