# async-covariance

Covariance-rate estimation from asynchronous log prices. A Rust core maintains
overlap moments through inserts, corrections, and deletions. Immutable batch
fits use direct interval scans. A native factor solver fits a positive-definite
low-rank-plus-diagonal model on selected pair estimates.

## Install

Python 3.10+. The [release wheel](https://github.com/epicycloids/async-covariance/releases)
supports Linux x86-64 with glibc 2.35+. Install a downloaded wheel with `pip`.
To build from source, use Rust 1.83+:

```sh
python -m pip install .
```

NumPy and SciPy handle array operations and optional sparse spectral
initialization. Tested on Linux x86-64.

## Use

```python
from async_covariance import OnlineCovariance, fit

# Numeric timestamps in one consistent unit; prices are already logarithms.
times = [0, .2, 1, 1.2, 2, 2.2]
assets = [0, 1, 0, 1, 0, 1]
log_prices = [0, 0, .1, .2, .15, .16]

raw = fit(times, assets, log_prices, method="hy")
model = raw.factor(rank=1).model
model.matvec([1., -1.])
model.solve([1., -1.])

online = OnlineCovariance(2, half_life=10, window=100)
online.update(times, assets, log_prices)
online.update(1., 0, .11)  # replace an earlier observation
online.remove(1, 1.2)
raw = online.snapshot(method="overlap", drift=[0., 0.], noise_variance=[0., 0.])
```

Asset indices are nonnegative integers. Use relative timestamps to preserve
precision; rates are per timestamp unit. Duplicate keys replace their previous
values. Batch fits sort by time; online updates follow supplied arrival order.
Instances use a single writer; the native work releases the GIL.

`edges=[(0, 1), ...]` restricts off-diagonal work; all diagonals remain included.
Large batch fits parallelize independent pair scans; `threads=1` selects serial
execution.
A window expires whole increments starting before `watermark-window` and rejects
older revisions. Omit it to retain history and permit unlimited backfills.

`"hy"` estimates products divided by overlap exposure. `"overlap"` uses an
inverse-duration overlap regression. Their efficiency depends on sampling and
noise. Both target a constant Brownian covariance rate under the assumptions in
[the method notes](docs/literature.md).
Known-drift batch fits center before accumulating. For a large online trend,
set `reference_drift=` at construction and supply that drift to `snapshot`.

Known `noise_variance` corrects diagonal bias under independent iid point noise.
Raw products can still become less precise with finer noisy sampling. `coarsen`
provides explicit fixed-bin batch thinning; `estimate_noise` provides the signed
adjacent-return moment. Neither silently chooses a bandwidth or clips an estimate.

`raw.dense()` uses NaN for missing pairs and preserves negative estimates. Factor
fitting needs positive diagonals; supply `variance=` explicitly for external
diagonal estimates. Missing pairs carry zero loss weight. A factor model's
predictions on those pairs depend on its assumptions and graph support.
`FactorFit` reports loss history, stopping reason, and support diagnostics.
`initial=previous_model` warm-starts fitting. Coverage weights are a heuristic,
not claimed inverse uncertainty. `spectral_factor` is a separate dense projection.

## Check and benchmark

```sh
python -m pip install '.[test]'
cargo test --locked --lib
OPENBLAS_NUM_THREADS=1 python -m pytest -q
OPENBLAS_NUM_THREADS=1 python benchmarks/engine.py
OPENBLAS_NUM_THREADS=1 python benchmarks/projection.py
python benchmarks/quality.py --reps 100
```

[Implementation](docs/design.md) covers indexing, corrections, decay, windows,
and factor constraints. [Measured results](docs/benchmarks.md) give costs and
statistical errors with their workloads. [Primary references](docs/literature.md)
explain the targets and alternatives.
