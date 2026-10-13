# Implementation

## Event state

The key of a point is `(asset, timestamp)`. A replacement changes that point;
deleting a point joins its two adjacent increments. Each edit removes the old
adjacent increments and adds the new ones. Only intersecting intervals on
requested asset pairs contribute. Touching endpoints have zero overlap.

The unrestricted graph uses an augmented interval treap. A selected graph uses
per-asset ordered interval lookup on each neighboring asset. The latter avoids
finding overlaps on pairs that the user did not request. A changed increment
costs `O(log M + q log(M/q+1))` when the unrestricted treap is balanced,
or `O(log M + d log m + q_selected)`
on a graph with incident degree `d`. Here `M` is all retained increments, `m`
is increments per neighboring asset, and `q` is the number of reported overlaps.
A point edit removes at most two old increments and adds at most two new ones.
The treap uses hashed sequence priorities; logarithmic height is an expected
behavior, not a worst-case guarantee. Window expiry costs additional work
proportional to expired intervals and their overlaps, plus point pruning.

The engine stores points, intervals, and one accumulator per observed requested
pair. Without a window, memory grows with history. With a window, it expires
whole increments whose start precedes `watermark-window`; a predecessor point
is retained per asset to define subsequent increments. Boundary-crossing
increments are excluded. Clipping just the denominator of a full return would
change its expectation.

The watermark never moves backward. A late observation changes its own local
intervals at the current watermark. An overlap's decay anchor is its midpoint;
accumulators decay lazily by `exp(-log(2)*elapsed/half_life)`. Addition and
subtraction evaluate the same historical weight. The twelve weighted
coefficients use compensated summation. Counts and unweighted coverage describe
retained support; they are not effective sample sizes.

Immutable batch fits group points into asset intervals and use a two-pointer
scan for each requested pair. Their work is `O(M + sum_edges(m_a+m_b))`, plus
sorting if inputs arrive unordered. No revision index is built. Large pair
scans can use a cached private CPU pool, up to eight available threads by
default. Work below 100,000 estimated interval visits stays serial; `threads=1`
always selects serial execution. Each pair retains the same summation order.

## Rate estimates

For increment lengths `l_a,l_b`, overlap `o`, returns `r_a,r_b`, supplied
drift rates `mu_a,mu_b`, and overlap weight `w`, define
`z=(r_a-mu_a*l_a)*(r_b-mu_b*l_b)-noise_covariance`.

* HY rate: `sum(w*z) / sum(w*o)`.
* Overlap slope: `sum(w*o*z/(l_a*l_b)) / sum(w*o**2/(l_a*l_b))`.

Under a constant Brownian covariance rate, exogenous observation times, fixed
weights, and correctly specified noise correction, both estimate the same
pairwise rate. Overlap slope's inverse-duration weighting is variance optimal
among linear product estimates at zero cross-covariance in the clean model.
That statement does not extend automatically to correlated or noisy products.
With time-varying rates the estimators have different temporal kernels; neither
output is an instantaneous covariance estimate.

Each pair retains product, drift-polynomial, denominator, and diagonal noise
coefficients for both estimates. Drift and known noise variance can therefore
change at snapshot time without replay. No fitted drift or noise is silently
substituted. Batch fitting centers each increment at the supplied drift before
forming products. Online construction can fix a `reference_drift` baseline;
snapshot recentering uses the difference between supplied drift and baseline.
This prevents a large known trend from swamping a small residual covariance.
Newly grown assets have a zero baseline. Recentring far from that baseline can
still lose precision; choose a baseline near the drift when the scales differ
substantially. Iid endpoint errors, independent of latent returns and other
assets, with point-noise variance `tau_a`, add `2*tau_a` to a same-asset squared
increment, and zero to a cross-asset product. Correcting that bias does not
prevent noise-driven variance growth as sampling becomes finer.

`coarsen` keeps the last actual point per fixed physical-time bin. It preserves
asynchronous timestamps and duplicate replacement semantics. It reduces the
number of noisy increments, but its width is a user choice and it is not
pre-averaging. `estimate_noise` uses the signed adjacent-product mean under its
declared model; a negative result remains visible.

## Factor output

Raw estimates may be indefinite and may have negative diagonals after noise
correction. The engine preserves them. Factor fitting requires positive
diagonals; it minimizes a weighted loss on the observed off-diagonal
correlations. Missing entries receive zero weight. Coverage is the default
heuristic fit weight, not inverse variance.

Each alternating row update solves a quadratic on a Euclidean ball. The ball
reserves a positive fraction of that asset's variance for the diagonal term.
The resulting `B B.T + diag(d)` is positive definite and preserves the chosen
diagonal. Accepted loss is nonincreasing; the nonconvex joint fit need not find
a global optimum. The native solver first tries a scaled Cholesky solve for an
unconstrained row minimizer. It accepts that fast path only inside the ball with
a passing KKT residual. Boundary solutions and singular or poorly conditioned
normal equations use a scaled Jacobi eigensolve and a secular multiplier solve,
with decomposition, orthogonality, and KKT residual checks. An independent SciPy
backend solves the same row equations. Seeded random initialization is the
default; warm starts reuse a prior factor model. Optional spectral initialization
uses a sparse eigensolver and seeds active rows that otherwise have zero loadings.
A dense spectral projection is separately available.

Graph connectivity alone does not identify missing covariances. Factor
predictions outside observed edges depend on the model and initialization.
Diagnostics report graph support. Multiplication and Woodbury solves use
`O(NK)` storage; a dense covariance is allocated only when explicitly requested.

## Hardware

Event ingestion is ordered, branch-heavy work over mutable interval indexes.
The initial implementation uses native CPU code and releases Python's GIL.
It does not round-trip each event through Python. Batch inputs are copied before
releasing the GIL to prevent concurrent NumPy mutation. Shape, dtype, finite
values, and timestamps against the initial window cutoff are checked before
ingestion. Events in an online update array then commit one at a time; a later
numerical error or a cutoff advanced by preceding events stops at that event.

Factor projection is separate from event updates and can run less frequently.
Sparse fitting and partial eigensolves avoid dense `N**2` allocation. See
`benchmarks/` for end-to-end costs. GPU work is justified only if a measured
projection workload dominates these paths; no GPU backend is claimed here.
