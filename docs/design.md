# Architecture and Implementation

This document describes the data structures, numerical invariants, and
algorithms used by the Rust moment engine (`src/`) and the factor projection
routines (`src/factor.rs` and `python/async_covariance/`).

---

## 1. Observation State and Local Increment Revisions

### Per-asset ordered points and increments

In `OnlineCovariance` (`src/engine.rs`), observations for each asset
$`a \in \{0, \dots, N-1\}`$ are stored in a `BTreeMap<Time, Point>`, where
`Time` wraps an `f64` ordered by `f64::total_cmp` after canonicalizing `-0.0`
to `0.0`. Each `Point` stores its `price` (natural log price) and an optional
handle `next_interval: Option<usize>` referencing the increment that starts at
that timestamp and ends at the next retained timestamp for the same asset.

Consecutive points $`(t_{k-1}, p_{k-1})`$ and $`(t_k, p_k)`$ of asset $`a`$ form
a half-open increment interval $`I_e = [t_{k-1}, t_k)`$ with duration
$`\ell_e = t_k - t_{k-1} > 0`$ and centered return
$`\check{r}_e = (p_k - p_{k-1}) - \mu^{\mathrm{ref}}_a \ell_e`$. Two intervals
$`I_e = [s_e, u_e)`$ and $`I_f = [s_f, u_f)`$ overlap if and only if
$`\max(s_e, s_f) < \min(u_e, u_f)`$; intervals that touch at a single endpoint
($`u_e = s_f`$) have zero overlap.

### Local revision mechanics

Because each point $`(a, t_k)`$ is incident to at most two increments—its left
increment $`[t_{k-1}, t_k)`$ and its right increment $`[t_k, t_{k+1})`$-any
point mutation modifies at most two existing increments and creates at most two
replacement increments:

- **Insertion at a new timestamp $`t`$**: If $`t`$ lies strictly between
  existing neighbors $`t_{k-1} < t < t_{k+1}`$, the bridging increment
  $`[t_{k-1}, t_{k+1})`$ is removed (subtracting its contributions from all
  overlapping pairs) and replaced by $`[t_{k-1}, t)`$ and $`[t, t_{k+1})`$. If
  $`t`$ is a new minimum or maximum timestamp for asset $`a`$, at most one new
  increment is added and none removed.
- **Price replacement at an existing timestamp $`t_k`$**: If the new log price
  equals the stored price, the update is a no-op. Otherwise, the incident
  increments $`[t_{k-1}, t_k)`$ and $`[t_k, t_{k+1})`$ are removed and
  re-created with the updated endpoint price.
- **Deletion at an existing timestamp $`t_k`$ (`remove`)**: The incident
  increments $`[t_{k-1}, t_k)`$ and $`[t_k, t_{k+1})`$ are removed, the point
  at $`t_k`$ is deleted, and if both neighbors $`t_{k-1}`$ and $`t_{k+1}`$ exist
  (and $`t_{k-1}`$ lies within the active window), the joined increment
  $`[t_{k-1}, t_{k+1})`$ is inserted.

### Atomic pre-validation

Before `update` or `remove` advances the watermark, expires old intervals, or
mutates any index, it constructs all candidate replacement `Interval` structs
and validates that their durations, centered returns, and diagonal moment
contributions are finite IEEE 754 `f64` numbers. If validation fails (for
example, due to subnormal duration underflow or squared-return overflow), the
method returns an error while leaving the engine state, watermark, and retained
window unchanged.

---

## 2. Overlap Indexing and Asymptotic Complexity

### Dynamic full-graph indexing (`src/interval_index.rs`)

When `edges=None` (all asset pairs tracked), active increments across all assets
are indexed in an augmented treap (`IntervalIndex`):

- **Node storage and handles**: Nodes are stored in a contiguous arena
  `Vec<Option<Node>>` with a free-list `Vec<usize>` of recycled slots. The slot
  index serves as a stable `usize` handle stored on the originating `Point`.
- **Tree ordering and heuristic balancing**: Nodes are ordered lexicographically
  by `(start, handle)`. Each inserted node receives a deterministic 64-bit
  priority computed by applying the SplitMix64 finalizer (`mix64`) to a
  monotonically increasing 64-bit insertion counter. Because priorities are
  deterministic hashes of insertion sequence numbers rather than independent
  random draws, logarithmic tree depth in the number of active intervals $`M`$
  is an expected heuristic behavior rather than a worst-case guarantee.
- **Subtree augmentation and pruning**: Each node caches `max_end`, the maximum
  interval end timestamp in its subtree. To report all intervals overlapping a
  query $`[s, u)`$ (`start < other.end && other.start < end`):
  1. If a subtree's `max_end <= s`, no interval in that subtree can extend past
     $`s`$, so the entire subtree is skipped.
  2. Otherwise, the left subtree is searched recursively.
  3. If the current node satisfies `node.start < u`, it is tested for
     `node.end > s` and its right subtree is searched; if `node.start >= u`, all
     nodes in the right subtree also have start $`\ge u`$ and are skipped.

When the treap is balanced over $`M`$ active intervals, inserting or removing an
interval and reporting its $`q`$ overlapping intervals takes
$`O(\log M + q \log(M/q + 1))`$ time.

### Dynamic selected-graph indexing (`edges=[...]`)

When `edges` restricts off-diagonal tracking to a sparse graph $`\mathcal{E}`$
where asset $`a`$ has incident degree
$`d_a = \lvert \mathcal{N}(a) \rvert \ll N`$, querying a global interval tree
across all $`N`$ assets would visit overlapping intervals belonging to assets
not connected to $`a`$ in $`\mathcal{E}`$.

Instead, when `edges` is provided, `Engine` builds sorted adjacency lists
`neighbors[a]` and resolves overlaps for a changed interval
$`I_e = [s_e, u_e)`$ of asset $`a`$ by querying `points[b]` directly for each
neighbor $`b \in \mathcal{N}(a)`$:

1. Because intervals of asset $`b`$ are pairwise disjoint, at most one interval
   of asset $`b`$ starting at or before $`s_e`$ can overlap $`I_e`$, namely the
   one starting at the predecessor `points[b].range(..=s_e).next_back()`, which
   overlaps $`I_e`$ if and only if its `end > s_e`.
2. Every interval of asset $`b`$ whose start timestamp lies strictly inside
   $`(s_e, u_e)`$ (`points[b].range((Excluded(s_e), Excluded(u_e)))`)
   automatically satisfies $`\max(s_e, s_f) = s_f < \min(u_e, u_f)`$ and
   therefore overlaps $`I_e`$.

The interval is still stored in `IntervalIndex` for handle lookup and expiry
management. When the treap is balanced, a single interval addition or removal on
a selected graph costs

$$
O(\log M + d_a \log m + q_{\mathcal{E}}),
$$

where $`m`$ is the number of retained points per neighboring asset and
$`q_{\mathcal{E}}`$ is the number of overlapping intervals across incident edges
in $`\mathcal{E}`$.

### Immutable batch scans (`src/batch.rs`)

For one-shot estimation (`fit`) on $`L`$ input observations across $`N`$ assets,
no dynamic revision index is built:

1. If the input `times` array is not already nondecreasing, Python applies a
   stable sort (`np.argsort(t, kind="stable")`) in $`O(L \log L)`$ time so that
   the last supplied value at any duplicate `(asset, time)` key appears last.
2. In Rust, observations are partitioned into $`N`$ per-asset vectors in
   $`O(N + L)`$ work, collapsing consecutive duplicate timestamps in place and
   building a contiguous `Vec<Interval>` of in-window increments
   ($`s_e \ge t_{\max} - W`$) for each asset, with $`m_a = \lvert \mathcal{I}_a \rvert`$
   retained increments for asset $`a`$ and $`M = \sum_{a=0}^{N-1} m_a \le L`$.
3. Diagonal statistics are accumulated in $`O(M)`$ work. For each requested
   off-diagonal pair $`\{a, b\} \in \mathcal{E}`$ where both assets have at
   least one active increment, `pair_statistics` runs a linear two-pointer scan
   over the sorted disjoint slices `intervals[a]` and `intervals[b]`, advancing
   the pointer whose interval ends first (`if a.end <= b.end { i += 1; }` and
   `if b.end <= a.end { j += 1; }`).

The off-diagonal pair-scanning phase performs

$$
O\!\left(\sum_{\{a,b\} \in \mathcal{E},\, m_a, m_b > 0} (m_a + m_b)\right)
$$

interval visits, followed by $`O(P \log P)`$ sorting of the $`P`$ observed
output pairs by `(row, col)`. When estimated interval visits exceed
`PARALLEL_SCAN_THRESHOLD = 100_000` and `threads != 1`, independent pair scans
are distributed across a cached Rayon thread pool (defaulting to
$`\min(\text{available\_parallelism}, 8)`$ threads). Because each pair
$`(a, b)`$ is accumulated sequentially by a single task and the final pair list
is sorted by `(a, b)`, parallel execution is bitwise identical to serial
execution (`threads=1`).

---

## 3. Watermark Advancement, Sliding Windows, and Sentinel Points

The engine's event-time `watermark` $`t_{\max}`$ is the maximum of all inserted
observation timestamps and all explicit `advance(time)` calls; it never moves
backward.

When `window = None`, all inserted points and intervals are retained
indefinitely, and historical revisions or deletions are permitted at arbitrary
past timestamps.

When `window = Some(W)`, the active cutoff is $`c = t_{\max} - W`$:

1. **Whole-increment retention**: An increment $`I_e = [s_e, u_e)`$ is included
   if and only if $`s_e \ge c`$. Any attempt to `update` or `remove` a point at
   a timestamp $`t < c`$ is rejected with a `ValueError`.
2. **Ordered expiration queues**: Active intervals are tracked in
   `interval_expiry: BTreeSet<(Time, usize)>` keyed by `(start, handle)`, and
   inserted points are tracked in `point_expiry: BTreeSet<(Time, usize)>` keyed
   by `(time, asset)`.
3. **Interval and point pruning (`advance_to`)**: Whenever $`t_{\max}`$
   advances:
   - Every interval in `interval_expiry` with `start < c` is popped and removed
     via `remove_interval`, subtracting its decayed sufficient statistics from
     all pairs it overlapped.
   - Every point in `point_expiry` with `time < c` is popped and its asset is
     marked for pruning.
   - For each marked asset $`a`$, `points[a].split_off(&Time(c))` separates
     points at or after $`c`$ from points strictly before $`c`$. All points
     before $`c`$ are discarded **except** the single latest point
     $`(t_{\text{sentinel}}, p_{\text{sentinel}})`$ with
     $`t_{\text{sentinel}} < c`$, which is re-inserted into `points[a]` with
     `next_interval = None`.

Retaining this single pre-cutoff sentinel per asset bounds memory to the active
window plus at most one sentinel point per asset, while preserving the exact
boundary invariant: if a subsequent backfill inserts a point at $`t' \ge c`$,
its left neighbor is $`t_{\text{sentinel}} < c`$, which fails
`included(start, cutoff)` and therefore does not create a boundary-crossing
increment $`[t_{\text{sentinel}}, t')`$.

---

## 4. Sufficient Statistics, Lazy Decay, and Reference Drift

### The 14-component pair accumulator (`PairStats`)

Let $`\mu^{\mathrm{ref}} \in \mathbb{R}^N`$ denote the fixed centering baseline
(`reference_drift` in `OnlineCovariance`, or `drift` in `fit`), and let
$`\check{r}_e = (p(u_e) - p(s_e)) - \mu^{\mathrm{ref}}_a \ell_e`$ denote the
baseline-centered return stored on each `Interval`. If a snapshot is requested
at target drift $`\mu \in \mathbb{R}^N`$ and point-noise variance
$`\tau^2 \in \mathbb{R}_{\ge 0}^N`$, let
$`\delta = \mu - \mu^{\mathrm{ref}}`$ be the residual drift shift. Then the
target-centered return is $`\tilde{r}_e = \check{r}_e - \delta_a \ell_e`$, and
the bilinear expansion

$$
\tilde{r}_e \tilde{r}_f
= \check{r}_e \check{r}_f - \delta_a (\ell_e \check{r}_f) - \delta_b (\ell_f \check{r}_e) + \delta_a \delta_b (\ell_e \ell_f)
$$

allows both estimators to be evaluated at any $`(\mu, \tau^2)`$ from 14 additive
sufficient statistics per active pair $`(a, b)`$ ($`a \le b`$):

| Index | Estimator / Role | Per-Overlap Contribution (before multiplying by $`w_{ef}`$ on `0..12`) |
| :---: | :--- | :--- |
| `0` | `"hy"` product | $`\check{r}_e \check{r}_f`$ |
| `1` | `"hy"` linear $`\delta_a`$ term | $`\ell_e \check{r}_f`$ |
| `2` | `"hy"` linear $`\delta_b`$ term | $`\ell_f \check{r}_e`$ |
| `3` | `"hy"` bilinear $`\delta_a \delta_b`$ term | $`\ell_e \ell_f`$ |
| `4` | `"hy"` denominator & `weighted_coverage` | $`o_{ef}`$ |
| `5` | `"hy"` diagonal noise coefficient | $`2`$ if $`a = b`$, else $`0`$ |
| `6` | `"overlap"` product | $`\frac{o_{ef}}{\ell_e \ell_f} \check{r}_e \check{r}_f`$ |
| `7` | `"overlap"` linear $`\delta_a`$ term | $`\frac{o_{ef}}{\ell_f} \check{r}_f`$ |
| `8` | `"overlap"` linear $`\delta_b`$ term | $`\frac{o_{ef}}{\ell_e} \check{r}_e`$ |
| `9` | `"overlap"` bilinear $`\delta_a \delta_b`$ term | $`o_{ef}`$ |
| `10` | `"overlap"` denominator | $`\frac{o_{ef}^2}{\ell_e \ell_f}`$ (exact $`1.0`$ on diagonal $`a = b`$) |
| `11` | `"overlap"` diagonal noise coefficient | $`\frac{2}{\ell_e}`$ if $`a = b`$, else $`0`$ |
| `12` | Unweighted pair count (`counts`) | $`1`$ (tracked as exact `usize`, exported as `f64`) |
| `13` | Unweighted overlap (`coverage`) | $`o_{ef}`$ (unweighted) |

At snapshot time (`_snapshot` in `python/async_covariance/api.py`), let
$`x_0, \dots, x_5`$ denote indices `0..6` for `method="hy"` or indices `6..12`
for `method="overlap"`. The pairwise covariance rate is evaluated as

$$
\widehat{\Sigma}_{ab}
= \frac{x_0 - \delta_a x_1 - \delta_b x_2 + \delta_a \delta_b x_3 - \mathbf{1}_{\{a=b\}} \tau_a^2 x_5}{x_4},
$$

returning `NaN` when the denominator $`x_4 \le 0`$. When all overlapping
intervals for a pair $`(a, b)`$ have been removed (`count == 0`), its
`PairStats` entry is removed from the hash table so empty pairs do not retain
stale rounding residuals.

### Lazy exponential decay and Neumaier summation

For an overlapping pair $`(e, f)`$ with intersection $`I_e \cap I_f = [s, u)`$
and overlap $`o_{ef} = u - s`$, the decay anchor is the overlap midpoint
$`m_{ef} = s + \frac{1}{2} o_{ef}`$. When added or removed at watermark $`t`$,
indices `0..12` of its contribution are multiplied by

$$
w_{ef}(t) = \exp\!\left(-\frac{\ln 2}{H}(t - m_{ef})\right).
$$

Each `PairStats` records `last_time`, the watermark at which its sums were last
updated. When touched at a newer watermark $`t' > \text{last\_time}`$,
`decay_to` multiplies both `sums[0..12]` and `compensation[0..12]` by
$`\exp(-\frac{\ln 2}{H}(t' - \text{last\_time}))`$ and updates
$`\text{last\_time} \leftarrow t'`$. Because

$$
\exp\!\left(-\frac{\ln 2}{H}(t - m_{ef})\right) \exp\!\left(-\frac{\ln 2}{H}(t' - t)\right)
= \exp\!\left(-\frac{\ln 2}{H}(t' - m_{ef})\right) = w_{ef}(t'),
$$

a subsequent removal of $`(e, f)`$ at watermark $`t'`$ evaluates and subtracts
the same decayed weight $`w_{ef}(t')`$ present in the accumulator.

All 13 floating-point accumulators (indices `0..12` and `13`) use Neumaier
compensated summation, which tracks the low-order rounding error whether the new
term is smaller or larger in magnitude than the running sum (the latter
occurring when a historical correction replaces a long interval). On the
diagonal ($`a = b`$), index `6` is computed as `(r / l) * r` rather than
`(r / sqrt(l))^2` to avoid premature underflow on short durations, and index
`10` is set to the exact constant `1.0` rather than `(l / sqrt(l) / sqrt(l))^2`.

### Numerical stability under large drift (`reference_drift`)

If an asset has a large drift rate $`\mu_a`$ such that
$`(\mu_a \ell_e)^2 \gg \Sigma_{aa} \ell_e`$, accumulating uncentered products
$`r_e^2 \approx (\mu_a \ell_e)^2`$ in `sums[0]` and subtracting
$`2\mu_a x_1 - \mu_a^2 x_3`$ at snapshot time cancels the leading
floating-point bits, losing precision proportional to
$`(\mu_a \ell_e)^2 / (\Sigma_{aa} \ell_e)`$.

- `fit` avoids this automatically by passing `drift` as `reference_drift` to the
  batch engine, centering every increment before products are formed and passing
  $`\delta = 0`$ at snapshot evaluation.
- `OnlineCovariance` accepts `reference_drift` at construction to fix
  $`\mu^{\mathrm{ref}}`$ prior to streaming ingestion; `snapshot(drift=...)`
  then applies only the small residual shift
  $`\delta = \mu - \mu^{\mathrm{ref}}`$. If asset indices dynamically grow
  beyond the initial `n_assets`, newly grown assets default to
  $`\mu^{\mathrm{ref}}_a = 0`$.

---

## 5. Constrained Factor Projection (`src/factor.rs` and `projection.py`)

### Standardization and CSR graph representation

Given marginal variances $`v \in \mathbb{R}_{>0}^N`$, rank
$`K \in \{0, \dots, N-1\}`$, specific-variance floor $`\eta \in (0, 1]`$, and
observed off-diagonal edges $`\mathcal{E}`$ with positive weights
$`w_{ij} > 0`$, `fit_factor` transforms the data to correlation scale:

$$
c_{ij} = \frac{\widehat{\Sigma}_{ij}}{\sqrt{v_i v_j}}, \qquad
w_{ij} \leftarrow \frac{w_{ij}}{\max_{\{p,q\} \in \mathcal{E}} w_{pq}} \in (0, 1], \qquad
R = \sqrt{1 - \eta} \in [0, 1).
$$

Normalizing $`w_{ij}`$ by its maximum leaves the objective $`f(Z)`$ unchanged
while preventing overflow in the row normal equations when raw overlap
coverages span large magnitudes. The undirected graph $`\mathcal{E}`$ is indexed
in a compressed sparse row (CSR) adjacency table storing
`(neighbor_asset, edge_id)` pairs so that iterating over $`\mathcal{N}(i)`$
requires $`O(\deg(i))`$ memory accesses and never allocates an $`N \times N`$
matrix.

### Degenerate cases and Gauss–Seidel row updates

When $`K = 0`$, $`\lvert \mathcal{E} \rvert = 0`$, or $`R = 0`$
($`\eta = 1`$), no factor update work is needed: the optimizer immediately
returns zero factor loadings (`specific = variance`) with `converged = True` and
`stop_reason = "no_factor_work"`.

In the nondegenerate regime ($`K \ge 1`$, $`\lvert \mathcal{E} \rvert \ge 1`$,
$`R > 0`$), each sweep updates rows $`i = 0, \dots, N-1`$ sequentially in
Gauss–Seidel order. Isolated rows ($`\mathcal{N}(i) = \emptyset`$) are fixed at
$`z_i = 0`$. For row $`i`$ with $`\mathcal{N}(i) \neq \emptyset`$ and normalized
incident weights
$`\tilde{w}_{ij} = w_{ij} / \sum_{j' \in \mathcal{N}(i)} w_{ij'}`$, the row
subproblem is

$$
\min_{\|z\|_2 \le R} \; z^\top G_i z - 2 b_i^\top z, \qquad
G_i = \sum_{j \in \mathcal{N}(i)} \tilde{w}_{ij} z_j z_j^\top \in \mathbb{R}^{K \times K}, \quad
b_i = \sum_{j \in \mathcal{N}(i)} \tilde{w}_{ij} c_{ij} z_j \in \mathbb{R}^K.
$$

If $`b_i = 0`$, the minimum is attained at $`z = 0`$ and `quadratic_ball`
returns immediately. When $`b_i \neq 0`$ and $`R > 0`$, `quadratic_ball` in
`src/factor.rs` solves this convex subproblem in two stages without adding an
artificial ridge penalty to $`G_i`$:

1. **Scaled Cholesky interior fast path (`cholesky_interior`)**:
   Let $`s = \max_{p,q} \lvert (G_i)_{pq} \rvert`$. The solver attempts an
   in-place Cholesky factorization of $`G_i / s`$, requiring every diagonal
   pivot to exceed $`64 \varepsilon_{\mathrm{mach}} \max(1, K)`$. If the
   factorization succeeds, it solves $`G_i z = b_i`$ by forward and backward
   substitution and accepts $`z`$ if and only if $`\|z\|_2 \le R`$ and the
   unconstrained KKT residual satisfies

$$
\|G_i z - b_i\|_2 \le 128\,\varepsilon_{\mathrm{mach}} \max(1, K)\,\big(\|G_i\|_F \|z\|_2 + \|b_i\|_2\big).
$$

2. **Eigenvalue and secular bisection path (`symmetric_eigen`)**:
   When the solution lies on the boundary $`\|z\|_2 = R`$ or $`G_i`$ is singular
   or ill-conditioned, `symmetric_eigen` diagonalizes
   $`G_i / s = Q \tilde{\Lambda} Q^\top`$ using up to 64 sweeps of cyclic Jacobi
   rotations, verifying both the decomposition residual
   $`\| (G_i/s) Q - Q \tilde{\Lambda} \|_F \le 64 \varepsilon_{\mathrm{mach}} \max(1, K) \|G_i/s\|_F`$
   and eigenvector orthogonality
   $`\| Q^\top Q - I_K \|_F \le 64 \varepsilon_{\mathrm{mach}} \max(1, K)`$.
   Let $`\sigma_k = \max(0, s \tilde{\Lambda}_{kk})`$ and $`g = Q^\top b_i`$.
   Eigenvalues satisfying
   $`\sigma_k > \varepsilon_{\mathrm{mach}} \max(1, K) \max_j \sigma_j`$ are
   treated as positive; if the projection of $`g`$ onto the remaining numerical
   nullspace is at most $`64 \varepsilon_{\mathrm{mach}} \|b_i\|_2`$ and the
   minimum-norm unconstrained vector
   $`z_{\mathrm{free}} = \sum_{\sigma_k > 0} (g_k / \sigma_k) q_k`$ satisfies
   $`\|z_{\mathrm{free}}\|_2 \le R`$, $`z_{\mathrm{free}}`$ is used. Otherwise,
   the unique root $`\lambda^* > 0`$ of the secular equation
   $`\|(\Lambda + \lambda I_K)^{-1} g\|_2 = R`$ lies in
   $`(0, \|b_i\|_2 / R]`$ and is located by 80 iterations of bisection
   (stopping early when the bracket width reaches
   $`4 \varepsilon_{\mathrm{mach}} \lambda`$). The solution
   $`z = Q (\Lambda + \lambda^* I_K)^{-1} g`$ is radially clamped if its norm
   exceeds $`R`$ by a last-bit rounding error and is verified against the KKT
   condition
   $`\|(G_i + \lambda^* I_K) z - b_i\|_2 \le 128 \varepsilon_{\mathrm{mach}} \max(1, K) (\|G_i\|_F \|z\|_2 + \|b_i\|_2 + \lambda^* \|z\|_2)`$.

An updated row $`z_i^{\mathrm{new}}`$ replaces $`z_i`$ only if its evaluated row
loss does not exceed the current row loss; otherwise
`rejected_numerical_updates` is incremented. Likewise, if a full sweep increases
the global objective due to floating-point roundoff near a stationary point, the
sweep is reverted and the solver terminates with
`stop_reason = "floating_point_stall"`.

An independent reference implementation (`backend="python"` in
`python/async_covariance/projection.py`) solves the same row equations using
`scipy.linalg.eigh`.

### Initialization and the Woodbury operator

- **Initialization (`fit_factor`)**:
  - `initialization="random"` (default): Draws standardized loadings from
    $`\mathcal{N}(0, 0.01)`$ using `np.random.default_rng(seed)` and projects
    rows into the ball of radius $`R`$.
  - `initialization="spectral"`: Forms the sparse symmetric weighted correlation
    matrix $`M_{ij} = M_{ji} = w_{ij} c_{ij}`$ on
    $`\{i, j\} \in \mathcal{E}`$, extracts the $`K`$ leading algebraic
    eigenpairs via `scipy.sparse.linalg.eigsh` (or dense `scipy.linalg.eigh`),
    verifies that the relative eigenpair residual
    $`\|M V - V \Lambda\|_F / \|\Lambda\|_2 \le 10^{-7}`$, and projects
    $`V \max(\Lambda, 0)^{1/2}`$ into the ball of radius $`R`$. Any row
    participating in nonzero targets whose spectral loading has norm
    $`\le 64 \varepsilon_{\mathrm{mach}}`$ (which occurs when leading
    eigenvectors localize on a stronger disconnected component) receives a
    seeded random vector so alternating updates do not remain trapped at the
    $`z_i = 0`$ saddle point.
  - `initial=...`: Warm-starts from a supplied `FactorCovariance` or $`(N, K)`$
    loading matrix in covariance units, rescaling row $`i`$ by $`1 / \sqrt{v_i}`$
    and projecting into the ball of radius $`R`$.
- **`FactorCovariance` (`python/async_covariance/model.py`)**:
  Stores `loadings` $`B \in \mathbb{R}^{N \times K}`$ and `specific`
  $`d \in \mathbb{R}_{>0}^N`$ in immutable bytes-backed NumPy arrays.
  `matvec(x)` computes $`D x + B (B^\top x)`$ in $`O(NK)`$ work per column.
  `solve(x)` applies the Woodbury matrix identity:

$$
(B B^\top + D)^{-1} x = D^{-1} x - D^{-1} B \left(I_K + B^\top D^{-1} B\right)^{-1} B^\top D^{-1} x.
$$

When $`K \ge 1`$, the first call to `solve` computes $`D^{-1} B \in \mathbb{R}^{N \times K}`$,
forms the $`K \times K`$ capacitance matrix $`I_K + B^\top D^{-1} B`$, and
caches its lower Cholesky factor in $`O(N K^2 + K^3)`$ work and $`O(N K + K^2)`$
storage. Each right-hand side column is then solved in $`O(N K + K^2)`$ work.

---

## 6. Python–Rust Boundary and Concurrency

All native entry points in `src/lib.rs` (`Core.update`, `Core.update_many`,
`Core.remove`, `Core.advance`, `Core.statistics`, `batch_statistics`, and
`_factor_optimize`) copy their input slices into owned Rust vectors before
calling `py.detach(...)` to release Python's Global Interpreter Lock (GIL).
Copying prior to releasing the GIL prevents data races if another Python thread
mutates a caller's NumPy array while the native computation is running.

In `OnlineCovariance.update(times, assets, log_prices)`, array dimensions,
integer asset bounds ($`0 \le a < 2^{31}`$), finiteness of all timestamps and
log prices, and the initial window cutoff ($`t \ge t_{\max} - W`$) are validated
before any state is mutated. Events in the batch are then committed sequentially
in arrival order; if a later event in the batch fails interval moment validation
or falls behind a window cutoff that was advanced by an earlier event in the
same batch, execution stops at that event and raises a `ValueError`.

Each `OnlineCovariance` instance is a mutable state machine intended for a
single writer thread; `MomentSnapshot`, `FactorCovariance`, and `FactorFit` are
frozen dataclasses backed by read-only arrays and can be shared freely across
threads.
