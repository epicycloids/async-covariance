"""Weighted graph factor fitting with explicit spectral alternatives.

The graph objective is nonconvex. Each alternating row update solves a convex
quadratic on a Euclidean ball; its accepted objective values are nonincreasing.
Missing edges contribute no loss. This is different from treating a missing
covariance as an observed zero.
"""

from __future__ import annotations

from dataclasses import dataclass
import operator
from types import MappingProxyType
from typing import Mapping

import numpy as np
from scipy.linalg import eigh
from scipy.sparse import coo_matrix, issparse
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import ArpackNoConvergence, eigsh

from .model import FactorCovariance, _immutable_array


@dataclass(frozen=True, eq=False)
class FactorFit:
    """Result of fitting a factor-plus-diagonal covariance model to an observed edge graph.

    ``model`` holds the fitted :class:`FactorCovariance`, ``objective_history``
    records the weighted mean-square correlation loss after initialization and
    each accepted sweep, ``converged`` indicates whether optimization stopped
    via ``'objective_tolerance'`` or ``'no_factor_work'``, and ``diagnostics``
    reports solver metadata and structural graph properties (component counts,
    bipartite components, and degree-deficient rows).
    """

    model: FactorCovariance
    objective_history: np.ndarray
    converged: bool
    iterations: int
    diagnostics: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "objective_history", _immutable_array(self.objective_history))
        object.__setattr__(self, "converged", bool(self.converged))
        object.__setattr__(self, "iterations", int(self.iterations))
        object.__setattr__(self, "diagnostics", MappingProxyType(dict(self.diagnostics)))

    @property
    def objective(self) -> float:
        return float(self.objective_history[-1])


def _integer(value: int, name: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer")
    try:
        value = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _variance_rank(variance: np.ndarray, rank: int) -> tuple[np.ndarray, int]:
    variance = np.asarray(variance, dtype=np.float64)
    if variance.ndim != 1 or not len(variance):
        raise ValueError("variance must be a nonempty vector")
    if not np.all(np.isfinite(variance)) or np.any(variance <= 0.0):
        raise ValueError("variance must be finite and positive")
    rank = _integer(rank, "rank")
    if rank >= len(variance):
        raise ValueError("rank must be less than n_assets")
    return variance, rank


def _floor(value: float) -> float:
    value = float(value)
    if not np.isfinite(value) or not 0.0 < value <= 1.0:
        raise ValueError("specific_floor must lie in (0, 1]")
    return value


def _indices(value: np.ndarray, name: str, size: int) -> np.ndarray:
    value = np.asarray(value)
    if value.ndim != 1:
        raise ValueError(f"{name} must be a vector")
    if value.size and value.dtype.kind not in "iu":
        raise TypeError(f"{name} must contain integer indices")
    if value.size and (np.any(value < 0) or np.any(value >= size)):
        raise ValueError(f"{name} indices must lie in [0, n_assets)")
    return value.astype(np.int64, copy=False)


def _graph(rows, cols, covariance, weights, variance):
    n = len(variance)
    rows, cols = _indices(rows, "rows", n), _indices(cols, "cols", n)
    covariance = np.asarray(covariance, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if cols.shape != rows.shape or covariance.shape != rows.shape or weights.shape != rows.shape:
        raise ValueError("rows, cols, covariance and weights must have the same vector shape")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("weights must be finite and nonnegative")
    if np.any(rows == cols):
        raise ValueError("diagonal entries belong in variance, not the edge arrays")
    observed = weights > 0.0
    rows, cols, covariance, weights = (a[observed] for a in (rows, cols, covariance, weights))
    if not np.all(np.isfinite(covariance)):
        raise ValueError("covariance must be finite on positive-weight edges")
    rows, cols = np.minimum(rows, cols), np.maximum(rows, cols)
    order = np.lexsort((cols, rows))
    rows, cols, covariance, weights = (a[order] for a in (rows, cols, covariance, weights))
    if len(rows) > 1 and np.any((rows[1:] == rows[:-1]) & (cols[1:] == cols[:-1])):
        raise ValueError("each observed unordered asset pair must occur once")
    if not len(rows):
        return rows, cols, covariance, weights
    std = np.sqrt(variance)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        correlation = (covariance / std[rows]) / std[cols]
    if not np.all(np.isfinite(correlation)):
        raise ValueError("standardized covariance must be finite")
    if np.any(np.abs(correlation) > np.sqrt(np.finfo(float).max) / 4.0):
        raise ValueError("standardized covariance exceeds the squared-loss floating-point range")
    # A common scaling leaves the minimizer unchanged and prevents exposure
    # magnitudes from overflowing the row normal equations.
    weights = weights / float(weights.max())
    if np.any(weights == 0.0):
        raise ValueError("positive weights have a dynamic range too large for float64 normalization")
    return rows, cols, correlation, weights


def _ball(value: np.ndarray, radius: float) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    norms = np.linalg.norm(value, axis=1)
    scale = np.minimum(1.0, radius / np.maximum(norms, np.finfo(float).tiny))
    return value * scale[:, None]


def _quadratic_ball(gram: np.ndarray, linear: np.ndarray, radius: float) -> np.ndarray:
    """Solve min ``z.T @ gram @ z - 2 * linear.T @ z`` on ``||z|| <= radius``.

    For a boundary solution the multiplier is the unique positive root of
    ``sum((q.T @ linear / (eigenvalues + multiplier)) ** 2) = radius ** 2``.
    This handles singular normal equations without changing the loss by adding
    a ridge. The caller checks the computed row loss before accepting a step.
    """
    if radius == 0.0 or not np.any(linear):
        return np.zeros_like(linear)
    values, vectors = eigh(0.5 * (gram + gram.T), check_finite=False)
    values = np.maximum(values, 0.0)
    coordinates = vectors.T @ linear
    threshold = np.finfo(float).eps * max(1, len(values)) * float(values.max(initial=0.0))
    positive = values > threshold
    free = np.divide(coordinates, values, out=np.zeros_like(coordinates), where=positive)
    null_linear = np.linalg.norm(coordinates[~positive])
    roundoff = 64.0 * np.finfo(float).eps * max(np.linalg.norm(linear), np.finfo(float).tiny)
    if null_linear <= roundoff and np.linalg.norm(free) <= radius:
        return vectors @ free
    lower = 0.0
    upper = max(np.linalg.norm(linear) / radius, np.finfo(float).tiny)
    for _ in range(80):
        multiplier = 0.5 * (lower + upper)
        norm = np.linalg.norm(coordinates / (values + multiplier))
        if norm > radius:
            lower = multiplier
        else:
            upper = multiplier
        if upper - lower <= 4.0 * np.finfo(float).eps * upper:
            break
    result = vectors @ (coordinates / (values + upper))
    # Remove a possible outward last-bit error in the orthogonal transform.
    norm = np.linalg.norm(result)
    if norm > radius:
        result *= radius / norm
    return result


def _top_eigenpairs(matrix, rank: int, method: str, seed: int):
    if method not in {"eigh", "eigsh"}:
        raise ValueError("eigensolver must be 'eigh' or 'eigsh'")
    n = matrix.shape[0]
    if rank == 0:
        return np.empty(0), np.empty((n, 0)), 0.0
    if method == "eigh":
        dense = matrix.toarray() if issparse(matrix) else np.asarray(matrix)
        values, vectors = eigh(dense, subset_by_index=(n - rank, n - 1), check_finite=False)
    else:
        start = np.random.default_rng(seed).normal(size=n)
        try:
            values, vectors = eigsh(matrix, k=rank, which="LA", v0=start, tol=1e-10)
        except ArpackNoConvergence as exc:
            raise RuntimeError("eigsh did not converge; choose eigensolver='eigh' for a dense solve") from exc
    residual = matrix @ vectors - vectors * values
    scale = max(float(np.linalg.norm(values)), np.finfo(float).tiny)
    relative_residual = float(np.linalg.norm(residual) / scale)
    if not np.isfinite(relative_residual) or relative_residual > 1e-7:
        raise RuntimeError("eigensolver residual exceeds 1e-7")
    order = np.argsort(values)[::-1]
    return values[order], vectors[:, order], relative_residual


def _initial_correlation(rows, cols, targets, weights, n, rank, radius, seed, method):
    if not len(rows) or not np.any(targets) or rank == 0 or radius == 0.0:
        return np.zeros((n, rank)), 0.0, 0
    graph = coo_matrix(
        (np.r_[targets * weights, targets * weights], (np.r_[rows, cols], np.r_[cols, rows])),
        shape=(n, n),
    ).tocsr()
    values, vectors, residual = _top_eigenpairs(graph, rank, method, seed)
    loadings = _ball(vectors * np.sqrt(np.maximum(values, 0.0)), radius)
    # A global leading spectrum can allocate every factor to the strongest
    # component. Zero factors in another component make both its row Gram and
    # linear term zero, so alternating updates cannot leave that saddle.
    # Seed numerically zero rows participating in nonzero targets; observed
    # zero-only rows and isolated assets retain their zero initialization.
    active = np.zeros(n, dtype=bool)
    nonzero = targets != 0.0
    active[rows[nonzero]] = True
    active[cols[nonzero]] = True
    unresolved = active & (np.linalg.norm(loadings, axis=1) <= 64.0 * np.finfo(float).eps)
    if np.any(unresolved):
        random = np.random.default_rng(seed).normal(size=(n, rank)) * 0.1
        loadings[unresolved] = _ball(random[unresolved], radius)
    return loadings, residual, int(unresolved.sum())


def _graph_diagnostics(neighbors, rank):
    adjacency = neighbors.copy()
    adjacency.data = np.ones(len(adjacency.data), dtype=np.int8)
    components, labels = connected_components(adjacency, directed=False)
    degree = np.diff(adjacency.indptr)
    sizes = np.bincount(labels, minlength=components)
    colors = np.full(len(degree), -1, dtype=np.int8)
    bipartite_components = 0
    for root in range(len(degree)):
        if degree[root] == 0 or colors[root] >= 0:
            continue
        colors[root] = 0
        stack = [root]
        bipartite = True
        while stack:
            vertex = stack.pop()
            for other in adjacency.indices[adjacency.indptr[vertex] : adjacency.indptr[vertex + 1]]:
                if colors[other] < 0:
                    colors[other] = 1 - colors[vertex]
                    stack.append(other)
                elif colors[other] == colors[vertex]:
                    bipartite = False
        bipartite_components += int(bipartite)
    issues = []
    if rank:
        if components > 1:
            issues.append("cross-component covariances are not identified by observed edges")
        if bipartite_components:
            issues.append("bipartite components permit reciprocal factor scaling")
        if np.any((degree > 0) & (degree < rank)):
            issues.append("some row updates have fewer observed neighbors than factor coordinates")
    return {
        "n_graph_components": components,
        "largest_graph_component": int(sizes.max(initial=0)),
        "n_bipartite_components": bipartite_components,
        "n_assets_with_degree_below_rank": int(np.count_nonzero(degree < rank)),
        "minimum_row_degree": int(degree.min(initial=len(degree))),
        "median_row_degree": float(np.median(degree)),
        "identifiability_issues": tuple(issues),
        "completion_identifiability": "fixed_diagonal_model" if rank == 0 else "not_established",
    }


def _python_optimize(rows, cols, targets, weights, loadings, neighbors, radius, max_iter, tol):
    """Independent NumPy/SciPy reference for the native row solver."""
    n, rank = loadings.shape
    total_weight = float(weights.sum())

    def objective(value):
        if not total_weight:
            return 0.0
        residual = np.einsum("ij,ij->i", value[rows], value[cols]) - targets
        return float(np.dot(weights / total_weight, residual * residual))

    history = [objective(loadings)]
    converged = rank == 0 or not len(rows) or radius == 0.0
    reason = "no_factor_work" if converged else "max_iter"
    rejected_updates = 0
    max_update = 0.0
    for _ in range(0 if converged else max_iter):
        previous = loadings.copy()
        max_update = 0.0
        for i in range(n):
            begin, end = neighbors.indptr[i : i + 2]
            if begin == end:
                continue
            edge = neighbors.data[begin:end]
            other = loadings[neighbors.indices[begin:end]]
            weight = weights[edge]
            target = targets[edge]
            normalized = weight / float(weight.sum())
            gram = other.T @ (normalized[:, None] * other)
            linear = other.T @ (normalized * target)
            trial = _quadratic_ball(gram, linear, radius)
            before = other @ loadings[i] - target
            after = other @ trial - target
            if np.dot(normalized, after * after) <= np.dot(normalized, before * before):
                max_update = max(max_update, float(np.linalg.norm(trial - loadings[i])))
                loadings[i] = trial
            else:
                rejected_updates += 1
        loss = objective(loadings)
        if loss > history[-1]:
            loadings = previous
            converged, reason = False, "floating_point_stall"
            break
        improvement = history[-1] - loss
        history.append(loss)
        if improvement <= tol * max(1.0, history[0]):
            converged, reason = True, "objective_tolerance"
            break
    return loadings, np.asarray(history), converged, reason, rejected_updates, max_update


def fit_factor(
    rows,
    cols,
    covariance,
    weights,
    variance,
    rank: int,
    *,
    specific_floor: float = 0.05,
    max_iter: int = 100,
    tol: float = 1e-8,
    seed: int = 0,
    initial: FactorCovariance | np.ndarray | None = None,
    initialization: str = "random",
    eigensolver: str = "eigsh",
    backend: str = "native",
) -> FactorFit:
    """Fit a positive-definite factor model to observed off-diagonal covariance rates.

    Minimizes ``sum(w_ij * (z_i @ z_j - c_ij)**2) / sum(w_ij)`` over observed
    off-diagonal edges ``(i, j)`` where ``c_ij = covariance_ij / sqrt(variance_i * variance_j)``,
    subject to ``||z_i||**2 <= 1 - specific_floor`` for each row ``i``. The
    resulting covariance ``B @ B.T + diag(d)`` preserves ``variance`` on the
    diagonal and satisfies ``d >= specific_floor * variance > 0``.

    Supply each unordered off-diagonal pair at most once. Zero-weight entries
    are omitted from the objective (including ``NaN`` covariances), whereas
    positive-weight zero covariances are fitted as observed zeros. ``rank``
    must lie in ``[0, n_assets)``. ``initial`` accepts a prior
    :class:`FactorCovariance` or ``(n_assets, rank)`` loadings in covariance
    units, rescaled to ``variance`` and projected into the row balls.

    ``initialization`` selects ``'random'`` (the seeded default) or
    ``'spectral'`` (using sparse ``eigensolver='eigsh'`` or dense ``'eigh'``,
    with seeded random fallback for active rows whose leading spectral loading
    is numerically zero). ``backend='native'`` runs the Gauss-Seidel row
    updates in compiled Rust; ``backend='python'`` runs the NumPy/SciPy
    reference implementation. Because the joint objective is nonconvex,
    alternating updates produce a local candidate with nonincreasing objective
    history; stopping on ``tol`` indicates small sweep-to-sweep objective
    change rather than a certificate of stationarity, global optimality, or
    unique missing-edge completion.
    """
    variance, rank = _variance_rank(variance, rank)
    floor = _floor(specific_floor)
    max_iter = _integer(max_iter, "max_iter")
    seed = _integer(seed, "seed")
    tol = float(tol)
    if not np.isfinite(tol) or tol < 0.0:
        raise ValueError("tol must be finite and nonnegative")
    if initialization not in {"spectral", "random"}:
        raise ValueError("initialization must be 'spectral' or 'random'")
    if eigensolver not in {"eigh", "eigsh"}:
        raise ValueError("eigensolver must be 'eigh' or 'eigsh'")
    if backend not in {"native", "python"}:
        raise ValueError("backend must be 'native' or 'python'")
    rows, cols, targets, weights = _graph(rows, cols, covariance, weights, variance)
    n, radius = len(variance), np.sqrt(1.0 - floor)
    eigen_residual = None
    seeded_rows = 0
    if initial is not None:
        raw = initial.loadings if isinstance(initial, FactorCovariance) else np.asarray(initial, dtype=float)
        if raw.shape != (n, rank) or not np.all(np.isfinite(raw)):
            raise ValueError("initial loadings must be finite with shape (n_assets, rank)")
        loadings = _ball(raw / np.sqrt(variance)[:, None], radius)
    elif initialization == "random":
        loadings = _ball(np.random.default_rng(seed).normal(size=(n, rank)) * 0.1, radius)
    else:
        loadings, eigen_residual, seeded_rows = _initial_correlation(
            rows, cols, targets, weights, n, rank, radius, seed, eigensolver
        )
    # CSR stores edge IDs so a row update obtains neighbors, targets and weights
    # together without materializing an n_assets x n_assets moment matrix.
    edge_ids = np.arange(len(rows), dtype=np.int64)
    neighbors = coo_matrix(
        (np.r_[edge_ids, edge_ids], (np.r_[rows, cols], np.r_[cols, rows])), shape=(n, n)
    ).tocsr()
    structural = _graph_diagnostics(neighbors, rank)
    isolated = np.diff(neighbors.indptr) == 0
    loadings[isolated] = 0.0
    if backend == "native":
        from ._native import _factor_optimize

        loadings, history, converged, reason, rejected_updates, max_update = _factor_optimize(
            rows, cols, targets, weights, np.ascontiguousarray(loadings).ravel(),
            n, rank, float(radius), max_iter, tol,
        )
        loadings = np.asarray(loadings).reshape(n, rank)
        history = np.asarray(history)
    else:
        loadings, history, converged, reason, rejected_updates, max_update = _python_optimize(
            rows, cols, targets, weights, loadings, neighbors, radius, max_iter, tol,
        )
    norm_sq = np.einsum("ij,ij->i", loadings, loadings)
    specific = variance * np.maximum(1.0 - norm_sq, floor)
    model = FactorCovariance(np.sqrt(variance)[:, None] * loadings, specific)
    return FactorFit(
        model=model,
        objective_history=np.asarray(history),
        converged=converged,
        iterations=len(history) - 1,
        diagnostics={
            **structural,
            "stop_reason": reason,
            "n_observed_pairs": len(rows),
            "n_isolated_assets": int(isolated.sum()),
            "initialization": "provided" if initial is not None else initialization,
            "eigensolver": eigensolver,
            "backend": backend,
            "initial_eigen_residual": eigen_residual,
            "n_seeded_spectral_rows": seeded_rows,
            "max_row_update": max_update,
            "rejected_numerical_updates": rejected_updates,
            "specific_floor": floor,
            "weighted_correlation_rmse": float(np.sqrt(history[-1])),
        },
    )


def spectral_factor(
    covariance,
    rank: int,
    *,
    variance=None,
    specific_floor: float = 0.05,
    eigensolver: str = "eigh",
    seed: int = 0,
) -> FactorCovariance:
    """A dense, explicitly spectral factor projection with fixed variances.

    This truncates the positive leading spectrum of the off-diagonal correlation
    matrix, caps row norms, and restores the diagonal with specific variances.
    It is an initialization/approximation, not a solution of the weighted graph
    objective. Both partial ``'eigh'`` and residual-checked ``'eigsh'`` are
    available. Use ``fit_factor`` when missing edges must have zero loss weight.
    """
    covariance = np.asarray(covariance, dtype=np.float64)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("covariance must be a square matrix")
    if not np.all(np.isfinite(covariance)):
        raise ValueError("covariance must be finite")
    if not np.allclose(covariance, covariance.T, rtol=1e-10, atol=1e-12):
        raise ValueError("covariance must be symmetric")
    variance, rank = _variance_rank(np.diag(covariance) if variance is None else variance, rank)
    if len(variance) != len(covariance):
        raise ValueError("variance must match the covariance dimension")
    floor = _floor(specific_floor)
    seed = _integer(seed, "seed")
    if eigensolver not in {"eigh", "eigsh"}:
        raise ValueError("eigensolver must be 'eigh' or 'eigsh'")
    std = np.sqrt(variance)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        correlation = (covariance / std[:, None]) / std[None, :]
    if not np.all(np.isfinite(correlation)):
        raise ValueError("standardized covariance must be finite")
    correlation = 0.5 * (correlation + correlation.T)
    np.fill_diagonal(correlation, 0.0)
    if rank and np.any(correlation) and floor < 1.0:
        values, vectors, _ = _top_eigenpairs(correlation, rank, eigensolver, seed)
        loadings = _ball(vectors * np.sqrt(np.maximum(values, 0.0)), np.sqrt(1.0 - floor))
    else:
        loadings = np.zeros((len(variance), rank))
    specific = variance * np.maximum(1.0 - np.einsum("ij,ij->i", loadings, loadings), floor)
    return FactorCovariance(std[:, None] * loadings, specific)
