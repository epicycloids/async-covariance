"""Factor-plus-diagonal covariance operators."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Hashable, Iterable

import numpy as np
from scipy.linalg import cho_factor, cho_solve


def _immutable_array(value: np.ndarray) -> np.ndarray:
    """Own an array in an immutable bytes buffer, including its shape."""
    array = np.ascontiguousarray(value, dtype=np.float64)
    return np.frombuffer(array.tobytes(), dtype=np.float64).reshape(array.shape)


@dataclass(frozen=True, eq=False)
class FactorCovariance:
    """Positive-definite covariance ``loadings @ loadings.T + diag(specific)``.

    ``specific`` contains variances, not standard deviations. Inputs are copied;
    the stored arrays cannot be made writable. The factor coordinates have no
    identifying rotation. Dense matrices are constructed only by ``covariance``
    and ``second_moment``.
    """

    loadings: np.ndarray
    specific: np.ndarray
    labels: Iterable[Hashable] | None = None
    _woodbury: tuple[np.ndarray, tuple[np.ndarray, bool]] | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        loadings = np.asarray(self.loadings, dtype=np.float64)
        specific = np.asarray(self.specific, dtype=np.float64)
        if loadings.ndim != 2:
            raise ValueError("loadings must have shape (n_assets, rank)")
        if specific.shape != (loadings.shape[0],):
            raise ValueError("specific must have shape (n_assets,)")
        if not np.all(np.isfinite(loadings)):
            raise ValueError("loadings must be finite")
        if not np.all(np.isfinite(specific)) or np.any(specific <= 0.0):
            raise ValueError("specific variances must be finite and positive")
        with np.errstate(over="ignore", invalid="ignore"):
            variance = specific + np.einsum("ij,ij->i", loadings, loadings)
        if not np.all(np.isfinite(variance)):
            raise ValueError("implied marginal variances must be finite")
        labels = None if self.labels is None else tuple(self.labels)
        if labels is not None:
            if len(labels) != len(specific):
                raise ValueError("labels must have one entry per asset")
            try:
                if len(set(labels)) != len(labels):
                    raise ValueError("labels must be unique")
            except TypeError as exc:
                raise TypeError("labels must be hashable") from exc
        object.__setattr__(self, "loadings", _immutable_array(loadings))
        object.__setattr__(self, "specific", _immutable_array(specific))
        object.__setattr__(self, "labels", labels)

    @property
    def n_assets(self) -> int:
        return len(self.specific)

    @property
    def rank(self) -> int:
        return self.loadings.shape[1]

    def _rhs(self, value: np.ndarray) -> np.ndarray:
        value = np.asarray(value, dtype=np.float64)
        if value.ndim not in (1, 2) or value.shape[0] != self.n_assets:
            raise ValueError("argument must have shape (n_assets,) or (n_assets, m)")
        if not np.all(np.isfinite(value)):
            raise ValueError("argument must be finite")
        return value

    def variance(self) -> np.ndarray:
        """Return the marginal variance vector without materializing covariance."""
        return self.specific + np.einsum("ij,ij->i", self.loadings, self.loadings)

    def matvec(self, value: np.ndarray) -> np.ndarray:
        """Multiply columns in ``O(n * rank)`` work per column."""
        value = self._rhs(value)
        diagonal = self.specific if value.ndim == 1 else self.specific[:, None]
        return diagonal * value + self.loadings @ (self.loadings.T @ value)

    def covariance(self) -> np.ndarray:
        """Materialize the dense covariance; this allocates ``n_assets ** 2`` entries."""
        result = self.loadings @ self.loadings.T
        result[np.diag_indices(self.n_assets)] += self.specific
        return result

    def solve(self, value: np.ndarray) -> np.ndarray:
        """Solve ``(loadings @ loadings.T + diag(specific)) @ x = value`` via Woodbury.

        On the first call with ``rank >= 1``, caches ``D^-1 @ loadings`` and the
        ``rank x rank`` Cholesky factor of ``I + loadings.T @ D^-1 @ loadings``
        in ``O(n_assets * rank**2 + rank**3)`` work. Each right-hand side column
        then solves in ``O(n_assets * rank + rank**2)`` work without forming a
        dense ``n_assets x n_assets`` matrix.
        """
        value = self._rhs(value)
        diagonal = self.specific if value.ndim == 1 else self.specific[:, None]
        scaled = value / diagonal
        if self.rank == 0:
            return scaled
        cached = self._woodbury
        if cached is None:
            weighted_loadings = self.loadings / self.specific[:, None]
            gram = np.eye(self.rank) + self.loadings.T @ weighted_loadings
            factor = cho_factor(gram, lower=True, check_finite=True)
            factor = (_immutable_array(factor[0]), factor[1])
            cached = (_immutable_array(weighted_loadings), factor)
            object.__setattr__(self, "_woodbury", cached)
        weighted_loadings, factor = cached
        coefficient = cho_solve(factor, self.loadings.T @ scaled, check_finite=False)
        return scaled - weighted_loadings @ coefficient

    def second_moment(
        self, horizon: float, drift: np.ndarray | None = None
    ) -> np.ndarray:
        """Materialize the increment second moment ``h * covariance + h² * μ μ.T``.

        ``horizon`` uses the covariance-rate time unit; ``drift`` is a mean rate
        in that same unit. This assumes constant rates over the horizon.
        """
        horizon = float(horizon)
        if not np.isfinite(horizon) or horizon < 0.0:
            raise ValueError("horizon must be finite and nonnegative")
        if drift is None:
            return horizon * self.covariance()
        drift = np.asarray(drift, dtype=np.float64)
        if drift.shape != (self.n_assets,) or not np.all(np.isfinite(drift)):
            raise ValueError("drift must be a finite vector of shape (n_assets,)")
        mean = horizon * drift
        return horizon * self.covariance() + np.outer(mean, mean)
