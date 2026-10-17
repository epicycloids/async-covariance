"""Incremental covariance-rate estimates from asynchronous log prices."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from ._native import Core

Method = Literal["hy", "overlap"]


def _observations(times, assets, log_prices):
    t = np.ascontiguousarray(times, dtype=np.float64)
    raw_assets = np.asarray(assets)
    p = np.ascontiguousarray(log_prices, dtype=np.float64)
    if t.ndim != 1 or raw_assets.ndim != 1 or p.ndim != 1:
        raise ValueError("observations must be one-dimensional arrays")
    if len(t) != len(raw_assets) or len(t) != len(p):
        raise ValueError("times, assets, and log_prices must have equal lengths")
    if raw_assets.dtype.kind not in "iu" and raw_assets.size:
        raise ValueError("assets must contain integer indices")
    if raw_assets.size and (np.any(raw_assets < 0) or np.any(raw_assets >= 2**31)):
        raise ValueError("assets must be nonnegative integer indices below 2**31")
    a = np.ascontiguousarray(raw_assets, dtype=np.int64)
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(p)):
        raise ValueError("times and log_prices must be finite")
    return t, a, p


def _vector(value, size, name, *, nonnegative=False):
    try:
        arr = np.array(np.broadcast_to(np.asarray(value, dtype=float), (size,)), copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must broadcast to a numeric vector of length {size}") from exc
    if not np.all(np.isfinite(arr)) or (nonnegative and np.any(arr < 0)):
        raise ValueError(f"{name} must be finite" + (" and nonnegative" if nonnegative else ""))
    return arr


def _readonly(array):
    array.setflags(write=False)
    return array


def _edge_pairs(edges):
    if edges is None:
        return None
    e = np.asarray(edges)
    if e.size == 0:
        return []
    if e.ndim != 2 or e.shape[1] != 2 or e.dtype.kind not in "iu" or np.any(e < 0):
        raise ValueError("edges must contain nonnegative integer index pairs")
    return [(int(a), int(b)) for a, b in e]


@dataclass(frozen=True)
class MomentSnapshot:
    """Pairwise covariance-rate estimates and overlap support diagnostics.

    Units are log-return covariance per input time unit. ``rows`` and ``cols``
    index observed asset pairs ``0 <= rows[k] <= cols[k] < n_assets`` with at
    least one overlapping increment pair. ``covariance`` holds the pairwise
    rate estimate (``NaN`` if ``denominator <= 0``), ``counts`` holds unweighted
    overlapping interval-pair counts, ``coverage`` holds unweighted total
    overlap duration, ``weighted_coverage`` holds decay-weighted overlap
    duration, and ``variance`` holds the length-``n_assets`` diagonal variance
    rates (``NaN`` for assets without positive diagonal denominator).
    """

    n_assets: int
    method: Method
    rows: np.ndarray
    cols: np.ndarray
    covariance: np.ndarray
    denominator: np.ndarray
    counts: np.ndarray
    coverage: np.ndarray
    weighted_coverage: np.ndarray
    variance: np.ndarray
    watermark: float | None

    def dense(self):
        """Materialize the symmetric ``(n_assets, n_assets)`` rate matrix, with ``NaN`` on unobserved pairs."""
        out = np.full((self.n_assets, self.n_assets), np.nan)
        out[self.rows, self.cols] = self.covariance
        out[self.cols, self.rows] = self.covariance
        return out

    def factor(self, rank, *, weights=None, variance=None, **kwargs):
        """Fit a positive-definite factor model to the observed off-diagonal pairs.

        By default, ``weights`` uses ``self.weighted_coverage`` (decay-weighted
        overlap duration) and ``variance`` uses ``self.variance``. Pass
        ``weights`` (one entry per snapshot pair) or ``variance`` (length
        ``n_assets``, all positive and finite) to supply custom edge weights or
        marginal variances. Additional keyword arguments are forwarded to
        :func:`fit_factor`.
        """
        from .projection import fit_factor

        mask = (self.rows != self.cols) & np.isfinite(self.covariance)
        w = self.weighted_coverage if weights is None else np.asarray(weights, dtype=float)
        if w.shape != self.covariance.shape:
            raise ValueError("weights must have one entry per snapshot pair")
        return fit_factor(
            self.rows[mask], self.cols[mask], self.covariance[mask], w[mask],
            self.variance if variance is None else variance, rank, **kwargs,
        )


class OnlineCovariance:
    """Streaming pairwise covariance-rate accumulator supporting historical revisions.

    Parameters
    ----------
    n_assets : int
        Initial number of assets. Asset indices in subsequent updates may grow
        beyond ``n_assets``.
    half_life : float, optional
        Exponential decay half-life in input time units, anchored at the
        midpoint of each interval overlap.
    window : float, optional
        Sliding retention window width in input time units. Whole increments
        starting at ``start >= watermark - window`` are retained; older
        increments are expired and edits prior to the cutoff are rejected.
        If ``None``, full history is retained.
    edges : array-like of shape (m, 2), optional
        2D integer array-like of nonnegative ``(asset_a, asset_b)`` index pairs
        (or an empty sequence) restricting off-diagonal tracking to the
        selected graph. All diagonal entries are always tracked. If ``None``,
        all overlapping pairs are tracked.
    reference_drift : float or array-like, default 0.0
        Fixed drift baseline subtracted from each increment as it is formed to
        prevent catastrophic cancellation under a large trend. Pass the target
        ``drift`` to :meth:`snapshot` to evaluate at any drift vector; newly
        grown assets use a baseline of zero.
    """

    def __init__(self, n_assets: int, *, half_life=None, window=None, edges=None,
                 reference_drift=0.0):
        if not isinstance(n_assets, (int, np.integer)) or n_assets < 0:
            raise ValueError("n_assets must be a nonnegative integer")
        reference = _vector(reference_drift, int(n_assets), "reference_drift")
        self._core = Core(int(n_assets), half_life, _edge_pairs(edges), window, reference.tolist())
        self._reference = _readonly(reference)

    @property
    def n_assets(self):
        return self._core.n_assets

    @property
    def watermark(self):
        return self._core.watermark

    @property
    def size(self):
        """Retained point, active increment, and pair accumulator counts."""
        return dict(points=self._core.n_points, intervals=self._core.n_intervals,
                    pairs=self._core.n_pairs)

    def update(self, times, assets, log_prices):
        """Insert or replace (asset, timestamp) points, in supplied arrival order.

        Shapes, finite values, and the initial window cutoff are checked before
        ingestion. Events then commit separately; numerical errors or a cutoff
        advanced by preceding events stop at that event.
        """
        if np.ndim(times) == 0 and np.ndim(assets) == 0 and np.ndim(log_prices) == 0:
            if not isinstance(assets, (int, np.integer)) or not 0 <= assets < 2**31:
                raise ValueError("asset must be a nonnegative integer index below 2**31")
            self._core.update(int(assets), float(times), float(log_prices))
        else:
            t, a, p = _observations(times, assets, log_prices)
            self._core.update_many(t, a, p)
        return self

    def remove(self, asset: int, time: float):
        """Delete one retained point and join its adjacent increments."""
        if not isinstance(asset, (int, np.integer)) or asset < 0:
            raise ValueError("asset must be a nonnegative integer")
        self._core.remove(int(asset), float(time))
        return self

    def advance(self, time: float):
        """Advance the event-time watermark, decay, and expire whole increments."""
        self._core.advance(float(time))
        return self

    def snapshot(self, *, method: Method = "hy", drift=0.0, noise_variance=0.0):
        """Evaluate a :class:`MomentSnapshot` at the supplied drift and point-noise variance.

        ``method`` selects ``'hy'`` (Hayashi-Yoshida overlap exposure ratio) or
        ``'overlap'`` (duration-normalized overlap regression). ``noise_variance``
        subtracts ``2 * noise_variance[a]`` per increment on the diagonal under
        independent additive point errors.
        """
        rows, cols, stats = self._core.statistics()
        if len(self._reference) < self.n_assets:
            self._reference = _readonly(np.pad(self._reference, (0, self.n_assets - len(self._reference))))
        centered_drift = (_vector(drift, self.n_assets, "drift")
                          - self._reference)
        return _snapshot(self.n_assets, self.watermark, rows, cols, stats, method,
                         centered_drift, noise_variance)


def _snapshot(n, watermark, rows, cols, stats, method, drift, noise_variance):
    if method not in ("hy", "overlap"):
        raise ValueError("method must be 'hy' or 'overlap'")
    mu = _vector(drift, n, "drift")
    tau = _vector(noise_variance, n, "noise_variance", nonnegative=True)
    offset = 0 if method == "hy" else 6
    x = stats[:, offset:offset + 6]
    numerator = (x[:, 0] - mu[rows] * x[:, 1] - mu[cols] * x[:, 2]
                 + mu[rows] * mu[cols] * x[:, 3])
    diag = rows == cols
    numerator[diag] -= tau[rows[diag]] * x[diag, 5]
    den = x[:, 4].copy()
    covariance = np.full(len(rows), np.nan)
    np.divide(numerator, den, out=covariance, where=den > 0)
    variance = np.full(n, np.nan)
    variance[rows[diag]] = covariance[diag]
    return MomentSnapshot(
        n, method, _readonly(rows), _readonly(cols), _readonly(covariance),
        _readonly(den), _readonly(stats[:, 12].copy()), _readonly(stats[:, 13].copy()),
        _readonly(stats[:, 4].copy()), _readonly(variance), watermark,
    )


def fit(times, assets, log_prices, *, n_assets=None, method: Method = "hy",
        drift=0.0, noise_variance=0.0, half_life=None, window=None, edges=None, threads=None):
    """Estimate pairwise covariance rates from batch arrays via two-pointer interval scans.

    Unordered timestamps are stably sorted so the last value at a duplicate
    ``(asset, time)`` key takes precedence. Increments are centered at ``drift``
    before accumulating sufficient statistics. Large pair scans execute in
    parallel across a cached CPU thread pool; pass ``threads=1`` for serial
    execution.
    """
    t, a, p = _observations(times, assets, log_prices)
    inferred = int(a.max()) + 1 if len(a) else 0
    if n_assets is None:
        n_assets = inferred
    elif not isinstance(n_assets, (int, np.integer)) or n_assets < inferred:
        raise ValueError("n_assets does not cover observed indices")
    from ._native import batch_statistics

    if method not in ("hy", "overlap"):
        raise ValueError("method must be 'hy' or 'overlap'")
    if threads is not None and (not isinstance(threads, (int, np.integer)) or threads < 1):
        raise ValueError("threads must be a positive integer or None")
    if np.any(t[1:] < t[:-1]):
        order = np.argsort(t, kind="stable")
        t, a, p = t[order], a[order], p[order]
    rows, cols, stats = batch_statistics(int(n_assets), t, a, p,
                                        half_life, window, _edge_pairs(edges), threads,
                                        _vector(drift, int(n_assets), "drift"))
    return _snapshot(int(n_assets), float(t.max()) if len(t) else None,
                     rows, cols, stats, method, 0., noise_variance)


def coarsen(times, assets, log_prices, *, width, origin=0.0):
    """Retain the last observation per asset in each fixed bucket ``[origin + k*width, origin + (k+1)*width)``.

    Preserves the selected observations' original timestamps and duplicate-key
    replacement semantics. Bounds retained points per bucket, though points on
    opposite sides of a bucket boundary may remain closer than ``width``.
    """
    t, a, p = _observations(times, assets, log_prices)
    if not np.isfinite(width) or width <= 0 or not np.isfinite(origin):
        raise ValueError("width must be positive and origin finite")
    bucket = np.floor((t - origin) / width)
    if not np.all(np.isfinite(bucket)):
        raise ValueError("time buckets overflow; use relative timestamps")
    order = np.lexsort((np.arange(len(t)), t, bucket, a))
    sorted_a, sorted_bucket = a[order], bucket[order]
    keep = np.ones(len(order), dtype=bool)
    if len(order) > 1:
        keep[:-1] = ((sorted_a[:-1] != sorted_a[1:]) |
                     (sorted_bucket[:-1] != sorted_bucket[1:]))
    chosen = order[keep]
    chosen = chosen[np.argsort(t[chosen], kind="stable")]
    return t[chosen], a[chosen], p[chosen]


def estimate_noise(times, assets, log_prices, *, n_assets=None, drift=0.0):
    """Estimate per-asset point-noise variance via ``-mean(r_k * r_{k+1})`` on drift-centered returns.

    Under exogenous sampling, independent Brownian increments, drift ``drift``,
    and mutually independent point errors, adjacent increments share only their
    common endpoint error with opposite signs. Returns the unclipped signed
    sample moment (or ``NaN`` for assets with fewer than two increments).
    """
    t, a, p = _observations(times, assets, log_prices)
    n = int(a.max()) + 1 if n_assets is None and len(a) else (n_assets or 0)
    if len(a) and n <= a.max():
        raise ValueError("n_assets does not cover observed indices")
    mu = _vector(drift, n, "drift")
    result = np.full(n, np.nan)
    for asset in np.unique(a):
        idx = np.flatnonzero(a == asset)
        idx = idx[np.argsort(t[idx], kind="stable")]
        # Last duplicate wins, matching fit/update semantics.
        idx = idx[np.r_[t[idx][:-1] != t[idx][1:], True]]
        r = np.diff(p[idx]) - mu[asset] * np.diff(t[idx])
        if len(r) >= 2:
            result[asset] = -np.mean(r[:-1] * r[1:])
    return result
