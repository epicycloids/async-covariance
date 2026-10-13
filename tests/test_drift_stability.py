"""Subtract a fixed large drift before accumulating small residual moments."""

import numpy as np
import pytest

from async_covariance import OnlineCovariance, fit


def prices_with_large_drift():
    rng = np.random.default_rng(854)
    times = np.arange(1001, dtype=float)
    drift = 1e6
    prices = drift * times + np.cumsum(rng.normal(scale=1e-3, size=len(times)))
    return times, prices, drift


def residual_variance(times, prices, drift, *, now, half_life, window, noise=0.):
    durations = np.diff(times)
    returns = np.diff(prices) - drift * durations
    keep = np.ones(len(returns), dtype=bool) if window is None else times[:-1] >= now - window
    weights = np.ones(len(returns)) if half_life is None else np.exp2(
        -(now - (times[:-1] + durations / 2)) / half_life
    )
    return np.sum(weights[keep] * (returns[keep] ** 2 - 2 * noise)) / np.sum(
        weights[keep] * durations[keep]
    )


@pytest.mark.parametrize("method", ["hy", "overlap"])
@pytest.mark.parametrize("half_life", [None, 2.7])
@pytest.mark.parametrize("window", [None, 17.5])
def test_batch_and_online_preserve_small_variance_below_large_known_drift(method, half_life, window):
    times, prices, drift = prices_with_large_drift()
    assets = np.zeros(len(times), dtype=int)
    expected = residual_variance(times, prices, drift, now=times[-1],
                                 half_life=half_life, window=window, noise=1e-8)
    batch = fit(times, assets, prices, method=method, drift=drift,
                half_life=half_life, window=window, noise_variance=1e-8)
    online = OnlineCovariance(1, reference_drift=drift,
                              half_life=half_life, window=window).update(times, assets, prices)
    observed = online.snapshot(method=method, drift=drift, noise_variance=1e-8)
    np.testing.assert_allclose(batch.variance, [expected], rtol=3e-12, atol=1e-15)
    np.testing.assert_allclose(observed.variance, [expected], rtol=3e-12, atol=1e-15)


@pytest.mark.parametrize("method", ["hy", "overlap"])
def test_centered_history_remains_stable_after_corrections_deletion_and_drift_adjustment(method):
    times, prices, drift = prices_with_large_drift()
    online = OnlineCovariance(1, reference_drift=drift, half_life=3.1)
    online.update(times, np.zeros(len(times), dtype=int), prices)
    online.update(500.5, 0, (prices[500] + prices[501]) / 2 + .2)
    online.update(500., 0, prices[500] + .1)
    online.update(500., 0, prices[500])
    online.remove(0, 500.5)
    # Revising the requested drift by a small amount uses the residual moments.
    adjusted = drift + .0002
    expected = residual_variance(times, prices, adjusted, now=1003.,
                                 half_life=3.1, window=None)
    online.advance(1003.)
    observed = online.snapshot(method=method, drift=adjusted)
    np.testing.assert_allclose(observed.variance, [expected], rtol=3e-10, atol=1e-15)


def test_centering_vector_validation_and_default_zero_for_growing_assets():
    for invalid in ([np.nan, 0.], [1., np.inf], [1., 2., 3.]):
        with pytest.raises(ValueError, match="reference_drift"):
            OnlineCovariance(2, reference_drift=invalid)
    engine = OnlineCovariance(1, reference_drift=[1e6])
    engine.update([0., 1., 0., 1.], [0, 0, 2, 2], [0., 1e6 + .01, 0., 2.])
    assert engine.n_assets == 3
    result = engine.snapshot(drift=[1e6, 0., 0.])
    np.testing.assert_allclose(result.variance[[0, 2]], [.01 ** 2, 4.], rtol=2e-7)
