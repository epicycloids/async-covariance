"""Native edit/expiry parity against a complete-history all-pairs oracle."""

import numpy as np
import pytest

from async_covariance import OnlineCovariance


def raw_reference(points, n_assets, now, *, window=None, half_life=None, edges=None):
    """Rebuild adjacency from full history; do not emulate native pruning."""
    increments = [[] for _ in range(n_assets)]
    for asset in range(n_assets):
        ordered = sorted((t, p) for (a, t), p in points.items() if a == asset)
        for (start, first), (end, last) in zip(ordered, ordered[1:]):
            if window is None or start >= now - window:
                increments[asset].append((start, end, last - first))

    rows = []
    for a in range(n_assets):
        for b in range(a, n_assets):
            if a != b and edges is not None and (a, b) not in edges:
                continue
            pairs = ((x, x) for x in increments[a]) if a == b else (
                (x, y) for x in increments[a] for y in increments[b]
            )
            total = np.zeros(14)
            for x, y in pairs:
                start, end = max(x[0], y[0]), min(x[1], y[1])
                overlap = end - start
                if overlap <= 0:
                    continue
                dx, dy = x[1] - x[0], y[1] - y[0]
                weight = 1. if half_life is None else 2. ** (
                    -(now - (start + overlap / 2)) / half_life
                )
                scale = overlap / (dx * dy)
                raw = np.array([
                    x[2] * y[2], dx * y[2], dy * x[2], dx * dy,
                    overlap, 2. if a == b else 0.,
                    scale * x[2] * y[2], scale * dx * y[2],
                    scale * dy * x[2], scale * dx * dy,
                    scale * overlap, scale * 2. if a == b else 0.,
                    1., overlap,
                ])
                raw[:12] *= weight
                total += raw
            if total[12]:
                rows.append((a, b, total))
    return rows


def assert_raw_parity(engine, points, *, window, half_life, edges):
    expected = raw_reference(points, engine.n_assets, engine.watermark,
                             window=window, half_life=half_life, edges=edges)
    a, b, actual = engine._core.statistics()
    np.testing.assert_array_equal(a, [row[0] for row in expected])
    np.testing.assert_array_equal(b, [row[1] for row in expected])
    values = np.asarray([row[2] for row in expected]).reshape(-1, 14)
    np.testing.assert_allclose(actual, values, rtol=2e-10, atol=2e-10)
    # Counts are exact integer bookkeeping; neither decay nor snapshots change them.
    np.testing.assert_array_equal(actual[:, 12], values[:, 12])
    if window is None:
        assert engine.size["points"] == len(points)
    else:
        cutoff = engine.watermark - window
        retained = sum(
            sum(t >= cutoff for (a, t) in points if a == asset)
            + any(t < cutoff for (a, t) in points if a == asset)
            for asset in range(engine.n_assets)
        )
        assert engine.size["points"] == retained


@pytest.mark.parametrize("window", [None, 4.75])
@pytest.mark.parametrize("half_life", [None, 1.3])
@pytest.mark.parametrize("sparse", [False, True])
def test_random_edits_windows_and_decay_match_complete_history(window, half_life, sparse):
    rng = np.random.default_rng(2461)
    edges = {(0, 1), (0, 3), (1, 4), (2, 5)} if sparse else None
    engine = OnlineCovariance(6 if sparse else 0, window=window,
                              half_life=half_life, edges=None if edges is None else list(edges))
    points = {}
    for asset in range(6):
        for t in [-1., -.5, 0.]:
            p = float(rng.normal())
            engine.update(t, asset, p)
            points[asset, t] = p

    for step in range(350):
        now = engine.watermark
        cutoff = now - window if window is not None else -8.
        available = [key for key in points if key[1] >= cutoff]
        action = rng.random()
        if action < .14:
            engine.advance(now + float(rng.integers(1, 7)) / 4)
        elif action < .32 and available:
            asset, t = available[int(rng.integers(len(available)))]
            engine.remove(asset, t)
            del points[asset, t]
        elif action < .59 and available:
            asset, t = available[int(rng.integers(len(available)))]
            p = float(rng.normal() * (25. if step % 11 == 0 else 1.))
            engine.update(t, asset, p)
            points[asset, t] = p
        else:
            lo = int(np.ceil(cutoff * 4))
            hi = int(np.floor((now + 1.5) * 4))
            t = float(rng.integers(lo, hi + 1)) / 4
            asset, p = int(rng.integers(6)), float(rng.normal())
            engine.update(t, asset, p)
            points[asset, t] = p
        if step % 7 == 0:
            assert_raw_parity(engine, points, window=window,
                              half_life=half_life, edges=edges)
    assert_raw_parity(engine, points, window=window, half_life=half_life, edges=edges)


def test_cutoff_boundary_crossing_and_sentinel_never_form_a_bridge():
    engine = OnlineCovariance(2, window=2., half_life=.8)
    points = {(0, 0.): 0., (0, 1.): .4, (0, 3.): 1.,
              (1, .5): .2, (1, 1.): .3, (1, 3.): .7}
    for (asset, t), p in sorted(points.items(), key=lambda item: item[0][1]):
        engine.update(t, asset, p)
    assert_raw_parity(engine, points, window=2., half_life=.8, edges=None)
    # Start==cutoff stays active; advancing one representable time step expires it.
    assert engine.size["intervals"] == 2
    engine.advance(np.nextafter(3., np.inf))
    assert engine.size["intervals"] == 0
    assert engine.size["points"] == 4
    assert_raw_parity(engine, points, window=2., half_life=.8, edges=None)
    # The first retained point can be removed, leaving only an old sentinel.
    engine.remove(0, 3.)
    del points[0, 3.]
    assert_raw_parity(engine, points, window=2., half_life=.8, edges=None)
    # This interval would start before the cutoff and must stay absent.
    engine.update(3.25, 0, 1.1)
    points[0, 3.25] = 1.1
    assert engine.size["intervals"] == 0
    assert_raw_parity(engine, points, window=2., half_life=.8, edges=None)
    # A new in-window predecessor makes one interval eligible.
    engine.update(2., 0, .8)
    points[0, 2.] = .8
    assert engine.size["intervals"] == 1
    assert_raw_parity(engine, points, window=2., half_life=.8, edges=None)


def test_signed_zero_is_one_key_and_all_underflowed_support_remains_counted():
    engine = OnlineCovariance(2, half_life=.01)
    engine.update(-0., 0, 1.).update(0., 0, 2.)
    engine.update(-0., 1, 0.).update(1., 0, 3.).update(1., 1, 2.)
    assert engine.size["points"] == 4
    before = engine.snapshot()
    np.testing.assert_array_equal(before.counts, [1., 1., 1.])
    engine.advance(100.)
    after = engine.snapshot()
    np.testing.assert_array_equal(after.counts, before.counts)
    np.testing.assert_array_equal(after.coverage, before.coverage)
    np.testing.assert_array_equal(after.denominator, np.zeros(3))
    assert np.isnan(after.covariance).all()
    engine.remove(0, -0.)
    assert engine.size == {"points": 3, "intervals": 1, "pairs": 1}


@pytest.mark.parametrize("window", [None, 2.5])
def test_large_exact_time_translation_preserves_decay_and_window(window):
    times = np.array([-1., -.5, .25, 1., 2., -.75, 0., 1.5, 2.])
    assets = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1])
    prices = np.array([0., -.2, .1, .3, .7, .2, .3, -.4, .8])
    order = np.argsort(times, kind="stable")
    engines = []
    for shift in [0., 2. ** 40]:
        engine = OnlineCovariance(2, window=window, half_life=1.75)
        engine.update(times[order] + shift, assets[order], prices[order])
        engine.update(.5 + shift, 1, -.3)
        engine.remove(0, .25 + shift)
        engine.advance(3. + shift)
        engines.append(engine)
    for method in ["hy", "overlap"]:
        left, right = [e.snapshot(method=method, drift=[.2, -.1],
                                  noise_variance=[.01, .02]) for e in engines]
        np.testing.assert_allclose(left.covariance, right.covariance, atol=2e-14)
        np.testing.assert_allclose(left.denominator, right.denominator, atol=2e-14)
        np.testing.assert_array_equal(left.counts, right.counts)
        np.testing.assert_array_equal(left.coverage, right.coverage)


def test_failed_future_event_does_not_advance_or_expire_window():
    engine = OnlineCovariance(1, window=2.).update([0., 1.], [0, 0], [0., 1.])
    before = engine._core.statistics()
    size = engine.size
    with pytest.raises(ValueError, match="floating point range"):
        engine.update(3., 0, 1e200)
    assert engine.watermark == 1.
    assert engine.size == size
    for actual, expected in zip(engine._core.statistics(), before):
        np.testing.assert_array_equal(actual, expected)
