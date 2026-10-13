"""Independent all-pairs oracle, deliberately without an interval index."""
import numpy as np
import pytest

from async_covariance import OnlineCovariance, coarsen, estimate_noise, fit


def reference(points, n, method, *, drift=0., noise=0., half_life=None, now=None,
              window=None, edges=None):
    mu = np.broadcast_to(drift, (n,))
    tau = np.broadcast_to(noise, (n,))
    intervals = [[] for _ in range(n)]
    if now is None:
        now = max((t for (_, t) in points), default=0.)
    for a in range(n):
        seq = sorted((t, p) for (asset, t), p in points.items() if asset == a)
        for (s, p0), (e, p1) in zip(seq, seq[1:]):
            if window is None or s >= now - window:
                intervals[a].append((s, e, p1 - p0 - mu[a] * (e - s)))
    cov = np.full((n, n), np.nan)
    coverage = np.zeros((n, n))
    for a in range(n):
        for b in range(a, n):
            if a != b and edges is not None and (a, b) not in edges:
                continue
            num, den = 0., 0.
            pairs = ((x, x) for x in intervals[a]) if a == b else (
                (x, y) for x in intervals[a] for y in intervals[b])
            for x, y in pairs:
                s, e = max(x[0], y[0]), min(x[1], y[1])
                o = e - s
                if o <= 0:
                    continue
                w = 1. if half_life is None else 2. ** (-(now - (s + e) / 2) / half_life)
                coeff = w if method == "hy" else w * o / ((x[1] - x[0]) * (y[1] - y[0]))
                product = x[2] * y[2] - (2 * tau[a] if a == b else 0)
                num += coeff * product
                den += coeff * o
                coverage[a, b] += o
            if den > 0:
                cov[a, b] = num / den
            cov[b, a] = cov[a, b]
            coverage[b, a] = coverage[a, b]
    return cov, coverage


def fixture(seed=32, n=5, m=17):
    rng = np.random.default_rng(seed)
    times = np.concatenate([np.cumsum(rng.uniform(.05, 1.5, m)) for _ in range(n)])
    assets = np.repeat(np.arange(n), m)
    prices = rng.normal(size=n * m)
    return times, assets, prices


@pytest.mark.parametrize("method", ["hy", "overlap"])
@pytest.mark.parametrize("half_life", [None, 2.5])
@pytest.mark.parametrize("sparse", [False, True])
def test_batch_against_independent_oracle(method, half_life, sparse):
    t, a, p = fixture()
    edges = [(0, 1), (1, 3), (2, 4)] if sparse else None
    drift = np.linspace(-.4, .7, 5)
    noise = np.linspace(0, .15, 5)
    snapshot = fit(t, a, p, method=method, half_life=half_life, edges=edges,
                   drift=drift, noise_variance=noise)
    points = {(int(a), float(t)): float(p) for t, a, p in zip(t, a, p)}
    expected, coverage = reference(points, 5, method, drift=drift, noise=noise,
                                   half_life=half_life, edges=edges)
    np.testing.assert_allclose(snapshot.dense(), expected, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(snapshot.coverage, coverage[snapshot.rows, snapshot.cols])


@pytest.mark.parametrize("method", ["hy", "overlap"])
def test_backfills_replacements_deletions_and_decay(method):
    t, a, p = fixture(m=11)
    rng = np.random.default_rng(847)
    order = rng.permutation(len(t))
    engine = OnlineCovariance(5, half_life=3.1)
    points = {}
    for i in order:
        engine.update(t[i], a[i], p[i])
        points[int(a[i]), float(t[i])] = float(p[i])
    engine.advance(float(t.max() + 4))
    for i in order[:13]:
        engine.update(t[i], a[i], p[i] + .75)
        points[int(a[i]), float(t[i])] = float(p[i] + .75)
    for i in order[13:28]:
        engine.remove(int(a[i]), float(t[i]))
        del points[int(a[i]), float(t[i])]
    for drift in [np.zeros(5), np.arange(5) / 10]:
        expected, _ = reference(points, 5, method, drift=drift, half_life=3.1,
                                now=engine.watermark)
        np.testing.assert_allclose(engine.snapshot(method=method, drift=drift).dense(),
                                   expected, rtol=3e-11, atol=3e-11)


def test_chained_nested_touching_gaps_and_growing_universe():
    t = [0., 2., 4., 1., 3., 5., -1., 6., 10., 12.]
    a = [0, 0, 0, 1, 1, 1, 2, 2, 0, 0]
    p = np.arange(10) ** 2 / 10
    engine = OnlineCovariance(0).update(t, a, p)
    assert engine.n_assets == 3
    points = {(a, t): p for t, a, p in zip(t, a, p)}
    for method in ["hy", "overlap"]:
        expected, _ = reference(points, 3, method)
        np.testing.assert_allclose(engine.snapshot(method=method).dense(), expected)
    # Touching intervals from distinct assets do not overlap.
    touching = fit([0, 1, 1, 2], [0, 0, 1, 1], [0, 1, 0, 1])
    assert np.isnan(touching.dense()[0, 1])


@pytest.mark.parametrize("method", ["hy", "overlap"])
def test_window_whole_intervals_and_bounded_points(method):
    t, a, p = fixture(n=3, m=60)
    engine = OnlineCovariance(3, half_life=8, window=7)
    order = np.argsort(t, kind="stable")
    engine.update(t[order], a[order], p[order])
    engine.advance(float(t.max() + 1))
    points = {(int(a), float(t)): float(p) for t, a, p in zip(t, a, p)}
    expected, _ = reference(points, 3, method, half_life=8, window=7, now=engine.watermark)
    np.testing.assert_allclose(engine.snapshot(method=method).dense(), expected,
                               rtol=5e-11, atol=5e-11)
    assert engine.size["points"] < 50
    with pytest.raises(ValueError, match="window"):
        engine.update(0., 0, 0.)
    with pytest.raises(ValueError, match="window"):
        engine.remove(0, 0.)
    engine.advance(engine.watermark + 100)
    assert engine.size["intervals"] == 0
    assert engine.size["points"] <= 3
    assert np.isnan(engine.snapshot().dense()).all()


def test_drift_noise_and_negative_raw_variance_are_explicit():
    t = [0, 1, 2, 3]
    p = [0, .5, 1., 1.5]
    snapshot = fit(t, [0] * 4, p, drift=.5)
    np.testing.assert_allclose(snapshot.variance, [0], atol=1e-14)
    assert fit(t, [0] * 4, p, drift=.5, noise_variance=1).variance[0] == -2
    with pytest.raises(ValueError):
        fit(t, [0] * 4, p, noise_variance=-1)


def test_invalid_arrays_do_not_partially_mutate():
    engine = OnlineCovariance(1).update([0., 1.], [0, 0], [0., 1.])
    before = engine.snapshot().dense()
    for args in [([2., np.nan], [0, 0], [2., 3.]), ([2, 3], [0., .5], [2, 3]),
                 ([2, 3], [0, -1], [2, 3]), ([[2]], [0], [2])]:
        with pytest.raises(ValueError):
            engine.update(*args)
        np.testing.assert_equal(engine.snapshot().dense(), before)
        assert engine.size["points"] == 2


def test_duplicate_keys_last_value_wins_and_empty_graph():
    snapshot = fit([0, 1, 1, 0, 1], [0, 0, 0, 1, 1], [0, 1, 2, 0, 3], edges=[])
    np.testing.assert_equal(snapshot.variance, [4, 9])
    assert np.isnan(snapshot.dense()[0, 1])
    assert snapshot.covariance.flags.writeable is False


def test_repeated_corrections_do_not_accumulate_drift():
    engine = OnlineCovariance(2).update([0, 1, 2, 0, 1, 2], [0, 0, 0, 1, 1, 1],
                                      [0, 1, 2, 0, .3, .8])
    expected = engine.snapshot().dense()
    for _ in range(500):
        engine.update(1., 0, 1e6)
        engine.update(1., 0, 1.)
    np.testing.assert_allclose(engine.snapshot().dense(), expected, atol=2e-10)


def test_coarsening_actual_timestamps_and_noise_moment():
    t, a, p = coarsen([1.2, .1, .9, 1.1, .2, .8, .9], [0, 0, 0, 0, 1, 1, 0],
                      [12, 1, 9, 11, 2, 8, 99], width=1)
    np.testing.assert_equal(t, [.8, .9, 1.2])
    np.testing.assert_equal(a, [1, 0, 0])
    np.testing.assert_equal(p, [8, 99, 12])
    rng = np.random.default_rng(33)
    noise = rng.normal(size=100_000)
    tau = estimate_noise(np.arange(len(noise)), np.zeros(len(noise), dtype=int), noise)
    assert abs(tau[0] - 1) < .03


def test_no_data_and_clock_without_data():
    engine = OnlineCovariance(3)
    assert engine.watermark is None
    engine.advance(-4.)
    assert engine.watermark == -4.
    assert np.isnan(engine.snapshot().variance).all()
    with pytest.raises(ValueError):
        engine.advance(-5.)
    assert fit([], [], [], n_assets=0).dense().shape == (0, 0)


def test_parallel_batch_preserves_pair_accumulation_order():
    t, a, p = fixture(n=64, m=40)
    for options in [{}, {"half_life": 5, "window": 8}]:
        serial = fit(t, a, p, threads=1, **options)
        for threads in [2, None]:
            parallel = fit(t, a, p, threads=threads, **options)
            np.testing.assert_equal(parallel.rows, serial.rows)
            np.testing.assert_equal(parallel.cols, serial.cols)
            np.testing.assert_equal(parallel.covariance, serial.covariance)
    for threads in [0, -1, 1.5]:
        with pytest.raises(ValueError, match="threads"):
            fit(t, a, p, threads=threads)
