"""Reference checks for numerical projection and covariance operators."""

import numpy as np
import pytest
from scipy.optimize import brentq

from async_covariance.model import FactorCovariance
from async_covariance.projection import _quadratic_ball, fit_factor, spectral_factor


def test_model_operators_and_immutability():
    rng = np.random.default_rng(1)
    factors = rng.normal(size=(8, 3))
    specific = rng.uniform(0.2, 2.0, 8)
    model = FactorCovariance(factors, specific, labels=range(8))
    dense = factors @ factors.T + np.diag(specific)
    vector = rng.normal(size=8)
    matrix = rng.normal(size=(8, 4))
    np.testing.assert_allclose(model.variance(), np.diag(dense))
    np.testing.assert_allclose(model.covariance(), dense)
    for rhs in (vector, matrix):
        np.testing.assert_allclose(model.matvec(rhs), dense @ rhs, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(model.solve(rhs), np.linalg.solve(dense, rhs), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(model.matvec(model.solve(rhs)), rhs, rtol=1e-12, atol=1e-12)
    factors[:] = 0.0
    specific[:] = 3.0
    np.testing.assert_allclose(model.covariance(), dense)
    with pytest.raises(ValueError):
        model.loadings[0, 0] = 0.0
    with pytest.raises(ValueError):
        model.specific.setflags(write=True)
    assert model.labels == tuple(range(8))
    drift = rng.normal(size=8)
    np.testing.assert_allclose(model.second_moment(2.0, drift), 2 * dense + 4 * np.outer(drift, drift))
    np.testing.assert_array_equal(model.second_moment(0.0, drift), np.zeros((8, 8)))


def test_rank_zero_operators():
    model = FactorCovariance(np.empty((3, 0)), np.array([1.0, 2.0, 4.0]))
    rhs = np.ones((3, 2))
    np.testing.assert_array_equal(model.solve(rhs), rhs / model.specific[:, None])
    np.testing.assert_array_equal(model.matvec(rhs), rhs * model.specific[:, None])


@pytest.mark.parametrize("loadings,specific", [
    ([1.0], [1.0]),
    ([[1.0]], [0.0]),
    ([[np.inf]], [1.0]),
    ([[1.0]], [np.nan]),
    ([[1.0]], [1.0, 2.0]),
])
def test_model_rejects_invalid_parameters(loadings, specific):
    with pytest.raises(ValueError):
        FactorCovariance(loadings, specific)


def test_model_argument_contracts():
    with pytest.raises(ValueError):
        FactorCovariance(np.zeros((2, 1)), np.ones(2), labels=["a", "a"])
    model = FactorCovariance(np.zeros((2, 1)), np.ones(2))
    for bad in ([1.0], np.ones((2, 1, 1)), [np.inf, 0.0]):
        with pytest.raises(ValueError):
            model.solve(bad)
        with pytest.raises(ValueError):
            model.matvec(bad)
    for horizon in (-1.0, np.inf, np.nan):
        with pytest.raises(ValueError):
            model.second_moment(horizon)
    with pytest.raises(ValueError):
        model.second_moment(1.0, [1.0])


def test_quadratic_row_update_interior_boundary_and_singular():
    gram = np.array([[2.0, 0.5], [0.5, 1.0]])
    linear = np.array([0.2, 0.1])
    np.testing.assert_allclose(_quadratic_ball(gram, linear, 1.0), np.linalg.solve(gram, linear), atol=1e-14)
    gram = np.diag([1.0, 4.0])
    linear = np.array([3.0, 1.0])
    radius = 0.7
    multiplier = brentq(lambda value: np.linalg.norm(linear / (np.diag(gram) + value)) - radius, 0.0, 20.0)
    result = _quadratic_ball(gram, linear, radius)
    np.testing.assert_allclose(result, linear / (np.diag(gram) + multiplier), rtol=2e-14, atol=2e-14)
    assert np.linalg.norm(result) <= radius + 1e-15
    np.testing.assert_allclose(_quadratic_ball(np.diag([1.0, 0.0]), [0.2, 0.0], 1.0), [0.2, 0.0], atol=1e-14)
    np.testing.assert_allclose(_quadratic_ball(np.zeros((2, 2)), [1.0, 2.0], 0.7), 0.7 * np.array([1.0, 2.0]) / np.sqrt(5), atol=1e-14)


def test_dense_factor_recovery_monotonicity_and_warm_start():
    rng = np.random.default_rng(71)
    n, rank = 12, 2
    standardized = rng.normal(size=(n, rank)) * 0.25
    variance = rng.uniform(0.1, 4.0, n)
    true = standardized * np.sqrt(variance)[:, None]
    rows, cols = np.triu_indices(n, 1)
    covariance = np.einsum("ij,ij->i", true[rows], true[cols])
    weights = rng.uniform(0.1, 2.0, len(rows))
    result = fit_factor(rows, cols, covariance, weights, variance, rank,
                        initial=true + rng.normal(size=true.shape) * 0.01,
                        max_iter=400, tol=1e-13)
    assert result.objective < 1e-9
    assert np.all(np.diff(result.objective_history) <= 0.0)
    assert np.all(result.model.specific >= variance * 0.05 * (1 - 1e-13))
    np.testing.assert_allclose(result.model.variance(), variance, rtol=1e-13, atol=1e-13)
    assert np.linalg.eigvalsh(result.model.covariance())[0] > 0.0
    warm = fit_factor(rows, cols, covariance, weights, variance, rank, initial=result.model, max_iter=2)
    assert warm.objective <= result.objective + 1e-18


def test_missing_edge_is_different_from_observed_zero():
    initial = np.full((3, 1), 0.5)
    args = ([0, 1, 0], [1, 2, 2], [0.25, 0.25, np.nan], [1.0, 1.0, 0.0], np.ones(3), 1)
    missing = fit_factor(*args, initial=initial, tol=1e-13)
    observed = fit_factor([0, 1, 0], [1, 2, 2], [0.25, 0.25, 0.0], [1.0] * 3,
                          np.ones(3), 1, initial=initial, tol=1e-13, max_iter=200)
    assert missing.objective < 1e-25
    assert missing.diagnostics["n_observed_pairs"] == 2
    assert missing.model.covariance()[0, 2] == pytest.approx(0.25)
    assert observed.model.covariance()[0, 2] < 0.25
    assert observed.diagnostics["n_observed_pairs"] == 3
    assert missing.diagnostics["n_bipartite_components"] == 1
    assert missing.diagnostics["completion_identifiability"] == "not_established"


def test_disconnected_graph_and_isolated_asset_diagnostics():
    result = fit_factor([0, 2], [1, 3], [0.2, -0.3], [1.0, 1.0], np.ones(5), 2,
                        initialization="random", seed=5, max_iter=50)
    assert result.diagnostics["n_graph_components"] == 3
    assert result.diagnostics["n_isolated_assets"] == 1
    assert result.diagnostics["n_assets_with_degree_below_rank"] == 5
    assert result.diagnostics["identifiability_issues"]
    np.testing.assert_array_equal(result.model.loadings[4], np.zeros(2))


@pytest.mark.parametrize("eigensolver", ["eigh", "eigsh"])
@pytest.mark.parametrize("backend", ["native", "python"])
def test_spectral_initialization_seeds_omitted_components(eigensolver, backend):
    result = fit_factor([0, 2, 4], [1, 3, 5], [0.8, 0.2, -0.4], np.ones(3),
                        np.ones(6), 1, eigensolver=eigensolver, backend=backend,
                        initialization="spectral", max_iter=20, tol=1e-13)
    assert result.objective < 1e-25
    assert result.diagnostics["n_seeded_spectral_rows"] == 4
    assert result.diagnostics["n_graph_components"] == 3
    assert result.diagnostics["completion_identifiability"] == "not_established"


def test_zero_rank_empty_edges_and_specific_floor_one():
    for rank, floor in ((0, 0.05), (1, 1.0)):
        result = fit_factor([], [], [], [], [1.0, 2.0], rank, specific_floor=floor)
        np.testing.assert_array_equal(result.model.covariance(), np.diag([1.0, 2.0]))
        assert result.objective == 0.0
        assert result.iterations == 0


def test_reproducible_sparse_spectral_initialization():
    rng = np.random.default_rng(72)
    rows, cols = np.triu_indices(20, 1)
    covariance = rng.normal(size=len(rows)) * 0.1
    first = fit_factor(rows, cols, covariance, np.ones(len(rows)), np.ones(20), 3,
                       seed=7, max_iter=2, initialization="spectral")
    second = fit_factor(rows, cols, covariance, np.ones(len(rows)), np.ones(20), 3,
                        seed=7, max_iter=2, initialization="spectral")
    np.testing.assert_array_equal(first.model.loadings, second.model.loadings)
    assert first.diagnostics["initial_eigen_residual"] < 1e-7


@pytest.mark.parametrize("updates,match", [
    ({"rows": [0.0]}, "integer"),
    ({"cols": [2]}, "indices"),
    ({"cols": [0]}, "diagonal"),
    ({"weights": [-1.0]}, "nonnegative"),
    ({"weights": [np.inf]}, "finite"),
    ({"covariance": [np.nan]}, "finite"),
    ({"variance": [0.0, 1.0]}, "positive"),
    ({"variance": [np.nan, 1.0]}, "finite"),
    ({"rank": 2}, "less than"),
    ({"rank": True}, "integer"),
    ({"max_iter": -1}, "at least"),
    ({"tol": np.nan}, "tol"),
    ({"specific_floor": 0.0}, "specific_floor"),
    ({"initialization": "unknown"}, "initialization"),
    ({"eigensolver": "unknown"}, "eigensolver"),
    ({"backend": "unknown"}, "backend"),
    ({"initial": np.ones((2, 2))}, "initial"),
    ({"covariance": [1e308]}, "floating-point range"),
])
def test_graph_input_contract(updates, match):
    arguments = dict(rows=[0], cols=[1], covariance=[0.2], weights=[1.0], variance=[1.0, 1.0], rank=1)
    arguments.update(updates)
    with pytest.raises((ValueError, TypeError), match=match):
        fit_factor(**arguments)


def test_duplicate_undirected_edges_are_rejected():
    with pytest.raises(ValueError, match="once"):
        fit_factor([0, 1], [1, 0], [0.1, 0.1], [1.0, 1.0], [1.0, 1.0], 1)


def test_positive_support_underflow_is_rejected():
    with pytest.raises(ValueError, match="dynamic range"):
        fit_factor([0, 1], [1, 2], [0.1, 0.1], [1e308, 1e-308], np.ones(3), 1)


def test_dense_spectral_methods_and_identity():
    rng = np.random.default_rng(4)
    factors = rng.normal(size=(30, 2)) * 0.3
    dense = factors @ factors.T + np.diag(rng.uniform(0.5, 1.0, 30))
    direct = spectral_factor(dense, 2, eigensolver="eigh")
    iterative = spectral_factor(dense, 2, eigensolver="eigsh", seed=3)
    np.testing.assert_allclose(direct.covariance(), iterative.covariance(), atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(direct.variance(), np.diag(dense), atol=1e-13, rtol=1e-13)
    np.testing.assert_array_equal(spectral_factor(np.eye(3), 1).covariance(), np.eye(3))
    with pytest.raises(ValueError, match="symmetric"):
        spectral_factor([[1.0, 0.2], [0.0, 1.0]], 1)


@pytest.mark.parametrize("rank", [1, 2, 4, 8])
@pytest.mark.parametrize("boundary", [False, True])
def test_native_matches_independent_python_reference(rank, boundary):
    rng = np.random.default_rng(941 + rank)
    n = 24
    rows, cols = np.triu_indices(n, 1)
    standardized = rng.normal(size=(n, rank)) * (0.2 / np.sqrt(rank))
    variance = rng.uniform(0.2, 3.0, n)
    targets = np.einsum("ij,ij->i", standardized[rows], standardized[cols])
    if boundary:
        targets = rng.uniform(-1.3, 1.3, len(rows))
    covariance = targets * np.sqrt(variance[rows] * variance[cols])
    weights = rng.uniform(0.1, 1.0, len(rows))
    initial = rng.normal(size=(n, rank)) * np.sqrt(variance)[:, None] * 0.1
    arguments = (rows, cols, covariance, weights, variance, rank)
    options = dict(initial=initial, max_iter=15, tol=0.0, specific_floor=0.2)
    native = fit_factor(*arguments, backend="native", **options)
    reference = fit_factor(*arguments, backend="python", **options)
    np.testing.assert_allclose(native.objective_history, reference.objective_history, rtol=2e-10, atol=2e-13)
    np.testing.assert_allclose(native.model.covariance(), reference.model.covariance(), rtol=2e-8, atol=2e-10)
    for result in (native, reference):
        assert np.all(np.diff(result.objective_history) <= 0.0)
        assert np.all(result.model.specific >= variance * 0.2 * (1.0 - 1e-13))
        np.testing.assert_allclose(result.model.variance(), variance, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("backend", ["native", "python"])
def test_singular_sparse_graph_and_extreme_targets(backend):
    # A degree-one graph makes the rank-three row Gram matrices singular.
    result = fit_factor([0, 2], [1, 3], [0.6, -0.8], [0.2, 1.0], np.ones(5), 3,
                        backend=backend, initialization="random", seed=19,
                        specific_floor=0.1, max_iter=40, tol=1e-13)
    assert result.objective < 1e-20
    assert np.all(np.diff(result.objective_history) <= 0.0)
    np.testing.assert_array_equal(result.model.loadings[4], np.zeros(3))
    # Normalizing before loss summation keeps the mean finite even when an
    # unnormalized sum of squared targets would overflow float64.
    n = 20
    rows, cols = np.triu_indices(n, 1)
    huge = np.sqrt(np.finfo(float).max) / 8.0
    result = fit_factor(rows, cols, np.full(len(rows), huge), np.ones(len(rows)),
                        np.ones(n), 1, backend=backend, initialization="random",
                        max_iter=2)
    assert np.isfinite(result.objective)
    np.testing.assert_allclose(result.objective, huge * huge, rtol=1e-13)


def test_native_rejects_malformed_low_level_arrays():
    from async_covariance._native import _factor_optimize

    good = [np.array([0], dtype=np.int64), np.array([1], dtype=np.int64),
            np.array([0.2]), np.array([1.0]), np.array([0.1, 0.1]),
            2, 1, 0.9, 3, 1e-8]
    for index, bad in ((1, np.array([], dtype=np.int64)),
                       (0, np.array([-1], dtype=np.int64)),
                       (1, np.array([0], dtype=np.int64)),
                       (2, np.array([np.nan])),
                       (3, np.array([0.0])),
                       (4, np.array([0.1])), (7, np.nan), (9, -1.0)):
        arguments = good.copy()
        arguments[index] = bad
        with pytest.raises(ValueError):
            _factor_optimize(*arguments)
