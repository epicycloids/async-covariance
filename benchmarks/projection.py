"""CPU projection comparisons; run with single-thread BLAS for a baseline."""

import argparse
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
import scipy

from async_covariance import fit_factor, spectral_factor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--assets", type=int, default=512)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--sweeps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=8271)
    args = parser.parse_args()
    if not 8 <= args.assets <= 2048:
        raise SystemExit("use between 8 and 2048 assets for this bounded benchmark")
    if not 1 <= args.rank < args.assets:
        raise SystemExit("rank must lie in [1, assets)")
    rng = np.random.default_rng(args.seed)
    n, rank = args.assets, args.rank
    loadings = rng.normal(size=(n, rank)) * (0.4 / np.sqrt(rank))
    row_norms = np.linalg.norm(loadings, axis=1)
    loadings *= np.minimum(1.0, 0.8 / np.maximum(row_norms, 1e-30))[:, None]
    variance = rng.uniform(0.2, 4.0, n)
    actual = loadings * np.sqrt(variance)[:, None]
    dense = actual @ actual.T
    dense[np.diag_indices(n)] = variance
    records = []
    for method in ("eigh", "eigsh"):
        start = time.perf_counter()
        model = spectral_factor(dense, rank, eigensolver=method, seed=args.seed)
        elapsed = time.perf_counter() - start
        residual = model.covariance() - dense
        np.fill_diagonal(residual, 0.0)
        records.append({
            "method": f"dense_spectral_{method}",
            "seconds": elapsed,
            "offdiagonal_rmse": float(np.linalg.norm(residual) / np.sqrt(n * (n - 1))),
            "minimum_specific_fraction": float(np.min(model.specific / variance)),
        })
    # A connected sparse graph containing triangles, plus independently sampled
    # edges. Targets have controlled noise; off-graph entries receive no weight.
    pairs = {(min(i, (i + offset) % n), max(i, (i + offset) % n))
             for i in range(n) for offset in (1, 2)}
    for _ in range(8 * n):
        a, b = rng.integers(0, n, 2)
        if a != b:
            pairs.add((min(a, b), max(a, b)))
    pairs = np.asarray(sorted(pairs), dtype=np.int64)
    rows, cols = pairs.T
    weights = rng.uniform(0.1, 1.0, len(rows))
    covariance = dense[rows, cols] + rng.normal(size=len(rows)) * 0.01 * np.sqrt(variance[rows] * variance[cols])
    for initialization, backend in ((initialization, backend)
                                    for initialization in ("spectral", "random")
                                    for backend in ("native", "python")):
        start = time.perf_counter()
        result = fit_factor(rows, cols, covariance, weights, variance, rank,
                            max_iter=args.sweeps, tol=1e-10, seed=args.seed,
                            initialization=initialization, eigensolver="eigsh", backend=backend)
        elapsed = time.perf_counter() - start
        if np.any(np.diff(result.objective_history) > 0.0):
            raise RuntimeError("observed-edge objective increased")
        probe = rng.normal(size=n)
        solve_residual = np.linalg.norm(result.model.matvec(result.model.solve(probe)) - probe) / np.linalg.norm(probe)
        records.append({
            "method": f"sparse_als_{initialization}_{backend}",
            "seconds": elapsed,
            "pairs": len(rows),
            "initial_objective": float(result.objective_history[0]),
            "final_objective": result.objective,
            "objective_history": result.objective_history.tolist(),
            "monotone": True,
            "sweeps": result.iterations,
            "optimization_converged": result.converged,
            "solve_relative_residual": float(solve_residual),
            "minimum_specific_fraction": float(np.min(result.model.specific / variance)),
            "diagnostics": dict(result.diagnostics),
        })
    result = {
        "assets": n, "rank": rank, "seed": args.seed,
        "python": platform.python_version(),
        "numpy": np.__version__, "scipy": scipy.__version__,
        "blas_thread_environment": {key: os.environ.get(key) for key in
                                    ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")},
        "dense_input_bytes": dense.nbytes,
        "records": records,
        "peak_rss_mib": float(next(line.split()[1] for line in Path("/proc/self/status").read_text().splitlines() if line.startswith("VmHWM:"))) / 1024,
        "scope": "single-run component timings; dense spectral and sparse weighted ALS solve different projection problems",
    }
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
