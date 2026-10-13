"""End-to-end native ingestion, historical revision, and snapshot timings.

Run: OPENBLAS_NUM_THREADS=1 python benchmarks/engine.py --output benchmarks/local/engine.json
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import resource
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
from async_covariance import OnlineCovariance, __version__, fit


def fixture(n, m, seed=17):
    rng = np.random.default_rng(seed)
    times = rng.uniform(.02, .98, size=(m, n)) + np.arange(m)[:, None]
    prices = rng.normal(scale=.01, size=(m, n)).cumsum(axis=0)
    assets = np.broadcast_to(np.arange(n), (m, n))
    order = np.argsort(times.ravel(), kind="stable")
    return times.ravel()[order], assets.ravel()[order], prices.ravel()[order]


def pair_graph(n, degree=8):
    return sorted({tuple(sorted((i, (i + j) % n))) for i in range(n)
                   for j in range(1, degree // 2 + 1)})


def benchmark(n, m, edges, repeats, *, window=None, half_life=None):
    t, a, p = fixture(n, m)
    build, batch, batch_serial, revisions, scalar, snapshot = [], [], [], [], [], []
    eligible = np.flatnonzero(t >= t.max() - window) if window is not None else np.arange(len(t))
    revise = eligible[np.linspace(1, len(eligible) - 2, min(512, len(eligible) // 4), dtype=int)]
    for _ in range(repeats):
        start = perf_counter()
        engine = OnlineCovariance(n, edges=edges, window=window, half_life=half_life).update(t, a, p)
        build.append(perf_counter() - start)
        start = perf_counter()
        engine.update(t[revise], a[revise], p[revise] + .001)
        revisions.append(perf_counter() - start)
        start = perf_counter()
        for i in revise:
            engine.update(t[i], a[i], p[i])
        scalar.append(perf_counter() - start)
        start = perf_counter()
        result = engine.snapshot(method="hy")
        snapshot.append(perf_counter() - start)
        start = perf_counter()
        direct = fit(t, a, p, n_assets=n, edges=edges, window=window, half_life=half_life)
        batch.append(perf_counter() - start)
        start = perf_counter()
        serial = fit(t, a, p, n_assets=n, edges=edges, window=window, half_life=half_life, threads=1)
        batch_serial.append(perf_counter() - start)
        np.testing.assert_equal(direct.covariance, serial.covariance)
        np.testing.assert_equal(direct.rows, result.rows)
        np.testing.assert_equal(direct.cols, result.cols)
        np.testing.assert_allclose(direct.covariance, result.covariance, rtol=1e-9, atol=1e-12)
    return dict(n_assets=n, points=len(t), graph="full" if edges is None else "selected",
                window=window, half_life=half_life, batch_seconds=median(batch),
                batch_serial_seconds=median(batch_serial),
                batch_parallel_speedup=median(batch_serial) / median(batch),
                batch_points_per_second=len(t) / median(batch),
                requested_edges=n * (n - 1) // 2 if edges is None else len(edges),
                build_seconds=median(build), points_per_second=len(t) / median(build),
                historical_revisions=len(revise), revision_seconds=median(revisions),
                revisions_per_second=len(revise) / median(revisions),
                scalar_revision_seconds=median(scalar),
                scalar_revisions_per_second=len(revise) / median(scalar),
                snapshot_seconds=median(snapshot), retained=engine.size,
                observed_pairs=int(np.isfinite(result.covariance).sum()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    cases = [(16, 128, None), (64, 128, None), (256, 64, None),
             (256, 128, pair_graph(256)), (1024, 64, pair_graph(1024))]
    if not args.quick:
        cases.append((4096, 32, pair_graph(4096)))
    records = [benchmark(n, m, edges, args.repeats) for n, m, edges in cases]
    records.append(benchmark(1024, 128, pair_graph(1024), args.repeats, window=16, half_life=32))
    cpu = next((line.partition(":")[2].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")), platform.processor())
    # ru_maxrss can retain a parent's pre-exec peak. Linux VmHWM measures this
    # executable's peak instead, which matters when launched by a large runner.
    status = Path("/proc/self/status")
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    if status.exists():
        peak = float(next(line.split()[1] for line in status.read_text().splitlines()
                          if line.startswith("VmHWM:"))) / 1024
    output = dict(version=__version__, cpu=cpu, python=platform.python_version(),
                  numpy=np.__version__, threads={k: os.environ.get(k) for k in
                  ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")},
                  repeats=args.repeats, peak_rss_mib=peak,
                  records=records)
    text = json.dumps(output, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
