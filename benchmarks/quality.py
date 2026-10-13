#!/usr/bin/env python3
"""Reproducible statistical checks; emit JSON without ranking assertions.

Run from an installed checkout::

    python benchmarks/quality.py --reps 100 --output quality.json

Times are in days. Covariances are rates per day; noise_variance contains
point-error variances. Paired methods and coarsening use the same simulated data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


METHODS = ("hy", "overlap")
SIGMA = np.array([[1.0, 0.6, 0.25], [0.6, 1.5, -0.1], [0.25, -0.1, 0.8]])


def _summary(values: np.ndarray) -> dict[str, float | int | None]:
    """Mean and its Monte Carlo standard error, using independent replicates."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    count = len(values)
    return {
        "n_finite": count,
        "mean": float(np.mean(values)) if count else None,
        "mc_se": float(np.std(values, ddof=1) / np.sqrt(count)) if count > 1 else None,
    }


def _rmse(squared_errors: np.ndarray) -> dict[str, float | int | None]:
    """RMSE and delta-method Monte Carlo SE of the replicate mean square."""
    mse = _summary(squared_errors)
    root = np.sqrt(mse["mean"]) if mse["mean"] is not None else None
    se = mse["mc_se"]
    return {
        "n_finite": mse["n_finite"],
        "rmse": float(root) if root is not None else None,
        "rmse_mc_se": float(se / (2 * root)) if se is not None and root > 0 else None,
    }


def _schedule(
    counts: tuple[int, ...], rng: np.random.Generator, *, synchronous: bool
) -> tuple[np.ndarray, np.ndarray]:
    times, assets = [], []
    for asset, count in enumerate(counts):
        ts = (
            # A half-open horizon avoids an unfinished singleton bucket at t=1.
            # There are still `count` observed intervals for each asset.
            np.arange(count + 1, dtype=float) / (count + 1)
            if synchronous
            else np.r_[0.0, np.sort(rng.uniform(0.0, 1.0, count - 1)), 1.0]
        )
        times.append(ts)
        assets.append(np.full(len(ts), asset, dtype=np.int64))
    times_array, assets_array = np.concatenate(times), np.concatenate(assets)
    order = np.lexsort((assets_array, times_array))
    return times_array[order], assets_array[order]


def _sample(
    counts: tuple[int, ...],
    sigma: np.ndarray,
    rng: np.random.Generator,
    *,
    synchronous: bool,
    drift: np.ndarray,
    noise_variance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    times, assets = _schedule(counts, rng, synchronous=synchronous)
    unique_times, inverse = np.unique(times, return_inverse=True)
    increments = rng.standard_normal((len(unique_times) - 1, len(counts)))
    increments = (increments @ np.linalg.cholesky(sigma).T) * np.sqrt(
        np.diff(unique_times)
    )[:, None]
    path = np.vstack((np.zeros(len(counts)), np.cumsum(increments, axis=0)))
    clean_prices = path[inverse, assets] + drift[assets] * times
    noisy_prices = clean_prices + rng.standard_normal(len(times)) * np.sqrt(
        noise_variance[assets]
    )
    return times, assets, clean_prices, noisy_prices


def _metrics(estimates: np.ndarray, sigma: np.ndarray) -> dict:
    errors = estimates - sigma
    complete = np.all(np.isfinite(errors), axis=(1, 2))
    full_squared = np.sum(errors[complete] ** 2, axis=(1, 2))
    pairs = []
    for a in range(len(sigma)):
        for b in range(a, len(sigma)):
            values = estimates[:, a, b]
            difference = values - sigma[a, b]
            pairs.append(
                {
                    "assets": [a, b],
                    "truth": float(sigma[a, b]),
                    "estimate": _summary(values),
                    "bias": _summary(difference),
                    **_rmse(difference**2),
                }
            )
    return {
        "n_complete_replicates": int(np.sum(complete)),
        "n_incomplete_replicates": int(np.sum(~complete)),
        "frobenius": _rmse(full_squared),
        "relative_frobenius": _rmse(full_squared / np.sum(sigma**2)),
        "pairs": pairs,
    }


def _case(
    *,
    name: str,
    counts: tuple[int, ...],
    sigma: np.ndarray,
    reps: int,
    seed: np.random.SeedSequence,
    synchronous: bool = False,
    drift: np.ndarray | None = None,
    noise_variance: np.ndarray | None = None,
    bin_width: float | None = None,
) -> dict:
    from async_covariance import coarsen, fit

    n_assets = len(counts)
    drift = np.zeros(n_assets) if drift is None else np.asarray(drift, dtype=float)
    noise_variance = (
        np.zeros(n_assets)
        if noise_variance is None
        else np.asarray(noise_variance, dtype=float)
    )
    modes = ("clean", "noisy", "coarsened") if bin_width is not None else ("observed",)
    estimates = {
        (mode, method): np.full((reps, n_assets, n_assets), np.nan)
        for mode in modes
        for method in METHODS
    }
    point_counts = {mode: [] for mode in modes}
    diagonal_mismatches = {key: [] for key in estimates}
    for rep, child_seed in enumerate(seed.spawn(reps)):
        rng = np.random.default_rng(child_seed)
        times, assets, clean, noisy = _sample(
            counts,
            sigma,
            rng,
            synchronous=synchronous,
            drift=drift,
            noise_variance=noise_variance,
        )
        for mode in modes:
            prices = clean if mode == "clean" else noisy
            effective_noise = np.zeros(n_assets) if mode == "clean" else noise_variance
            data = (times, assets, prices)
            if mode == "coarsened":
                data = coarsen(*data, width=bin_width, origin=0.0)
            point_counts[mode].append(len(data[0]))
            for method in METHODS:
                snapshot = fit(
                    *data,
                    n_assets=n_assets,
                    method=method,
                    noise_variance=effective_noise,
                    drift=drift,
                )
                matrix = np.asarray(snapshot.dense(), dtype=float)
                if matrix.shape != sigma.shape:
                    raise ValueError(f"unexpected dense shape: {matrix.shape}")
                estimates[mode, method][rep] = matrix
                mismatch = np.abs(np.diag(matrix) - np.asarray(snapshot.variance))
                finite = mismatch[np.isfinite(mismatch)]
                diagonal_mismatches[mode, method].append(
                    float(np.max(finite)) if len(finite) else np.nan
                )
    results = []
    for (mode, method), values in estimates.items():
        results.append(
            {
                "mode": mode,
                "method": method,
                "n_points": _summary(np.asarray(point_counts[mode])),
                "snapshot_diagonal_disagreement": _summary(
                    np.asarray(diagonal_mismatches[mode, method])
                ),
                **_metrics(values, sigma),
            }
        )
    paired_mse = []
    for mode in modes:
        hy_error = np.sum((estimates[mode, "hy"] - sigma) ** 2, axis=(1, 2))
        slope_error = np.sum((estimates[mode, "overlap"] - sigma) ** 2, axis=(1, 2))
        paired_mse.append(
            {
                "mode": mode,
                "quantity": "overlap_minus_hy_squared_frobenius_error",
                **_summary(slope_error - hy_error),
            }
        )
    coarsening_mse = []
    if bin_width is not None:
        for method in METHODS:
            coarse_error = np.sum((estimates["coarsened", method] - sigma) ** 2, axis=(1, 2))
            raw_error = np.sum((estimates["noisy", method] - sigma) ** 2, axis=(1, 2))
            coarsening_mse.append(
                {
                    "method": method,
                    "quantity": "coarsened_minus_raw_squared_frobenius_error",
                    **_summary(coarse_error - raw_error),
                }
            )
    return {
        "name": name,
        "interval_counts": list(counts),
        "schedule": "synchronous_uniform" if synchronous else "independent_uniform_given_count",
        "horizon_days": 1.0,
        "latest_observation_day": counts[0] / (counts[0] + 1) if synchronous else 1.0,
        "endpoint_policy": "half_open_horizon" if synchronous else "both_horizon_endpoints",
        "covariance_rate": sigma.tolist(),
        "drift_rate": drift.tolist(),
        "known_point_noise_variance": noise_variance.tolist(),
        "coarsening_width_days": bin_width,
        "results": results,
        "paired_method_mse_difference": paired_mse,
        "paired_coarsening_mse_difference": coarsening_mse,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261012)
    parser.add_argument("--output", type=Path, help="also save the JSON report to this path")
    parser.add_argument("--json", action="store_true", help="print the complete JSON report")
    args = parser.parse_args()
    if args.reps < 2:
        parser.error("--reps must be at least 2 to estimate Monte Carlo uncertainty")
    specifications = [
        dict(name="stationary_balanced", counts=(512, 512, 512), sigma=SIGMA),
        dict(name="sampling_imbalance", counts=(2048, 128, 16), sigma=SIGMA),
        dict(
            name="stationary_drift_corrected",
            counts=(512, 512, 512),
            sigma=SIGMA,
            drift=np.array([2.0, -1.0, 0.5]),
        ),
    ]
    for count in (128, 512, 2048, 8192):
        specifications.append(
            dict(
                name=f"point_noise_infill_{count}",
                counts=(count, count),
                sigma=SIGMA[:2, :2],
                synchronous=True,
                noise_variance=np.array([0.05, 0.05]),
                bin_width=1.0 / 64,
            )
        )
    seeds = np.random.SeedSequence(args.seed).spawn(len(specifications))
    report = {
        "schema_version": 1,
        "seed": args.seed,
        "replications": args.reps,
        "units": "time in days; covariance and drift rates per day",
        "uncertainty": "mc_se is replicate SD / sqrt(R); RMSE SE uses the delta method",
        "scope": "raw constant-rate moments; no factor projection or superiority assertions",
        "cases": [
            _case(**specification, reps=args.reps, seed=seed)
            for specification, seed in zip(specifications, seeds, strict=True)
        ],
    }
    encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    if args.json:
        print(encoded, end="")
    else:
        print("Case / data / method: Frobenius RMSE +/- Monte Carlo SE")
        for case in report["cases"]:
            for result in case["results"]:
                score = result["frobenius"]
                print(f"{case['name']} / {result['mode']} / {result['method']}: "
                      f"{score['rmse']:.5g} +/- {score['rmse_mc_se']:.3g}")


if __name__ == "__main__":
    main()
