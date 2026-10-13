//! Immutable batch moments using disjoint per-asset interval scans.
//!
//! No event index or historical revision state is needed for a batch fit. Each
//! requested off-diagonal pair is traversed once with two interval pointers.

use crate::engine::{contribution, make_interval, make_interval_centered, PairStats};
use crate::interval_index::Interval;
use rayon::prelude::*;
use rayon::{ThreadPool, ThreadPoolBuilder};
use std::collections::{BTreeSet, HashMap};
use std::sync::{Arc, Mutex, OnceLock};

const PARALLEL_SCAN_THRESHOLD: usize = 100_000;
static POOLS: OnceLock<Mutex<HashMap<usize, Arc<ThreadPool>>>> = OnceLock::new();

pub(crate) fn batch(
    times: &[f64],
    assets: &[i64],
    prices: &[f64],
    n_assets: usize,
    half_life: Option<f64>,
    window: Option<f64>,
    edges: Option<Vec<(usize, usize)>>,
    threads: Option<usize>,
    reference_drift: Option<&[f64]>,
) -> Result<Vec<(usize, usize, [f64; 14])>, String> {
    if times.len() != assets.len() || times.len() != prices.len() {
        return Err("times, assets, and log_prices must have equal lengths".into());
    }
    if threads == Some(0) {
        return Err("threads must be positive".into());
    }
    if let Some(drift) = reference_drift {
        if drift.len() != n_assets {
            return Err("reference_drift must have length n_assets".into());
        }
        if drift.iter().any(|value| !value.is_finite()) {
            return Err("reference_drift must be finite".into());
        }
    }
    for (value, name) in [(half_life, "half_life"), (window, "window")] {
        if value.is_some_and(|value| !value.is_finite() || value <= 0.0) {
            return Err(format!("{name} must be finite and positive"));
        }
    }
    let selected = match edges {
        None => None,
        Some(edges) => {
            let mut selected = BTreeSet::new();
            for (a, b) in edges {
                if a >= n_assets || b >= n_assets {
                    return Err("selected edge asset is outside n_assets".into());
                }
                if a != b {
                    selected.insert((a.min(b), a.max(b)));
                }
            }
            Some(selected)
        }
    };
    let mut points = vec![Vec::<(f64, f64)>::new(); n_assets];
    let mut previous_time = None;
    for ((&time, &asset), &price) in times.iter().zip(assets).zip(prices) {
        if !time.is_finite() || !price.is_finite() {
            return Err("times and log_prices must be finite".into());
        }
        if asset < 0 || asset > i32::MAX as i64 || asset as usize >= n_assets {
            return Err("assets must be nonnegative indices covered by n_assets".into());
        }
        let time = if time == 0.0 { 0.0 } else { time };
        if previous_time.is_some_and(|previous| time < previous) {
            return Err("batch times must be sorted in nondecreasing order".into());
        }
        previous_time = Some(time);
        let row = &mut points[asset as usize];
        if let Some((last_time, last_price)) = row.last_mut() {
            if *last_time == time {
                *last_price = price;
                continue;
            }
        }
        row.push((time, price));
    }
    let Some(now) = previous_time else {
        return Ok(Vec::new());
    };
    let cutoff = window.map(|window| now - window);
    let mut intervals: Vec<Vec<Interval>> = Vec::with_capacity(n_assets);
    for (asset, row) in points.into_iter().enumerate() {
        let reference = reference_drift.map_or(0.0, |drift| drift[asset]);
        let mut increments = Vec::with_capacity(row.len().saturating_sub(1));
        for adjacent in row.windows(2) {
            let ((start, first), (end, last)) = (adjacent[0], adjacent[1]);
            if cutoff.is_none_or(|cutoff| start >= cutoff) {
                increments.push(if reference == 0.0 {
                    make_interval(asset, start, end, first, last)?
                } else {
                    make_interval_centered(asset, start, end, first, last, reference)?
                });
            }
        }
        intervals.push(increments);
    }

    let mut result = Vec::new();
    let active: Vec<_> = intervals
        .iter()
        .enumerate()
        .filter_map(|(asset, row)| (!row.is_empty()).then_some(asset))
        .collect();
    for &asset in &active {
        let mut stats = PairStats::new(now);
        for &interval in &intervals[asset] {
            stats.add(&contribution(interval, interval, true, half_life, now), 1.0);
        }
        result.push((asset, asset, stats.snapshot(now, half_life)));
    }
    let thread_count = threads.unwrap_or_else(|| {
        std::thread::available_parallelism()
            .map_or(1, |count| count.get())
            .min(8)
    });
    if let Some(selected) = selected {
        let selected: Vec<_> = selected
            .into_iter()
            .filter(|&(a, b)| !intervals[a].is_empty() && !intervals[b].is_empty())
            .collect();
        let scan_work = selected.iter().fold(0usize, |work, &(a, b)| {
            work.saturating_add(intervals[a].len().saturating_add(intervals[b].len()))
        });
        if thread_count > 1 && scan_work >= PARALLEL_SCAN_THRESHOLD {
            let pool = thread_pool(thread_count)?;
            let rows: Vec<_> = pool.install(|| {
                selected
                    .par_iter()
                    .filter_map(|&(a, b)| {
                        pair_statistics(&intervals[a], &intervals[b], half_life, now)
                            .map(|stats| (a, b, stats))
                    })
                    .collect()
            });
            result.extend(rows);
        } else {
            for (a, b) in selected {
                if let Some(stats) = pair_statistics(&intervals[a], &intervals[b], half_life, now) {
                    result.push((a, b, stats));
                }
            }
        }
    } else {
        let scan_work = active
            .iter()
            .fold(0usize, |work, &a| work.saturating_add(intervals[a].len()))
            .saturating_mul(active.len().saturating_sub(1));
        if thread_count > 1 && scan_work >= PARALLEL_SCAN_THRESHOLD {
            let pool = thread_pool(thread_count)?;
            let rows: Vec<_> = pool.install(|| {
                let intervals = &intervals;
                active
                    .par_iter()
                    .enumerate()
                    .flat_map(|(position, &a)| {
                        active[position + 1..].par_iter().filter_map(move |&b| {
                            pair_statistics(&intervals[a], &intervals[b], half_life, now)
                                .map(|stats| (a, b, stats))
                        })
                    })
                    .collect()
            });
            result.extend(rows);
        } else {
            for (position, &a) in active.iter().enumerate() {
                for &b in &active[position + 1..] {
                    if let Some(stats) =
                        pair_statistics(&intervals[a], &intervals[b], half_life, now)
                    {
                        result.push((a, b, stats));
                    }
                }
            }
        }
    }
    result.sort_unstable_by_key(|&(a, b, _)| (a, b));
    Ok(result)
}

fn thread_pool(threads: usize) -> Result<Arc<ThreadPool>, String> {
    let mut pools = POOLS
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .map_err(|_| "batch thread pool cache is unavailable".to_string())?;
    if let Some(pool) = pools.get(&threads) {
        return Ok(Arc::clone(pool));
    }
    let pool = Arc::new(
        ThreadPoolBuilder::new()
            .num_threads(threads)
            .thread_name(move |index| format!("async-covariance-{threads}-{index}"))
            .build()
            .map_err(|error| format!("cannot create batch thread pool: {error}"))?,
    );
    pools.insert(threads, Arc::clone(&pool));
    Ok(pool)
}

fn pair_statistics(
    left: &[Interval],
    right: &[Interval],
    half_life: Option<f64>,
    now: f64,
) -> Option<[f64; 14]> {
    let (mut i, mut j, mut count) = (0, 0, 0);
    let mut stats = PairStats::new(now);
    while i < left.len() && j < right.len() {
        let (a, b) = (left[i], right[j]);
        if a.start.max(b.start) < a.end.min(b.end) {
            stats.add(&contribution(a, b, false, half_life, now), 1.0);
            count += 1;
        }
        if a.end <= b.end {
            i += 1;
        }
        if b.end <= a.end {
            j += 1;
        }
    }
    (count > 0).then(|| stats.snapshot(now, half_life))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::Engine;

    fn assert_close(actual: &[(usize, usize, [f64; 14])], expected: &[(usize, usize, [f64; 14])]) {
        assert_eq!(actual.len(), expected.len());
        for ((a, b, values), (x, y, target)) in actual.iter().zip(expected) {
            assert_eq!((a, b), (x, y));
            for index in 0..14 {
                assert!(
                    (values[index] - target[index]).abs() <= 3e-12 * (1.0 + target[index].abs()),
                    "pair {a}/{b} statistic {index}: {} != {}",
                    values[index],
                    target[index]
                );
            }
        }
    }

    #[test]
    fn direct_scan_matches_event_engine_across_modes() {
        let times = [
            -1., -0.5, 0., 0., 0.25, 0.5, 1., 1., 1.5, 2., 2., 3., 3., 4.,
        ];
        let assets = [0, 2, 0, 1, 2, 1, 0, 2, 1, 0, 1, 2, 0, 1];
        let prices = [
            0., 0.3, 0.1, 0.2, 0.4, 0.1, 0.2, 0.5, 0.3, 0.4, 0.6, 0.9, 0.5, 0.7,
        ];
        for half_life in [None, Some(1.3)] {
            for window in [None, Some(2.5)] {
                for edges in [None, Some(vec![]), Some(vec![(2, 0), (0, 2), (1, 1)])] {
                    let actual = batch(
                        &times,
                        &assets,
                        &prices,
                        4,
                        half_life,
                        window,
                        edges.clone(),
                        None,
                        None,
                    )
                    .unwrap();
                    let mut engine = Engine::new(4, half_life, edges).unwrap();
                    engine.set_window(window).unwrap();
                    for ((&t, &a), &p) in times.iter().zip(&assets).zip(&prices) {
                        engine.update(a as usize, t, p).unwrap();
                    }
                    assert_close(&actual, &engine.snapshot());
                }
            }
        }
    }

    #[test]
    fn signed_zero_duplicates_use_last_value_and_touching_intervals_are_disjoint() {
        let times = [-0., 0., 0., 1., 1., 2.];
        let assets = [0, 0, 0, 0, 1, 1];
        let prices = [100., 1., 2., 4., 0., 3.];
        let actual = batch(&times, &assets, &prices, 2, None, None, None, None, None).unwrap();
        assert_eq!(actual.len(), 2);
        assert_eq!((actual[0].0, actual[0].1), (0, 0));
        assert_eq!((actual[1].0, actual[1].1), (1, 1));
        assert_eq!(actual[0].2[0], 4.);
        assert_eq!(actual[1].2[0], 9.);
        assert_eq!(actual[0].2[12], 1.);
    }

    #[test]
    fn empty_and_invalid_batches_are_explicit() {
        assert!(batch(&[], &[], &[], 0, None, None, None, None, None)
            .unwrap()
            .is_empty());
        assert!(batch(
            &[1., 0.],
            &[0, 0],
            &[1., 0.],
            1,
            None,
            None,
            None,
            None,
            None
        )
        .is_err());
        assert!(batch(&[0.], &[-1], &[0.], 1, None, None, None, None, None).is_err());
        assert!(batch(&[0.], &[1], &[0.], 1, None, None, None, None, None).is_err());
        assert!(batch(&[0.], &[0], &[f64::NAN], 1, None, None, None, None, None).is_err());
        assert!(batch(&[0.], &[0], &[], 1, None, None, None, None, None).is_err());
        assert!(batch(&[], &[], &[], 0, Some(0.), None, None, None, None).is_err());
        assert!(batch(&[], &[], &[], 0, None, None, Some(vec![(0, 1)]), None, None).is_err());
        assert!(batch(
            &[0., 1.],
            &[0, 0],
            &[0., 1e200],
            1,
            None,
            None,
            None,
            None,
            None
        )
        .is_err());
        assert!(batch(&[], &[], &[], 0, None, None, None, Some(0), None).is_err());
        assert!(batch(&[], &[], &[], 1, None, None, None, None, Some(&[])).is_err());
        assert!(batch(&[], &[], &[], 1, None, None, None, None, Some(&[f64::NAN])).is_err());
    }

    #[test]
    fn parallel_full_and_selected_scans_are_bitwise_identical_to_serial() {
        let n = 32;
        let mut times = Vec::new();
        let mut assets = Vec::new();
        let mut prices = Vec::new();
        for step in 0..128 {
            for asset in 0..n {
                times.push(step as f64 + asset as f64 / n as f64);
                assets.push(asset as i64);
                prices.push((step as f64 * 0.07 + asset as f64 * 0.19).sin());
            }
        }
        let edges: Vec<_> = (0..n)
            .flat_map(|a| (a + 1..n).map(move |b| (a, b)))
            .collect();
        for half_life in [None, Some(3.2)] {
            for window in [None, Some(120.)] {
                for selected in [None, Some(edges.clone())] {
                    let serial = batch(
                        &times,
                        &assets,
                        &prices,
                        n,
                        half_life,
                        window,
                        selected.clone(),
                        Some(1),
                        None,
                    )
                    .unwrap();
                    let parallel = batch(
                        &times,
                        &assets,
                        &prices,
                        n,
                        half_life,
                        window,
                        selected,
                        Some(2),
                        None,
                    )
                    .unwrap();
                    assert_eq!(serial, parallel);
                }
            }
        }
    }

    #[test]
    fn fixed_large_reference_drift_matches_centered_event_statistics() {
        let drift = [1e6, -2e6];
        let mut times = Vec::new();
        let mut assets = Vec::new();
        let mut prices = Vec::new();
        for step in 0..200 {
            for asset in 0..2 {
                let time = step as f64 + asset as f64 * 0.5;
                times.push(time);
                assets.push(asset as i64);
                prices.push(drift[asset] * time + (time * 0.37 + asset as f64).sin() * 0.001);
            }
        }
        for half_life in [None, Some(5.0)] {
            for window in [None, Some(20.0)] {
                let actual = batch(
                    &times,
                    &assets,
                    &prices,
                    2,
                    half_life,
                    window,
                    None,
                    Some(1),
                    Some(&drift),
                )
                .unwrap();
                let mut engine = Engine::new(2, half_life, None).unwrap();
                engine.set_window(window).unwrap();
                engine.set_reference_drift(drift.to_vec()).unwrap();
                for ((&time, &asset), &price) in times.iter().zip(&assets).zip(&prices) {
                    engine.update(asset as usize, time, price).unwrap();
                }
                assert_close(&actual, &engine.snapshot());
                assert!(actual.iter().all(|(_, _, stats)| stats[0].abs() < 1e-4));
            }
        }
    }
}
