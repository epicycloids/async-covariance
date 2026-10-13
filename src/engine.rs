//! Add/remove adjacent-increment sufficient statistics.
//!
//! Observations are ordered by measurement time inside each asset. The global
//! watermark advances independently of late edits. Every changed interval is
//! subtracted before replacements are added, giving the same raw moments as a
//! batch traversal of the retained observations.

use crate::interval_index::{Interval, IntervalIndex};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::ops::Bound::{Excluded, Unbounded};

#[derive(Clone, Copy, Debug, PartialEq)]
struct Time(f64);

impl Eq for Time {}

impl PartialOrd for Time {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Time {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.total_cmp(&other.0)
    }
}

#[derive(Clone, Copy, Debug)]
struct Point {
    price: f64,
    next_interval: Option<usize>,
}

#[derive(Debug)]
pub(crate) struct PairStats {
    sums: [f64; 14],
    compensation: [f64; 14],
    count: usize,
    last_time: f64,
}

impl PairStats {
    pub(crate) fn new(time: f64) -> Self {
        Self {
            sums: [0.0; 14],
            compensation: [0.0; 14],
            count: 0,
            last_time: time,
        }
    }

    fn decay_to(&mut self, time: f64, half_life: Option<f64>) {
        // Local replacement edges repeatedly touch a pair at one watermark.
        // Disabled decay also needs no accumulator reads or writes.
        if time == self.last_time || half_life.is_none() {
            self.last_time = time;
            return;
        }
        let factor = decay(half_life, time, self.last_time);
        for index in 0..12 {
            self.sums[index] *= factor;
            self.compensation[index] *= factor;
        }
        self.last_time = time;
    }

    pub(crate) fn add(&mut self, values: &[f64; 14], sign: f64) {
        // Neumaier summation also handles contributions larger than the current
        // sum, which occurs when a late correction replaces a long interval.
        for index in (0..12).chain(std::iter::once(13)) {
            let value = sign * values[index];
            let sum = self.sums[index];
            let updated = sum + value;
            self.compensation[index] += if sum.abs() >= value.abs() {
                (sum - updated) + value
            } else {
                (value - updated) + sum
            };
            self.sums[index] = updated;
        }
        if sign > 0.0 {
            self.count += 1;
        } else {
            self.count = self.count.checked_sub(1).expect("matched interval removal");
        }
    }

    pub(crate) fn snapshot(&self, time: f64, half_life: Option<f64>) -> [f64; 14] {
        let mut values = [0.0; 14];
        let factor = decay(half_life, time, self.last_time);
        for (index, value) in values.iter_mut().enumerate().take(12) {
            *value = (self.sums[index] + self.compensation[index]) * factor;
        }
        values[12] = self.count as f64;
        values[13] = self.sums[13] + self.compensation[13];
        values
    }
}

#[derive(Debug)]
pub struct Engine {
    points: Vec<BTreeMap<Time, Point>>,
    reference_drift: Vec<f64>,
    intervals: IntervalIndex,
    pairs: HashMap<(usize, usize), PairStats>,
    selected_edges: Option<HashSet<(usize, usize)>>,
    neighbors: Option<Vec<Vec<usize>>>,
    half_life: Option<f64>,
    window: Option<f64>,
    watermark: Option<f64>,
    n_points: usize,
    interval_expiry: BTreeSet<(Time, usize)>,
    point_expiry: BTreeSet<(Time, usize)>,
}

impl Engine {
    pub fn new(
        n_assets: usize,
        half_life: Option<f64>,
        edges: Option<Vec<(usize, usize)>>,
    ) -> Result<Self, String> {
        validate_positive_option(half_life, "half_life")?;
        let selected_edges = match edges {
            None => None,
            Some(edges) => {
                let mut selected = HashSet::new();
                for (left, right) in edges {
                    if left >= n_assets || right >= n_assets {
                        return Err("selected edge asset is outside n_assets".into());
                    }
                    if left != right {
                        selected.insert(ordered_pair(left, right));
                    }
                }
                Some(selected)
            }
        };
        let neighbors = selected_edges.as_ref().map(|edges| {
            let mut neighbors = vec![Vec::new(); n_assets];
            for &(left, right) in edges {
                neighbors[left].push(right);
                neighbors[right].push(left);
            }
            for row in &mut neighbors {
                row.sort_unstable();
            }
            neighbors
        });
        Ok(Self {
            points: vec![BTreeMap::new(); n_assets],
            reference_drift: vec![0.0; n_assets],
            intervals: IntervalIndex::default(),
            pairs: HashMap::new(),
            selected_edges,
            neighbors,
            half_life,
            window: None,
            watermark: None,
            n_points: 0,
            interval_expiry: BTreeSet::new(),
            point_expiry: BTreeSet::new(),
        })
    }

    pub fn set_window(&mut self, window: Option<f64>) -> Result<(), String> {
        validate_positive_option(window, "window")?;
        if self.n_points > 0 {
            return Err("window must be configured before observations are inserted".into());
        }
        self.window = window;
        Ok(())
    }

    pub fn window(&self) -> Option<f64> {
        self.window
    }

    pub fn set_reference_drift(&mut self, drift: Vec<f64>) -> Result<(), String> {
        if drift.len() != self.points.len() {
            return Err("reference_drift must have length n_assets".into());
        }
        if drift.iter().any(|value| !value.is_finite()) {
            return Err("reference_drift must be finite".into());
        }
        if self.n_points > 0 {
            return Err(
                "reference_drift must be configured before observations are inserted".into(),
            );
        }
        self.reference_drift = drift;
        Ok(())
    }

    pub fn reference_drift(&self) -> &[f64] {
        &self.reference_drift
    }

    pub fn n_assets(&self) -> usize {
        self.points.len()
    }

    pub fn watermark(&self) -> Option<f64> {
        self.watermark
    }

    pub fn n_points(&self) -> usize {
        self.n_points
    }

    pub fn n_intervals(&self) -> usize {
        self.intervals.len()
    }

    pub fn n_pairs(&self) -> usize {
        self.pairs.len()
    }

    pub fn update(&mut self, asset: usize, time: f64, logprice: f64) -> Result<(), String> {
        let key = validate_time(time)?;
        if !logprice.is_finite() {
            return Err("logprice must be finite".into());
        }
        self.check_retained_time(key.0)?;
        let future_time = self.watermark.map_or(key.0, |current| current.max(key.0));
        let cutoff = self.window.map(|window| future_time - window);

        // Validate all replacement intervals before changing the clock, expiry
        // sets, or topology. This keeps numerical rejections atomic.
        let (left, current, right) = self.neighborhood(asset, key);
        let reference = self.reference_drift.get(asset).copied().unwrap_or(0.0);
        let mut replacements = Vec::with_capacity(2);
        if let Some((start, point)) = left {
            if included(start.0, cutoff) {
                replacements.push(make_interval_centered(
                    asset,
                    start.0,
                    key.0,
                    point.price,
                    logprice,
                    reference,
                )?);
            }
        }
        if let Some((end, point)) = right {
            if included(key.0, cutoff) {
                replacements.push(make_interval_centered(
                    asset,
                    key.0,
                    end.0,
                    logprice,
                    point.price,
                    reference,
                )?);
            }
        }
        if current.is_some_and(|point| point.price == logprice) {
            return Ok(());
        }

        asset.checked_add(1).ok_or("asset index overflow")?;
        self.advance_to(future_time);
        self.ensure_asset(asset);
        // Expiry may have removed old edges, so obtain their current handles.
        let (left, current, _) = self.neighborhood(asset, key);
        if let Some((_, point)) = left {
            if let Some(handle) = point.next_interval {
                self.remove_interval(handle);
            }
        }
        if let Some(point) = current {
            if let Some(handle) = point.next_interval {
                self.remove_interval(handle);
            }
        }
        if let Some((start, _)) = left {
            self.points[asset].get_mut(&start).unwrap().next_interval = None;
        }
        if current.is_none() {
            self.n_points += 1;
            if self.window.is_some() {
                self.point_expiry.insert((key, asset));
            }
        }
        self.points[asset].insert(
            key,
            Point {
                price: logprice,
                next_interval: None,
            },
        );
        for interval in replacements {
            let start = Time(interval.start);
            let handle = self.add_interval(interval);
            self.points[asset].get_mut(&start).unwrap().next_interval = Some(handle);
        }
        Ok(())
    }

    pub fn remove(&mut self, asset: usize, time: f64) -> Result<(), String> {
        let key = validate_time(time)?;
        self.check_retained_time(key.0)?;
        let (left, current, right) = self.neighborhood(asset, key);
        let current = current.ok_or("no observation at the requested asset and time")?;
        let cutoff = self.cutoff();
        let replacement = match (left, right) {
            (Some((start, first)), Some((end, last))) if included(start.0, cutoff) => {
                Some(make_interval_centered(
                    asset,
                    start.0,
                    end.0,
                    first.price,
                    last.price,
                    self.reference_drift[asset],
                )?)
            }
            _ => None,
        };
        if let Some((_, point)) = left {
            if let Some(handle) = point.next_interval {
                self.remove_interval(handle);
            }
        }
        if let Some(handle) = current.next_interval {
            self.remove_interval(handle);
        }
        self.points[asset].remove(&key);
        self.point_expiry.remove(&(key, asset));
        self.n_points -= 1;
        if let Some((start, _)) = left {
            self.points[asset].get_mut(&start).unwrap().next_interval = None;
        }
        if let Some(interval) = replacement {
            let start = Time(interval.start);
            let handle = self.add_interval(interval);
            self.points[asset].get_mut(&start).unwrap().next_interval = Some(handle);
        }
        Ok(())
    }

    pub fn advance(&mut self, time: f64) -> Result<(), String> {
        let time = validate_time(time)?.0;
        if self.watermark.is_some_and(|current| time < current) {
            return Err("advance time cannot precede the watermark".into());
        }
        self.advance_to(time);
        Ok(())
    }

    pub fn snapshot(&self) -> Vec<(usize, usize, [f64; 14])> {
        let Some(time) = self.watermark else {
            return Vec::new();
        };
        let mut rows: Vec<_> = self
            .pairs
            .iter()
            .map(|(&(left, right), stats)| (left, right, stats.snapshot(time, self.half_life)))
            .collect();
        rows.sort_unstable_by_key(|&(left, right, _)| (left, right));
        rows
    }

    fn ensure_asset(&mut self, asset: usize) {
        if asset >= self.points.len() {
            self.points.resize_with(asset + 1, BTreeMap::new);
            self.reference_drift.resize(asset + 1, 0.0);
            if let Some(neighbors) = &mut self.neighbors {
                neighbors.resize_with(asset + 1, Vec::new);
            }
        }
    }

    fn neighborhood(&self, asset: usize, time: Time) -> Neighborhood {
        let Some(points) = self.points.get(asset) else {
            return (None, None, None);
        };
        let left = points
            .range(..time)
            .next_back()
            .map(|(&key, &value)| (key, value));
        let current = points.get(&time).copied();
        let right = points
            .range((Excluded(time), Unbounded))
            .next()
            .map(|(&key, &value)| (key, value));
        (left, current, right)
    }

    fn cutoff(&self) -> Option<f64> {
        self.watermark
            .zip(self.window)
            .map(|(time, window)| time - window)
    }

    fn check_retained_time(&self, time: f64) -> Result<(), String> {
        if self.cutoff().is_some_and(|cutoff| time < cutoff) {
            return Err("event time precedes the retained correction window".into());
        }
        Ok(())
    }

    fn advance_to(&mut self, time: f64) {
        self.watermark = Some(time);
        let Some(cutoff) = self.cutoff() else {
            return;
        };
        while let Some(&(start, handle)) = self.interval_expiry.first() {
            if start.0 >= cutoff {
                break;
            }
            let interval = self.intervals.get(handle).unwrap();
            self.remove_interval(handle);
            self.points[interval.asset]
                .get_mut(&start)
                .unwrap()
                .next_interval = None;
        }
        let mut prune_assets = HashSet::new();
        while let Some(&(event_time, asset)) = self.point_expiry.first() {
            if event_time.0 >= cutoff {
                break;
            }
            self.point_expiry.pop_first();
            prune_assets.insert(asset);
        }
        for asset in prune_assets {
            let newer = self.points[asset].split_off(&Time(cutoff));
            let sentinel = self.points[asset].pop_last();
            self.n_points -= self.points[asset].len();
            self.points[asset] = newer;
            if let Some((key, mut point)) = sentinel {
                point.next_interval = None;
                self.points[asset].insert(key, point);
            }
        }
    }

    fn matching_intervals(&self, interval: Interval) -> Vec<Interval> {
        if let Some(neighbors) = &self.neighbors {
            // A sparse requested graph should not pay for every unrelated asset
            // whose interval overlaps this event. Per-asset adjacency is disjoint.
            let mut matched = Vec::new();
            for &asset in &neighbors[interval.asset] {
                let points = &self.points[asset];
                if let Some((_, point)) = points.range(..=Time(interval.start)).next_back() {
                    if let Some(handle) = point.next_interval {
                        let other = self.intervals.get(handle).unwrap();
                        if other.end > interval.start {
                            matched.push(other);
                        }
                    }
                }
                for (_, point) in
                    points.range((Excluded(Time(interval.start)), Excluded(Time(interval.end))))
                {
                    if let Some(handle) = point.next_interval {
                        matched.push(self.intervals.get(handle).unwrap());
                    }
                }
            }
            matched
        } else {
            self.intervals.overlapping(interval.start, interval.end)
        }
    }

    fn apply_interval(&mut self, interval: Interval, sign: f64) {
        let time = self.watermark.unwrap();
        let diagonal = contribution(interval, interval, true, self.half_life, time);
        self.apply_pair((interval.asset, interval.asset), diagonal, sign);
        for other in self.matching_intervals(interval) {
            if other.asset == interval.asset {
                continue;
            }
            let pair = ordered_pair(interval.asset, other.asset);
            if self
                .selected_edges
                .as_ref()
                .is_some_and(|edges| !edges.contains(&pair))
            {
                continue;
            }
            let (left, right) = if interval.asset < other.asset {
                (interval, other)
            } else {
                (other, interval)
            };
            let values = contribution(left, right, false, self.half_life, time);
            self.apply_pair(pair, values, sign);
        }
    }

    fn apply_pair(&mut self, pair: (usize, usize), values: [f64; 14], sign: f64) {
        let time = self.watermark.unwrap();
        let stats = self
            .pairs
            .entry(pair)
            .or_insert_with(|| PairStats::new(time));
        stats.decay_to(time, self.half_life);
        stats.add(&values, sign);
        if stats.count == 0 {
            self.pairs.remove(&pair);
        }
    }

    fn add_interval(&mut self, interval: Interval) -> usize {
        self.apply_interval(interval, 1.0);
        let handle = self.intervals.insert(interval);
        if self.window.is_some() {
            self.interval_expiry.insert((Time(interval.start), handle));
        }
        handle
    }

    fn remove_interval(&mut self, handle: usize) {
        let interval = self.intervals.get(handle).expect("live point interval");
        self.apply_interval(interval, -1.0);
        self.intervals.remove(handle).unwrap();
        self.interval_expiry.remove(&(Time(interval.start), handle));
    }
}

type Neighborhood = (Option<(Time, Point)>, Option<Point>, Option<(Time, Point)>);

fn validate_time(time: f64) -> Result<Time, String> {
    if !time.is_finite() {
        return Err("time must be finite".into());
    }
    // Signed zero describes one physical timestamp, not two observations.
    Ok(Time(if time == 0.0 { 0.0 } else { time }))
}

fn validate_positive_option(value: Option<f64>, name: &str) -> Result<(), String> {
    if value.is_some_and(|value| !value.is_finite() || value <= 0.0) {
        return Err(format!("{name} must be finite and positive"));
    }
    Ok(())
}

fn included(start: f64, cutoff: Option<f64>) -> bool {
    cutoff.is_none_or(|cutoff| start >= cutoff)
}

fn ordered_pair(left: usize, right: usize) -> (usize, usize) {
    if left <= right {
        (left, right)
    } else {
        (right, left)
    }
}

fn decay(half_life: Option<f64>, time: f64, anchor: f64) -> f64 {
    match half_life {
        Some(half_life) => (-std::f64::consts::LN_2 * ((time - anchor) / half_life)).exp(),
        None => 1.0,
    }
}

pub(crate) fn make_interval(
    asset: usize,
    start: f64,
    end: f64,
    first: f64,
    last: f64,
) -> Result<Interval, String> {
    make_interval_centered(asset, start, end, first, last, 0.0)
}

pub(crate) fn make_interval_centered(
    asset: usize,
    start: f64,
    end: f64,
    first: f64,
    last: f64,
    reference: f64,
) -> Result<Interval, String> {
    let duration = end - start;
    let raw_return = last - first;
    let interval = Interval {
        asset,
        start,
        end,
        duration,
        return_value: if reference == 0.0 {
            raw_return
        } else {
            (-reference).mul_add(duration, raw_return)
        },
    };
    if !interval.duration.is_finite()
        || interval.duration <= 0.0
        || !interval.return_value.is_finite()
    {
        return Err("interval duration and return must be finite, with positive duration".into());
    }
    if contribution(interval, interval, true, None, end)
        .iter()
        .any(|value| !value.is_finite())
    {
        return Err("interval moments exceed floating point range".into());
    }
    Ok(interval)
}

pub(crate) fn contribution(
    left: Interval,
    right: Interval,
    diagonal: bool,
    half_life: Option<f64>,
    time: f64,
) -> [f64; 14] {
    let start = left.start.max(right.start);
    let end = left.end.min(right.end);
    let overlap = (end - start).max(0.0);
    let anchor = start + 0.5 * overlap;
    let weight = decay(half_life, time, anchor);
    // A diagonal increment has unit normalized overlap. Computing it through
    // two square roots unnecessarily perturbs its exact support denominator.
    let (slope_product, slope_denominator) = if diagonal {
        ((left.return_value / left.duration) * left.return_value, 1.0)
    } else {
        let normalized_overlap = overlap / left.duration.sqrt() / right.duration.sqrt();
        let normalized_left = left.return_value / left.duration.sqrt();
        let normalized_right = right.return_value / right.duration.sqrt();
        (
            normalized_overlap * normalized_left * normalized_right,
            normalized_overlap * normalized_overlap,
        )
    };
    let mut values = [
        left.return_value * right.return_value,
        left.duration * right.return_value,
        right.duration * left.return_value,
        left.duration * right.duration,
        overlap,
        if diagonal { 2.0 } else { 0.0 },
        slope_product,
        (overlap / right.duration) * right.return_value,
        (overlap / left.duration) * left.return_value,
        overlap,
        slope_denominator,
        if diagonal { 2.0 / left.duration } else { 0.0 },
        1.0,
        overlap,
    ];
    for value in values.iter_mut().take(12) {
        *value *= weight;
    }
    values
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: &[(usize, usize, [f64; 14])], expected: &[(usize, usize, [f64; 14])]) {
        assert_eq!(actual.len(), expected.len());
        for ((left, right, values), (expected_left, expected_right, expected_values)) in
            actual.iter().zip(expected)
        {
            assert_eq!((left, right), (expected_left, expected_right));
            for index in 0..14 {
                let tolerance = 3e-12 * (1.0 + expected_values[index].abs());
                assert!(
                    (values[index] - expected_values[index]).abs() < tolerance,
                    "pair {left}/{right} coefficient {index}: {} != {}",
                    values[index],
                    expected_values[index]
                );
            }
        }
    }

    fn fitted(rows: &[(usize, f64, f64)], half_life: Option<f64>) -> Engine {
        let mut engine = Engine::new(3, half_life, None).unwrap();
        for &(asset, time, price) in rows {
            engine.update(asset, time, price).unwrap();
        }
        engine
    }

    #[test]
    fn scalar_statistics_and_drift_noise_coefficients() {
        let mut engine = Engine::new(1, None, None).unwrap();
        engine.update(0, -1.0, 2.0).unwrap();
        engine.update(0, 1.0, 5.0).unwrap();
        assert_eq!(
            engine.snapshot()[0].2,
            [9.0, 6.0, 6.0, 4.0, 2.0, 2.0, 4.5, 3.0, 3.0, 2.0, 1.0, 1.0, 1.0, 2.0]
        );
    }

    #[test]
    fn fixed_drift_centering_preserves_small_variance_and_local_revisions() {
        let drift = 1e6;
        let mut engine = Engine::new(1, None, None).unwrap();
        engine.set_reference_drift(vec![drift]).unwrap();
        let mut expected = 0.0;
        let mut previous = None;
        for step in 0..1001 {
            let price = drift * step as f64 + (step as f64 * 0.37).sin() * 0.001;
            if let Some(first) = previous {
                let residual = price - first - drift;
                expected += residual * residual;
            }
            engine.update(0, step as f64, price).unwrap();
            previous = Some(price);
        }
        let original = engine.snapshot();
        assert!((original[0].2[0] / expected - 1.0).abs() < 1e-12);
        assert!((original[0].2[6] / expected - 1.0).abs() < 1e-12);
        engine.update(0, 0.5, drift * 0.5 + 0.1).unwrap();
        engine.update(0, 11.0, drift * 11.0 + 0.3).unwrap();
        engine.remove(0, 0.5).unwrap();
        engine
            .update(0, 11.0, drift * 11.0 + (11.0_f64 * 0.37).sin() * 0.001)
            .unwrap();
        assert_close(&engine.snapshot(), &original);
    }

    #[test]
    fn reference_drift_is_validated_fixed_and_grows_with_zero_baselines() {
        let mut engine = Engine::new(1, None, None).unwrap();
        assert!(engine.set_reference_drift(vec![]).is_err());
        assert!(engine.set_reference_drift(vec![f64::NAN]).is_err());
        assert_eq!(engine.reference_drift(), &[0.0]);
        engine.set_reference_drift(vec![2.0]).unwrap();
        engine.update(2, 0.0, 1.0).unwrap();
        engine.update(2, 1.0, 3.0).unwrap();
        assert_eq!(engine.reference_drift(), &[2.0, 0.0, 0.0]);
        assert_eq!(engine.snapshot()[0].2[0], 4.0);
        assert!(engine.set_reference_drift(vec![0.0; 3]).is_err());
        assert_eq!(engine.reference_drift(), &[2.0, 0.0, 0.0]);
    }

    #[test]
    fn diagonal_slope_avoids_squaring_underflow_before_normalization() {
        let mut engine = Engine::new(1, None, None).unwrap();
        engine.update(0, 0.0, 0.0).unwrap();
        engine.update(0, 1e-300, 1e-200).unwrap();
        let values = engine.snapshot()[0].2;
        assert!((values[6] / 1e-100 - 1.0).abs() < 1e-14);
        assert_eq!(values[10], 1.0);
    }

    #[test]
    fn insertion_correction_deletion_and_decay_match_final_batch() {
        let final_rows = [
            (0, 0.0, 0.0),
            (0, 2.0, 0.4),
            (0, 3.0, 0.2),
            (1, 0.5, 1.0),
            (1, 1.5, 1.1),
            (1, 4.0, 0.9),
            (2, 0.0, 0.2),
            (2, 4.0, 0.7),
        ];
        for half_life in [None, Some(2.0)] {
            let mut engine = fitted(&final_rows, half_life);
            engine.update(0, 1.0, 0.8).unwrap();
            engine.update(1, 1.5, 2.3).unwrap();
            engine.update(0, 2.0, -0.2).unwrap();
            engine.remove(0, 1.0).unwrap();
            engine.update(1, 1.5, 1.1).unwrap();
            engine.update(0, 2.0, 0.4).unwrap();
            engine.advance(7.0).unwrap();
            let mut batch = fitted(&final_rows, half_life);
            batch.advance(7.0).unwrap();
            assert_close(&engine.snapshot(), &batch.snapshot());
        }
    }

    #[test]
    fn selected_edges_match_full_graph_subset() {
        let rows = [
            (0, 0.0, 0.0),
            (0, 2.0, 1.0),
            (1, 0.0, 0.1),
            (1, 1.0, 0.2),
            (1, 3.0, 0.5),
            (2, 0.5, 0.7),
            (2, 3.0, 0.2),
        ];
        let full = fitted(&rows, Some(4.0));
        let mut selected = Engine::new(3, Some(4.0), Some(vec![(2, 0)])).unwrap();
        for &(asset, time, price) in rows.iter().rev() {
            selected.update(asset, time, price).unwrap();
        }
        let expected: Vec<_> = full
            .snapshot()
            .into_iter()
            .filter(|(a, b, _)| a == b || (*a == 0 && *b == 2))
            .collect();
        assert_close(&selected.snapshot(), &expected);
    }

    #[test]
    fn window_expiry_is_whole_interval_and_preserves_sentinel() {
        let mut engine = Engine::new(2, Some(2.0), None).unwrap();
        engine.set_window(Some(3.0)).unwrap();
        for &(asset, time, price) in &[
            (0, 0.0, 0.0),
            (0, 1.0, 0.3),
            (1, 1.0, 0.1),
            (0, 2.0, 0.5),
            (1, 3.0, 0.2),
            (0, 4.0, 0.9),
            (1, 4.0, 0.4),
        ] {
            engine.update(asset, time, price).unwrap();
        }
        engine.advance(5.0).unwrap();
        let mut batch = fitted(
            &[(0, 2.0, 0.5), (0, 4.0, 0.9), (1, 3.0, 0.2), (1, 4.0, 0.4)],
            Some(2.0),
        );
        batch.advance(5.0).unwrap();
        assert_close(&engine.snapshot(), &batch.snapshot());
        assert_eq!(engine.n_points(), 6); // two active points plus one sentinel per asset
        engine.update(0, 2.5, 0.55).unwrap();
        assert!(engine.update(0, 1.0, 0.9).is_err());
        engine.advance(9.0).unwrap();
        assert_eq!(engine.n_intervals(), 0);
        assert_eq!(engine.n_points(), 2);
        assert!(engine.snapshot().is_empty());
    }

    #[test]
    fn repeated_corrections_do_not_accumulate_and_failures_are_atomic() {
        let rows = [
            (0, 0.0, 0.1),
            (0, 1.0, 0.2),
            (0, 2.0, 0.4),
            (1, 0.0, 0.2),
            (1, 2.0, 0.7),
        ];
        let mut engine = fitted(&rows, None);
        let expected = engine.snapshot();
        for _ in 0..1000 {
            engine.update(0, 1.0, 100_000.0).unwrap();
            engine.update(0, 1.0, 0.2).unwrap();
        }
        assert_close(&engine.snapshot(), &expected);
        for (time, price) in [(f64::NAN, 0.0), (3.0, f64::INFINITY), (3.0, 1e200)] {
            assert!(engine.update(0, time, price).is_err());
            assert_close(&engine.snapshot(), &expected);
            assert_eq!(engine.watermark(), Some(2.0));
        }
        assert!(engine.update(0, 1e-320, 0.0).is_err());
        assert_close(&engine.snapshot(), &expected);
    }
}
