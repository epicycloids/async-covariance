//! Dynamic interval reporting index with pooled treap nodes.
//!
//! Keys are `(start, handle)`. Each subtree stores its maximum interval end so
//! queries skip expired or disjoint subtrees. Handles are internal and reused
//! only after removal; callers must not retain removed handles.

use std::cmp::Ordering;

#[derive(Clone, Copy, Debug)]
pub(crate) struct Interval {
    pub asset: usize,
    pub start: f64,
    pub end: f64,
    pub duration: f64,
    pub return_value: f64,
}

#[derive(Debug)]
struct Node {
    interval: Interval,
    left: Option<usize>,
    right: Option<usize>,
    priority: u64,
    max_end: f64,
}

#[derive(Debug, Default)]
pub(crate) struct IntervalIndex {
    root: Option<usize>,
    nodes: Vec<Option<Node>>,
    free: Vec<usize>,
    sequence: u64,
    len: usize,
}

impl IntervalIndex {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn get(&self, handle: usize) -> Option<Interval> {
        self.nodes.get(handle)?.as_ref().map(|node| node.interval)
    }

    pub fn insert(&mut self, interval: Interval) -> usize {
        let handle = self.free.pop().unwrap_or_else(|| {
            self.nodes.push(None);
            self.nodes.len() - 1
        });
        self.sequence = self.sequence.wrapping_add(1);
        self.nodes[handle] = Some(Node {
            interval,
            left: None,
            right: None,
            priority: mix64(self.sequence),
            max_end: interval.end,
        });
        self.root = Some(self.insert_at(self.root, handle));
        self.len += 1;
        handle
    }

    pub fn remove(&mut self, handle: usize) -> Option<Interval> {
        let interval = self.get(handle)?;
        self.root = self.remove_at(self.root, interval.start, handle);
        self.nodes[handle] = None;
        self.free.push(handle);
        self.len -= 1;
        Some(interval)
    }

    pub fn overlapping(&self, start: f64, end: f64) -> Vec<Interval> {
        let mut result = Vec::new();
        self.report(self.root, start, end, &mut result);
        result
    }

    fn node(&self, handle: usize) -> &Node {
        self.nodes[handle].as_ref().expect("live interval handle")
    }

    fn node_mut(&mut self, handle: usize) -> &mut Node {
        self.nodes[handle].as_mut().expect("live interval handle")
    }

    fn update_max(&mut self, handle: usize) {
        let node = self.node(handle);
        let mut maximum = node.interval.end;
        if let Some(left) = node.left {
            maximum = maximum.max(self.node(left).max_end);
        }
        if let Some(right) = node.right {
            maximum = maximum.max(self.node(right).max_end);
        }
        self.node_mut(handle).max_end = maximum;
    }

    fn compare(&self, left: usize, right: usize) -> Ordering {
        self.node(left)
            .interval
            .start
            .total_cmp(&self.node(right).interval.start)
            .then_with(|| left.cmp(&right))
    }

    fn insert_at(&mut self, root: Option<usize>, handle: usize) -> usize {
        let Some(root) = root else {
            return handle;
        };
        if self.compare(handle, root) == Ordering::Less {
            let child = self.insert_at(self.node(root).left, handle);
            self.node_mut(root).left = Some(child);
            if self.node(child).priority > self.node(root).priority {
                return self.rotate_right(root);
            }
        } else {
            let child = self.insert_at(self.node(root).right, handle);
            self.node_mut(root).right = Some(child);
            if self.node(child).priority > self.node(root).priority {
                return self.rotate_left(root);
            }
        }
        self.update_max(root);
        root
    }

    fn remove_at(&mut self, root: Option<usize>, start: f64, handle: usize) -> Option<usize> {
        let root = root?;
        let order = start
            .total_cmp(&self.node(root).interval.start)
            .then_with(|| handle.cmp(&root));
        match order {
            Ordering::Less => {
                let child = self.remove_at(self.node(root).left, start, handle);
                self.node_mut(root).left = child;
                self.update_max(root);
                Some(root)
            }
            Ordering::Greater => {
                let child = self.remove_at(self.node(root).right, start, handle);
                self.node_mut(root).right = child;
                self.update_max(root);
                Some(root)
            }
            Ordering::Equal => self.join(self.node(root).left, self.node(root).right),
        }
    }

    fn join(&mut self, left: Option<usize>, right: Option<usize>) -> Option<usize> {
        match (left, right) {
            (None, other) | (other, None) => other,
            (Some(left), Some(right)) => {
                if self.node(left).priority > self.node(right).priority {
                    let child = self.join(self.node(left).right, Some(right));
                    self.node_mut(left).right = child;
                    self.update_max(left);
                    Some(left)
                } else {
                    let child = self.join(Some(left), self.node(right).left);
                    self.node_mut(right).left = child;
                    self.update_max(right);
                    Some(right)
                }
            }
        }
    }

    fn rotate_right(&mut self, root: usize) -> usize {
        let child = self.node(root).left.expect("left rotation child");
        self.node_mut(root).left = self.node(child).right;
        self.node_mut(child).right = Some(root);
        self.update_max(root);
        self.update_max(child);
        child
    }

    fn rotate_left(&mut self, root: usize) -> usize {
        let child = self.node(root).right.expect("right rotation child");
        self.node_mut(root).right = self.node(child).left;
        self.node_mut(child).left = Some(root);
        self.update_max(root);
        self.update_max(child);
        child
    }

    fn report(&self, root: Option<usize>, start: f64, end: f64, out: &mut Vec<Interval>) {
        let Some(root) = root else {
            return;
        };
        let node = self.node(root);
        if node.max_end <= start {
            return;
        }
        self.report(node.left, start, end, out);
        if node.interval.start < end {
            if node.interval.end > start {
                out.push(node.interval);
            }
            self.report(node.right, start, end, out);
        }
    }
}

fn mix64(mut value: u64) -> u64 {
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn interval(asset: usize, start: f64, end: f64) -> Interval {
        Interval {
            asset,
            start,
            end,
            duration: end - start,
            return_value: asset as f64,
        }
    }

    #[test]
    fn half_open_overlap_and_handle_reuse() {
        let mut index = IntervalIndex::default();
        let first = index.insert(interval(0, 0.0, 1.0));
        index.insert(interval(1, 1.0, 3.0));
        index.insert(interval(2, 0.0, 4.0));
        let mut matched: Vec<_> = index
            .overlapping(1.0, 2.0)
            .iter()
            .map(|x| x.asset)
            .collect();
        matched.sort_unstable();
        assert_eq!(matched, [1, 2]);
        index.remove(first).unwrap();
        assert!(index.remove(first).is_none());
        let reused = index.insert(interval(3, 10.0, 20.0));
        assert_eq!(first, reused);
        assert_eq!(index.len(), 3);
        assert_eq!(index.overlapping(9.0, 11.0)[0].asset, 3);
    }

    #[test]
    fn queries_and_removals_match_brute_force() {
        let mut index = IntervalIndex::default();
        let mut live = Vec::new();
        for id in 0..1000 {
            let start = (mix64(id + 29) % 2000) as f64 / 10.0;
            let length = (mix64(id + 103) % 200 + 1) as f64 / 10.0;
            let value = interval(id as usize, start, start + length);
            let handle = index.insert(value);
            live.push((handle, value));
        }
        for step in 0..100 {
            let start = step as f64 * 2.0;
            let end = start + 0.7;
            let mut expected: Vec<_> = live
                .iter()
                .filter(|(_, x)| x.start < end && x.end > start)
                .map(|(_, x)| x.asset)
                .collect();
            let mut actual: Vec<_> = index
                .overlapping(start, end)
                .iter()
                .map(|x| x.asset)
                .collect();
            expected.sort_unstable();
            actual.sort_unstable();
            assert_eq!(actual, expected);
            for _ in 0..3 {
                let (handle, _) = live.pop().unwrap();
                index.remove(handle).unwrap();
            }
        }
        assert_eq!(index.len(), live.len());
    }
}
