//! `FilterTickAggregator` -- k-of-d threshold detector primitive.
//!
//!
//!

use serde::{Deserialize, Serialize};

use super::MultiStreamChangePoint;

/// k-of-d threshold detector. Fires when ≥ `k` of `d` streams have
/// `|observation| > threshold` in a single step, subject to a
/// cooldown after each fire.
pub struct FilterTickAggregator {
    d: usize,
    k: usize,
    threshold: f64,
    cooldown: usize,
    last_emit: Option<usize>,
    step_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterTickAggregatorState {
    pub d: usize,
    pub k: usize,
    pub threshold: f64,
    pub cooldown: usize,
    pub last_emit: Option<usize>,
    pub step_count: usize,
}

impl FilterTickAggregator {
    ///
    /// # Panics
    /// - `d == 0` or `k == 0`
    /// - `k > d`
    /// - `threshold <= 0`
    pub fn new(d: usize, k: usize, threshold: f64) -> Self {
        assert!(d > 0, "d must be > 0");
        assert!(k > 0, "k must be > 0");
        assert!(k <= d, "k ({k}) must be ≤ d ({d})");
        assert!(threshold > 0.0, "threshold must be > 0, got {threshold}");
        Self {
            d,
            k,
            threshold,
            cooldown: 15,
            last_emit: None,
            step_count: 0,
        }
    }

    pub fn with_cooldown(mut self, c: usize) -> Self {
        self.cooldown = c;
        self
    }

    pub fn step(&mut self, observations: &[Vec<f64>]) -> Vec<MultiStreamChangePoint> {
        let mut out = Vec::new();
        for obs in observations {
            assert_eq!(
                obs.len(),
                self.d,
                "observation dim {} ≠ aggregator dim {}",
                obs.len(),
                self.d
            );
            let i = self.step_count;
            self.step_count += 1;

            let in_cd = matches!(
                self.last_emit,
                Some(le) if i.saturating_sub(le) <= self.cooldown
            );
            if in_cd {
                continue;
            }

            let mut active_indices: Vec<usize> = Vec::with_capacity(self.d);
            for (idx, &x) in obs.iter().enumerate() {
                if x.is_finite() && x.abs() > self.threshold {
                    active_indices.push(idx);
                }
            }
            if active_indices.len() >= self.k {
                out.push(MultiStreamChangePoint {
                    index: i,
                    confidence: 1.0,
                    streams: active_indices,
                    per_stream_weights: obs.clone(),
                });
                self.last_emit = Some(i);
            }
        }
        out
    }

    pub fn save_state(&self) -> FilterTickAggregatorState {
        FilterTickAggregatorState {
            d: self.d,
            k: self.k,
            threshold: self.threshold,
            cooldown: self.cooldown,
            last_emit: self.last_emit,
            step_count: self.step_count,
        }
    }

    pub fn restore(state: FilterTickAggregatorState) -> Self {
        Self {
            d: state.d,
            k: state.k,
            threshold: state.threshold,
            cooldown: state.cooldown,
            last_emit: state.last_emit,
            step_count: state.step_count,
        }
    }

    pub fn step_count(&self) -> usize {
        self.step_count
    }

    pub fn d(&self) -> usize {
        self.d
    }

    pub fn k(&self) -> usize {
        self.k
    }

    pub fn threshold(&self) -> f64 {
        self.threshold
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic(expected = "k (4) must be ≤ d (3)")]
    fn refuses_k_greater_than_d() {
        let _ = FilterTickAggregator::new(3, 4, 0.01);
    }

    #[test]
    #[should_panic(expected = "threshold must be > 0")]
    fn refuses_zero_threshold() {
        let _ = FilterTickAggregator::new(3, 2, 0.0);
    }

    #[test]
    fn fires_when_k_streams_cross_threshold() {
        let mut agg = FilterTickAggregator::new(3, 2, 0.05);
        // Step 0: 2 streams active (≥ k=2) → should fire.
        let fires = agg.step(&[vec![0.1, 0.06, 0.01]]);
        assert_eq!(fires.len(), 1);
        assert_eq!(fires[0].streams, vec![0, 1]);
        assert_eq!(fires[0].confidence, 1.0);
        assert_eq!(fires[0].per_stream_weights, vec![0.1, 0.06, 0.01]);
    }

    #[test]
    fn does_not_fire_under_threshold() {
        let mut agg = FilterTickAggregator::new(3, 2, 0.05);
        let fires = agg.step(&[vec![0.04, 0.03, 0.01]]);
        assert!(fires.is_empty(), "no stream above threshold → no fire");
    }

    #[test]
    fn does_not_fire_when_only_one_stream_active() {
        let mut agg = FilterTickAggregator::new(3, 2, 0.05);
        let fires = agg.step(&[vec![0.1, 0.01, 0.01]]);
        assert!(fires.is_empty(), "k=2 but only 1 active → no fire");
    }

    #[test]
    fn cooldown_blocks_consecutive_fires() {
        let mut agg = FilterTickAggregator::new(3, 2, 0.05).with_cooldown(5);
        // Step 0: fires.
        let f0 = agg.step(&[vec![0.1, 0.1, 0.0]]);
        assert_eq!(f0.len(), 1);
        // Steps 1-5: in cooldown, no fire even with above-threshold obs.
        let dummy: Vec<Vec<f64>> = (0..5).map(|_| vec![0.1, 0.1, 0.0]).collect();
        let f15 = agg.step(&dummy);
        assert!(f15.is_empty(), "cooldown=5 → 5 steps suppressed");
        // Step 6: cooldown elapsed (6 - 0 > 5), should fire again.
        let f6 = agg.step(&[vec![0.1, 0.1, 0.0]]);
        assert_eq!(f6.len(), 1);
    }

    #[test]
    fn nan_observations_excluded_from_active_count() {
        let mut agg = FilterTickAggregator::new(3, 2, 0.05);
        // Stream 0 is NaN (missing); streams 1 + 2 above threshold.
        let fires = agg.step(&[vec![f64::NAN, 0.1, 0.06]]);
        assert_eq!(fires.len(), 1);
        assert_eq!(fires[0].streams, vec![1, 2]);
    }

    #[test]
    fn nan_observations_can_block_fire_below_k() {
        let mut agg = FilterTickAggregator::new(3, 2, 0.05);
        // Stream 0 NaN, only stream 1 above; k=2 not met.
        let fires = agg.step(&[vec![f64::NAN, 0.1, 0.01]]);
        assert!(fires.is_empty(), "NaN excluded; only 1 active < k=2");
    }

    #[test]
    fn save_restore_round_trips() {
        let mut agg = FilterTickAggregator::new(3, 2, 0.05).with_cooldown(7);
        let _ = agg.step(&[vec![0.1, 0.1, 0.0], vec![0.0; 3], vec![0.0; 3]]);
        let state = agg.save_state();
        let json = serde_json::to_string(&state).unwrap();
        let restored: FilterTickAggregatorState = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.step_count, 3);
        assert_eq!(restored.cooldown, 7);
        assert_eq!(restored.last_emit, Some(0));
        let _ = FilterTickAggregator::restore(restored);
    }
}
