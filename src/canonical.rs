//! Convenience factories for multi-stream detectors.
//!
//! Parameters are starting points; calibrate them with representative data.

use crate::multistream::{FilterTickAggregator, SumCusumAggregator};
use crate::streaming::StreamingDetector;
/// Build `d` BOCPD streams (lambda 200, maximum run length 250) feeding a
/// sum-CUSUM aggregator with threshold 0.1. Each observation has `d` values.
///
/// ```no_run
/// let mut detector = cesura::canonical::recommended_streams(2);
/// let changes = detector.step(&vec![vec![0.0; 2]; 200]);
/// ```
pub fn recommended_streams(d: usize) -> SumCusumAggregator<StreamingDetector> {
    let streams: Vec<StreamingDetector> =
        (0..d).map(|_| StreamingDetector::new(200.0, 250)).collect();
    SumCusumAggregator::new(streams).with_threshold(0.1)
}
/// Build `d` streams with lambda 2000 and maximum run length 250.
/// The sum-CUSUM threshold is `0.3 * sqrt(d)`. This scaling is a heuristic;
/// dependent channels and different sampling rates require calibration.
#[must_use]
pub fn recommended_streams_tick(d: usize) -> SumCusumAggregator<StreamingDetector> {
    let streams: Vec<StreamingDetector> =
        (0..d).map(|_| StreamingDetector::new(2000.0, 250)).collect();
    let tau_base = 0.3;
    let tau = tau_base * (d as f64).sqrt();
    SumCusumAggregator::new(streams).with_threshold(tau)
}
/// Build a direct threshold filter that fires when at least `k` of `d`
/// observations exceed `threshold` in absolute value, with a 15-step cooldown.
/// This binary filter does not estimate change-point probabilities.
/// Missing observations may be passed as NaN and are excluded from the count.
#[must_use]
pub fn recommended_filter_tick(d: usize, k: usize, threshold: f64) -> FilterTickAggregator {
    FilterTickAggregator::new(d, k, threshold)
}
/// Build the tick-stream configuration with an explicit sum-CUSUM threshold.
/// Use representative stationary and shifted fixtures to choose `tau`.
#[must_use]
pub fn recommended_streams_tick_with_threshold(d: usize, tau: f64) -> SumCusumAggregator<StreamingDetector> {
    let streams: Vec<StreamingDetector> =
        (0..d).map(|_| StreamingDetector::new(2000.0, 250)).collect();
    SumCusumAggregator::new(streams).with_threshold(tau)
}

/// Bench-harness convenience over [`recommended_streams`].
///
/// Returns a [`SumCusumAggregatorAdapter`] labelled
/// `"SumCusum(CpProb, τ=0.1)"` so the canonical detector slots
/// into the same `CpDetector` trait as every other adapter in
/// `cesura::bench`. Behind `feature = "test-utils"` to keep the
/// production library decoupled from dev-only types.
///
/// For non-bench use, prefer [`recommended_streams`] directly.
#[cfg(feature = "test-utils")]
pub fn recommended_detector() -> crate::bench::multistream_adapter::SumCusumAggregatorAdapter {
    crate::bench::multistream_adapter::SumCusumAggregatorAdapter::with_map_streams(0.1)
        .label("SumCusum(CpProb, τ=0.1)")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::Rng;

    #[test]
    fn recommended_streams_tick_runs_on_synthetic_shift() {
        let mut rng = Rng::new(0xC0FFEEu64);
        let n = 300usize;
        let mut data: Vec<Vec<f64>> = Vec::with_capacity(n);
        for t in 0..n {
            let s = if t < 150 { 0.0 } else { 5.0 };
            data.push(vec![rng.normal(s, 1.0)]);
        }
        let mut agg = recommended_streams_tick(1);
        let fires = agg.step(&data);
        assert!(
            !fires.is_empty(),
            "recommended_streams_tick produced 0 fires on a 5σ-shift fixture"
        );
        let near_shift = fires.iter().any(|m| (m.index as i64 - 150).abs() <= 30);
        assert!(
            near_shift,
            "recommended_streams_tick got no fire near GT=150; fires={fires:?}"
        );
    }

    #[test]
    fn recommended_streams_runs_on_synthetic_shift() {
        let mut rng = Rng::new(0xCAFE_F00Du64);
        let n = 300usize;
        let mut data: Vec<Vec<f64>> = Vec::with_capacity(n);
        for t in 0..n {
            let s = if t < 150 { 0.0 } else { 5.0 };
            data.push(vec![rng.normal(s, 1.0), rng.normal(s, 1.0)]);
        }

        let mut agg = recommended_streams(2);
        let fires = agg.step(&data);
        assert!(
            !fires.is_empty(),
            "recommended_streams produced 0 fires on a 5σ-shift fixture"
        );
        let near_shift = fires.iter().any(|m| (m.index as i64 - 150).abs() <= 30);
        assert!(
            near_shift,
            "recommended_streams got no fire near GT=150; fires={fires:?}"
        );
    }

    #[cfg(feature = "test-utils")]
    #[test]
    fn recommended_detector_runs_on_synthetic_shift() {
        use crate::bench::{CpDetector, Fixture};
        let mut rng = Rng::new(0xCAFE_F00Du64);
        let n = 300usize;
        let mut data: Vec<Vec<f64>> = Vec::with_capacity(n);
        for t in 0..n {
            let s = if t < 150 { 0.0 } else { 5.0 };
            data.push(vec![rng.normal(s, 1.0), rng.normal(s, 1.0)]);
        }
        let fix = Fixture {
            name: "canonical-smoke".into(),
            version: 1,
            d: 2,
            data,
            epochs: None,
            ground_truth: vec![150],
            seed: None,
            margin: 30,
        };

        let det = recommended_detector();
        let cps = det.detect(&fix);
        assert!(
            !cps.is_empty(),
            "recommended_detector produced 0 CPs on a 5σ-shift fixture"
        );
        let near_shift = cps.iter().any(|c| (c.index as i64 - 150).abs() <= 30);
        assert!(
            near_shift,
            "recommended_detector got no CP near GT=150; cps={cps:?}"
        );
    }
}
