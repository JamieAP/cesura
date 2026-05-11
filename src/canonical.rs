//!
//!
//!
//!
//! # Factories
//!

use crate::multistream::SumCusumAggregator;
use crate::streaming::StreamingDetector;

/// Cesura's canonical recommended aggregator for multivariate streams.
///
/// Builds `d` per-channel [`StreamingDetector`]s (λ = 200, max_rl =
/// 250, NIG predictive, no Bayes-factor rule) and wires them into a
/// [`SumCusumAggregator`] with threshold `τ = 0.1`. Drive it by
/// calling `step(&data)` where `data` is a slice of `d`-dim
/// observations.
///
///
///
/// # Example
///
/// ```no_run
/// use cesura::canonical::recommended_streams;
///
/// let mut agg = recommended_streams(4);  // d = 4 channels
/// let data: Vec<Vec<f64>> = vec![vec![0.0; 4]; 200];
/// let cps = agg.step(&data);
/// ```
pub fn recommended_streams(d: usize) -> SumCusumAggregator<StreamingDetector> {
    let streams: Vec<StreamingDetector> =
        (0..d).map(|_| StreamingDetector::new(200.0, 250)).collect();
    SumCusumAggregator::new(streams).with_threshold(0.1)
}

/// Cesura's recommended aggregator for **1s-tick microstructure** streams.
///
/// Same architecture as [`recommended_streams`] -- per-channel
/// [`StreamingDetector`] feeding a [`SumCusumAggregator`] over
/// `ScoreKind::CpProbability` -- but with parameters tuned for the
/// 1-second log-return regime where the canonical macro defaults
/// (τ=0.1, λ=200) saturate the confidence histogram. Builds `d`
/// streams with λ = 2000, max_rl = 250, threshold τ = 0.3.
///
///
///
///
///
///
///
/// # Use this when
///
/// - Per-second (or sub-second) bars over crypto/equity tick data.
/// - `FeatureKind::LogReturn` or similar narrow-band per-bar scalar.
///
/// For hourly / daily macro bars, prefer [`recommended_streams`].
#[must_use]
pub fn recommended_streams_tick(d: usize) -> SumCusumAggregator<StreamingDetector> {
    let streams: Vec<StreamingDetector> =
        (0..d).map(|_| StreamingDetector::new(2000.0, 250)).collect();
    SumCusumAggregator::new(streams).with_threshold(0.3)
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
