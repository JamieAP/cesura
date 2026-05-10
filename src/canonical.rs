//!
//!
//!

use crate::bench::multistream_adapter::SumCusumAggregatorAdapter;

///
/// Returns a configured [`SumCusumAggregatorAdapter`] over per-channel
/// `BocpdDetector` streams (NIG predictive, MAP-rule per-channel
/// CP-probability outputs aggregated via sum-CUSUM with threshold
/// `τ = 0.1`). Implements [`crate::bench::CpDetector`] so it slots
/// into the existing bench harness.
///
/// # Example
///
/// ```no_run
/// use cesura::canonical::recommended_detector;
/// use cesura::bench::CpDetector;
/// # use cesura::bench::Fixture;
/// # let fixture = Fixture { name: "demo".into(), version: 1, d: 4,
/// #   data: vec![vec![0.0; 4]; 200], epochs: None,
/// #   ground_truth: vec![100], seed: None, margin: 30 };
///
/// let detector = recommended_detector();
/// let cps = detector.detect(&fixture);
/// ```
///
///
///
pub fn recommended_detector() -> SumCusumAggregatorAdapter {
    SumCusumAggregatorAdapter::with_map_streams(0.1).label("SumCusum(CpProb, τ=0.1)")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bench::{CpDetector, Fixture};
    use crate::eval::Rng;

    #[test]
    fn recommended_detector_runs_on_synthetic_shift() {
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
            "recommended detector produced 0 CPs on a 5σ-shift fixture"
        );
        let near_shift = cps.iter().any(|c| (c.index as i64 - 150).abs() <= 30);
        assert!(
            near_shift,
            "recommended detector got no CP near GT=150; cps={cps:?}"
        );
    }
}
