//! `CpDetector` trait -- the lowest common denominator across all
//! detector strategies. Univariate detectors handle `d=1` fixtures;
//! the trait does not split uni/multi (callers always get
//! `&[Vec<f64>]` via `Fixture::data`).
//!
//! Attribution: aggregator detectors (HC) carry per-stream provenance
//! ("which streams crossed the threshold") on each detection. The
//! attribution channel is opt-in via [`CpDetector::detect_full`] --
//! univariate / joint detectors return empty per-CP lists by default.

use crate::ChangePoint;
use crate::bench::fixture::Fixture;

/// Detection result with optional per-CP attribution.
///
/// `attribution[i]` lists stream indices (sorted ascending) that
/// contributed to `cps[i]`. Empty for univariate detectors and joint
/// aggregators (e.g. sum-CUSUM) where attribution is not separable.
/// Always aligned: `attribution.len() == cps.len()`.
#[derive(Debug, Clone, Default)]
pub struct DetectionResult {
    pub cps: Vec<ChangePoint>,
    pub attribution: Vec<Vec<usize>>,
}

impl DetectionResult {
    pub fn from_cps(cps: Vec<ChangePoint>) -> Self {
        let n = cps.len();
        Self {
            cps,
            attribution: vec![Vec::new(); n],
        }
    }
}

pub trait CpDetector {
    fn name(&self) -> String;
    fn detect(&self, fixture: &Fixture) -> Vec<ChangePoint>;

    /// Detection with attribution. Default impl wraps [`Self::detect`]
    /// with empty per-CP attribution -- correct for univariate
    /// detectors and for joint aggregators (sum-CUSUM) by their own
    /// contract. Override on aggregators that surface provenance (HC).
    fn detect_full(&self, fixture: &Fixture) -> DetectionResult {
        DetectionResult::from_cps(self.detect(fixture))
    }
}
