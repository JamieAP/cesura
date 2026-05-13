//! Multi-stream change-point aggregators for cesura.
//!
//!
//! - [`HcAggregator`] -- Higher-Criticism over per-stream scores.
//!   Sparse-aware (Gong-Kipnis-Xie 2024, arXiv:2409.15597). Output
//!   includes per-stream attribution so the user knows *which* streams
//!   crossed the threshold. Optimal when the change is sparse (k of d
//!   streams shift, k ≪ d, k unknown).
//!
//! - [`SumCusumAggregator`] -- sum of per-stream CUSUMs (Mei 2010,
//!   *Biometrika*). Joint detector, no per-stream attribution. Optimal
//!   when the change is dense (most streams shift together).
//!

mod filter_tick;
mod hc;
mod sum_cusum;

pub use filter_tick::{FilterTickAggregator, FilterTickAggregatorState};
pub use hc::{HcAggregator, HcAggregatorState};
pub use sum_cusum::{SumCusumAggregator, SumCusumAggregatorState};

use serde::{Deserialize, Serialize};

/// Per-step score from a univariate detector. **Convention: higher =
/// more change-evidence.** Implementors are responsible for emitting a
/// monotone score. HC applies an empirical rank transform; sum-CUSUM
/// selects its reference and threshold from `score_kind()`.
pub trait ScoreStream {
    /// Process one observation, return the per-step change-evidence
    /// score. Higher = more evidence.
    fn step_score(&mut self, x: f64) -> f64;

    /// What flavour of score this stream emits. Aggregators use this
    /// to check compatible score kinds and choose sum-CUSUM defaults.
    /// HC uses an empirical rank transform rather than an analytic mapping.
    fn score_kind(&self) -> ScoreKind;

    /// Total observations processed (NaN-skipped or not -- whatever
    /// the detector's own counter reports).
    fn step_count(&self) -> usize;
}

/// What flavour of score a [`ScoreStream`] emits. Determines the
/// neutral reference and threshold sum-CUSUM uses. HC uses empirical
/// ranks. All streams in a single aggregator must agree on score kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScoreKind {
    /// Bayes factor on aggregated short-run-length mass (`StreamingDetector`
    /// with `with_bayes_factor_rule`). Range: \[0, ∞). Illustrative analytic mapping:
    /// `p = 1 / (1 + score)` (Vovk/Sellke calibration). Sum-CUSUM
    /// neutral reference: 1.0 (BF=1 = equal posterior odds).
    BayesFactor,
    CpProbability,
}

/// Multi-stream change-point detection. Carries per-stream attribution
/// when the aggregator supports it (HC); empty otherwise (sum-CUSUM).
#[derive(Debug, Clone)]
pub struct MultiStreamChangePoint {
    /// Step index where the aggregator fired.
    pub index: usize,
    /// Aggregator confidence in \[0, 1\]. HC: normalised statistic over
    /// τ. sum-CUSUM: normalised cumulative sum over threshold.
    pub confidence: f64,
    /// Streams that contributed to this detection. Sorted ascending.
    /// Empty for joint detectors (sum-CUSUM) where attribution is
    /// not separable.
    pub streams: Vec<usize>,
    /// Per-stream change-evidence at the fire instant. Length = d for
    /// both aggregators.
    ///
    /// - **sum-CUSUM**: CUSUM W_i, snapshotted BEFORE the post-fire
    ///   reset. Higher = more sustained evidence.
    /// - **HC**: per-stream rank-based p-value at the fire's terminal
    ///   step. Lower = more evidence. The `streams` field carries the
    ///   index set that crossed below the HC threshold; this field
    ///   carries the underlying continuous score.
    ///
    /// Reporting-only -- not part of the detector logic. The Mei 2010
    /// asymptotic design says sum-CUSUM has no per-stream attribution,
    /// but operationally consumers need to know which stream drove a
    /// fire.
    pub per_stream_weights: Vec<f64>,
}

impl ScoreKind {
    /// Default neutral reference for sum-CUSUM: the score level where
    /// evidence is approximately balanced. Per-step contribution is
    /// `score - neutral_reference`; cumulative sum reset on negative.
    /// Calibrated against typical BOCPD output: BF ≈ 1 under H_0;
    /// `P(r=0) ≈ 1/(1+λ) ≈ 0.005` for λ ≈ 200.
    pub(crate) fn neutral_reference(self) -> f64 {
        match self {
            ScoreKind::BayesFactor => 1.0,
            ScoreKind::CpProbability => 0.005,
        }
    }

    ///
    pub(crate) fn default_sum_cusum_threshold(self) -> f64 {
        match self {
            ScoreKind::BayesFactor => 20.0,
            ScoreKind::CpProbability => 0.1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neutral_reference_per_kind() {
        assert_eq!(ScoreKind::BayesFactor.neutral_reference(), 1.0);
        // CpProbability ref calibrated to typical BOCPD H_0:
        // P(r=0) ≈ 1/(1+λ) for λ ≈ 200.
        assert_eq!(ScoreKind::CpProbability.neutral_reference(), 0.005);
    }
}
