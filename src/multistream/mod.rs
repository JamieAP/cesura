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

mod hc;
mod sum_cusum;

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
}

impl ScoreKind {
    /// Analytic score → p-value mapping. **Currently unused** -- HC
    /// uses rank-transform p-values (empirical CDF) which are robust
    /// to score-kind misspecification. Retained as a reference
    /// calibration for users who want analytic mappings; may be wired
    /// into a future HC variant that uses a fixed reference distribution
    /// instead of a sliding window.
    #[allow(dead_code)]
    pub(crate) fn to_pvalue(self, score: f64) -> f64 {
        let raw = match self {
            // Vovk/Sellke approximate Bayes-factor-to-p-value:
            // p = 1 / (1 + BF). Conservative; bounded above by 1.
            ScoreKind::BayesFactor => 1.0 / (1.0 + score.max(0.0)),
            // cp_probs ∈ [0, 1] already; complement is the p-value.
            ScoreKind::CpProbability => (1.0 - score).max(0.0),
        };
        raw.clamp(1e-12, 1.0)
    }

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
    fn pvalue_mapping_bf_monotone_decreasing() {
        let p_low = ScoreKind::BayesFactor.to_pvalue(0.5);
        let p_mid = ScoreKind::BayesFactor.to_pvalue(2.0);
        let p_high = ScoreKind::BayesFactor.to_pvalue(10.0);
        assert!(p_low > p_mid, "p({}) ≤ p({})", 0.5, 2.0);
        assert!(p_mid > p_high, "p({}) ≤ p({})", 2.0, 10.0);
    }

    #[test]
    fn pvalue_mapping_cp_monotone_decreasing() {
        let p_low = ScoreKind::CpProbability.to_pvalue(0.05);
        let p_high = ScoreKind::CpProbability.to_pvalue(0.95);
        assert!(p_low > p_high);
    }

    #[test]
    fn pvalue_mapping_floor_avoids_zero() {
        let p = ScoreKind::CpProbability.to_pvalue(1.0);
        assert!(p >= 1e-12, "p must be ≥ ε floor, got {p}");
        let p = ScoreKind::BayesFactor.to_pvalue(f64::INFINITY);
        assert!(p >= 1e-12);
    }

    #[test]
    fn neutral_reference_per_kind() {
        assert_eq!(ScoreKind::BayesFactor.neutral_reference(), 1.0);
        // CpProbability ref calibrated to typical BOCPD H_0:
        // P(r=0) ≈ 1/(1+λ) for λ ≈ 200.
        assert_eq!(ScoreKind::CpProbability.neutral_reference(), 0.005);
    }
}
