//! BOCPD + FOCuS confirmation ensemble.
//!
//! Wraps a [`BocpdDetector`] with a [`FocusDetector`] confirmation arm.
//! A BOCPD detection is kept iff it is **confident enough on its own**
//! OR **confirmed by FOCuS** within a tolerance window.
//!
//! ```text
//!     keep CP   ⇔   conf ≥ floor   ∨   ∃ FOCuS CP within ±tol of it
//! ```
//!
//! The pre-condition for either branch is a BOCPD emission; FOCuS does
//! not contribute novel detections, only suppresses solo low-confidence
//! BOCPD calls. Calibrate the confirmation threshold and tolerance
//! on representative fixtures before choosing an ensemble policy.
//!
//!
//!
//!

use crate::bocpd::BocpdDetector;
use crate::conformal::{MvScoredDetect, ScoredDetect};
use crate::detrend::{dominant_period_via_acf, seasonal_difference};
use crate::focus::FocusDetector;
use crate::nig::Nig;
use crate::predictive::Predictive;
use crate::ChangePoint;

/// Ensemble detector: BOCPD with FOCuS confirmation.
///
/// `confidence_floor` is the BOCPD-confidence threshold above which a
/// detection is kept regardless of FOCuS. `tolerance` is the maximum
/// step-distance between a BOCPD index and a FOCuS index for the latter
/// to count as confirmation.
///
pub struct EnsembleDetector<P: Predictive = Nig> {
    bocpd: BocpdDetector<P>,
    focus_threshold: f64,
    confidence_floor: f64,
    tolerance: usize,
    auto_detrend: bool,
}

impl EnsembleDetector<Nig> {
    /// Construct with BOCPD lambda + max run-length and the documented
    /// ensemble defaults. Uses the standard `Nig` prior; for non-default
    /// predictives (e.g. `BocpdDetector::<NigAr1>::with_prior(...)`),
    /// pass through [`EnsembleDetector::from_bocpd`].
    pub fn new(lambda: f64, max_run_length: usize) -> Self {
        Self::from_bocpd(BocpdDetector::new(lambda, max_run_length))
    }
}

impl<P: Predictive> EnsembleDetector<P> {
    /// Construct from an existing [`BocpdDetector<P>`] -- preserves any
    /// builder customisation (mass cutoff, β-divergence, prior) the
    /// caller has already applied. Generic over the predictive `P`.
    pub fn from_bocpd(bocpd: BocpdDetector<P>) -> Self {
        Self {
            bocpd,
            focus_threshold: 8.0,
            confidence_floor: 0.40,
            tolerance: 25,
            auto_detrend: false,
        }
    }

    pub fn with_focus_threshold(mut self, threshold: f64) -> Self {
        assert!(threshold > 0.0, "focus threshold must be > 0");
        self.focus_threshold = threshold;
        self
    }

    pub fn with_confidence_floor(mut self, floor: f64) -> Self {
        assert!(
            (0.0..=1.0).contains(&floor),
            "confidence floor must be in [0, 1], got {floor}"
        );
        self.confidence_floor = floor;
        self
    }

    pub fn with_tolerance(mut self, tolerance: usize) -> Self {
        self.tolerance = tolerance;
        self
    }

    ///
    ///
    pub fn with_auto_detrend(mut self, on: bool) -> Self {
        self.auto_detrend = on;
        self
    }

    /// Run the ensemble. The returned `ChangePoint` values are the
    /// **BOCPD** detections that survived the rule -- their `confidence`
    /// and `shift_sigma` come from BOCPD's posterior. FOCuS contributes
    /// only confirmation; its threshold-statistic is not exposed here.
    ///
    /// When [`with_auto_detrend`](Self::with_auto_detrend) is enabled
    /// and `cesura::detrend::dominant_period_via_acf` returns
    /// `Some(period)`, the input is replaced with its seasonal
    /// difference `y_t - y_{t-period}` before the recursion runs;
    /// emitted indices are shifted back into the original sample
    /// space. Aperiodic streams skip this step automatically.
    /// Multivariate ensemble: `BocpdDetector::detect_multivariate`
    /// (NIW + Mahalanobis whitening) confirmed by
    /// `FocusDetector::detect_multivariate` (per-dim FOCuS OR-union).
    /// Same `confidence_or_confirmed` rule as the univariate path.
    ///
    /// **Asymmetric arms.** BOCPD MV is the load-bearing detector --
    /// it catches the anti-correlated-shift case the per-dim FOCuS
    /// path misses (see `tests/correctness.rs::multivariate_detects_correlated_shift_per_dim_invisible`
    /// and the per-dim-OR-misses pin in `src/focus.rs::tests::mv_focus_per_dim_union_misses_anti_correlated_at_low_shift`).
    /// FOCuS-MV serves only as a CONFIRMATION filter, suppressing
    /// low-confidence BOCPD calls that no per-dim FOCuS supports. On
    /// shifts that BOCPD alone catches but per-dim FOCuS misses, the
    /// confidence-floor branch still passes them through.
    ///
    /// `auto_detrend` is **not** applied here -- `seasonal_difference`
    /// and `dominant_period_via_acf` are univariate-only. Multivariate
    /// inputs ignore the flag silently.
    pub fn detect_multivariate(&self, data: &[Vec<f64>]) -> Vec<ChangePoint> {
        let bocpd_cps = self.bocpd.detect_multivariate(data);
        let focus_cps = FocusDetector::new(self.focus_threshold)
            .detect_multivariate(data);
        let tol = self.tolerance as i64;
        bocpd_cps
            .into_iter()
            .filter(|cp| {
                if cp.confidence >= self.confidence_floor {
                    return true;
                }
                focus_cps
                    .iter()
                    .any(|f| (f.index as i64 - cp.index as i64).abs() <= tol)
            })
            .collect()
    }

    pub fn detect(&self, data: &[f64]) -> Vec<ChangePoint> {
        let (working_data, shift): (std::borrow::Cow<'_, [f64]>, usize) = if self.auto_detrend {
            match dominant_period_via_acf(data) {
                Some(p) if p > 0 && p < data.len() / 4 => {
                    (std::borrow::Cow::Owned(seasonal_difference(data, p)), p)
                }
                _ => (std::borrow::Cow::Borrowed(data), 0),
            }
        } else {
            (std::borrow::Cow::Borrowed(data), 0)
        };

        let bocpd_cps = self.bocpd.detect(&working_data);
        let mut focus = FocusDetector::new(self.focus_threshold);
        let focus_cps = focus.detect(&working_data);
        let tol = self.tolerance as i64;
        bocpd_cps
            .into_iter()
            .filter(|cp| {
                if cp.confidence >= self.confidence_floor {
                    return true;
                }
                focus_cps
                    .iter()
                    .any(|f| (f.index as i64 - cp.index as i64).abs() <= tol)
            })
            .map(|cp| ChangePoint {
                index: cp.index + shift,
                ..cp
            })
            .collect()
    }
}

impl<P: Predictive> ScoredDetect for EnsembleDetector<P> {
    /// Mirrors [`EnsembleDetector::detect`] but forwards the underlying
    /// BOCPD score through the FOCuS-confirmation filter. Score units
    /// follow [`BocpdDetector::detect_with_score`]: trigger-to-MAP-CP
    /// offset on the **working series**.
    ///
    fn detect_with_score(&self, data: &[f64]) -> Vec<(ChangePoint, f64)> {
        let (working_data, shift): (std::borrow::Cow<'_, [f64]>, usize) = if self.auto_detrend {
            match dominant_period_via_acf(data) {
                Some(p) if p > 0 && p < data.len() / 4 => {
                    (std::borrow::Cow::Owned(seasonal_difference(data, p)), p)
                }
                _ => (std::borrow::Cow::Borrowed(data), 0),
            }
        } else {
            (std::borrow::Cow::Borrowed(data), 0)
        };

        let bocpd_scored = self.bocpd.detect_with_score(&working_data);
        let mut focus = FocusDetector::new(self.focus_threshold);
        let focus_cps = focus.detect(&working_data);
        let tol = self.tolerance as i64;
        bocpd_scored
            .into_iter()
            .filter(|(cp, _)| {
                if cp.confidence >= self.confidence_floor {
                    return true;
                }
                focus_cps
                    .iter()
                    .any(|f| (f.index as i64 - cp.index as i64).abs() <= tol)
            })
            .map(|(cp, score)| {
                (
                    ChangePoint {
                        index: cp.index + shift,
                        ..cp
                    },
                    score,
                )
            })
            .collect()
    }
}

impl<P: Predictive> MvScoredDetect for EnsembleDetector<P> {
    /// Multivariate analogue of the `ScoredDetect` impl. Forwards
    /// BOCPD's per-CP score through the FOCuS-MV-confirmation filter.
    /// Score units follow `BocpdDetector::detect_multivariate_with_score`
    /// (trigger-to-MAP-CP-mass-peak offset, computed on the run-length
    /// posterior which is 1-D regardless of input dim). `auto_detrend`
    /// is ignored on MV input -- same as `detect_multivariate`.
    fn detect_multivariate_with_score(
        &self,
        data: &[Vec<f64>],
    ) -> Vec<(ChangePoint, f64)> {
        let bocpd_scored = self.bocpd.detect_multivariate_with_score(data);
        let focus_cps = FocusDetector::new(self.focus_threshold)
            .detect_multivariate(data);
        let tol = self.tolerance as i64;
        bocpd_scored
            .into_iter()
            .filter(|(cp, _)| {
                if cp.confidence >= self.confidence_floor {
                    return true;
                }
                focus_cps
                    .iter()
                    .any(|f| (f.index as i64 - cp.index as i64).abs() <= tol)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ensemble_clean_shift() {
        let data: Vec<f64> = std::iter::repeat_n(0.0, 100)
            .chain(std::iter::repeat_n(5.0, 100))
            .collect();
        let det = EnsembleDetector::new(200.0, 250);
        let cps = det.detect(&data);
        assert!(!cps.is_empty(), "expected at least one CP");
        assert!(
            cps.iter().any(|c| (c.index as i64 - 100).abs() < 25),
            "expected a CP near index 100, got {:?}",
            cps.iter().map(|c| c.index).collect::<Vec<_>>()
        );
    }

    #[test]
    fn ensemble_high_confidence_passes_solo() {
        // Strong shift: BOCPD confidence will exceed any reasonable floor
        // even if FOCuS is intentionally muted with a huge threshold.
        let data: Vec<f64> = std::iter::repeat_n(0.0, 100)
            .chain(std::iter::repeat_n(20.0, 100))
            .collect();
        let det = EnsembleDetector::new(200.0, 250)
            .with_focus_threshold(1e9) // FOCuS effectively disabled
            .with_confidence_floor(0.30);
        let cps = det.detect(&data);
        assert!(
            cps.iter().any(|c| (c.index as i64 - 100).abs() < 25),
            "high-confidence BOCPD CP must pass without FOCuS"
        );
    }

    #[test]
    fn ensemble_no_panic_on_short_input() {
        let det = EnsembleDetector::new(200.0, 50);
        let cps = det.detect(&[0.0; 5]);
        assert!(cps.is_empty());
    }

    // ── MV ensemble ─────────────────────────────────────────────

    #[test]
    fn mv_ensemble_clean_axis_aligned_shift() {
        // Both dims shift +5σ at t=100. BOCPD MV catches (high
        // confidence); per-dim FOCuS confirms. Either arm alone would
        // pass.
        let mut data: Vec<Vec<f64>> = (0..100)
            .map(|_| vec![0.0_f64, 0.0])
            .collect();
        data.extend((0..100).map(|_| vec![5.0_f64, 5.0]));
        let det = EnsembleDetector::new(200.0, 250);
        let cps = det.detect_multivariate(&data);
        assert!(!cps.is_empty(), "expected MV ensemble to fire on axis-aligned 5σ shift");
        assert!(
            cps.iter().any(|c| (c.index as i64 - 100).abs() < 25),
            "expected a CP near 100, got {:?}",
            cps.iter().map(|c| c.index).collect::<Vec<_>>()
        );
    }

    #[test]
    fn mv_ensemble_anti_correlated_passes_via_high_confidence() {
        // ρ=0.95, 1σ anti-correlated shift at t=200. Mahalanobis BOCPD
        // catches with high confidence (per `where_cesura_shines §1`);
        // per-dim FOCuS misses. The ensemble should still emit because
        // BOCPD's confidence ≥ floor=0.40. This is the load-bearing
        // claim: MV ensemble doesn't degrade BOCPD MV's anti-
        // correlated-shift performance just because FOCuS-MV is weak.
        use crate::eval::Rng;
        let rho = 0.95_f64;
        let mut rng = Rng::new(0xE001);
        let mut data: Vec<Vec<f64>> = Vec::with_capacity(400);
        for _ in 0..200 {
            let z0 = rng.normal(0.0, 1.0);
            let z1 = rng.normal(0.0, 1.0);
            data.push(vec![z0, rho * z0 + (1.0 - rho * rho).sqrt() * z1]);
        }
        for _ in 0..200 {
            let z0 = rng.normal(0.0, 1.0);
            let z1 = rng.normal(0.0, 1.0);
            data.push(vec![z0 + 0.5, rho * z0 + (1.0 - rho * rho).sqrt() * z1 - 0.5]);
        }
        let det = EnsembleDetector::new(200.0, 350);
        let cps = det.detect_multivariate(&data);
        assert!(
            cps.iter().any(|c| (c.index as i64 - 200).abs() < 50),
            "MV ensemble should inherit BOCPD's anti-correlated catch via confidence-floor branch; got {:?}",
            cps.iter().map(|c| c.index).collect::<Vec<_>>(),
        );
    }

    #[test]
    fn mv_ensemble_empty_input_no_panic() {
        let det = EnsembleDetector::new(200.0, 50);
        let cps = det.detect_multivariate(&[]);
        assert!(cps.is_empty());
    }

    #[test]
    fn mv_ensemble_scored_emits_pairs_for_high_conf_cps() {
        let mut data: Vec<Vec<f64>> = (0..100)
            .map(|_| vec![0.0_f64, 0.0])
            .collect();
        data.extend((0..100).map(|_| vec![5.0_f64, 5.0]));
        let det = EnsembleDetector::new(200.0, 250);
        let scored = det.detect_multivariate_with_score(&data);
        assert!(!scored.is_empty());
        // Score must be finite and ≥ 0 (it's a step-offset).
        assert!(scored.iter().all(|(_, s)| s.is_finite() && *s >= 0.0));
    }
}
