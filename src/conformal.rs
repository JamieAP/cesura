//! Calibrated empirical timing intervals for change-point detectors.
//!
//! [`ConformalCpWrapper`] wraps any [`ScoredDetect`] and emits a
//! [`ConformalCp`] per detection, augmenting the underlying
//! [`ChangePoint`] with a `timing_interval: (i64, i64)` derived from
//! the empirical-quantile span of a rolling FIFO calibration buffer
//! of non-conformity scores at the chosen `coverage` level (default
//! 90%).
//!
//! # Example
//!
//! ```
//! use cesura::{BocpdDetector, ConformalCpWrapper};
//!
//! let data: Vec<f64> = std::iter::repeat(0.0).take(100)
//!     .chain(std::iter::repeat(5.0).take(100))
//!     .collect();
//!
//! let mut wrapper = ConformalCpWrapper::new(BocpdDetector::new(200.0, 250))
//!     .with_coverage(0.9)
//!     .with_calibration_capacity(8);
//! let cps = wrapper.detect(&data);
//!
//! assert!(!cps.is_empty());
//! let cp = &cps[0];
//! assert!((cp.cp.index as i64 - 100).abs() < 15);
//! let (lo, hi) = cp.timing_interval;
//! assert!(lo <= hi);
//! ```
//!
//! ## Coverage semantics
//!
//! The wrapper ships **marginal-across-regimes** calibration: a single
//! rolling-FIFO buffer accumulates non-conformity scores from every
//! emitted CP, regardless of which regime they originated from.
//! Sun & Yu 2025 (CPTC, arXiv:2509.02844) §4.1 names this the naive
//! case for online streams whose data distribution shifts; their fix
//! is per-regime ACI-style adaptation for forecast intervals (the
//! method is for *forecasting* prediction intervals on time-series
//! with CPs, not timing-uncertainty for CP detectors -- it is
//! adjacent prior art, not a direct timing-CP follow-up). One
//! mitigation specific to this wrapper: scores enter the buffer only
//! at *triggered* CPs, so quiet-period contamination is auto-rejected;
//! the residual hazard is mixing scores across regimes of differing
//! signal magnitude. Marginal-coverage parity across an SNR
//! transition is pinned by
//! `tests/statistical.rs::conformal_wrapper_coverage_under_snr_transition`.
//! Per-regime conditioning is a possible extension.
//!
//! Bayesian dual: arXiv:2508.06385 (Aug 2025) proposes a credible
//! interval on BOCPD's MAP run length `r* = argmax_r p(r_t = r | y_{1:t})`
//! directly from the posterior, avoiding any calibration buffer.
//! Coverage there is posterior, conditional on model correctness
//! rather than score exchangeability.
//!
//! ## Score semantics
//!
//! [`ScoredDetect`]'s scalar must be on a scale comparable across CPs
//! from the same detector instance. For [`crate::BocpdDetector`] the
//! score is the **trigger-to-CP-mass-peak offset**
//! `i − argmax_{k ∈ lookback} P(r_k = 0 | y_{1:k}) ≥ 0`, where the
//! lookback is the cooldown window. This is the time step within the
//! lookback at which BOCPD's per-step run-length-zero posterior was
//! largest -- i.e. the model's **MAP CP-trigger time** within the
//! window. (This is *not* the standard MAP run length
//! `r* = argmax_r p(r_t = r | y_{1:t})`; the score takes argmax over
//! `t` at fixed `r = 0`, not over `r` at fixed `t`. The two are
//! related but distinct quantities -- see Adams & MacKay 2007 for
//! the run-length posterior, and arXiv:2508.06385 for the Bayesian
//! credible-interval angle.) Exchangeability across regimes is an assumption;
//! synthetic regression tests do not establish a statistical coverage guarantee
//! for arbitrary streams.
//!
//! The wrapper's `timing_interval` is `(i − q_high, i − q_low)`, an
//! absolute-index range that brackets the MAP CP-trigger time, not
//! the true latent CP. The two coincide when the model is well-
//! specified; on misspecified within-regime distributions they
//! differ by a model-bias offset that the symmetric unconditional
//! calibration here does not correct.

use crate::ChangePoint;

/// Detector trait the conformal wrapper consumes. Each emitted
/// change point comes paired with a non-conformity score.
///
/// The score must be on a scale comparable across CPs from the **same**
/// detector instance. Detectors with different score conventions must
/// not share a calibration buffer.
pub trait ScoredDetect {
    /// Run detection. For each emitted CP, return the non-conformity
    /// score alongside it. Score units are detector-defined.
    fn detect_with_score(&self, data: &[f64]) -> Vec<(ChangePoint, f64)>;
}

/// Multivariate analogue of [`ScoredDetect`]. Wraps a detector that
/// consumes `&[Vec<f64>]` (rows = observations, columns = dimensions)
/// and emits the same `(ChangePoint, score)` pair shape.
///
/// Scores must remain comparable across CPs from the same detector
/// instance, the same way univariate scores do. The conformal
/// exchangeability assumption is on the score sequence and is
/// dimension-agnostic -- BOCPD's run-length posterior is 1-D
/// regardless of input dimensionality, so the score's interpretation
/// (the trigger-to-CP-mass-peak offset) carries over from the
/// univariate path unchanged.
pub trait MvScoredDetect {
    fn detect_multivariate_with_score(
        &self,
        data: &[Vec<f64>],
    ) -> Vec<(ChangePoint, f64)>;
}

/// Change point with a calibrated empirical timing interval.
///
/// `timing_interval` is `(low, high)` in absolute-index space:
/// `low = cp.index − q_high`, `high = cp.index − q_low`, where
/// `(q_low, q_high)` is the empirical quantile range at the chosen
/// coverage over the wrapper's rolling-FIFO calibration buffer of
/// non-conformity scores. See module docs for coverage semantics
/// and the score's reference point.
#[derive(Debug, Clone)]
pub struct ConformalCp {
    pub cp: ChangePoint,
    pub timing_interval: (i64, i64),
    pub coverage: f64,
}

/// Wrap a [`ScoredDetect`] with rolling-quantile timing intervals.
///
/// Defaults: `coverage = 0.9`, `calibration_capacity = 500`.
pub struct ConformalCpWrapper<D> {
    inner: D,
    calibration: RingBuffer,
    coverage: f64,
}

impl<D> ConformalCpWrapper<D> {
    pub fn new(inner: D) -> Self {
        Self {
            inner,
            calibration: RingBuffer::new(500),
            coverage: 0.9,
        }
    }

    /// Set the nominal coverage in `(0, 1)`.
    ///
    /// # Panics
    /// Panics if `c` is not in `(0, 1)`.
    pub fn with_coverage(mut self, c: f64) -> Self {
        assert!(c > 0.0 && c < 1.0, "coverage must be in (0, 1), got {c}");
        self.coverage = c;
        self
    }

    /// Set the calibration buffer capacity. The buffer is rolling FIFO
    /// once full; older scores age out.
    ///
    /// # Panics
    /// Panics if `n == 0`.
    pub fn with_calibration_capacity(mut self, n: usize) -> Self {
        assert!(n > 0, "calibration capacity must be > 0");
        self.calibration = RingBuffer::new(n);
        self
    }

    /// Run the inner detector and emit calibrated timing intervals.
    /// Online split-conformal-style: each emission's interval is
    /// computed over the calibration buffer of **prior** scores; the
    /// new score is pushed *after* the interval is queried, so it
    /// never participates in its own bracket. The first emission's
    /// buffer is empty and yields the degenerate `(cp.index,
    /// cp.index)` interval; widths grow as the buffer fills and
    /// stabilise once it reaches capacity.
    /// Shared post-detection conformal-quantile loop. Pulled out so
    /// both `detect` (univariate, on `D: ScoredDetect`) and
    /// `detect_multivariate` (on `D: MvScoredDetect`) share identical
    /// calibration logic.
    fn consume_scored(&mut self, scored: Vec<(ChangePoint, f64)>) -> Vec<ConformalCp> {
        let mut out = Vec::with_capacity(scored.len());
        for (cp, score) in scored {
            let (q_low, q_high) = self.calibration.quantile_interval(self.coverage);
            let timing_interval = (
                cp.index as i64 - q_high.round() as i64,
                cp.index as i64 - q_low.round() as i64,
            );
            out.push(ConformalCp {
                cp,
                timing_interval,
                coverage: self.coverage,
            });
            self.calibration.push(score);
        }
        out
    }
}

impl<D: ScoredDetect> ConformalCpWrapper<D> {
    /// Run the inner detector and emit calibrated timing intervals.
    /// Online split-conformal-style: each emission's interval is
    /// computed over the calibration buffer of **prior** scores; the
    /// new score is pushed *after* the interval is queried, so it
    /// never participates in its own bracket. The first emission's
    /// buffer is empty and yields the degenerate `(cp.index,
    /// cp.index)` interval; widths grow as the buffer fills and
    /// stabilise once it reaches capacity.
    pub fn detect(&mut self, data: &[f64]) -> Vec<ConformalCp> {
        let scored = self.inner.detect_with_score(data);
        self.consume_scored(scored)
    }
}

/// Multivariate dispatch. Same wrapper struct, separate `impl` block
/// so a detector that implements `MvScoredDetect` (but not `ScoredDetect`)
/// can still be wrapped, and a detector that implements both -- like
/// `BocpdDetector` -- gets `detect` and `detect_multivariate` on the
/// same wrapper instance.
///
/// The calibration buffer is shared between the two paths: scores
/// from univariate and multivariate emissions go into the same FIFO.
/// In practice you'd run a `ConformalCpWrapper` on either univariate
/// or multivariate data, not both -- the buffer-scope warning in the
/// `ScoredDetect` doc applies here too: don't mix score conventions
/// in one wrapper instance.
impl<D: MvScoredDetect> ConformalCpWrapper<D> {
    /// Run multivariate detection and emit calibrated timing intervals.
    /// Same online split-conformal-style update as the univariate
    /// `detect`: each emission's interval is computed over the
    /// calibration buffer of *prior* scores; the new score is pushed
    /// after the interval is queried, so it never participates in its
    /// own bracket.
    pub fn detect_multivariate(&mut self, data: &[Vec<f64>]) -> Vec<ConformalCp> {
        let scored = self.inner.detect_multivariate_with_score(data);
        self.consume_scored(scored)
    }
}

/// Rolling-FIFO buffer with a sorted side-vector. `O(log n)`
/// binary-search to find insert/remove positions plus an `O(n)`
/// `Vec::insert`/`Vec::remove` shift; quantile lookup is `O(1)`.
/// Linear shifts may be expensive at large capacity; alternative data
/// structures require profiling on the intended workload.
struct RingBuffer {
    cap: usize,
    fifo: std::collections::VecDeque<f64>,
    sorted: Vec<f64>,
}

impl RingBuffer {
    fn new(cap: usize) -> Self {
        Self {
            cap,
            fifo: std::collections::VecDeque::with_capacity(cap),
            sorted: Vec::with_capacity(cap),
        }
    }

    fn push(&mut self, x: f64) {
        if self.fifo.len() == self.cap {
            let old = self.fifo.pop_front().expect("non-empty when at cap");
            let pos = self
                .sorted
                .binary_search_by(|v| v.total_cmp(&old))
                .expect("invariant: every fifo element is in sorted");
            self.sorted.remove(pos);
        }
        self.fifo.push_back(x);
        let idx = self
            .sorted
            .binary_search_by(|v| v.total_cmp(&x))
            .unwrap_or_else(|i| i);
        self.sorted.insert(idx, x);
    }

    /// Returns `(q_α/2, q_{1-α/2})` over the buffer at coverage `c`,
    /// using nearest-rank index. Empty buffer returns `(0.0, 0.0)`.
    fn quantile_interval(&self, c: f64) -> (f64, f64) {
        if self.sorted.is_empty() {
            return (0.0, 0.0);
        }
        let n = self.sorted.len();
        let alpha = 1.0 - c;
        let lo_idx = ((alpha / 2.0) * n as f64).floor() as usize;
        let hi_raw = ((1.0 - alpha / 2.0) * n as f64).ceil() as usize;
        let hi_idx = hi_raw.saturating_sub(1).min(n - 1);
        let lo_idx = lo_idx.min(n - 1);
        (self.sorted[lo_idx], self.sorted[hi_idx])
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.fifo.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rolling_quantile_matches_sort() {
        let cap = 200;
        let mut buf = RingBuffer::new(cap);
        let mut state = 0xC0FFEEu64;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f64) / ((1u64 << 31) as f64)
        };
        let mut all: Vec<f64> = Vec::new();
        for _ in 0..(cap + 50) {
            let x = next();
            buf.push(x);
            all.push(x);
        }
        let recent: Vec<f64> = all.iter().rev().take(cap).rev().copied().collect();
        let mut sorted = recent.clone();
        sorted.sort_by(|a, b| a.total_cmp(b));
        let n = sorted.len();
        for &c in &[0.5_f64, 0.8, 0.9, 0.95] {
            let (lo, hi) = buf.quantile_interval(c);
            let alpha = 1.0 - c;
            let lo_idx = ((alpha / 2.0) * n as f64).floor() as usize;
            let hi_raw = ((1.0 - alpha / 2.0) * n as f64).ceil() as usize;
            let hi_idx = hi_raw.saturating_sub(1).min(n - 1);
            assert_eq!(lo, sorted[lo_idx], "lo at coverage {c}");
            assert_eq!(hi, sorted[hi_idx], "hi at coverage {c}");
        }
    }

    #[test]
    fn ring_buffer_evicts_oldest_fifo() {
        let mut buf = RingBuffer::new(3);
        for v in [1.0, 2.0, 3.0, 4.0] {
            buf.push(v);
        }
        let fifo: Vec<f64> = buf.fifo.iter().copied().collect();
        assert_eq!(fifo, vec![2.0, 3.0, 4.0]);
        assert_eq!(buf.sorted, vec![2.0, 3.0, 4.0]);
        assert_eq!(buf.len(), 3);
    }

    #[test]
    #[should_panic(expected = "calibration capacity must be > 0")]
    fn with_calibration_capacity_zero_panics() {
        struct Stub;
        impl ScoredDetect for Stub {
            fn detect_with_score(&self, _: &[f64]) -> Vec<(ChangePoint, f64)> {
                vec![]
            }
        }
        let _ = ConformalCpWrapper::new(Stub).with_calibration_capacity(0);
    }

    #[test]
    #[should_panic(expected = "coverage must be in (0, 1)")]
    fn with_coverage_zero_panics() {
        struct Stub;
        impl ScoredDetect for Stub {
            fn detect_with_score(&self, _: &[f64]) -> Vec<(ChangePoint, f64)> {
                vec![]
            }
        }
        let _ = ConformalCpWrapper::new(Stub).with_coverage(0.0);
    }

    #[test]
    #[should_panic(expected = "coverage must be in (0, 1)")]
    fn with_coverage_one_panics() {
        struct Stub;
        impl ScoredDetect for Stub {
            fn detect_with_score(&self, _: &[f64]) -> Vec<(ChangePoint, f64)> {
                vec![]
            }
        }
        let _ = ConformalCpWrapper::new(Stub).with_coverage(1.0);
    }

    /// MV stub for the wrapper-impl-bound test. Confirms a detector
    /// that ONLY implements `MvScoredDetect` (no `ScoredDetect`) can
    /// still be wrapped and called via `detect_multivariate`.
    struct MvStub {
        emits: Vec<(usize, f64)>,
    }

    impl MvScoredDetect for MvStub {
        fn detect_multivariate_with_score(
            &self,
            _data: &[Vec<f64>],
        ) -> Vec<(ChangePoint, f64)> {
            self.emits
                .iter()
                .map(|&(idx, score)| {
                    (
                        ChangePoint {
                            index: idx,
                            confidence: 0.5,
                            shift_sigma: 1.0,
                        },
                        score,
                    )
                })
                .collect()
        }
    }

    #[test]
    fn mv_wrapper_calibrates_intervals_from_prior_scores() {
        let stub = MvStub {
            emits: vec![(100, 0.0), (200, 1.0), (300, 2.0), (400, 3.0)],
        };
        let mut wrapper = ConformalCpWrapper::new(stub)
            .with_coverage(0.9)
            .with_calibration_capacity(10);
        let dummy: Vec<Vec<f64>> = vec![vec![0.0]; 500];
        let cps = wrapper.detect_multivariate(&dummy);
        assert_eq!(cps.len(), 4);
        // First emission: empty buffer → degenerate (cp.index, cp.index).
        assert_eq!(cps[0].timing_interval, (100, 100));
        // Subsequent emissions: non-degenerate intervals derived from
        // prior scores; the new score is pushed AFTER the interval is
        // queried so it never participates in its own bracket.
        assert!(
            cps[3].timing_interval.0 <= cps[3].timing_interval.1,
            "interval lo {} should be ≤ hi {}",
            cps[3].timing_interval.0,
            cps[3].timing_interval.1
        );
    }

    #[test]
    fn mv_wrapper_compiles_without_scored_detect() {
        // This is a compile-time check: MvStub does NOT implement
        // ScoredDetect (no impl above), only MvScoredDetect. If the
        // wrapper still required ScoredDetect, this wouldn't compile.
        let stub = MvStub { emits: vec![] };
        let wrapper = ConformalCpWrapper::new(stub).with_coverage(0.5);
        let _ = wrapper; // suppress unused
    }

    #[test]
    fn ring_buffer_handles_duplicates() {
        let mut buf = RingBuffer::new(4);
        for v in [1.0, 2.0, 2.0, 3.0, 2.0] {
            buf.push(v);
        }
        let fifo: Vec<f64> = buf.fifo.iter().copied().collect();
        assert_eq!(fifo, vec![2.0, 2.0, 3.0, 2.0]);
        let mut sorted = fifo.clone();
        sorted.sort_by(|a, b| a.total_cmp(b));
        assert_eq!(buf.sorted, sorted);
    }
}
