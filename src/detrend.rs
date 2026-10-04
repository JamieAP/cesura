//! Seasonal-trend decomposition for time series detrending.
//!
//! Reduces predictable periodic structure such as daily traffic cycles
//! and batch schedules before change-point detection.
//!
//! Algorithm: Theil-Sen linear trend (robust to step changes) + median-
//! based seasonal estimation (STL-lite). O(n²) fit at small n; pruned
//! pair sampling above 600 observations. O(k) incremental update.
//!
//! Yoshizawa (2022, arXiv:2201.02325) tackles the same baseline-shift
//! problem inside BOCPD itself by reinitialising the NIG posterior on
//! detected CP. cesura's approach is complementary -- detrending
//! happens at the input layer; Yoshizawa's reset happens at the
//! posterior layer. The two compose; users monitoring drifting
//! baselines should consider applying both.

use serde::{Deserialize, Serialize};

/// Seasonal + trend decomposition model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Detrender {
    /// Period length in observations (e.g., 1440 for daily at 1m step).
    period: usize,
    /// Estimated seasonal component per period offset.
    seasonal: Vec<f64>,
    /// Linear trend coefficient (slope per observation).
    trend: f64,
    /// Intercept (mean of detrended data).
    intercept: f64,
    /// Number of complete periods seen.
    periods_seen: usize,
    /// Total samples processed (for correct trend baseline in update).
    #[serde(default)]
    samples_seen: usize,
}

impl Detrender {
    /// Fit a seasonal + trend model from historical data.
    ///
    /// Requires at least one full period of data. With less, returns
    /// a no-op detrender (seasonal = zeros, trend = 0).
    pub fn fit(data: &[f64], period: usize) -> Self {
        if period == 0 || data.len() < period {
            return Self {
                period: period.max(1),
                seasonal: vec![0.0; period.max(1)],
                trend: 0.0,
                intercept: 0.0,
                periods_seen: 0,
                samples_seen: 0,
            };
        }

        // Step 1: Theil-Sen slope. Median of pairwise slopes -- robust to
        // step changes (a least-squares fit absorbs the step into the slope).
        let trend = theil_sen_slope(data);

        // Step 2: Remove trend
        let detrended: Vec<f64> = data
            .iter()
            .enumerate()
            .map(|(i, &y)| y - trend * i as f64)
            .collect();

        // Step 3: Estimate seasonal component via median per offset
        let mut seasonal = vec![0.0; period];
        for (offset, s) in seasonal.iter_mut().enumerate() {
            let mut values: Vec<f64> = detrended
                .iter()
                .skip(offset)
                .step_by(period)
                .copied()
                .collect();
            if values.is_empty() {
                continue;
            }
            values.sort_by(|a, b| a.total_cmp(b));
            *s = median(&values);
        }

        // Step 4: Center seasonal so mean = 0, absorb offset into intercept
        let seasonal_mean = seasonal.iter().sum::<f64>() / seasonal.len() as f64;
        for s in &mut seasonal {
            *s -= seasonal_mean;
        }

        // Intercept: the overall level after removing trend at index 0
        let intercept = seasonal_mean;
        let periods_seen = data.len() / period;

        Self {
            period,
            seasonal,
            trend,
            intercept,
            periods_seen,
            samples_seen: data.len(),
        }
    }

    /// Remove seasonal + trend components, returning residuals.
    ///
    /// `start_offset` is the *absolute* position of `data[0]` measured from
    /// the start of the fit window (e.g., for a continuation immediately
    /// following the fit data, `start_offset = fit_len`). Used for both
    /// seasonal phase (`(start_offset + i) % period`) and trend baseline
    /// (`trend * (start_offset + i)`). Pass `0` when detrending the same
    /// data the model was fit on.
    pub fn detrend(&self, data: &[f64], start_offset: usize) -> Vec<f64> {
        data.iter()
            .enumerate()
            .map(|(i, &y)| {
                let abs_pos = start_offset + i;
                let offset = abs_pos % self.period;
                y - self.seasonal[offset] - self.trend * abs_pos as f64 - self.intercept
            })
            .collect()
    }

    /// Update the seasonal model incrementally with new observations.
    ///
    /// Uses exponential moving average to blend new seasonal estimates
    /// with existing ones. Alpha controls adaptation speed.
    pub fn update(&mut self, new_points: &[f64], start_offset: usize) {
        if new_points.is_empty() {
            return;
        }
        // Backward compat: reconstruct samples_seen from periods_seen if not set
        if self.samples_seen == 0 && self.periods_seen > 0 {
            self.samples_seen = self.periods_seen * self.period;
        }
        // Blend factor: trust new data more when we have little history
        let alpha = if self.periods_seen < 3 { 0.3 } else { 0.1 };

        for (i, &y) in new_points.iter().enumerate() {
            let offset = (start_offset + i) % self.period;
            let abs_pos = self.samples_seen + i;
            let residual = y - self.trend * abs_pos as f64 - self.intercept;
            self.seasonal[offset] = (1.0 - alpha) * self.seasonal[offset] + alpha * residual;
        }

        self.samples_seen += new_points.len();
        self.periods_seen = self.samples_seen / self.period;
    }

    /// Detrend by seasonal differencing: `y_t - y_{t-period}`.
    ///
    /// Recommended over [`Detrender::detrend`] (median + Theil-Sen) for use
    /// inputs to a change-point detector, because differencing converts a
    /// step regime change into a clean transient pulse without requiring a
    /// fitted seasonal model. Returns `data.len() - period` values; the
    /// first `period` observations are consumed.
    pub fn detrend_diff(&self, data: &[f64]) -> Vec<f64> {
        seasonal_difference(data, self.period)
    }

    /// Whether the detrender has enough data to be meaningful.
    pub fn is_fitted(&self) -> bool {
        self.periods_seen >= 1
    }

    /// Measure how much variance the seasonal model explains.
    /// Returns a ratio in [0, 1] -- higher means stronger seasonality.
    /// Use this to decide whether detrending is worthwhile for a signal.
    pub fn seasonal_strength(&self, data: &[f64]) -> f64 {
        if data.len() < self.period || self.seasonal.iter().all(|&s| s.abs() < 1e-10) {
            return 0.0;
        }
        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let total_var = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>();
        if total_var < 1e-20 {
            return 0.0;
        }
        let residuals = self.detrend(data, 0);
        let resid_var = residuals.iter().map(|x| x.powi(2)).sum::<f64>();
        (1.0 - resid_var / total_var).max(0.0)
    }

    /// The period length.
    pub fn period(&self) -> usize {
        self.period
    }
}

// ── Period detection ─────────────────────────────────────────

/// Estimate the dominant period via autocorrelation peak.
///
/// Searches lags in `[min_lag, max_lag]`, returning the lag with the
/// highest local-maximum autocorrelation that exceeds `min_corr`.
/// Returns `None` if no candidate clears the threshold or if `data`
/// has fewer than `2 * max_lag` samples.
///
/// `min_lag = 4` excludes high-frequency noise; `max_lag = data.len()/3`
/// gives at least 3 cycles of evidence at the longest tested period.
/// `min_corr = 0.3` excludes weakly-correlated peaks that would not
/// produce a useful seasonal model.
pub fn dominant_period_via_acf(data: &[f64]) -> Option<usize> {
    let n = data.len();
    let min_lag = 4;
    let max_lag = (n / 3).max(min_lag + 1);
    let min_corr = 0.3;

    if n < 2 * max_lag || max_lag <= min_lag {
        return None;
    }

    let mean = data.iter().sum::<f64>() / n as f64;
    let var = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
    if var < 1e-20 {
        return None;
    }

    let acf = |lag: usize| -> f64 {
        let mut s = 0.0;
        for i in 0..(n - lag) {
            s += (data[i] - mean) * (data[i + lag] - mean);
        }
        s / ((n - lag) as f64 * var)
    };

    let mut prev = acf(min_lag);
    let mut prev_prev = acf(min_lag.saturating_sub(1).max(1));
    for lag in (min_lag + 1)..max_lag {
        let cur = acf(lag);
        if prev > prev_prev && prev > cur && prev >= min_corr {
            return Some(lag - 1);
        }
        prev_prev = prev;
        prev = cur;
    }
    None
}

impl Detrender {
    /// Auto-detect the dominant period via autocorrelation, then fit.
    ///
    /// Returns `None` when no period meets the autocorrelation threshold,
    /// signalling that the signal is non-periodic (or that `data` is too
    /// short relative to its periodicity).
    pub fn auto_fit(data: &[f64]) -> Option<Self> {
        dominant_period_via_acf(data).map(|p| Self::fit(data, p))
    }
}

// ── Seasonal differencing ────────────────────────────────────

/// Seasonal differencing: `y_t - y_{t-P}`.
///
/// Removes both trend and seasonality in one operation without model fitting.
/// Change points appear as a transient step of length P in the output.
/// Returns `data.len() - period` values (first P values are consumed).
pub fn seasonal_difference(data: &[f64], period: usize) -> Vec<f64> {
    if period == 0 || data.len() <= period {
        return vec![];
    }
    data.iter()
        .enumerate()
        .skip(period)
        .map(|(i, &y)| y - data[i - period])
        .collect()
}

/// Dual-detector: run BOCPD on both raw and seasonally-differenced data,
/// return only change points detected in both (intersection).
///
/// The intersection is a heuristic for reducing seasonal and detrending
/// artifacts. It can retain false positives and miss real changes;
/// agreement between the two runs does not establish a real change.
///
/// Returns indices in the raw data coordinate system.
///
/// Pass `period = None` to auto-detect via [`dominant_period_via_acf`].
/// When auto-detection returns no period, the guard short-circuits to
/// the raw-detector output (no false positives are filtered, but no
/// real detections are lost either).
pub fn detect_with_seasonal_guard<P: crate::Predictive>(
    data: &[f64],
    period: Option<usize>,
    detector: &crate::BocpdDetector<P>,
) -> Vec<crate::ChangePoint> {
    let raw_cps = detector.detect(data);

    let p = match period.or_else(|| dominant_period_via_acf(data)) {
        Some(p) if p > 0 && data.len() >= 3 * p => p,
        _ => return raw_cps,
    };

    let diffed = seasonal_difference(data, p);
    if diffed.len() < 20 {
        return raw_cps;
    }

    let diff_cps = detector.detect(&diffed);

    raw_cps
        .into_iter()
        .filter(|cp| {
            diff_cps.iter().any(|dcp| {
                let adj = dcp.index + p;
                (adj as i64 - cp.index as i64).unsigned_abs() as usize <= p / 2
            })
        })
        .collect()
}

/// Theil-Sen slope estimator: median of pairwise slopes `(y_j - y_i)/(j - i)`.
/// Uses a robust median slope estimate. Step changes and the sampling
/// scheme can still affect the estimate; there is no universal zero-bias
/// guarantee for a stepped series.
///
/// O(n²) pairs; for large `n` we subsample to keep fit cost bounded.
fn theil_sen_slope(data: &[f64]) -> f64 {
    let n = data.len();
    if n < 2 {
        return 0.0;
    }

    // Pair budget: cap at MAX_PAIRS to keep fit O(MAX_PAIRS · log MAX_PAIRS).
    // For n ≤ 600 this is exact (all pairs); for larger n we stride uniformly.
    const MAX_PAIRS: usize = 200_000;
    let total_pairs = n * (n - 1) / 2;

    let mut slopes: Vec<f64> = Vec::with_capacity(total_pairs.min(MAX_PAIRS));
    if total_pairs <= MAX_PAIRS {
        for i in 0..n {
            for j in (i + 1)..n {
                slopes.push((data[j] - data[i]) / (j - i) as f64);
            }
        }
    } else {
        let stride = ((total_pairs as f64) / (MAX_PAIRS as f64)).sqrt().ceil() as usize;
        let stride = stride.max(1);
        let mut i = 0;
        while i < n {
            let mut j = i + 1;
            while j < n {
                slopes.push((data[j] - data[i]) / (j - i) as f64);
                j += stride;
            }
            i += stride;
        }
    }

    slopes.sort_by(|a, b| a.total_cmp(b));
    median(&slopes)
}

fn median(sorted: &[f64]) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return 0.0;
    }
    if n.is_multiple_of(2) {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    } else {
        sorted[n / 2]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn detrend_continuation_subtracts_trend_at_correct_position() {
        // Fit on [0..200], then detrend [200..400] of the SAME generative process.
        // Residuals must be small. If `detrend()` uses the slice index `i` for
        // trend subtraction (instead of absolute position since fit), the trend
        // baseline for the continuation will be wrong by `trend * fit_len`.
        let period = 60;
        let trend = 0.1;
        let amp = 3.0;
        let full: Vec<f64> = (0..400)
            .map(|i| trend * i as f64 + amp * (i as f64 * 2.0 * PI / period as f64).sin())
            .collect();

        let det = Detrender::fit(&full[..200], period);

        // Sanity: detrending the FULL series should produce near-zero residuals.
        let res_full = det.detrend(&full, 0);
        let max_full = res_full.iter().map(|r| r.abs()).fold(0.0f64, f64::max);
        assert!(
            max_full < 2.0,
            "full-series detrend max |r|={max_full:.2}"
        );

        // Now detrend the continuation [200..400] with start_offset=200.
        // If `start_offset` is interpreted as absolute position since fit
        // (correct contract), residuals are equally small. If it is only
        // interpreted modulo period (current implementation), the trend
        // baseline is wrong by `trend * 200 = 20`.
        let res_cont = det.detrend(&full[200..], 200);
        let max_cont = res_cont.iter().map(|r| r.abs()).fold(0.0f64, f64::max);
        assert!(
            max_cont < 2.0,
            "continuation detrend max |r|={max_cont:.2} -- `start_offset` should \
             control trend baseline as well as seasonal phase"
        );
    }

    #[test]
    fn fit_pure_sinusoid() {
        let period = 60;
        let data: Vec<f64> = (0..300)
            .map(|i| (i as f64 * 2.0 * PI / period as f64).sin() * 5.0)
            .collect();
        let det = Detrender::fit(&data, period);
        let residuals = det.detrend(&data, 0);

        // Residuals should be small for a pure periodic signal.
        // Median estimation with 5 samples per offset has inherent error.
        let max_residual = residuals.iter().map(|r| r.abs()).fold(0.0f64, f64::max);
        assert!(
            max_residual < 1.0,
            "pure sinusoid residuals should be <1.0, got {max_residual:.2}"
        );
    }

    #[test]
    fn fit_sinusoid_plus_shift() {
        let period = 60;
        let data: Vec<f64> = (0..300)
            .map(|i| {
                let seasonal = (i as f64 * 2.0 * PI / period as f64).sin() * 3.0;
                let level = if i < 150 { 0.0 } else { 5.0 };
                seasonal + level
            })
            .collect();

        // Fit on first half (no shift)
        let det = Detrender::fit(&data[..150], period);
        let residuals = det.detrend(&data, 0);

        // First half residuals should be small
        let first_half_max = residuals[..150]
            .iter()
            .map(|r| r.abs())
            .fold(0.0f64, f64::max);
        assert!(
            first_half_max < 1.0,
            "first half residuals should be <1.0, got {first_half_max:.2}"
        );

        // Second half residuals should show the shift
        let second_half_mean = residuals[150..].iter().sum::<f64>() / residuals[150..].len() as f64;
        assert!(
            second_half_mean > 3.0,
            "second half should show shift in residuals, mean={second_half_mean:.2}"
        );
    }

    #[test]
    fn detrend_removes_linear_trend() {
        let data: Vec<f64> = (0..200).map(|i| i as f64 * 0.1 + 5.0).collect();
        let det = Detrender::fit(&data, 50);
        let residuals = det.detrend(&data, 0);

        // Variance of residuals should be much smaller than variance of raw data
        let raw_var = data.iter().map(|x| (x - 15.0).powi(2)).sum::<f64>() / data.len() as f64;
        let res_var = residuals.iter().map(|x| x.powi(2)).sum::<f64>() / residuals.len() as f64;
        assert!(
            res_var < raw_var * 0.1,
            "residual variance should be <10% of raw: raw_var={raw_var:.2}, res_var={res_var:.2}"
        );
    }

    #[test]
    fn insufficient_data_returns_noop() {
        let data = vec![1.0, 2.0, 3.0];
        let det = Detrender::fit(&data, 60);
        assert!(!det.is_fitted());
        let residuals = det.detrend(&data, 0);
        // No-op detrender: residuals ≈ original (minus intercept)
        assert_eq!(residuals.len(), 3);
    }

    #[test]
    fn diurnal_pattern_detrended() {
        // Simulate 6h of data at 1m step with a "day/night" pattern
        let period = 360; // 6h cycle for testing
        let data: Vec<f64> = (0..720)
            .map(|i| {
                let hour_frac = (i as f64 / 60.0) % 6.0;
                if (2.0..4.0).contains(&hour_frac) {
                    0.6
                } else {
                    0.1
                }
            })
            .collect();

        let det = Detrender::fit(&data, period);
        let residuals = det.detrend(&data, 0);

        let res_var = residuals.iter().map(|r| r.powi(2)).sum::<f64>() / residuals.len() as f64;
        assert!(
            res_var < 0.01,
            "diurnal pattern residual variance should be <0.01, got {res_var:.4}"
        );
    }

    #[test]
    fn update_adapts_seasonal() {
        let period = 60;
        let data: Vec<f64> = (0..120)
            .map(|i| (i as f64 * 2.0 * PI / period as f64).sin() * 3.0)
            .collect();
        let mut det = Detrender::fit(&data, period);

        // Update with new data that has a different amplitude
        let new_data: Vec<f64> = (0..60)
            .map(|i| (i as f64 * 2.0 * PI / period as f64).sin() * 6.0)
            .collect();
        det.update(&new_data, 0);

        // Seasonal should have shifted toward the new amplitude
        let peak = det.seasonal.iter().copied().fold(0.0f64, f64::max);
        assert!(
            peak > 3.0,
            "seasonal peak should adapt upward, got {peak:.2}"
        );
    }

    #[test]
    fn state_serde_roundtrip() {
        let data: Vec<f64> = (0..200)
            .map(|i| (i as f64 * 0.1).sin() * 2.0 + i as f64 * 0.01)
            .collect();
        let det = Detrender::fit(&data, 60);
        let json = serde_json::to_string(&det).unwrap();
        let restored: Detrender = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.period, 60);
        assert_eq!(restored.seasonal.len(), 60);
        assert!((restored.trend - det.trend).abs() < 1e-10);
    }

    #[test]
    fn detrend_then_bocpd_on_periodic() {
        // Detrending a periodic signal should produce small residuals
        // that BOCPD treats as stationary noise.
        use crate::BocpdDetector;

        let period = 60;
        // Periodic signal with enough periods for good estimation
        let data: Vec<f64> = (0..600)
            .map(|i| (i as f64 * 2.0 * PI / period as f64).sin() * 3.0 + 5.0)
            .collect();

        let detrender = Detrender::fit(&data, period);
        let residuals = detrender.detrend(&data, 0);

        // Residuals should have much lower variance than raw
        let raw_var = {
            let mean = data.iter().sum::<f64>() / data.len() as f64;
            data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64
        };
        let res_var = residuals.iter().map(|x| x.powi(2)).sum::<f64>() / residuals.len() as f64;

        eprintln!("periodic: raw_var={raw_var:.2}, res_var={res_var:.4}");
        assert!(
            res_var < raw_var * 0.2,
            "residual variance should be <20% of raw"
        );

        // BOCPD on residuals should have few/no detections
        let detector = BocpdDetector::new(200.0, 650);
        let cps = detector.detect(&residuals);
        assert!(
            cps.len() <= 2,
            "detrended periodic should have ≤2 FPs, got {}",
            cps.len()
        );
    }

    #[test]
    fn detrend_preserves_real_shift() {
        // Detrending should NOT hide a real change point
        use crate::BocpdDetector;

        let period = 60;
        let data: Vec<f64> = (0..300)
            .map(|i| {
                let seasonal = (i as f64 * 2.0 * PI / period as f64).sin() * 2.0;
                let level = if i < 150 { 0.0 } else { 5.0 };
                seasonal + level
            })
            .collect();

        // Fit on first half only (before shift)
        let detrender = Detrender::fit(&data[..150], period);
        let residuals = detrender.detrend(&data, 0);

        let detector = BocpdDetector::new(200.0, 350);
        let cps = detector.detect(&residuals);

        assert!(!cps.is_empty(), "detrending should preserve the real shift");
        assert!(
            (cps[0].index as i64 - 150).abs() < 20,
            "shift should be detected near 150, got {}",
            cps[0].index
        );
    }

    #[test]
    fn acf_finds_period_within_lag_resolution() {
        for true_period in [12_usize, 24, 60, 100] {
            let n = 8 * true_period;
            let data: Vec<f64> = (0..n)
                .map(|i| (i as f64 * 2.0 * PI / true_period as f64).sin())
                .collect();
            let detected = dominant_period_via_acf(&data).expect("should detect period");
            let err = (detected as i64 - true_period as i64).abs();
            assert!(
                err <= 2,
                "period {true_period}: detected {detected}, |err|={err}"
            );
        }
    }

    #[test]
    fn acf_returns_none_for_white_noise() {
        let mut rng = crate::eval::Rng::new(0xACF_F00D);
        let data: Vec<f64> = (0..1000).map(|_| rng.normal(0.0, 1.0)).collect();
        let p = dominant_period_via_acf(&data);
        assert!(p.is_none(), "white noise should not yield a period: {p:?}");
    }

    #[test]
    fn auto_fit_matches_manual_fit() {
        let period = 60;
        let data: Vec<f64> = (0..600)
            .map(|i| (i as f64 * 2.0 * PI / period as f64).sin() * 3.0)
            .collect();
        let auto = Detrender::auto_fit(&data).expect("should auto-detect");
        let manual = Detrender::fit(&data, auto.period());
        let auto_res = auto.detrend(&data, 0);
        let manual_res = manual.detrend(&data, 0);
        for (a, m) in auto_res.iter().zip(&manual_res) {
            assert!((a - m).abs() < 1e-12, "auto vs manual diverged: {a} vs {m}");
        }
    }

    #[test]
    fn seasonal_guard_with_none_auto_detects() {
        use crate::BocpdDetector;
        let period = 60;
        let data: Vec<f64> = (0..600)
            .map(|i| {
                let s = (i as f64 * 2.0 * PI / period as f64).sin() * 3.0;
                let level = if i < 300 { 0.0 } else { 5.0 };
                s + level
            })
            .collect();
        let detector = BocpdDetector::new(200.0, 650);

        let with_explicit = super::detect_with_seasonal_guard(&data, Some(period), &detector);
        let with_auto = super::detect_with_seasonal_guard(&data, None, &detector);

        assert!(!with_auto.is_empty(), "auto-detect path should still detect the shift");
        let close = with_auto.iter().any(|cp| (cp.index as i64 - 300).abs() <= period as i64);
        assert!(close, "auto-detect should keep the real CP at ~300");
        assert_eq!(
            with_explicit.len(),
            with_auto.len(),
            "explicit and auto paths should agree on this clean periodic signal"
        );
    }
}
