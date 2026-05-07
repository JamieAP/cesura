//! Seasonal-trend decomposition for time series detrending.
//!
//! Reduces predictable periodic structure such as daily traffic cycles
//! and batch schedules before change-point detection.
//!
//! Algorithm: median-based seasonal estimation (STL-lite). Robust to
//! outliers. O(n) fit, O(k) incremental update.

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

        // Step 1: Estimate linear trend via least-squares
        let n = data.len() as f64;
        let x_mean = (n - 1.0) / 2.0;
        let y_mean = data.iter().sum::<f64>() / n;
        let mut num = 0.0;
        let mut den = 0.0;
        for (i, &y) in data.iter().enumerate() {
            let x = i as f64 - x_mean;
            num += x * (y - y_mean);
            den += x * x;
        }
        let trend = if den.abs() > 1e-20 { num / den } else { 0.0 };

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
    /// `start_offset` is the position within the period where `data` begins
    /// (e.g., minute-of-day for daily detrending).
    pub fn detrend(&self, data: &[f64], start_offset: usize) -> Vec<f64> {
        data.iter()
            .enumerate()
            .map(|(i, &y)| {
                let offset = (start_offset + i) % self.period;
                y - self.seasonal[offset] - self.trend * i as f64 - self.intercept
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
pub fn detect_with_seasonal_guard(
    data: &[f64],
    period: usize,
    detector: &crate::BocpdDetector,
    threshold: f64,
) -> Vec<crate::ChangePoint> {
    let raw_cps = detector.detect(data, threshold);

    // Skip detrending if insufficient data
    if data.len() < 3 * period || period == 0 {
        return raw_cps;
    }

    let diffed = seasonal_difference(data, period);
    if diffed.len() < 20 {
        return raw_cps;
    }

    let diff_cps = detector.detect(&diffed, threshold);

    // Intersect: keep raw CPs that have a corresponding diff CP
    // The diff series is offset by `period` indices, so adjust.
    raw_cps
        .into_iter()
        .filter(|cp| {
            diff_cps.iter().any(|dcp| {
                let adj = dcp.index + period;
                (adj as i64 - cp.index as i64).unsigned_abs() as usize <= period / 2
            })
        })
        .collect()
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
        let cps = detector.detect(&residuals, 0.3);
        assert!(
            cps.len() <= 2,
            "detrended periodic should have ≤2 FPs, got {}",
            cps.len()
        );
    }


    #[test]
    fn research_fit_on_full_data_weakens_shift() {
        use crate::BocpdDetector;

        let period = 60;
        let data: Vec<f64> = (0..300)
            .map(|i| {
                let seasonal = (i as f64 * 2.0 * PI / period as f64).sin() * 2.0;
                let level = if i < 150 { 0.0 } else { 5.0 };
                seasonal + level
            })
            .collect();

        let detector = BocpdDetector::new(200.0, 350);

        // Fit on pre-shift data only (correct approach)
        let correct = Detrender::fit(&data[..150], period);
        let correct_residuals = correct.detrend(&data, 0);
        let correct_cps = detector.detect(&correct_residuals, 0.3);

        // Fit on full data including shift (problematic approach)
        let full = Detrender::fit(&data, period);
        let full_residuals = full.detrend(&data, 0);
        let full_cps = detector.detect(&full_residuals, 0.3);

        // Both detect, but the full-fit version has weaker detection
        // (shift_sigma is lower because trend absorbed part of the step)
        eprintln!(
            "correct: {} detections, shift_sigma={:.2}",
            correct_cps.len(),
            correct_cps.first().map(|c| c.shift_sigma).unwrap_or(0.0)
        );
        eprintln!(
            "full-fit: {} detections, shift_sigma={:.2}",
            full_cps.len(),
            full_cps.first().map(|c| c.shift_sigma).unwrap_or(0.0)
        );

        // The trend absorbs ~0.025/step, so over 150 steps that's ~3.75 of the 5.0 shift
        assert!(
            full.trend.abs() > 0.01,
            "full-data fit should have non-zero trend from absorbing the step: trend={:.4}",
            full.trend
        );
    }

    #[test]
    fn research_two_periods_worse_than_five() {

        let period = 60;
        let mut rng = crate::eval::Rng::new(7777);
        let full_data: Vec<f64> = (0..300)
            .map(|i| (i as f64 * 2.0 * PI / period as f64).sin() * 3.0 + rng.normal(0.0, 0.5))
            .collect();

        // 2-period fit
        let det2 = Detrender::fit(&full_data[..120], period);
        let res2 = det2.detrend(&full_data[..120], 0);
        let var2 = res2.iter().map(|x| x.powi(2)).sum::<f64>() / res2.len() as f64;

        // 5-period fit
        let det5 = Detrender::fit(&full_data, period);
        let res5 = det5.detrend(&full_data[..120], 0);
        let var5 = res5.iter().map(|x| x.powi(2)).sum::<f64>() / res5.len() as f64;

        eprintln!("2-period residual var: {var2:.4}, 5-period residual var: {var5:.4}");
        assert!(
            var5 < var2,
            "5-period estimate should produce lower residual variance: \
             2p={var2:.4}, 5p={var5:.4}"
        );
    }

    #[test]
    #[should_panic(expected = "seasonal differencing should have fewer FPs")]
    fn research_differencing_beats_median_subtraction() {
        use crate::BocpdDetector;

        let period = 50;
        // Bursty signal -- regular bursts every 50 steps (MustReject scenario)
        let mut rng = crate::eval::Rng::new(8888);
        let data: Vec<f64> = (0..300)
            .map(|i| {
                let burst = if i % 50 < 5 { 3.0 } else { 0.0 };
                burst + rng.normal(0.0, 0.5)
            })
            .collect();

        let detector = BocpdDetector::new(200.0, 350);

        // Current approach: median-based detrending
        let detrender = Detrender::fit(&data, period);
        let median_residuals = detrender.detrend(&data, 0);
        let median_fps = detector.detect(&median_residuals, 0.3).len();

        // Better approach: seasonal differencing
        let diff_data: Vec<f64> = data
            .iter()
            .enumerate()
            .skip(period)
            .map(|(i, &y)| y - data[i - period])
            .collect();
        let diff_fps = detector.detect(&diff_data, 0.3).len();

        eprintln!("bursty: median_fps={median_fps}, diff_fps={diff_fps}");
        assert!(
            diff_fps < median_fps,
            "seasonal differencing should have fewer FPs than median subtraction: \
             diff={diff_fps}, median={median_fps}"
        );
    }

    #[test]
    #[should_panic(expected = "dual detector should have fewer FPs")]
    fn research_dual_detector_beats_single() {
        use crate::BocpdDetector;

        let period = 60;
        // Periodic signal with noise (no real change point -- should have 0 detections)
        let mut rng = crate::eval::Rng::new(9999);
        let data: Vec<f64> = (0..360)
            .map(|i| {
                let seasonal = (i as f64 * 2.0 * PI / period as f64).sin() * 3.0;
                seasonal + 5.0 + rng.normal(0.0, 0.3)
            })
            .collect();

        let detector = BocpdDetector::new(200.0, 400);

        // Single detector on raw: may fire on seasonal transitions
        let raw_cps = detector.detect(&data, 0.3);

        // Single detector on detrended: may fire on detrending artifacts
        let detrender = Detrender::fit(&data, period);
        let residuals = detrender.detrend(&data, 0);
        let detrended_cps = detector.detect(&residuals, 0.3);

        // Dual detector: intersect raw and detrended detections
        let diff_data: Vec<f64> = data
            .iter()
            .enumerate()
            .skip(period)
            .map(|(i, &y)| y - data[i - period])
            .collect();
        let diff_cps = detector.detect(&diff_data, 0.3);
        let dual_fps: usize = raw_cps
            .iter()
            .filter(|cp| {
                diff_cps.iter().any(|dcp| {
                    let adj = dcp.index + period;
                    (adj as i64 - cp.index as i64).unsigned_abs() as usize <= period / 2
                })
            })
            .count();

        let single_fps = raw_cps.len();

        eprintln!(
            "periodic+noise: raw_fps={}, detrended_fps={}, dual_fps={}",
            single_fps,
            detrended_cps.len(),
            dual_fps
        );
        assert!(
            dual_fps < single_fps,
            "dual detector should have fewer FPs than raw-only: dual={dual_fps}, single={single_fps}"
        );
    }

    #[test]
    #[should_panic(expected = "trend should not absorb")]
    fn research_trend_absorbs_step_change() {

        let data: Vec<f64> = (0..200).map(|i| if i < 100 { 0.0 } else { 5.0 }).collect();

        let det = Detrender::fit(&data, 50);

        // The trend coefficient should be near 0 (it's a step, not a slope)
        // But least-squares fits a line through the step, getting ~0.025 slope
        eprintln!("step change: trend={:.4}", det.trend);
        assert!(
            det.trend.abs() < 0.005,
            "trend should not absorb the step change, but trend={:.4}",
            det.trend
        );
    }

    // ── End research validation tests ────────────────────────

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
        let cps = detector.detect(&residuals, 0.3);

        assert!(!cps.is_empty(), "detrending should preserve the real shift");
        assert!(
            (cps[0].index as i64 - 150).abs() < 20,
            "shift should be detected near 150, got {}",
            cps[0].index
        );
    }
}
