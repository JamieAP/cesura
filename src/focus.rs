//! FOCuS-style frequentist online change-point detector for univariate
//! Gaussian-mean shifts.
//!
//! Reference: Romano, Eckley, Fearnhead, Rigaill (2023), "Fast Online
//! Change Point Detection via Functional Pruning CUSUM Statistics";
//! Ward et al. (2024) "Faster Online Changepoint Detection ..."
//! (arXiv:2402.05989) for the multivariate generalisation.
//!
//! What ships here is the univariate Gaussian-mean variant with a naive
//! O(n) per-step inner loop -- complementary to BOCPD, frequentist where
//! BOCPD is Bayesian. The functional-pruning O(log n) data structure
//! that gives FOCuS its name is deferred (see CHANGELOG / KNOWN_LIMITATIONS).
//!
//! Operating-characteristic note: FOCuS thresholds ARE NOT comparable to
//! BOCPD confidences. Use FOCuS as a sanity-check parallel detector,
//! not as a precision/recall replacement.

use crate::ChangePoint;

/// Window used to compute `shift_sigma` once a candidate is committed.
const SHIFT_WINDOW: usize = 20;

/// Univariate Gaussian-mean FOCuS detector.
///
/// Tracks cumulative sums since the last reset and computes the
/// generalized-likelihood-ratio statistic
/// `½ · max_τ [τ(t−τ)/t] · (x̄_{0..τ} − x̄_{τ..t})²`
/// at each step. When the maximum exceeds `threshold`, emits a change
/// point at the maximizing `τ` and resets baselines past that index
/// (subsequent observations re-accumulate from there).
///
/// Inputs are z-normalised against the data fed so far before the GLR
/// is computed -- matches BOCPD's input convention so threshold
/// calibration on N(0,1) carries over to scaled inputs.
pub struct FocusDetector {
    threshold: f64,
    cooldown: usize,
    /// Cumulative running sum for the *current* segment (post-last-reset).
    /// At index i the entry is `Σ x_{seg_start..=i}` after z-norm.
    seg_sums: Vec<f64>,
    /// `seg_sums.len() = i+1−seg_start` once we reach absolute step i.
    seg_start: usize,
    /// Welford state on raw input for streaming z-norm.
    mean: f64,
    m2: f64,
    n: usize,
    /// Most recently emitted CP, for cooldown bookkeeping.
    last_emit: Option<usize>,
}

impl FocusDetector {
    /// `threshold`: GLR cutoff. Higher → fewer false alarms. Calibrate
    /// to a target ARL₀ via [`arl0_calibrate`].
    pub fn new(threshold: f64) -> Self {
        assert!(
            threshold > 0.0 && threshold.is_finite(),
            "threshold must be finite and positive, got {threshold}"
        );
        Self {
            threshold,
            cooldown: 15,
            seg_sums: Vec::new(),
            seg_start: 0,
            mean: 0.0,
            m2: 0.0,
            n: 0,
            last_emit: None,
        }
    }

    /// Detect on a complete batch. Mirrors `BocpdDetector::detect`.
    pub fn detect(&mut self, data: &[f64]) -> Vec<ChangePoint> {
        let mut out = Vec::new();
        for (raw_idx, &x) in data.iter().enumerate() {
            if let Some(idx) = self.step_inner(x) {
                let cp = build_cp(data, idx, self.threshold);
                out.push(cp);
                let _ = raw_idx; // raw_idx tracked via self.n
            }
        }
        out
    }

    /// Online step. Returns `Some(t)` when a change point at absolute
    /// time `t` has fired; else `None`. To produce a full `ChangePoint`
    /// (with `confidence` / `shift_sigma`), use [`detect`] which has the
    /// data window required for `shift_sigma`.
    pub fn step(&mut self, x: f64) -> Option<usize> {
        self.step_inner(x)
    }

    fn step_inner(&mut self, x: f64) -> Option<usize> {
        // Welford update on raw input for streaming z-norm.
        self.n += 1;
        let n = self.n as f64;
        let delta = x - self.mean;
        self.mean += delta / n;
        self.m2 += delta * (x - self.mean);

        let std = if self.n >= 2 {
            (self.m2 / n).sqrt().max(1e-10)
        } else {
            1.0
        };
        let z = (x - self.mean) / std;

        let abs_idx = self.n - 1;
        // Drop observations strictly inside the cooldown window.
        if let Some(last) = self.last_emit {
            if abs_idx <= last + self.cooldown {
                return None;
            }
        }
        // Append z-normed observation to the segment cumsum.
        let prev = self.seg_sums.last().copied().unwrap_or(0.0);
        self.seg_sums.push(prev + z);

        // Need at least 2 observations in the segment to define a split.
        let m = self.seg_sums.len();
        if m < 4 {
            return None;
        }
        let total = *self.seg_sums.last().unwrap();
        let mut best_stat = 0.0f64;
        let mut best_tau = 0usize;
        // τ = number of obs before the split, in [1, m−1].
        for tau in 1..m {
            let s_left = self.seg_sums[tau - 1];
            let s_right = total - s_left;
            let n_l = tau as f64;
            let n_r = (m - tau) as f64;
            let mean_l = s_left / n_l;
            let mean_r = s_right / n_r;
            let diff = mean_l - mean_r;
            let stat = 0.5 * (n_l * n_r / m as f64) * diff * diff;
            if stat > best_stat {
                best_stat = stat;
                best_tau = tau;
            }
        }

        if best_stat >= self.threshold {
            let cp_abs = self.seg_start + best_tau;
            self.last_emit = Some(cp_abs);
            // Reset segment baseline to start *after* the detected split.
            // Carry forward the right-segment cumsum so the next step's
            // statistic is computed from the new regime.
            let s_left = self.seg_sums[best_tau - 1];
            let mut new_sums = Vec::with_capacity(self.seg_sums.len() - best_tau);
            for v in &self.seg_sums[best_tau..] {
                new_sums.push(v - s_left);
            }
            self.seg_sums = new_sums;
            self.seg_start = cp_abs;
            Some(cp_abs)
        } else {
            None
        }
    }

    /// Number of observations seen so far.
    pub fn total_steps(&self) -> usize {
        self.n
    }
}

/// Build a `ChangePoint` from the data slice and detected index. Uses
/// the same before/after-window math BOCPD uses for `shift_sigma`. The
/// `confidence` is `1 − exp(−glr_at_index / threshold)` -- a monotone
/// score in `[0, 1]`, NOT a posterior probability.
fn build_cp(data: &[f64], idx: usize, threshold: f64) -> ChangePoint {
    let n = data.len();
    let lo = idx.saturating_sub(SHIFT_WINDOW);
    let hi = (idx + SHIFT_WINDOW).min(n);
    let before = &data[lo..idx];
    let after = &data[idx..hi];
    let shift_sigma = if before.is_empty() || after.is_empty() {
        0.0
    } else {
        let mean_b: f64 = before.iter().sum::<f64>() / before.len() as f64;
        let mean_a: f64 = after.iter().sum::<f64>() / after.len() as f64;
        let var_b: f64 =
            before.iter().map(|x| (x - mean_b).powi(2)).sum::<f64>() / before.len() as f64;
        let var_a: f64 =
            after.iter().map(|x| (x - mean_a).powi(2)).sum::<f64>() / after.len() as f64;
        let pooled = ((var_b + var_a) / 2.0).sqrt().max(1e-10);
        ((mean_a - mean_b) / pooled).abs()
    };
    // Confidence: monotone-in-margin transform of the GLR statistic.
    // `glr_at_index` recomputes the per-side cumulative-mean form with the
    // emitted slice -- proxy for "how strong is this evidence vs threshold".
    let glr = if !before.is_empty() && !after.is_empty() {
        let mean_b: f64 = before.iter().sum::<f64>() / before.len() as f64;
        let mean_a: f64 = after.iter().sum::<f64>() / after.len() as f64;
        let n_l = before.len() as f64;
        let n_r = after.len() as f64;
        let m = n_l + n_r;
        0.5 * (n_l * n_r / m) * (mean_a - mean_b).powi(2)
    } else {
        threshold
    };
    let conf = (1.0 - (-glr / threshold).exp()).clamp(0.0, 1.0);
    ChangePoint {
        index: idx,
        confidence: conf,
        shift_sigma,
    }
}

/// Pick a threshold for a target false-alarm rate (`ARL₀`).
///
/// Empirical: simulate `trials` × `length` Gaussian streams, bisect on
/// the threshold so that the average run length to first detection
/// approximately matches `target_arl0`. No analytic shortcut. Slow --
/// expect ~seconds at default settings. Caller-friendly defaults:
/// `trials = 30`, `length = (target_arl0 * 3.0) as usize`.
///
/// Available with the `test-utils` feature -- the simulator depends on
/// `crate::eval::Rng` which is itself feature-gated.
#[cfg(any(test, feature = "test-utils"))]
pub fn arl0_calibrate(target_arl0: f64) -> f64 {
    use crate::eval::Rng;
    let trials = 30;
    let length = (target_arl0 * 3.0) as usize;
    let mut lo = 1.0;
    let mut hi = 50.0;
    for _ in 0..18 {
        let mid = 0.5 * (lo + hi);
        let mut total_run_length = 0.0;
        for t in 0..trials {
            let mut rng = Rng::new(7000 + t as u64);
            let mut det = FocusDetector::new(mid);
            let mut fired_at = None;
            for i in 0..length {
                if det.step(rng.normal(0.0, 1.0)).is_some() {
                    fired_at = Some(i);
                    break;
                }
            }
            total_run_length += fired_at.map(|i| i as f64).unwrap_or(length as f64);
        }
        let avg = total_run_length / trials as f64;
        if avg < target_arl0 {
            // Too sensitive: raise threshold.
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::Rng;

    #[test]
    fn focus_detects_clean_mean_shift() {
        let mut rng = Rng::new(11);
        let mut data: Vec<f64> = (0..150).map(|_| rng.normal(0.0, 1.0)).collect();
        data.extend((0..150).map(|_| rng.normal(5.0, 1.0)));
        let mut det = FocusDetector::new(8.0);
        let cps = det.detect(&data);
        assert!(!cps.is_empty(), "must detect a clean 5σ shift");
        assert!(
            (cps[0].index as i64 - 150).abs() < 30,
            "first CP at {} too far from truth 150",
            cps[0].index
        );
    }

    #[test]
    fn focus_no_detection_on_stationary_noise() {
        let mut rng = Rng::new(42);
        let data: Vec<f64> = (0..1000).map(|_| rng.normal(0.0, 1.0)).collect();
        let mut det = FocusDetector::new(8.0);
        let cps = det.detect(&data);
        // FOCuS can fire spuriously; threshold-8 on N(0,1) over 1k should
        // produce at most a handful. A hard cap catches a calibration regression.
        assert!(
            cps.len() <= 3,
            "stationary N(0,1) produced {} CPs at threshold=8",
            cps.len()
        );
    }

    #[test]
    fn focus_detection_power_increases_with_shift() {
        // Larger shifts must fire faster. Compare median first-CP index
        // between 5σ and 1σ shifts at the same threshold; the 5σ case
        // should detect noticeably earlier than 1σ.
        let mut delay_strong = Vec::new();
        let mut delay_weak = Vec::new();
        for seed in 0..20 {
            let mut rng = Rng::new(seed);
            let mut strong: Vec<f64> = (0..150).map(|_| rng.normal(0.0, 1.0)).collect();
            strong.extend((0..150).map(|_| rng.normal(5.0, 1.0)));
            let mut weak: Vec<f64> = (0..150).map(|_| rng.normal(0.0, 1.0)).collect();
            weak.extend((0..150).map(|_| rng.normal(1.0, 1.0)));
            let mut a = FocusDetector::new(8.0);
            let mut b = FocusDetector::new(8.0);
            if let Some(cp) = a.detect(&strong).first() {
                if cp.index >= 150 {
                    delay_strong.push(cp.index - 150);
                }
            }
            if let Some(cp) = b.detect(&weak).first() {
                if cp.index >= 150 {
                    delay_weak.push(cp.index - 150);
                }
            }
        }
        assert!(
            !delay_strong.is_empty() && !delay_weak.is_empty(),
            "expected at least one detection in each arm"
        );
        let med_strong = median(&mut delay_strong);
        let med_weak = median(&mut delay_weak);
        eprintln!("median delay 5σ={med_strong} 1σ={med_weak}");
        assert!(
            med_strong < med_weak,
            "5σ median delay ({med_strong}) must be < 1σ median delay ({med_weak})"
        );
    }

    fn median(xs: &mut [usize]) -> usize {
        xs.sort();
        xs[xs.len() / 2]
    }

    #[test]
    fn focus_arl0_at_threshold_8_is_high() {
        // Light gate: at threshold 8, the average run length to first false
        // alarm should be large (≥ 200) over 5 trials × 1000 samples N(0,1).
        // The longer 30×1000 trial is gated `#[ignore]`.
        let mut total = 0.0;
        let trials = 5;
        for t in 0..trials {
            let mut rng = Rng::new(2000 + t);
            let mut det = FocusDetector::new(8.0);
            let mut fired = None;
            for i in 0..1000 {
                if det.step(rng.normal(0.0, 1.0)).is_some() {
                    fired = Some(i);
                    break;
                }
            }
            total += fired.map(|i| i as f64).unwrap_or(1000.0);
        }
        let arl0 = total / trials as f64;
        eprintln!("FOCuS ARL₀ at threshold=8 over {trials} × 1000 = {arl0}");
        assert!(arl0 >= 200.0, "ARL₀ at threshold=8 too low: {arl0}");
    }
}
