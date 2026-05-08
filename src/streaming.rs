//! Streaming BOCPD -- incremental change point detection.
//!
//! Maintains run-length distribution and NIG sufficient statistics
//! between calls to [`StreamingDetector::step`]. Processes only new
//! observations incrementally -- O(k) per call where k = new points.

use serde::{Deserialize, Serialize};

use crate::{log_add_exp, ChangePoint, Nig};

/// State that can be serialized for persistence between daemon restarts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectorState {
    pub rl_log: Vec<f64>,
    pub stats: Vec<NigState>,
    pub map_rls: Vec<usize>,
    /// Per-step posterior P(r_t = 0). Optional for forward-compat with v0.2 snapshots.
    #[serde(default)]
    pub cp_probs: Vec<f64>,
    pub total_steps: usize,
    pub last_detection: usize,
    pub welford: WelfordState,
    pub hazard_log: f64,
    pub growth_log: f64,
    pub max_rl: usize,
    /// Total observations including NaN (for correct index reporting).
    #[serde(default)]
    pub raw_steps: usize,
    /// Maps model step → raw input position.
    #[serde(default)]
    pub raw_index_map: Vec<usize>,
    /// β-divergence robustness parameter. 0 = standard Bayesian update.
    /// Forward-compat: pre-0.6 snapshots without this field default to 0.
    #[serde(default)]
    pub beta: f64,
}

/// Serializable NIG sufficient statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NigState {
    pub mu: f64,
    pub kappa: f64,
    pub alpha: f64,
    pub beta: f64,
}

impl From<&Nig> for NigState {
    fn from(nig: &Nig) -> Self {
        Self {
            mu: nig.mu,
            kappa: nig.kappa,
            alpha: nig.alpha,
            beta: nig.beta,
        }
    }
}

impl From<&NigState> for Nig {
    fn from(s: &NigState) -> Self {
        Self {
            mu: s.mu,
            kappa: s.kappa,
            alpha: s.alpha,
            beta: s.beta,
        }
    }
}

/// Welford's online algorithm for running mean and variance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WelfordState {
    pub count: u64,
    pub mean: f64,
    pub m2: f64,
}

impl WelfordState {
    pub fn new() -> Self {
        Self {
            count: 0,
            mean: 0.0,
            m2: 0.0,
        }
    }

    pub fn update(&mut self, x: f64) {
        self.count += 1;
        let delta = x - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = x - self.mean;
        self.m2 += delta * delta2;
    }

    pub fn std(&self) -> f64 {
        if self.count < 2 {
            return 1.0;
        }
        let var = self.m2 / self.count as f64;
        let s = var.sqrt();
        if s < 1e-10 {
            1.0
        } else {
            s
        }
    }

    /// Normalize a value using running mean/std.
    pub fn normalize(&self, x: f64) -> f64 {
        (x - self.mean) / self.std()
    }
}

impl Default for WelfordState {
    fn default() -> Self {
        Self::new()
    }
}

/// Detection awaiting its post-window. Held until enough observations
/// have accumulated to compute `shift_sigma` from a real before/after window.
#[derive(Debug, Clone)]
struct PendingCp {
    raw_index: usize,
    fire_step: usize,
    confidence: f64,
}

/// Streaming BOCPD detector -- maintains state between calls to `step()`.
pub struct StreamingDetector {
    hazard_log: f64,
    growth_log: f64,
    max_rl: usize,
    log_mass_cutoff: f64,
    /// β-divergence robustness parameter (see [`crate::BocpdDetector::with_beta`]).
    /// `0.0` ⇒ standard BOCPD (default, short-circuited).
    beta: f64,
    prior: Nig,
    // Mutable state
    rl_log: Vec<f64>,
    stats: Vec<Nig>,
    map_rls: Vec<usize>,
    /// Per-step posterior P(r_t = 0). Same length as `map_rls`.
    cp_probs: Vec<f64>,
    total_steps: usize,
    last_detection: usize,
    welford: WelfordState,
    /// Total observations including NaN-skipped ones.
    raw_steps: usize,
    /// Maps model step index → raw input position.
    raw_index_map: Vec<usize>,
    // Pre-allocated scratch buffers (avoid per-step allocation)
    scratch_rl: Vec<f64>,
    scratch_stats: Vec<Nig>,
    /// Ring buffer of last `2 * SHIFT_WINDOW` normalised observations,
    /// used to compute `shift_sigma` from a proper before/after window.
    norm_ring: std::collections::VecDeque<f64>,
    /// Detections fired but awaiting a full post-window.
    pending: Vec<PendingCp>,
}

const SHIFT_WINDOW: usize = 20;

/// Compute |mean(after) - mean(before)| from the ring buffer.
/// `total_steps` is the step count after the most recent observation.
/// `fire_step` is the step count at the moment the candidate was triggered.
fn compute_shift_sigma(
    ring: &std::collections::VecDeque<f64>,
    total_steps: usize,
    fire_step: usize,
) -> f64 {
    let len = ring.len();
    let post = total_steps.saturating_sub(fire_step).min(SHIFT_WINDOW);
    let pre = (len.saturating_sub(post)).min(SHIFT_WINDOW);
    if pre == 0 || post == 0 {
        return 0.0;
    }
    let split = len - post;
    let before_start = split - pre;
    let mean_b: f64 = ring.iter().skip(before_start).take(pre).sum::<f64>() / pre as f64;
    let mean_a: f64 = ring.iter().skip(split).take(post).sum::<f64>() / post as f64;
    (mean_a - mean_b).abs()
}

impl StreamingDetector {
    /// Create a new streaming detector.
    ///
    /// # Panics
    /// Panics if `lambda <= 1.0` (would produce -inf or NaN hazard rates).
    pub fn new(lambda: f64, max_run_length: usize) -> Self {
        assert!(lambda > 1.0, "lambda must be > 1.0, got {lambda}");
        let h = 1.0 / lambda;
        let prior = Nig {
            mu: 0.0,
            kappa: 1.0,
            alpha: 1.0,
            beta: 1.0,
        };
        let mut rl_log = vec![f64::NEG_INFINITY; max_run_length + 1];
        rl_log[0] = 0.0;
        Self {
            hazard_log: h.ln(),
            growth_log: (1.0 - h).ln(),
            max_rl: max_run_length,
            log_mass_cutoff: crate::DEFAULT_MASS_CUTOFF.ln(),
            beta: 0.0,
            prior: prior.clone(),
            rl_log,
            stats: vec![prior.clone(); max_run_length + 1],
            map_rls: Vec::new(),
            cp_probs: Vec::new(),
            total_steps: 0,
            last_detection: 0,
            welford: WelfordState::new(),
            raw_steps: 0,
            raw_index_map: Vec::new(),
            scratch_rl: vec![f64::NEG_INFINITY; max_run_length + 1],
            scratch_stats: vec![prior; max_run_length + 1],
            norm_ring: std::collections::VecDeque::with_capacity(2 * SHIFT_WINDOW),
            pending: Vec::new(),
        }
    }

    /// Opt into β-divergence robust BOCPD (see [`crate::BocpdDetector::with_beta`]).
    /// `beta = 0.0` is the default and short-circuits to standard cesura.
    /// β is persisted in [`DetectorState`] and survives `save_state`/`restore`.
    ///
    /// # Panics
    /// Panics if `beta < 0.0` or `beta > 1.0`.
    pub fn with_beta(mut self, beta: f64) -> Self {
        assert!(
            (0.0..=1.0).contains(&beta),
            "beta must be in [0.0, 1.0], got {beta}"
        );
        self.beta = beta;
        self
    }

    /// Set the mass-pruning cutoff (see [`crate::BocpdDetector::with_mass_cutoff`]).
    pub fn with_mass_cutoff(mut self, cutoff: f64) -> Self {
        self.log_mass_cutoff = if cutoff > 0.0 {
            cutoff.ln()
        } else {
            f64::NEG_INFINITY
        };
        self
    }

    /// Process new observations incrementally. Returns any new change points.
    pub fn step(&mut self, observations: &[f64]) -> Vec<ChangePoint> {
        let mut result = Vec::new();

        for &x in observations {
            let current_raw = self.raw_steps;
            self.raw_steps += 1;

            if !x.is_finite() {
                continue;
            }
            // Online normalization
            self.welford.update(x);
            let xn = self.welford.normalize(x);

            let t = self.total_steps;
            let active = (t + 1).min(self.max_rl);

            // Reset scratch buffers
            for v in self.scratch_rl.iter_mut() {
                *v = f64::NEG_INFINITY;
            }
            let mut prev_mass = f64::NEG_INFINITY;

            for r in 0..=active.min(self.max_rl.saturating_sub(1)) {
                if self.rl_log[r] == f64::NEG_INFINITY {
                    continue;
                }
                let pred = self.stats[r].log_predictive_robust(xn, self.beta);
                if !pred.is_finite() {
                    continue;
                }
                if r < self.max_rl {
                    self.scratch_rl[r + 1] =
                        log_add_exp(self.scratch_rl[r + 1], self.rl_log[r] + pred + self.growth_log);
                }
                prev_mass = log_add_exp(prev_mass, self.rl_log[r]);
            }
            let prior_pred = self.prior.log_predictive_robust(xn, self.beta);
            self.scratch_rl[0] = if prior_pred.is_finite() && prev_mass.is_finite() {
                prev_mass + self.hazard_log + prior_pred
            } else {
                f64::NEG_INFINITY
            };

            // Normalize
            let evidence = self
                .scratch_rl
                .iter()
                .copied()
                .filter(|v| v.is_finite())
                .fold(f64::NEG_INFINITY, log_add_exp);
            if evidence.is_finite() {
                for v in self.scratch_rl.iter_mut() {
                    *v -= evidence;
                }
            }

            // Mass-prune the tail: data-adaptive truncation.
            if self.log_mass_cutoff > f64::NEG_INFINITY {
                for r in (1..=self.max_rl).rev() {
                    if self.scratch_rl[r] >= self.log_mass_cutoff {
                        break;
                    }
                    self.scratch_rl[r] = f64::NEG_INFINITY;
                }
            }

            // MAP run length
            let map_r = self
                .scratch_rl
                .iter()
                .enumerate()
                .filter(|(_, v)| v.is_finite())
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(r, _)| r)
                .unwrap_or(0);
            self.map_rls.push(map_r);
            self.cp_probs.push(if self.scratch_rl[0].is_finite() {
                self.scratch_rl[0].exp()
            } else {
                0.0
            });
            self.raw_index_map.push(current_raw);

            // Update sufficient stats in scratch buffer
            for s in self.scratch_stats.iter_mut() {
                *s = self.prior.clone();
            }
            for r in 0..=active.min(self.max_rl.saturating_sub(1)) {
                if r < self.max_rl && self.scratch_rl[r + 1] > f64::NEG_INFINITY {
                    self.scratch_stats[r + 1] = self.stats[r].update(xn);
                }
            }

            // Swap scratch into live state
            std::mem::swap(&mut self.rl_log, &mut self.scratch_rl);
            std::mem::swap(&mut self.stats, &mut self.scratch_stats);
            self.total_steps += 1;

            // Maintain ring buffer of last 2*SHIFT_WINDOW normalised values
            // for proper shift_sigma computation at detection emission time.
            self.norm_ring.push_back(xn);
            while self.norm_ring.len() > 2 * SHIFT_WINDOW {
                self.norm_ring.pop_front();
            }

            // Drain any pending detections whose post-window is now full.
            self.pending.retain(|p| {
                if self.total_steps >= p.fire_step + SHIFT_WINDOW {
                    let shift_sigma = compute_shift_sigma(
                        &self.norm_ring,
                        self.total_steps,
                        p.fire_step,
                    );
                    if shift_sigma >= 1e-9 {
                        result.push(ChangePoint {
                            index: p.raw_index,
                            confidence: p.confidence,
                            shift_sigma,
                        });
                    }
                    false
                } else {
                    true
                }
            });

            // Change point detection (same trigger logic as batch).
            let drop_to = 3;
            let min_prev_rl = 30;
            let cooldown = 15;
            let i = self.total_steps - 1;

            if i >= min_prev_rl && self.map_rls[i] <= drop_to && i - self.last_detection >= cooldown
            {
                let prev_max = self.map_rls[i.saturating_sub(15)..i]
                    .iter()
                    .copied()
                    .max()
                    .unwrap_or(0);
                if prev_max >= min_prev_rl {
                    let look_back = cooldown.min(i);
                    let confidence = self.cp_probs[i.saturating_sub(look_back)..=i]
                        .iter()
                        .copied()
                        .fold(0.0_f64, f64::max)
                        .clamp(0.0, 1.0);
                    self.last_detection = i;
                    self.pending.push(PendingCp {
                        raw_index: self.raw_index_map[i],
                        fire_step: self.total_steps,
                        confidence,
                    });
                }
            }
        }

        result
    }

    /// Serialize detector state for persistence.
    /// Non-finite values in rl_log are stored as f64::MIN for JSON compatibility.
    pub fn save_state(&self) -> DetectorState {
        DetectorState {
            rl_log: self
                .rl_log
                .iter()
                .map(|&v| if v.is_finite() { v } else { f64::MIN })
                .collect(),
            stats: self.stats.iter().map(NigState::from).collect(),
            map_rls: self.map_rls.clone(),
            cp_probs: self.cp_probs.clone(),
            total_steps: self.total_steps,
            last_detection: self.last_detection,
            welford: self.welford.clone(),
            hazard_log: self.hazard_log,
            growth_log: self.growth_log,
            max_rl: self.max_rl,
            raw_steps: self.raw_steps,
            raw_index_map: self.raw_index_map.clone(),
            beta: self.beta,
        }
    }

    /// Restore detector from saved state.
    ///
    /// Checks rl_log and stats lengths against max_rl + 1 and
    /// returns an error when either check fails.
    pub fn restore(state: DetectorState) -> Result<Self, String> {
        let expected = state.max_rl + 1;
        if state.rl_log.len() != expected {
            return Err(format!(
                "rl_log length {} does not match max_rl + 1 = {expected}",
                state.rl_log.len()
            ));
        }
        if state.stats.len() != expected {
            return Err(format!(
                "stats length {} does not match max_rl + 1 = {expected}",
                state.stats.len()
            ));
        }
        let prior = Nig {
            mu: 0.0,
            kappa: 1.0,
            alpha: 1.0,
            beta: 1.0,
        };
        Ok(Self {
            hazard_log: state.hazard_log,
            growth_log: state.growth_log,
            max_rl: state.max_rl,
            log_mass_cutoff: crate::DEFAULT_MASS_CUTOFF.ln(),
            beta: state.beta,
            prior: prior.clone(),
            rl_log: state
                .rl_log
                .iter()
                .map(|&v| {
                    if v <= f64::MIN + 1.0 {
                        f64::NEG_INFINITY
                    } else {
                        v
                    }
                })
                .collect(),
            stats: state.stats.iter().map(Nig::from).collect(),
            map_rls: state.map_rls,
            cp_probs: if state.cp_probs.is_empty() {
                // v0.2 forward-compat: synthesise zeros so old snapshots still load.
                vec![0.0; state.total_steps]
            } else {
                state.cp_probs
            },
            total_steps: state.total_steps,
            last_detection: state.last_detection,
            welford: state.welford,
            raw_steps: state.raw_steps,
            raw_index_map: state.raw_index_map,
            scratch_rl: vec![f64::NEG_INFINITY; state.max_rl + 1],
            scratch_stats: vec![prior; state.max_rl + 1],
            norm_ring: std::collections::VecDeque::with_capacity(2 * SHIFT_WINDOW),
            pending: Vec::new(),
        })
    }

    /// Total observations processed.
    pub fn total_steps(&self) -> usize {
        self.total_steps
    }

    /// Current MAP run length.
    pub fn current_map_rl(&self) -> usize {
        self.map_rls.last().copied().unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn streaming_detects_clean_shift() {
        let mut det = StreamingDetector::new(200.0, 250);
        let data: Vec<f64> = std::iter::repeat_n(0.0, 100)
            .chain(std::iter::repeat_n(5.0, 100))
            .collect();
        let cps = det.step(&data);
        assert!(!cps.is_empty(), "should detect the mean shift");
        assert!(
            (cps[0].index as i64 - 100).abs() < 20,
            "change point near 100, got {}",
            cps[0].index
        );
    }

    #[test]
    fn streaming_no_detection_on_constant() {
        let mut det = StreamingDetector::new(200.0, 250);
        let data = vec![1.0; 200];
        let cps = det.step(&data);
        assert!(cps.is_empty(), "constant signal should have no detections");
    }

    #[test]
    fn streaming_incremental_equals_batch() {
        // Feed data in chunks vs all at once -- should detect at same location
        let mut data = vec![0.0; 100];
        data.extend(vec![5.0; 100]);

        // Batch
        let mut det_batch = StreamingDetector::new(200.0, 250);
        let batch_cps = det_batch.step(&data);

        // Incremental (10-point chunks)
        let mut det_inc = StreamingDetector::new(200.0, 250);
        let mut inc_cps = Vec::new();
        for chunk in data.chunks(10) {
            inc_cps.extend(det_inc.step(chunk));
        }

        assert_eq!(
            det_batch.total_steps(),
            det_inc.total_steps(),
            "total steps should match"
        );

        // Both should detect, at the same location
        assert!(!batch_cps.is_empty(), "batch should detect");
        assert!(!inc_cps.is_empty(), "incremental should detect");
        assert!(
            (batch_cps[0].index as i64 - inc_cps[0].index as i64).abs() <= 1,
            "detection locations should match: batch={}, inc={}",
            batch_cps[0].index,
            inc_cps[0].index
        );
    }

    #[test]
    fn state_save_restore() {
        let mut det = StreamingDetector::new(200.0, 250);
        let phase1: Vec<f64> = std::iter::repeat_n(0.0, 80).collect();
        det.step(&phase1);

        // Save state
        let state = det.save_state();
        let json = serde_json::to_string(&state).unwrap();
        assert!(!json.is_empty());

        // Restore
        let restored_state: DetectorState = serde_json::from_str(&json).unwrap();
        let mut det2 = StreamingDetector::restore(restored_state).unwrap();

        // Continue with same data on both
        let phase2: Vec<f64> = std::iter::repeat_n(5.0, 120).collect();
        let cps1 = det.step(&phase2);
        let cps2 = det2.step(&phase2);

        assert_eq!(det.total_steps(), det2.total_steps());
        assert_eq!(
            cps1.len(),
            cps2.len(),
            "restored detector should match original"
        );
        for (a, b) in cps1.iter().zip(cps2.iter()) {
            assert_eq!(
                a.index, b.index,
                "detection indices should match after restore"
            );
        }
    }

    #[test]
    fn state_serde_roundtrip() {
        let mut det = StreamingDetector::new(200.0, 250);
        det.step(&[1.0, 2.0, 3.0]);
        let state = det.save_state();
        let json = serde_json::to_string(&state).unwrap();
        let restored: DetectorState = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.total_steps, 3);
        assert_eq!(restored.max_rl, 250);
    }

    #[test]
    fn nan_in_step_does_not_panic() {
        let mut det = StreamingDetector::new(200.0, 250);
        let mut data: Vec<f64> = vec![0.0; 100];
        data.extend(vec![5.0; 100]);
        data[20] = f64::NAN;
        data[110] = f64::NAN;
        // Must not panic
        let cps = det.step(&data);
        assert!(
            !cps.is_empty(),
            "NaN-containing data should still produce detections"
        );
    }

    #[test]
    fn all_nan_step_returns_empty_without_panic() {
        let mut det = StreamingDetector::new(200.0, 100);
        let data = vec![f64::NAN; 50];
        let cps = det.step(&data);
        assert!(cps.is_empty(), "all-NaN step should return empty vec");
    }

    #[test]
    fn welford_matches_batch_stats() {
        // Streaming Welford must converge to the population mean and
        // population std (not sample std) within machine ε after N updates.
        // Use a fixed-seed sequence; population sigma = 2.0, mu = 7.0.
        let mut rng = crate::eval::Rng::new(31_337);
        let data: Vec<f64> = (0..2000).map(|_| rng.normal(7.0, 2.0)).collect();

        let mut w = WelfordState::new();
        for &x in &data {
            w.update(x);
        }

        let n = data.len() as f64;
        let batch_mean = data.iter().sum::<f64>() / n;
        let batch_var = data.iter().map(|x| (x - batch_mean).powi(2)).sum::<f64>() / n;
        let batch_std = batch_var.sqrt();

        // Population stats -- should match exactly (modulo float reorder).
        assert!(
            (w.mean - batch_mean).abs() < 1e-10,
            "welford mean drifted: {} vs {}",
            w.mean,
            batch_mean
        );
        assert!(
            (w.std() - batch_std).abs() < 1e-10,
            "welford std drifted: {} vs {}",
            w.std(),
            batch_std
        );
    }

    #[test]
    fn welford_online_stats() {
        let mut w = WelfordState::new();
        for x in [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0] {
            w.update(x);
        }
        assert!(
            (w.mean - 5.0).abs() < 0.01,
            "mean should be 5.0, got {}",
            w.mean
        );
        assert!(
            (w.std() - 2.0).abs() < 0.1,
            "std should be ~2.0, got {}",
            w.std()
        );
    }

    #[test]
    fn streaming_multiple_regimes() {
        let mut det = StreamingDetector::new(200.0, 300);
        let mut data = vec![0.0; 80];
        data.extend(vec![5.0; 80]);
        data.extend(vec![-3.0; 80]);
        let cps = det.step(&data);
        assert!(
            cps.len() >= 2,
            "should detect ≥2 regime changes, got {}",
            cps.len()
        );
    }

    // ── Acceptance criteria tests ────────────────────────────

    #[test]
    fn step_matches_batch_detect() {
        // AC#1: step() produces results at same location as detect()
        // Note: streaming uses Welford online normalization vs batch global normalization,
        // so we test location equivalence within ±5 steps, not bitwise equality.
        use crate::BocpdDetector;

        let mut data = vec![0.0; 100];
        data.extend(vec![5.0; 100]);

        let batch = BocpdDetector::new(200.0, 250);
        let batch_cps = batch.detect(&data);

        let mut streaming = StreamingDetector::new(200.0, 250);
        let stream_cps = streaming.step(&data);

        assert!(!batch_cps.is_empty(), "batch should detect");
        assert!(!stream_cps.is_empty(), "streaming should detect");
        assert!(
            (batch_cps[0].index as i64 - stream_cps[0].index as i64).abs() <= 5,
            "detection location should be within ±5 steps: batch={}, streaming={}",
            batch_cps[0].index,
            stream_cps[0].index
        );
    }

    #[test]
    fn step_matches_batch_detect_noisy() {
        // AC#1 with noisy data
        use crate::eval::Rng;
        use crate::BocpdDetector;

        let mut rng = Rng::new(42);
        let mut data: Vec<f64> = (0..150).map(|_| rng.normal(0.0, 1.0)).collect();
        data.extend((0..150).map(|_| rng.normal(5.0, 1.0)));

        let batch = BocpdDetector::new(200.0, 350);
        let batch_cps = batch.detect(&data);

        let mut streaming = StreamingDetector::new(200.0, 350);
        let stream_cps = streaming.step(&data);

        assert!(!batch_cps.is_empty(), "batch should detect noisy shift");
        assert!(
            !stream_cps.is_empty(),
            "streaming should detect noisy shift"
        );
        assert!(
            (batch_cps[0].index as i64 - stream_cps[0].index as i64).abs() <= 10,
            "noisy detection within ±10 steps: batch={}, streaming={}",
            batch_cps[0].index,
            stream_cps[0].index
        );
    }

    #[test]
    fn step_matches_batch_detect_no_false_positives() {
        // AC#1: both should agree on "no detection" for stationary noise
        use crate::eval::Rng;
        use crate::BocpdDetector;

        let mut rng = Rng::new(999);
        let data: Vec<f64> = (0..300).map(|_| rng.normal(0.0, 1.0)).collect();

        let batch = BocpdDetector::new(200.0, 350);
        let batch_cps = batch.detect(&data);

        let mut streaming = StreamingDetector::new(200.0, 350);
        let stream_cps = streaming.step(&data);

        // Both should have very few (ideally zero) false positives
        assert!(
            batch_cps.len() <= 1,
            "batch FPs on stationary: {}",
            batch_cps.len()
        );
        assert!(
            stream_cps.len() <= 1,
            "streaming FPs on stationary: {}",
            stream_cps.len()
        );
    }

    #[test]
    fn streaming_eval_suite_quality() {
        // AC#1 aggregate: run streaming on the eval suite, verify quality doesn't degrade
        use crate::eval::{self, Category};

        let scenarios = eval::all_scenarios();
        let mut metrics = Vec::new();

        for s in &scenarios {
            let mut det = StreamingDetector::new(200.0, 350);
            let cps = det.step(&s.data);
            let detected: Vec<usize> = cps.iter().map(|c| c.index).collect();
            let mut m = eval::match_detections(&detected, &s.ground_truth, 25); // wider tolerance for streaming
            m.name = s.name.to_string();
            m.category = s.category;
            metrics.push(m);
        }

        eval::print_report(&metrics);
        let agg = eval::aggregate(&metrics);

        // Streaming may have slightly lower quality than batch due to
        // online normalization, but should still pass reasonable baselines
        assert!(agg.f1 >= 0.35, "streaming F1={:.2}, need ≥0.35", agg.f1);
        assert!(
            agg.recall >= 0.50,
            "streaming recall={:.2}, need ≥0.50",
            agg.recall
        );

        // MustDetect scenarios should still work
        let md_recall: f64 = {
            let md: Vec<_> = metrics
                .iter()
                .filter(|m| m.category == Category::MustDetect)
                .collect();
            let tp: usize = md.iter().map(|m| m.tp).sum();
            let fn_count: usize = md.iter().map(|m| m.r#fn).sum();
            if tp + fn_count > 0 {
                tp as f64 / (tp + fn_count) as f64
            } else {
                1.0
            }
        };
        assert!(
            md_recall >= 0.50,
            "streaming MustDetect recall={:.2}, need ≥0.50",
            md_recall
        );
    }

    #[test]
    fn restore_rejects_mismatched_rl_log() {
        let mut det = StreamingDetector::new(200.0, 100);
        det.step(&[1.0, 2.0, 3.0]);
        let mut state = det.save_state();
        state.rl_log.push(0.0); // make it too long
        match StreamingDetector::restore(state) {
            Ok(_) => panic!("expected error for mismatched rl_log"),
            Err(e) => assert!(e.contains("rl_log length"), "got: {e}"),
        }
    }

    #[test]
    fn restore_rejects_mismatched_stats() {
        let mut det = StreamingDetector::new(200.0, 100);
        det.step(&[1.0, 2.0, 3.0]);
        let mut state = det.save_state();
        state.stats.pop(); // make it too short
        match StreamingDetector::restore(state) {
            Ok(_) => panic!("expected error for mismatched stats"),
            Err(e) => assert!(e.contains("stats length"), "got: {e}"),
        }
    }

    #[test]
    #[should_panic(expected = "lambda must be > 1.0")]
    fn streaming_rejects_lambda_one() {
        StreamingDetector::new(1.0, 100);
    }

    #[test]
    #[should_panic(expected = "lambda must be > 1.0")]
    fn streaming_rejects_lambda_below_one() {
        StreamingDetector::new(0.5, 100);
    }

    #[test]
    fn nan_gaps_use_raw_indices() {
        // Input with NaN at positions 5..10 -- change point indices must reference
        // original input positions, not compressed model steps.
        let mut data: Vec<f64> = vec![0.0; 100];
        data.extend(vec![5.0; 100]);
        // Insert NaN gap
        for v in data.iter_mut().take(10).skip(5) {
            *v = f64::NAN;
        }
        let mut det = StreamingDetector::new(200.0, 250);
        let cps = det.step(&data);
        assert!(!cps.is_empty(), "should detect the shift");
        // raw_steps should equal total input length (including NaN)
        assert_eq!(det.raw_steps, 200);
        // The detected index should be near 100 in raw coordinates
        assert!(
            (cps[0].index as i64 - 100).abs() < 20,
            "change point index should be near 100 in raw coords, got {}",
            cps[0].index
        );
        // Index must be >= 100 (the shift is at raw position 100)
        // and must not be shifted by the 5 NaN values
        assert!(
            cps[0].index >= 90,
            "index {} is too low -- NaN gap shifted it incorrectly",
            cps[0].index
        );
    }

    #[test]
    fn benchmark_streaming_tick() {
        // AC#4: streaming tick < 10ms for 100 signals
        use std::time::Instant;

        // Simulate 100 signals, each getting 5 new observations per tick
        let mut detectors: Vec<StreamingDetector> = (0..100)
            .map(|_| StreamingDetector::new(200.0, 400))
            .collect();

        // Warm up with 50 points each
        for det in &mut detectors {
            let warmup: Vec<f64> = (0..50).map(|i| (i as f64 * 0.1).sin()).collect();
            det.step(&warmup);
        }

        // Benchmark: 5 new points per signal (simulating 5m capture at 1m step)
        let tick_data: Vec<f64> = vec![1.0, 1.1, 0.9, 1.2, 0.8];
        let start = Instant::now();
        for det in &mut detectors {
            det.step(&tick_data);
        }
        let elapsed = start.elapsed();

        eprintln!(
            "streaming tick: 100 signals × 5 points = {:?} ({:.2}ms)",
            elapsed,
            elapsed.as_secs_f64() * 1000.0
        );
        assert!(
            elapsed.as_millis() < 60,
            "streaming tick took {}ms, need <60ms (debug) / <10ms (release)",
            elapsed.as_millis()
        );
    }
}
