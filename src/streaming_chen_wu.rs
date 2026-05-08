//! Streaming Chen & Wu joint CP + collective-anomaly detector.
//! Online counterpart to [`crate::chen_wu::ChenWuDetector`]; same
//! recursion (paper § 4.1 BOCD-AR sequential), same emission rules
//! (eq. 12 / 13), same § 4.3 Υ_c^t persistence.
//!
//! Raw observations and retained ℍ_c history are bounded: respectively
//! O(u_c + u_a·Δt) and O(u_a·Δt), with history truncated to the first Δt
//! run lengths (paper § 4.2). Exclusion ranges and emitted-anomaly
//! bookkeeping can grow with detections. Save / restore uses
//! [`ChenWuDetectorState`].

use std::collections::{HashSet, VecDeque};

use serde::{Deserialize, Serialize};

use crate::chen_wu::{a_constraint_pub, rebuild_stats_full_pub, Detection};
use crate::streaming::NigState;
use crate::{log_add_exp, ChangePoint, Nig};

const NEG_INF: f64 = f64::NEG_INFINITY;
const COOLDOWN: usize = 15;

/// Streaming Chen & Wu BOCD-AR detector. Mirrors
/// [`crate::chen_wu::ChenWuDetector`] with bounded observation/history buffers
/// across calls to [`step`](Self::step) so production daemons can
/// process observations incrementally and persist state across
/// restarts.
pub struct StreamingChenWuDetector {
    p0: f64,
    q0: f64,
    delta_t: usize,
    u_c: usize,
    u_a: usize,
    lambda_a: f64,
    lambda_c: f64,
    delta: usize,
    min_post_change_obs: usize,
    prior: Nig,
    beta: f64,

    log_p0: f64,
    log_1_minus_p0: f64,
    log_q0: f64,
    log_1_minus_q0: f64,

    // Recursion state. log_h_a / log_h_c hold the most-recently-computed
    // ℍ_a^t(r) / ℍ_c^t(r) vectors (length n_c+1 at step t).
    log_h_a: Vec<f64>,
    log_h_c: Vec<f64>,
    // Past steps' log_h_c truncated to the first Δt run lengths --
    // exactly the slice the r2 argmax (paper § 4.2) reads. front =
    // oldest (paper-time = total_steps - h_c_history.len()), back =
    // newest (paper-time = total_steps - 1 at end of step t).
    h_c_history: VecDeque<Vec<f64>>,

    // § 4.3 persistence + observation buffer.
    raw_history: VecDeque<f64>,
    abs_origin: usize,
    excluded_ranges: Vec<(usize, usize)>,

    // Detection bookkeeping.
    emitted_anom: HashSet<(usize, usize)>,
    last_cp: Option<(usize, usize)>,
    total_steps: usize,
}

/// Serialised streaming-detector state. `#[serde(default)]` on
/// extension fields keeps older snapshots loadable as the schema
/// evolves; see [`crate::streaming::DetectorState`] for the same
/// pattern.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChenWuDetectorState {
    pub p0: f64,
    pub q0: f64,
    pub delta_t: usize,
    pub u_c: usize,
    pub u_a: usize,
    pub lambda_a: f64,
    pub lambda_c: f64,
    pub delta: usize,
    pub min_post_change_obs: usize,
    pub prior: NigState,
    #[serde(default)]
    pub beta: f64,

    pub log_h_a: Vec<f64>,
    pub log_h_c: Vec<f64>,
    pub h_c_history: Vec<Vec<f64>>,

    pub raw_history: Vec<f64>,
    pub abs_origin: usize,
    pub excluded_ranges: Vec<(usize, usize)>,

    pub emitted_anom: Vec<(usize, usize)>,
    pub last_cp: Option<(usize, usize)>,
    pub total_steps: usize,
}

impl StreamingChenWuDetector {
    /// Five core hyperparameters; defaults match
    /// [`crate::chen_wu::ChenWuDetector::new`].
    pub fn new(p0: f64, q0: f64, delta_t: usize, lambda_a: f64, lambda_c: f64) -> Self {
        assert!((0.0..1.0).contains(&p0), "p0 must be in [0, 1), got {p0}");
        assert!((0.0..1.0).contains(&q0), "q0 must be in [0, 1), got {q0}");
        assert!(delta_t >= 1, "delta_t must be >= 1, got {delta_t}");
        assert!(
            (0.0..1.0).contains(&lambda_a),
            "lambda_a must be in [0, 1), got {lambda_a}"
        );
        assert!(
            (0.0..1.0).contains(&lambda_c),
            "lambda_c must be in [0, 1), got {lambda_c}"
        );
        Self {
            p0,
            q0,
            delta_t,
            u_c: 256,
            u_a: 32,
            lambda_a,
            lambda_c,
            delta: 0,
            min_post_change_obs: 5,
            prior: Nig::new(0.0, 0.01, 0.5, 0.125),
            beta: 0.0,
            log_p0: p0.ln(),
            log_1_minus_p0: (1.0 - p0).ln(),
            log_q0: q0.ln(),
            log_1_minus_q0: (1.0 - q0).ln(),
            log_h_a: Vec::new(),
            log_h_c: Vec::new(),
            h_c_history: VecDeque::new(),
            raw_history: VecDeque::new(),
            abs_origin: 1,
            excluded_ranges: Vec::new(),
            emitted_anom: HashSet::new(),
            last_cp: None,
            total_steps: 0,
        }
    }

    pub fn with_search_windows(mut self, u_c: usize, u_a: usize) -> Self {
        assert!(u_c >= u_a, "u_c must be >= u_a");
        self.u_c = u_c;
        self.u_a = u_a;
        self
    }

    /// Replace the `q_0` passed to [`new`](Self::new) with the largest
    /// value satisfying paper § 5 / eq. 14 for the current
    /// `(p_0, λ_a, Δt)`. Mirrors
    /// [`crate::chen_wu::ChenWuDetector::with_auto_q0`]; rebuilds the
    /// `log q_0` / `log (1 - q_0)` caches the streaming recursion uses
    /// per step.
    ///
    /// Idempotent after a single call.
    pub fn with_auto_q0(mut self) -> Self {
        self.q0 = crate::auto_q0::q0_upper_bound(self.p0, self.lambda_a, self.delta_t);
        self.log_q0 = self.q0.ln();
        self.log_1_minus_q0 = (1.0 - self.q0).ln();
        self
    }

    pub fn with_localisation_tolerance(mut self, delta: usize) -> Self {
        self.delta = delta;
        self
    }

    pub fn with_min_post_change(mut self, n: usize) -> Self {
        self.min_post_change_obs = n;
        self
    }

    pub fn with_prior(mut self, mu: f64, kappa: f64, alpha: f64, beta: f64) -> Self {
        self.prior = Nig::new(mu, kappa, alpha, beta);
        self
    }

    /// Opt the within-regime likelihood into β-divergence (Knoblauch
    /// et al. 2018). See
    /// [`crate::chen_wu::ChenWuDetector::with_robust`] for full notes.
    pub fn with_robust(mut self, beta: f64) -> Self {
        assert!(
            (0.0..=1.0).contains(&beta),
            "beta must be in [0.0, 1.0], got {beta}"
        );
        self.beta = beta;
        self
    }

    pub fn total_steps(&self) -> usize {
        self.total_steps
    }

    fn raw_capacity(&self) -> usize {
        self.u_c + self.u_a * self.delta_t + 1
    }

    fn likelihood(&self, n: &Nig, x: f64) -> f64 {
        if self.beta > 0.0 {
            return n.log_predictive_robust(x, self.beta);
        }
        n.log_predictive(x)
    }

    /// Process a chunk of observations. Returns the timeline of
    /// detections that fired during this call, in the same firing
    /// order [`crate::chen_wu::ChenWuDetector::detect`] would emit.
    ///
    /// Non-finite samples (NaN, ±∞) are silently skipped: they do not
    /// advance `total_steps`, do not enter `raw_history`, and produce
    /// no detection. Mirrors [`crate::streaming::StreamingDetector::step`]'s
    /// guard. The joint detector does not maintain a separate raw-step
    /// counter, so emitted `Detection` indices reference the post-skip
    /// step count, not the original chunk position.
    pub fn step(&mut self, observations: &[f64]) -> Vec<Detection> {
        let mut detections = Vec::new();
        for &y in observations {
            if !y.is_finite() {
                continue;
            }
            self.step_one(y, &mut detections);
        }
        detections
    }

    fn step_one(&mut self, y_t: f64, out: &mut Vec<Detection>) {
        let t_paper = self.total_steps + 1;

        if t_paper == 1 {
            // Paper t = 1 initialisation, mirrors batch detect().
            let log_l_y1 = self.likelihood(&self.prior, y_t);
            self.log_h_a = vec![NEG_INF];
            self.log_h_c = vec![log_l_y1];
            self.raw_history.push_back(y_t);
            self.total_steps = 1;
            // No detections fire at t = 1; a CP needs r* ≥ min_post_change.
            return;
        }

        let n_c = (t_paper - 1).min(self.u_c);

        // Stats reflect every exclusion accumulated through step t-1.
        let stats = rebuild_stats_full_pub(
            &self.prior,
            &self.raw_history,
            self.abs_origin,
            &self.excluded_ranges,
            t_paper,
            n_c,
        );

        let log_l_y_t = self.likelihood(&self.prior, y_t);
        let mut log_pred = vec![NEG_INF; n_c + 1];
        for r in 1..=n_c {
            if r < stats.len() {
                log_pred[r] = self.likelihood(&stats[r], y_t);
            }
        }

        let mut new_log_h_a = vec![NEG_INF; n_c + 1];
        let mut new_log_h_c = vec![NEG_INF; n_c + 1];

        // Eq. 7
        for r in 1..=n_c {
            let prev_idx = r - 1;
            let prev_log_h_a = self.log_h_a.get(prev_idx).copied().unwrap_or(NEG_INF);
            new_log_h_a[r] = prev_log_h_a + log_pred[r] + self.log_1_minus_p0;
        }
        let a_bound = a_constraint_pub(t_paper, t_paper - 1, self.delta_t);
        if a_bound >= 0 {
            let upper = (a_bound as usize).min(self.log_h_c.len().saturating_sub(1));
            let mut log_sum = NEG_INF;
            for rp in 0..=upper {
                log_sum = log_add_exp(log_sum, self.log_h_c[rp]);
            }
            new_log_h_a[0] = log_sum + log_l_y_t + self.log_q0;
        }

        // Eq. 8
        for r in 1..=n_c {
            let prev_log_h_c = self.log_h_c.get(r - 1).copied().unwrap_or(NEG_INF);
            let log_factor = if r > self.delta_t || r == t_paper - 1 {
                self.log_1_minus_p0
            } else {
                self.log_1_minus_q0
            };
            new_log_h_c[r] = prev_log_h_c + log_pred[r] + log_factor;
        }
        // r = 0
        let mut log_sum_h_a_prev = NEG_INF;
        for &v in &self.log_h_a {
            log_sum_h_a_prev = log_add_exp(log_sum_h_a_prev, v);
        }
        let log_r0_factor = if t_paper >= self.delta_t + 3 {
            let mut log_sum_c = NEG_INF;
            let lo = self.delta_t;
            for rp in lo..self.log_h_c.len() {
                log_sum_c = log_add_exp(log_sum_c, self.log_h_c[rp]);
            }
            log_add_exp(log_sum_c, log_sum_h_a_prev)
        } else {
            let last_idx = (t_paper - 2).min(self.log_h_c.len().saturating_sub(1));
            let log_h_c_last = self.log_h_c.get(last_idx).copied().unwrap_or(NEG_INF);
            log_add_exp(log_h_c_last, log_sum_h_a_prev)
        };
        new_log_h_c[0] = log_r0_factor + log_l_y_t + self.log_p0;

        // Renormalise.
        renormalise(&mut new_log_h_a, &mut new_log_h_c);

        let n_a = (t_paper - 1).min(self.u_a);
        let mut mask = vec![false; n_c + 1];
        let mut latest_r_star: usize = 0;

        // Detection emission (paper Algorithm 1 + § 4.2). Mirrors batch
        // `ChenWuDetector::detect`'s log-space inner loop -- argmax,
        // ratios, and bounded sums all stay in log-space; `.exp()` lands
        // only at emission. See chen_wu.rs for the full commentary.
        for _iter in 0..=n_a {
            let mut r_star = 0;
            let mut best = NEG_INF;
            let mut found = false;
            for r in 0..=n_c {
                if mask[r] {
                    continue;
                }
                let combined = log_add_exp(new_log_h_a[r], new_log_h_c[r]);
                if combined > best {
                    best = combined;
                    r_star = r;
                    found = true;
                }
            }
            if !found {
                break;
            }
            latest_r_star = r_star;

            if r_star > n_a || r_star < 2 {
                break;
            }
            let lo = r_star.saturating_sub(self.delta_t);
            let hi = r_star;
            let mut log_num = NEG_INF;
            let mut log_den = NEG_INF;
            for r in lo..=hi {
                if r >= new_log_h_a.len() || mask[r] {
                    continue;
                }
                log_num = log_add_exp(log_num, new_log_h_a[r]);
                let combined = log_add_exp(new_log_h_a[r], new_log_h_c[r]);
                log_den = log_add_exp(log_den, combined);
            }
            if log_den == NEG_INF {
                break;
            }
            let log_lambda_a = self.lambda_a.ln();
            if log_num - log_den <= log_lambda_a {
                break;
            }
            let posterior = (log_num - log_den).exp();

            // Locate r1 via argmax log_h_a; argmax in log preserves linear.
            let mut r1 = lo;
            let mut best1 = NEG_INF;
            for r in lo..=hi {
                if r < new_log_h_a.len() && !mask[r] && new_log_h_a[r] > best1 {
                    best1 = new_log_h_a[r];
                    r1 = r;
                }
            }
            let prior_t = t_paper.saturating_sub(r1).saturating_sub(1);
            let r2 = self.r2_lookup(prior_t);
            let end_paper = t_paper - r1;
            let start_paper = end_paper.saturating_sub(r2 + 1);
            if end_paper < 1 || start_paper < 1 {
                break;
            }
            let new_window = (start_paper - 1, end_paper - 1);
            if !self.emitted_anom.contains(&new_window) {
                out.push(Detection::CollectiveAnomaly {
                    start: new_window.0,
                    end: new_window.1,
                    confidence: posterior,
                });
                self.emitted_anom.insert(new_window);
            }

            let r_lo = t_paper.saturating_sub(end_paper);
            let r_hi = t_paper.saturating_sub(start_paper);

            self.excluded_ranges.push((start_paper, end_paper));
            for r in r_lo..=r_hi.min(n_c) {
                mask[r] = true;
                new_log_h_a[r] = NEG_INF;
                new_log_h_c[r] = NEG_INF;
            }

            renormalise(&mut new_log_h_a, &mut new_log_h_c);
        }

        // Eq. 13 CP test, log-space mirror of batch.
        let r_star = latest_r_star;
        let lo = r_star.saturating_sub(self.delta);
        let hi = (r_star + self.delta).min(n_c);
        let mut log_numer = NEG_INF;
        let mut log_denom = NEG_INF;
        for r in 0..=n_c {
            if r >= new_log_h_a.len() {
                break;
            }
            let m_log = if mask[r] {
                NEG_INF
            } else {
                log_add_exp(new_log_h_a[r], new_log_h_c[r])
            };
            log_denom = log_add_exp(log_denom, m_log);
            if r >= lo && r <= hi {
                log_numer = log_add_exp(log_numer, m_log);
            }
        }
        let p_band = if log_denom > NEG_INF {
            (log_numer - log_denom).exp()
        } else {
            0.0
        };

        if p_band > self.lambda_c && r_star >= self.min_post_change_obs {
            let cp_paper = t_paper - r_star;
            if cp_paper >= 1 {
                let cp_idx = cp_paper - 1;
                let suppress = match self.last_cp {
                    Some((idx, t_prev)) => {
                        cp_idx.abs_diff(idx) <= COOLDOWN || t_paper - t_prev < COOLDOWN
                    }
                    None => false,
                };
                if !suppress {
                    let shift_sigma = streaming_shift_sigma(
                        &self.raw_history,
                        self.abs_origin,
                        y_t,
                        t_paper,
                        cp_idx,
                    );
                    out.push(Detection::ChangePoint(ChangePoint {
                        index: cp_idx,
                        confidence: p_band,
                        shift_sigma,
                    }));
                    self.last_cp = Some((cp_idx, t_paper));
                }
            }
        }

        // Push y_t to bounded raw_history.
        self.raw_history.push_back(y_t);
        let cap = self.raw_capacity();
        while self.raw_history.len() > cap {
            self.raw_history.pop_front();
            self.abs_origin += 1;
        }

        // Roll the h_c retention ring: push the previous step's
        // log_h_c (truncated) to the back, drop the front if longer
        // than u_a.
        let mut prev_hc_truncated: Vec<f64> =
            self.log_h_c.iter().take(self.delta_t).copied().collect();
        prev_hc_truncated.shrink_to_fit();
        self.h_c_history.push_back(prev_hc_truncated);
        while self.h_c_history.len() > self.u_a {
            self.h_c_history.pop_front();
        }

        // Promote new step's vectors.
        self.log_h_a = new_log_h_a;
        self.log_h_c = new_log_h_c;
        self.total_steps = t_paper;
    }

    /// Look up `ℍ_c^(prior_t)(r)` for `r ∈ [0, Δt-1]` and return the
    /// argmax index. Mirrors batch's r2 sub-routine.
    fn r2_lookup(&self, prior_t: usize) -> usize {
        if prior_t == 0 {
            return 0;
        }
        // Prior-step entry. Identify which slice in
        // h_c_history (or self.log_h_c if prior_t == total_steps) maps
        // to prior_t.
        let slice: &[f64] = if prior_t == self.total_steps {
            &self.log_h_c
        } else if prior_t > self.total_steps {
            return 0;
        } else {
            // h_c_history is rolled at the end of each step: after
            // step t completes the just-overwritten log_h_c (which
            // held paper-time t-1's vector) is pushed onto the ring.
            // After step T, the ring holds log_h_c at paper-times
            //   [T - h_c_history.len() .. T - 1]
            // (clamped to [1, T-1] when the ring isn't yet full).
            let oldest_pt = self.total_steps.saturating_sub(self.h_c_history.len());
            if prior_t < oldest_pt {
                return 0;
            }
            let idx = prior_t - oldest_pt;
            if idx >= self.h_c_history.len() {
                return 0;
            }
            &self.h_c_history[idx]
        };
        let mut best2 = NEG_INF;
        let mut idx = 0;
        let bound = self.delta_t.saturating_sub(1).min(slice.len().saturating_sub(1));
        for (r, &v) in slice.iter().enumerate().take(bound + 1) {
            if v > best2 {
                best2 = v;
                idx = r;
            }
        }
        idx
    }

    pub fn save_state(&self) -> ChenWuDetectorState {
        // serde_json renders f64::NEG_INFINITY as `null`, which then
        // fails to deserialise back into f64. Match StreamingDetector::
        // save_state and clamp non-finite log values to f64::MIN; the
        // restore path maps them back to NEG_INFINITY.
        let pack = |v: f64| if v.is_finite() { v } else { f64::MIN };
        ChenWuDetectorState {
            p0: self.p0,
            q0: self.q0,
            delta_t: self.delta_t,
            u_c: self.u_c,
            u_a: self.u_a,
            lambda_a: self.lambda_a,
            lambda_c: self.lambda_c,
            delta: self.delta,
            min_post_change_obs: self.min_post_change_obs,
            prior: NigState::from(&self.prior),
            beta: self.beta,
            log_h_a: self.log_h_a.iter().map(|&v| pack(v)).collect(),
            log_h_c: self.log_h_c.iter().map(|&v| pack(v)).collect(),
            h_c_history: self
                .h_c_history
                .iter()
                .map(|v| v.iter().map(|&x| pack(x)).collect())
                .collect(),
            raw_history: self.raw_history.iter().copied().collect(),
            abs_origin: self.abs_origin,
            excluded_ranges: self.excluded_ranges.clone(),
            emitted_anom: self.emitted_anom.iter().copied().collect(),
            last_cp: self.last_cp,
            total_steps: self.total_steps,
        }
    }

    /// Reconstruct a detector from a previously-serialised state.
    ///
    /// Performs the scalar-range and vector-geometry checks below
    /// before constructing the detector. These checks do not establish
    /// every possible state invariant. Uses the length-and-range pattern in
    /// [`crate::streaming::StreamingDetector::restore`].
    pub fn restore(state: ChenWuDetectorState) -> Result<Self, String> {
        // Scalar range / finiteness checks. Run before construction so
        // a hostile state never reaches `new()` / builders that panic.
        if !(0.0..1.0).contains(&state.p0) {
            return Err(format!("p0 out of range: {}", state.p0));
        }
        if !(0.0..1.0).contains(&state.q0) {
            return Err(format!("q0 out of range: {}", state.q0));
        }
        if state.delta_t < 1 {
            return Err("delta_t must be >= 1".into());
        }
        if !(0.0..1.0).contains(&state.lambda_a) {
            return Err(format!("lambda_a out of range: {}", state.lambda_a));
        }
        if !(0.0..1.0).contains(&state.lambda_c) {
            return Err(format!("lambda_c out of range: {}", state.lambda_c));
        }
        if !(0.0..=1.0).contains(&state.beta) {
            return Err(format!("beta out of range: {}", state.beta));
        }
        if state.u_c < state.u_a {
            return Err(format!(
                "u_c ({}) must be >= u_a ({})",
                state.u_c, state.u_a
            ));
        }
        if state.delta > state.u_c {
            return Err(format!(
                "delta ({}) must be <= u_c ({})",
                state.delta, state.u_c
            ));
        }
        if state.min_post_change_obs < 1 {
            return Err("min_post_change_obs must be >= 1".into());
        }
        if state.min_post_change_obs > state.u_c {
            return Err(format!(
                "min_post_change_obs ({}) must be <= u_c ({})",
                state.min_post_change_obs, state.u_c
            ));
        }
        for (label, v) in [
            ("p0", state.p0),
            ("q0", state.q0),
            ("lambda_a", state.lambda_a),
            ("lambda_c", state.lambda_c),
            ("beta", state.beta),
            ("prior.mu", state.prior.mu),
            ("prior.kappa", state.prior.kappa),
            ("prior.alpha", state.prior.alpha),
            ("prior.beta", state.prior.beta),
        ] {
            if !v.is_finite() {
                return Err(format!("{label} is not finite: {v}"));
            }
        }

        // Vector geometry checks. The recursion indexes log_h_a/log_h_c
        // by r ∈ [0, n_c]; the per-step inner loops at
        // streaming_chen_wu.rs:248-282 panic on direct `[rp]` indexing
        // if these vectors are empty or mismatched.
        if state.log_h_a.len() != state.log_h_c.len() {
            return Err(format!(
                "log_h_a length {} != log_h_c length {}",
                state.log_h_a.len(),
                state.log_h_c.len()
            ));
        }
        if state.total_steps == 0 {
            if !state.log_h_a.is_empty()
                || !state.log_h_c.is_empty()
                || !state.h_c_history.is_empty()
                || !state.raw_history.is_empty()
            {
                return Err(
                    "total_steps == 0 requires log_h_a / log_h_c / h_c_history / raw_history all empty"
                        .into(),
                );
            }
        } else if state.log_h_a.is_empty() {
            return Err(format!(
                "total_steps {} > 0 requires non-empty log_h_a / log_h_c",
                state.total_steps
            ));
        }
        if state.log_h_a.len() > state.u_c + 1 {
            return Err(format!(
                "log_h_a length {} exceeds u_c + 1 = {}",
                state.log_h_a.len(),
                state.u_c + 1
            ));
        }
        if state.h_c_history.len() > state.u_a {
            return Err(format!(
                "h_c_history length {} exceeds u_a = {}",
                state.h_c_history.len(),
                state.u_a
            ));
        }
        for (idx, entry) in state.h_c_history.iter().enumerate() {
            if entry.len() > state.delta_t {
                return Err(format!(
                    "h_c_history[{idx}] length {} exceeds delta_t = {}",
                    entry.len(),
                    state.delta_t
                ));
            }
        }
        let raw_capacity = state.u_c + state.u_a * state.delta_t + 1;
        if state.raw_history.len() > raw_capacity {
            return Err(format!(
                "raw_history length {} exceeds raw_capacity = {raw_capacity}",
                state.raw_history.len()
            ));
        }
        if state.abs_origin < 1 {
            return Err(format!("abs_origin must be >= 1, got {}", state.abs_origin));
        }

        for (idx, &(start, end)) in state.excluded_ranges.iter().enumerate() {
            if start < 1 {
                return Err(format!(
                    "excluded_ranges[{idx}] start {start} must be >= 1"
                ));
            }
            if start > end {
                return Err(format!(
                    "excluded_ranges[{idx}] start {start} > end {end}"
                ));
            }
        }

        if let Some((idx, t_paper)) = state.last_cp {
            if idx >= state.total_steps {
                return Err(format!(
                    "last_cp idx {idx} must be < total_steps {}",
                    state.total_steps
                ));
            }
            if t_paper < 1 || t_paper > state.total_steps {
                return Err(format!(
                    "last_cp t_paper {t_paper} must be in [1, total_steps={}]",
                    state.total_steps
                ));
            }
        }

        // Validation passed -- now construct.
        let mut det = StreamingChenWuDetector::new(
            state.p0,
            state.q0,
            state.delta_t,
            state.lambda_a,
            state.lambda_c,
        )
        .with_search_windows(state.u_c, state.u_a)
        .with_localisation_tolerance(state.delta)
        .with_min_post_change(state.min_post_change_obs)
        .with_prior(
            state.prior.mu,
            state.prior.kappa,
            state.prior.alpha,
            state.prior.beta,
        );
        if state.beta > 0.0 {
            det = det.with_robust(state.beta);
        }
        let unpack = |v: f64| if v <= f64::MIN + 1.0 { NEG_INF } else { v };
        det.log_h_a = state.log_h_a.iter().map(|&v| unpack(v)).collect();
        det.log_h_c = state.log_h_c.iter().map(|&v| unpack(v)).collect();
        det.h_c_history = state
            .h_c_history
            .into_iter()
            .map(|v| v.into_iter().map(unpack).collect())
            .collect();
        det.raw_history = state.raw_history.into_iter().collect();
        det.abs_origin = state.abs_origin;
        det.excluded_ranges = state.excluded_ranges;
        det.emitted_anom = state.emitted_anom.into_iter().collect();
        det.last_cp = state.last_cp;
        det.total_steps = state.total_steps;
        Ok(det)
    }
}

fn renormalise(log_h_a: &mut [f64], log_h_c: &mut [f64]) {
    let mut log_total = NEG_INF;
    for &v in log_h_a.iter() {
        log_total = log_add_exp(log_total, v);
    }
    for &v in log_h_c.iter() {
        log_total = log_add_exp(log_total, v);
    }
    if log_total.is_finite() {
        for v in log_h_a.iter_mut() {
            *v -= log_total;
        }
        for v in log_h_c.iter_mut() {
            *v -= log_total;
        }
    }
}

/// shift_sigma at fire time, computed against the bounded raw_history
/// + the current `y_t` (which has not yet been pushed to history).
///
/// Equivalent to `shift_sigma_at(&data[..t_paper], cp_idx)` in batch
/// when raw_history covers the relevant window -- the contract
/// `matches_batch_detect` checks.
fn streaming_shift_sigma(
    raw_history: &VecDeque<f64>,
    abs_origin: usize,
    y_t: f64,
    t_paper: usize,
    cp_idx: usize,
) -> f64 {
    const W: usize = 20;
    let n_data = t_paper; // virtual prefix length
    let lo = cp_idx.saturating_sub(W);
    let hi = (cp_idx + W).min(n_data);
    if cp_idx <= lo || hi <= cp_idx {
        return 0.0;
    }
    let get_obs = |data_idx: usize| -> Option<f64> {
        let pt = data_idx + 1;
        if pt > t_paper {
            // Strictly future obs. hi clamps to n_data = t_paper so
            // this branch shouldn't fire in normal use, but the guard
            // keeps the closure total.
            None
        } else if pt == t_paper {
            Some(y_t)
        } else if pt >= abs_origin && pt < abs_origin + raw_history.len() {
            Some(raw_history[pt - abs_origin])
        } else {
            None
        }
    };
    let before: Vec<f64> = (lo..cp_idx).filter_map(get_obs).collect();
    let after: Vec<f64> = (cp_idx..hi).filter_map(get_obs).collect();
    if before.is_empty() || after.is_empty() {
        return 0.0;
    }
    let mean_b: f64 = before.iter().sum::<f64>() / before.len() as f64;
    let mean_a: f64 = after.iter().sum::<f64>() / after.len() as f64;
    let var_b: f64 =
        before.iter().map(|x| (x - mean_b).powi(2)).sum::<f64>() / before.len() as f64;
    let var_a: f64 = after.iter().map(|x| (x - mean_a).powi(2)).sum::<f64>() / after.len() as f64;
    let pooled = ((var_b + var_a) / 2.0).sqrt().max(1e-10);
    ((mean_a - mean_b) / pooled).abs()
}
