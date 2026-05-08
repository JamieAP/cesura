//! Chen & Wu (2025, arXiv:2508.06385) joint CP + collective-anomaly
//! detector. BOCD-AR sequential variant (paper § 4).
//!
//! Distinguishes "collective anomaly" (short reverting deviation) from
//! "genuine change point" (persistent shift) in one online recursion.
//! Output is a single ordered timeline of `Detection`s -- callers see
//! the same temporal interleaving the algorithm computed.
//!
//! Log-space recursion. Linear posterior probabilities are computed
//! when constructing emitted detections.

use std::collections::{HashSet, VecDeque};

use crate::{log_add_exp, ChangePoint, Nig};

const NEG_INF: f64 = f64::NEG_INFINITY;

/// Joint detection output. A timeline mixes both kinds; preserving the
/// order matters because Chen & Wu's anomaly-vs-CP classification can
/// flip late in the per-step loop (paper § 3.3).
#[derive(Debug, Clone)]
pub enum Detection {
    /// Genuine change point per paper eq. 13.
    ChangePoint(ChangePoint),
    /// Collective anomaly between `start` and `end` (inclusive),
    /// emitted when the per-step anomaly test eq. 12 exceeds `lambda_a`.
    /// `confidence` is the eq. 12 posterior at the firing step.
    CollectiveAnomaly {
        start: usize,
        end: usize,
        confidence: f64,
    },
}

/// BOCD-AR sequential detector. Univariate, NIG conjugate prior.
#[allow(clippy::needless_range_loop)]
pub struct ChenWuDetector {
    /// `p_0` -- prior change probability per step (paper § 2).
    p0: f64,
    /// `q_0` -- prior anomaly-end probability conditional on an open
    /// anomaly (paper § 2).
    q0: f64,
    /// `Δt` -- maximum collective-anomaly duration (paper § 2).
    delta_t: usize,
    /// `u_c` -- CP search window upper limit (paper § 3.1).
    u_c: usize,
    /// `u_a` -- anomaly search window upper limit (paper § 3.1).
    u_a: usize,
    /// `λ_a` -- anomaly alarm threshold (paper eq. 12).
    lambda_a: f64,
    /// `λ_c` -- CP alarm threshold (paper eq. 13).
    lambda_c: f64,
    /// `δ` -- CP localisation tolerance (paper eq. 13).
    delta: usize,
    /// Minimum post-change observations before a CP alert can fire.
    /// Paper § 6.1: "Change alerts are triggered after observing at least
    /// five post-change data points." Default 5.
    min_post_change_obs: usize,
    /// Conjugate prior on the within-regime distribution.
    prior: Nig,
    /// β-divergence robustness parameter for the within-regime
    /// likelihood. `0.0` is the default (standard Bayesian update).
    /// See `Nig::log_predictive_robust` for the predictive-Gaussian
    /// approximation of the β-power integral.
    beta: f64,
}

#[allow(clippy::needless_range_loop)]
impl ChenWuDetector {
    /// Five core hyperparameters. Defaults: `u_c = 256`, `u_a = 32`,
    /// `δ = 0`. Override via the `with_*` builders.
    pub fn new(p0: f64, q0: f64, delta_t: usize, lambda_a: f64, lambda_c: f64) -> Self {
        assert!(
            (0.0..1.0).contains(&p0),
            "p0 must be in [0, 1), got {p0}"
        );
        assert!(
            (0.0..1.0).contains(&q0),
            "q0 must be in [0, 1), got {q0}"
        );
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
            // Paper § 6.1 hyperparameters: σ_0² = 0.25, v_0 = 1, k_0 = 0.01.
            // Matches cesura Nig: alpha=v_0/2, beta=v_0·σ_0²/2, mu=0, kappa=k_0.
            prior: Nig::new(0.0, 0.01, 0.5, 0.125),
            beta: 0.0,
        }
    }

    /// Within-regime log-likelihood. Dispatches to `Nig::log_predictive`
    /// when `β = 0` (standard) or `Nig::log_predictive_robust` otherwise.
    fn likelihood(&self, n: &Nig, x: f64) -> f64 {
        if self.beta > 0.0 {
            return n.log_predictive_robust(x, self.beta);
        }
        n.log_predictive(x)
    }

    /// Opt the within-regime likelihood into β-divergence (Knoblauch
    /// et al. 2018, arXiv:1806.02261). Bounds the influence of any one
    /// observation; useful when fine-grained data has within-regime
    /// kurtosis > 3. Reasonable range `[0.05, 0.20]`. `0.0` short-
    /// circuits to standard cesura.
    ///
    /// Composes with the joint detector: anomalies are still classified
    /// via the run-length recursion, but the predictive that drives
    /// `r_star` is robustified.
    pub fn with_robust(mut self, beta: f64) -> Self {
        assert!(
            (0.0..=1.0).contains(&beta),
            "beta must be in [0.0, 1.0], got {beta}"
        );
        self.beta = beta;
        self
    }

    pub fn with_search_windows(mut self, u_c: usize, u_a: usize) -> Self {
        assert!(u_c >= u_a, "u_c must be >= u_a");
        self.u_c = u_c;
        self.u_a = u_a;
        self
    }

    /// Replace the `q_0` passed to [`new`](Self::new) with the largest
    /// value satisfying paper § 5 / eq. 14 for the current
    /// `(p_0, λ_a, Δt)`. The picker line-searches via bisection (no
    /// closed form). See [`crate::auto_q0`] for the constraint and a
    /// note on why the picker returns the *strict* upper bound, which
    /// can land below the paper's permissive Figure 4(a) read-off.
    ///
    /// Idempotent after a single call -- the bound depends only on
    /// `(p_0, λ_a, Δt)`, so repeated invocations return the same `q_0`.
    pub fn with_auto_q0(mut self) -> Self {
        self.q0 = crate::auto_q0::q0_upper_bound(self.p0, self.lambda_a, self.delta_t);
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

    pub fn with_prior(mut self, prior_mu: f64, prior_kappa: f64, prior_alpha: f64, prior_beta: f64) -> Self {
        self.prior = Nig::new(prior_mu, prior_kappa, prior_alpha, prior_beta);
        self
    }

    /// Run the joint detector on `data`. Returns the time-ordered
    /// sequence of `Detection`s, deduplicated so each underlying event
    /// emits at most once.
    pub fn detect(&self, data: &[f64]) -> Vec<Detection> {
        let n = data.len();
        if n < 4 {
            return Vec::new();
        }

        // Cooldown / dedup state: emit each event once. An anomaly is the
        // same event if it shares a start/end pair with the most recent
        // emission. A CP is the same if it shares the index. Both also
        // suppressed for `cooldown` steps after the previous emission of
        // the same kind to absorb trailing noise.
        let cooldown = 15usize;
        let mut emitted_anom: HashSet<(usize, usize)> = HashSet::new();
        let mut last_cp: Option<(usize, usize)> = None; // (index, t_paper)

        // 1-indexed internally to match the paper. Cell `[t]` holds the
        // value at paper-time `t`. Index 0 is unused.
        // log_h_a^t and log_h_c^t indexed by r ∈ [0, n_c^t]. Empty entries
        // hold f64::NEG_INFINITY (log 0).
        let mut log_h_a: Vec<Vec<f64>> = vec![Vec::new()];
        let mut log_h_c: Vec<Vec<f64>> = vec![Vec::new()];

        // Paper § 4.3 Υ_c^t persistence. `excluded_ranges` is the list of
        // paper-time intervals (1-indexed, inclusive) that have been
        // removed by previously-emitted collective anomalies. `stats[r]`
        // is the NIG posterior given the r most-recent *non-excluded*
        // observations, so future steps see the recursion as if those
        // points never happened. The per-step inner loop also recomputes
        // `stats[r]` for affected `r` after each new anomaly emission so
        // re-detection within the same step uses the post-removal
        // likelihood.
        let mut excluded_ranges: Vec<(usize, usize)> = Vec::new();

        // Bounded ring of past raw observations. Capacity matches the
        // paper's space bound `O(u_c + u_a·Δt)` so the walk-back for
        // `stats[r]` can land enough non-excluded obs even when several
        // anomaly windows are interleaved with the regime history.
        // `abs_origin` is the paper-time of `raw_history[0]`.
        let raw_capacity = self.u_c + self.u_a * self.delta_t + 1;
        let mut raw_history: VecDeque<f64> = VecDeque::with_capacity(raw_capacity);
        let mut abs_origin: usize = 1;

        // stats[r] = Nig posterior given r most-recent non-excluded
        // observations from y_1, …, y_{t_paper-1}; supplies the predictive
        // ℙ(y^t | y^((t-r):(t-1))) via stats[r].log_predictive(y_t).
        let mut stats: Vec<Nig> = vec![self.prior.clone()];

        // Paper t = 1 corresponds to data[0]. Initialise log-space.
        let log_l_y1 = self.likelihood(&self.prior, data[0]);
        log_h_a.push(vec![NEG_INF]);
        log_h_c.push(vec![log_l_y1]);
        stats.push(self.prior.update(data[0]));
        raw_history.push_back(data[0]);

        // Pre-compute log constants used inside the per-step recursion.
        let log_p0 = self.p0.ln();
        let log_1_minus_p0 = (1.0 - self.p0).ln();
        let log_q0 = self.q0.ln();
        let log_1_minus_q0 = (1.0 - self.q0).ln();

        let mut detections = Vec::new();

        for t_paper in 2..=n {
            let y_t = data[t_paper - 1];
            let n_c = (t_paper - 1).min(self.u_c);

            // Rebuild stats[0..=n_c] from raw_history + excluded_ranges so
            // it reflects every exclusion accumulated through step t-1.
            // O(n_c) per step; the partial rebuild inside the inner loop
            // adds O(Δt) per anomaly emission.
            stats = rebuild_stats_full(
                &self.prior,
                &raw_history,
                abs_origin,
                &excluded_ranges,
                t_paper,
                n_c,
            );

            // Log-predictives. log_pred[r] = ℙ(y^t | y^((t-r):(t-1))) in
            // log-space. log_pred[0] is unused; index 0 means "fresh prior".
            let log_l_y_t = self.likelihood(&self.prior, y_t);
            let mut log_pred = vec![NEG_INF; n_c + 1];
            for r in 1..=n_c {
                if r < stats.len() {
                    log_pred[r] = self.likelihood(&stats[r], y_t);
                }
            }

            let mut new_log_h_a = vec![NEG_INF; n_c + 1];
            let mut new_log_h_c = vec![NEG_INF; n_c + 1];

            // Eq. 7: log ℍ_a^t(r)
            //   r > 0: log_h_a^(t-1)(r-1) + log_pred[r] + log(1-p_0)
            //   r = 0: log Σ_{r'=0}^{A(t, t-1)} exp(log_h_c^(t-1)(r')) + log_l_y_t + log_q_0
            for r in 1..=n_c {
                let prev_idx = r - 1;
                let prev_log_h_a = log_h_a[t_paper - 1].get(prev_idx).copied().unwrap_or(NEG_INF);
                new_log_h_a[r] = prev_log_h_a + log_pred[r] + log_1_minus_p0;
            }
            // r = 0: A(t, t-1)
            let a_bound = a_constraint(t_paper, t_paper - 1, self.delta_t);
            if a_bound >= 0 {
                let prev_log_h_c_full = &log_h_c[t_paper - 1];
                let upper = (a_bound as usize).min(prev_log_h_c_full.len().saturating_sub(1));
                let mut log_sum = NEG_INF;
                for rp in 0..=upper {
                    log_sum = log_add_exp(log_sum, prev_log_h_c_full[rp]);
                }
                new_log_h_a[0] = log_sum + log_l_y_t + log_q0;
            }

            // Eq. 8: log ℍ_c^t(r)
            //   r > Δt OR r = t-1: log_h_c^(t-1)(r-1) + log_pred + log(1-p_0)
            //   0 < r ≤ Δt AND r ≠ t-1: log_h_c^(t-1)(r-1) + log_pred + log(1-q_0)
            //   r = 0 AND t ≥ Δt+3: log_add_exp(log Σ_{Δt..} h_c, log Σ_all h_a) + log_l + log p_0
            //   r = 0 otherwise: log_add_exp(log_h_c(t-2), log Σ_all h_a) + log_l + log p_0
            for r in 1..=n_c {
                let prev_log_h_c = log_h_c[t_paper - 1].get(r - 1).copied().unwrap_or(NEG_INF);
                let log_factor = if r > self.delta_t || r == t_paper - 1 {
                    log_1_minus_p0
                } else {
                    log_1_minus_q0
                };
                new_log_h_c[r] = prev_log_h_c + log_pred[r] + log_factor;
            }
            // r = 0 case
            let prev_log_h_c_full = &log_h_c[t_paper - 1];
            let prev_log_h_a_full = &log_h_a[t_paper - 1];
            let mut log_sum_h_a_prev = NEG_INF;
            for &v in prev_log_h_a_full {
                log_sum_h_a_prev = log_add_exp(log_sum_h_a_prev, v);
            }
            let log_r0_factor = if t_paper >= self.delta_t + 3 {
                let mut log_sum_c = NEG_INF;
                let lo = self.delta_t;
                for rp in lo..prev_log_h_c_full.len() {
                    log_sum_c = log_add_exp(log_sum_c, prev_log_h_c_full[rp]);
                }
                log_add_exp(log_sum_c, log_sum_h_a_prev)
            } else {
                let last_idx = (t_paper - 2).min(prev_log_h_c_full.len().saturating_sub(1));
                let log_h_c_last = prev_log_h_c_full.get(last_idx).copied().unwrap_or(NEG_INF);
                log_add_exp(log_h_c_last, log_sum_h_a_prev)
            };
            new_log_h_c[0] = log_r0_factor + log_l_y_t + log_p0;

            // Log-renormalise so the joint mass Σ_r (h_a + h_c) = 1.
            // Compute total = log_add_exp over both vectors.
            let mut log_total = NEG_INF;
            for &v in &new_log_h_a {
                log_total = log_add_exp(log_total, v);
            }
            for &v in &new_log_h_c {
                log_total = log_add_exp(log_total, v);
            }
            if log_total.is_finite() {
                for v in new_log_h_a.iter_mut() {
                    *v -= log_total;
                }
                for v in new_log_h_c.iter_mut() {
                    *v -= log_total;
                }
            }

            // Spec-level invariant: post-renorm joint mass = 1, i.e.
            // log_total over ℍ_a ∪ ℍ_c is 0. Linear sum is what eqs. 10-13
            // operate on conceptually; in log-space that becomes
            // `log_add_exp` over both vectors landing within ε of 0.
            #[cfg(debug_assertions)]
            {
                let mut log_total = NEG_INF;
                for &v in &new_log_h_a {
                    log_total = log_add_exp(log_total, v);
                }
                for &v in &new_log_h_c {
                    log_total = log_add_exp(log_total, v);
                }
                debug_assert!(
                    log_total.abs() < 1e-9,
                    "log Σ_r (ℍ_a + ℍ_c) = {log_total} at t = {t_paper}, expected ≈ 0"
                );
            }

            // Detection emission (paper Algorithm 1 + § 4.2). Operates
            // entirely in log-space; the only `.exp()` calls land at
            // emission time when a `Detection` carries a linear posterior
            // through to the caller.
            //
            // Mask tracks r values whose corresponding paper-time t-r has
            // been excluded by a previously-detected anomaly window in
            // this step. After the loop, the most recent unmasked r* is
            // what eq. 13 uses for CP testing -- this is the paper's
            // "updated most recent change point".
            let n_a = (t_paper - 1).min(self.u_a);
            let mut mask = vec![false; n_c + 1];
            let mut latest_r_star: usize = 0;

            // Loop: detect, remove, redetect. At most n_a iterations
            // (each removed window costs at least one r index).
            for _iter in 0..=n_a {
                // r* = argmax (ℍ_a + ℍ_c) over unmasked indices (eq. 10).
                // log_add_exp(log_h_a, log_h_c) is monotone in the linear
                // sum, so argmax in log preserves argmax in linear.
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

                // Anomaly test eq. 12, only if r* ≤ n_a^t (paper § 4.2).
                // r_star >= 2 is a defensive minimum: r_star ∈ {0, 1} means
                // the change is at t or t-1, no time for an anomaly to form.
                if r_star > n_a || r_star < 2 {
                    break;
                }
                let lo = r_star.saturating_sub(self.delta_t);
                let hi = r_star;
                // Numerator: log Σ ℍ_a^t(r) over the band.
                // Denominator: log Σ (ℍ_a^t(r) + ℍ_c^t(r)) over the band.
                // Posterior in linear is exp(log_num - log_den); the
                // λ_a comparison stays in log-space to avoid catastrophic
                // cancellation when log_num and log_den share large terms.
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

                // Locate endpoint r1 via argmax ℍ_a^t over [lo, hi]. Same
                // monotonicity argument: argmax log_h_a ≡ argmax h_a.
                let mut r1 = lo;
                let mut best1 = NEG_INF;
                for r in lo..=hi {
                    if r < new_log_h_a.len() && !mask[r] && new_log_h_a[r] > best1 {
                        best1 = new_log_h_a[r];
                        r1 = r;
                    }
                }
                let prior_t = t_paper.saturating_sub(r1).saturating_sub(1);
                let r2 = if prior_t >= 1 && prior_t < log_h_c.len() {
                    let prior_log_h_c = &log_h_c[prior_t];
                    let mut best2 = NEG_INF;
                    let mut idx = 0;
                    let bound = self.delta_t.saturating_sub(1).min(prior_log_h_c.len().saturating_sub(1));
                    for r in 0..=bound {
                        if prior_log_h_c[r] > best2 {
                            best2 = prior_log_h_c[r];
                            idx = r;
                        }
                    }
                    idx
                } else {
                    0
                };
                let end_paper = t_paper - r1;
                let start_paper = end_paper.saturating_sub(r2 + 1);
                if end_paper < 1 || start_paper < 1 {
                    break;
                }
                let new_window = (start_paper - 1, end_paper - 1);
                // Dedup against all previously-emitted windows. Back-to-
                // back anomalies (paper § 3.1 point 3) can be 1-4 steps
                // apart, so a time cooldown would swallow them; window
                // identity is the right dedup key.
                if !emitted_anom.contains(&new_window) {
                    detections.push(Detection::CollectiveAnomaly {
                        start: new_window.0,
                        end: new_window.1,
                        confidence: posterior,
                    });
                    emitted_anom.insert(new_window);
                }

                // Mask the detected window's r-range so the next argmax
                // skips it. The window covers paper-times start_paper..=end_paper,
                // which corresponds to r values t_paper - end_paper ..= t_paper - start_paper.
                let r_lo = t_paper.saturating_sub(end_paper);
                let r_hi = t_paper.saturating_sub(start_paper);

                // Paper § 4.3 Υ_c^t / Υ_a^t persistence. Two coupled
                // changes per detected window:
                //
                //   (a) Record the paper-time window in excluded_ranges
                //       so future steps' stats walk-back skips it. Past
                //       observations are removed from the conditioning
                //       set; future predictives respect that.
                //   (b) Remove the window's r-range from this step's
                //       Υ -- mask + zero the corresponding ℍ_a / ℍ_c
                //       cells. Zeroing matters because eq. 7-8 propagate
                //       these cells into the next step's r+1 entries; if
                //       we leave the mass in, an anomaly endpoint
                //       resurfaces as r* = 1 at t+1 and fires a spurious
                //       CP. The previous heuristic (`inside_emitted_anomaly`)
                //       was a correctness band-aid for that leak; the
                //       paper's rule eliminates it.
                excluded_ranges.push((start_paper, end_paper));
                for r in r_lo..=r_hi.min(n_c) {
                    mask[r] = true;
                    new_log_h_a[r] = NEG_INF;
                    new_log_h_c[r] = NEG_INF;
                }

                // Re-renormalise so the joint mass Σ_r (h_a + h_c) = 1
                // post-removal. Stays in log-space; the next inner-loop
                // iteration's argmax / anomaly test reads the same
                // `new_log_h_a` / `new_log_h_c` slices directly.
                let mut log_total = NEG_INF;
                for &v in &new_log_h_a {
                    log_total = log_add_exp(log_total, v);
                }
                for &v in &new_log_h_c {
                    log_total = log_add_exp(log_total, v);
                }
                if log_total.is_finite() {
                    for v in new_log_h_a.iter_mut() {
                        *v -= log_total;
                    }
                    for v in new_log_h_c.iter_mut() {
                        *v -= log_total;
                    }
                }
            }

            // Eq. 13 (paper Algorithm 1, post-removal): CP posterior over
            // the band [r* - δ, r* + δ] / total, using the post-removal r*.
            // Denominator is the SUM over all r including masked entries
            // (the algorithm exists in linear-mass space; mask = 0 there).
            // In log-space, masked r entries contribute NEG_INF to the
            // accumulator, which `log_add_exp` correctly treats as zero
            // contribution.
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

            // Paper § 6.1 gate: wait at least min_post_change_obs steps
            // post-change before alarming. r_star is the run length since
            // the most recent change, so r_star >= min_post_change_obs.
            if p_band > self.lambda_c && r_star >= self.min_post_change_obs {
                let cp_paper = t_paper - r_star;
                if cp_paper >= 1 {
                    let cp_idx = cp_paper - 1;
                    // With paper § 4.3 Υ_c^t persistence in place, removed
                    // anomaly windows no longer leak through as spurious
                    // CP candidates -- the recursion's mass at those r
                    // values has already been corrected via the inner
                    // loop's stats / log_pred / ℍ rebuild. Only the
                    // cooldown remains, to absorb trailing noise across
                    // the cp's own neighbourhood.
                    let suppress = match last_cp {
                        Some((idx, t_prev)) => {
                            cp_idx.abs_diff(idx) <= cooldown
                                || t_paper - t_prev < cooldown
                        }
                        None => false,
                    };
                    if !suppress {
                        // Prefix-data shift_sigma: use only data through
                        // the current step (same window the streaming
                        // path can supply at fire time). Both paths now
                        // produce identical shift_sigma values -- this
                        // is the contract `matches_batch_detect` checks.
                        detections.push(Detection::ChangePoint(ChangePoint {
                            index: cp_idx,
                            confidence: p_band,
                            shift_sigma: shift_sigma_at(&data[..t_paper], cp_idx),
                        }));
                        last_cp = Some((cp_idx, t_paper));
                    }
                }
            }

            // Push y_t to bounded raw_history. abs_origin tracks the
            // paper-time of raw_history[0]; pop_front shifts it.
            raw_history.push_back(y_t);
            while raw_history.len() > raw_capacity {
                raw_history.pop_front();
                abs_origin += 1;
            }

            log_h_a.push(new_log_h_a);
            log_h_c.push(new_log_h_c);
        }

        detections
    }
}

/// Test whether a paper-time index falls inside any excluded range.
/// Linear scan; total length bounded by `u_a` (one entry per emitted
/// collective anomaly), so cost is small in practice.
fn is_excluded_paper_time(abs_time: usize, excluded_ranges: &[(usize, usize)]) -> bool {
    excluded_ranges.iter().any(|&(s, e)| s <= abs_time && abs_time <= e)
}

/// Crate-internal re-exports so [`crate::streaming_chen_wu`] can share
/// the exact recursion helpers the batch path uses. Keeping them
/// private to the crate (not the public API) preserves the option to
/// refactor the helper signatures without a semver bump.
#[doc(hidden)]
pub(crate) fn a_constraint_pub(t: usize, d: usize, delta_t: usize) -> i64 {
    a_constraint(t, d, delta_t)
}

#[doc(hidden)]
pub(crate) fn rebuild_stats_full_pub(
    prior: &Nig,
    raw_history: &std::collections::VecDeque<f64>,
    abs_origin: usize,
    excluded_ranges: &[(usize, usize)],
    t_paper: usize,
    n_c: usize,
) -> Vec<Nig> {
    rebuild_stats_full(prior, raw_history, abs_origin, excluded_ranges, t_paper, n_c)
}

/// Build `stats[0..=n_c]` from the prior plus the most recent `n_c`
/// non-excluded observations in `raw_history`. Mirrors the per-step
/// recursion but applied lazily over a possibly sparse history.
///
/// NIG conjugate updates are commutative in their final state -- the
/// posterior depends only on the sufficient statistics `n, Σx, Σx²`,
/// not the order in which observations were applied -- so the
/// incremental form `stats[r] = stats[r-1].update(backward[r-1])`
/// produces the same `Nig` as a fresh prior updated by the chronological
/// sequence.
fn rebuild_stats_full(
    prior: &Nig,
    raw_history: &VecDeque<f64>,
    abs_origin: usize,
    excluded_ranges: &[(usize, usize)],
    t_paper: usize,
    n_c: usize,
) -> Vec<Nig> {
    let mut stats = vec![prior.clone(); n_c + 1];
    rebuild_stats_full_into(
        &mut stats,
        prior,
        raw_history,
        abs_origin,
        excluded_ranges,
        t_paper,
        n_c,
    );
    stats
}

fn rebuild_stats_full_into(
    stats: &mut Vec<Nig>,
    prior: &Nig,
    raw_history: &VecDeque<f64>,
    abs_origin: usize,
    excluded_ranges: &[(usize, usize)],
    t_paper: usize,
    n_c: usize,
) {
    if stats.len() < n_c + 1 {
        stats.resize(n_c + 1, prior.clone());
    }
    stats[0] = prior.clone();

    let history_len = raw_history.len();
    let mut backward: Vec<f64> = Vec::with_capacity(n_c);
    for delta in 1..=history_len {
        if backward.len() >= n_c {
            break;
        }
        let idx = history_len - delta;
        let abs_time = abs_origin + idx; // 1-indexed paper-time
        if abs_time >= t_paper {
            continue;
        }
        if is_excluded_paper_time(abs_time, excluded_ranges) {
            continue;
        }
        backward.push(raw_history[idx]);
    }
    // backward[k] = (k+1)-th most recent non-excluded observation.
    // stats[r] uses the r most recent (backward[0..r]) applied in
    // chronological order; the marginal step from r-1 to r adds
    // backward[r-1].
    let collected = backward.len();
    for r in 1..=n_c {
        if r - 1 < collected {
            stats[r] = stats[r - 1].update(backward[r - 1]);
        } else {
            stats[r] = stats[r - 1].clone();
        }
    }
}

/// Pooled-σ effect size between the 20-obs window before `idx` and the
/// 20-obs window starting at `idx`. Mirrors the shape `FocusDetector`
/// uses; chen_wu inlines rather than coupling modules.
fn shift_sigma_at(data: &[f64], idx: usize) -> f64 {
    const W: usize = 20;
    let n = data.len();
    let lo = idx.saturating_sub(W);
    let hi = (idx + W).min(n);
    let before = &data[lo..idx];
    let after = &data[idx..hi];
    if before.is_empty() || after.is_empty() {
        return 0.0;
    }
    let mean_b: f64 = before.iter().sum::<f64>() / before.len() as f64;
    let mean_a: f64 = after.iter().sum::<f64>() / after.len() as f64;
    let var_b: f64 = before.iter().map(|x| (x - mean_b).powi(2)).sum::<f64>() / before.len() as f64;
    let var_a: f64 = after.iter().map(|x| (x - mean_a).powi(2)).sum::<f64>() / after.len() as f64;
    let pooled = ((var_b + var_a) / 2.0).sqrt().max(1e-10);
    ((mean_a - mean_b) / pooled).abs()
}

/// `A(t, d) = min(Δt - 1, R(t, d) - 1)` per paper § 3.1.
/// Returns -1 for empty ranges (early `t`).
fn a_constraint(t: usize, d: usize, delta_t: usize) -> i64 {
    let r_t_d = if d == t.saturating_sub(1) {
        d as i64 - 1
    } else {
        d as i64 - delta_t as i64 - 1
    };
    let upper = (delta_t as i64).saturating_sub(1).min(r_t_d.saturating_sub(1));
    upper.max(-1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::Rng;

    #[test]
    fn detector_runs_without_panic() {
        let mut rng = Rng::new(1);
        let data: Vec<f64> = (0..200).map(|_| rng.normal(0.0, 1.0)).collect();
        let det = ChenWuDetector::new(0.1, 0.2, 4, 0.5, 0.5);
        let _ = det.detect(&data);
    }

    #[test]
    fn long_stationary_run_no_nan_no_panic() {
        // 1000 N(0,1) observations. The recursion multiplies probabilities
        // step-by-step in linear space; without the per-step renormaliser the
        // numbers underflow well before t = 1000. Pin that the renormaliser
        // keeps everything finite. No assertion on detection counts -- this is
        // a stability test, not an ARL₀ test.
        let mut rng = Rng::new(7);
        let data: Vec<f64> = (0..1000).map(|_| rng.normal(0.0, 1.0)).collect();
        let det = ChenWuDetector::new(0.1, 0.2, 4, 0.5, 0.5);
        let dets = det.detect(&data);
        for d in &dets {
            match d {
                Detection::ChangePoint(cp) => {
                    assert!(cp.confidence.is_finite(), "non-finite confidence");
                    assert!(cp.shift_sigma.is_finite(), "non-finite shift_sigma");
                    assert!((0.0..=1.0).contains(&cp.confidence));
                }
                Detection::CollectiveAnomaly { confidence, .. } => {
                    assert!(confidence.is_finite(), "non-finite anomaly confidence");
                    assert!((0.0..=1.0).contains(confidence));
                }
            }
        }
    }

    #[test]
    fn detects_collective_anomaly_not_cp() {
        // Short reverting deviation: 100 N(0,1) → 4 N(5,1) → 100 N(0,1).
        // Must emit a CollectiveAnomaly, NOT two ChangePoints.
        let mut rng = Rng::new(3);
        let mut data: Vec<f64> = (0..100).map(|_| rng.normal(0.0, 1.0)).collect();
        data.extend((0..4).map(|_| rng.normal(5.0, 1.0)));
        data.extend((0..100).map(|_| rng.normal(0.0, 1.0)));
        let det = ChenWuDetector::new(0.1, 0.2, 4, 0.5, 0.5);
        let dets = det.detect(&data);
        eprintln!("single-anomaly detections: {dets:?}");
        let n_anom = dets
            .iter()
            .filter(|d| matches!(d, Detection::CollectiveAnomaly { .. }))
            .count();
        assert!(
            n_anom >= 1,
            "expected ≥ 1 CollectiveAnomaly, got {dets:?}"
        );
    }

    #[test]
    fn detects_back_to_back_anomalies() {
        // Two collective anomalies in close succession. Per paper § 3.1
        // point (3), this synthetic fixture checks both emissions.
        // The removal loop permits detection of the second anomaly.
        let mut rng = Rng::new(4);
        let mut data: Vec<f64> = (0..100).map(|_| rng.normal(0.0, 1.0)).collect();
        data.extend((0..3).map(|_| rng.normal(5.0, 1.0))); // anomaly #1
        data.extend((0..1).map(|_| rng.normal(0.0, 1.0)));
        data.extend((0..3).map(|_| rng.normal(-5.0, 1.0))); // anomaly #2
        data.extend((0..100).map(|_| rng.normal(0.0, 1.0)));
        let det = ChenWuDetector::new(0.1, 0.2, 4, 0.5, 0.5);
        let dets = det.detect(&data);
        eprintln!("back-to-back detections: {dets:?}");
        let n_anom = dets
            .iter()
            .filter(|d| matches!(d, Detection::CollectiveAnomaly { .. }))
            .count();
        assert!(
            n_anom >= 2,
            "expected ≥ 2 CollectiveAnomaly emissions, got {n_anom} in {dets:?}"
        );
    }

    #[test]
    fn rebuild_stats_skips_excluded_observations() {
        // Anomaly-window invariant: after pushing an anomaly window into
        // excluded_ranges, rebuild_stats_full produces stats[r] equal to
        // a hand-computed Nig that walks the prior over the
        // non-excluded observations only. Affected r values pull in
        // older non-excluded data; un-affected r values match the
        // no-exclusion baseline exactly.
        use std::collections::VecDeque;

        let prior = Nig::new(0.0, 0.01, 0.5, 0.125);
        // Paper-times 1..=10. Inject an "anomaly" at times 6..=8 with
        // 10× shift so the post-removal walk-back lands on different
        // observations than the no-exclusion path.
        let raw: Vec<f64> = (1..=10)
            .map(|t| if (6..=8).contains(&t) { 50.0 } else { t as f64 })
            .collect();
        let raw_history: VecDeque<f64> = raw.iter().copied().collect();
        let abs_origin = 1usize;
        let t_paper = 11; // about to compute stats for step 11

        // Without exclusions: stats[3] = prior + obs at times 8, 9, 10.
        let no_excl: Vec<(usize, usize)> = Vec::new();
        let stats_no_excl = rebuild_stats_full(&prior, &raw_history, abs_origin, &no_excl, t_paper, 5);
        let mut hand_no_excl = prior.clone();
        for &x in &raw[7..10] {
            hand_no_excl = hand_no_excl.update(x);
        }
        assert!((stats_no_excl[3].mu - hand_no_excl.mu).abs() < 1e-12);
        assert!((stats_no_excl[3].kappa - hand_no_excl.kappa).abs() < 1e-12);
        assert!((stats_no_excl[3].alpha - hand_no_excl.alpha).abs() < 1e-12);
        assert!((stats_no_excl[3].beta - hand_no_excl.beta).abs() < 1e-12);

        // With anomaly window [6, 8] excluded: stats[3] walks back from
        // time 10, skipping times 8, 7, 6. The 3 most recent non-
        // excluded observations are at times 10, 9, 5 (in that order).
        let with_excl = vec![(6usize, 8usize)];
        let stats_excl = rebuild_stats_full(&prior, &raw_history, abs_origin, &with_excl, t_paper, 5);
        // Stats are sufficient statistics (n, Σx, Σx²) so the order of
        // application doesn't matter -- compare against a plain
        // chronological replay over the non-excluded set.
        let nonexcl: Vec<f64> = raw
            .iter()
            .enumerate()
            .filter_map(|(i, &x)| {
                let t = i + 1;
                if (6..=8).contains(&t) || t >= t_paper {
                    None
                } else {
                    Some(x)
                }
            })
            .collect();
        // stats[r] uses the last r non-excluded -- here last 3 of nonexcl.
        let last_three = &nonexcl[nonexcl.len() - 3..];
        let mut hand_excl = prior.clone();
        for &x in last_three {
            hand_excl = hand_excl.update(x);
        }
        assert!(
            (stats_excl[3].mu - hand_excl.mu).abs() < 1e-12,
            "rebuild_stats_full mu drift: got {} hand-computed {}",
            stats_excl[3].mu,
            hand_excl.mu
        );
        assert!((stats_excl[3].kappa - hand_excl.kappa).abs() < 1e-12);
        assert!((stats_excl[3].alpha - hand_excl.alpha).abs() < 1e-12);
        assert!(
            (stats_excl[3].beta - hand_excl.beta).abs() < 1e-12,
            "rebuild_stats_full beta drift: got {} hand-computed {}",
            stats_excl[3].beta,
            hand_excl.beta
        );

        // Sanity: with exclusion the predictive moves -- exclusion
        // pulled in y=5 (older normal obs) instead of the y=50
        // anomaly-region obs that the no-exclusion path used.
        assert!(
            stats_no_excl[3].mu != stats_excl[3].mu,
            "exclusion should change stats[3] -- got identical mu = {}",
            stats_excl[3].mu
        );
    }

    #[test]
    fn detects_clean_cp() {
        // Strong shift mid-stream. Should emit a CP near 100.
        let mut rng = Rng::new(2);
        let mut data: Vec<f64> = (0..100).map(|_| rng.normal(0.0, 1.0)).collect();
        data.extend((0..100).map(|_| rng.normal(5.0, 1.0)));
        let det = ChenWuDetector::new(0.1, 0.2, 4, 0.5, 0.5);
        let dets = det.detect(&data);
        let any_cp_near_100 = dets.iter().any(|d| match d {
            Detection::ChangePoint(cp) => (cp.index as i64 - 100).abs() < 30,
            _ => false,
        });
        assert!(
            any_cp_near_100,
            "expected a ChangePoint within 30 steps of 100, got {dets:?}"
        );
    }
}
