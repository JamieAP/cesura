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
    /// Bayes-factor decision rule state. `None` ⇒ MAP-drop trigger (default).
    /// Forward-compat: pre-0.9 snapshots default to None.
    #[serde(default)]
    pub bf_rule: Option<BfRuleState>,
    /// Optional pre-detector observation transform. `Identity` (default)
    /// preserves byte-identical behaviour with earlier snapshots.
    /// `Logit` unbounds prices in (0, 1) for use against the NIG/NIW
    /// Gaussian likelihood; intended for bounded-data callers (for example, probabilities).
    /// Forward-compat: earlier snapshots default to Identity.
    #[serde(default)]
    pub transform: PriceTransform,
}

/// Observation-stream transform applied before BOCPD sees the data.
///
/// `Identity` is the default and preserves byte-identical behaviour with
/// existing callers. `Logit` is intended
/// for bounded callers whose prices live in (0, 1); it unbounds the
/// stream so the NIG/NIW Gaussian likelihood is less misspecified near
/// the boundaries.
///
/// Bounded probabilities can be transformed with Logit before Gaussian modeling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PriceTransform {
    /// No transform. Default. Behaviour identical to earlier cesura.
    #[default]
    Identity,
    /// `y = log(x / (1 - x))`, clamped to a tiny epsilon away from 0 and 1
    /// so endpoints don't produce ±infinity.
    Logit,
}

impl PriceTransform {
    /// Apply the transform to one observation. `Identity` returns `x`
    /// verbatim. `Logit` clamps `x` to `[eps, 1-eps]` before the log-odds
    /// transform.
    #[inline]
    pub fn apply(self, x: f64) -> f64 {
        match self {
            PriceTransform::Identity => x,
            PriceTransform::Logit => {
                const EPS: f64 = 1e-9;
                let xc = x.clamp(EPS, 1.0 - EPS);
                (xc / (1.0 - xc)).ln()
            }
        }
    }
}

/// Bayes-factor decision-rule state. When attached to a [`StreamingDetector`]
/// via [`StreamingDetector::with_bayes_factor_rule`], the streaming detector
/// uses BF on aggregated short-run-length mass instead of MAP-drop. Mirrors
/// the offline [`crate::BocpdDetector::detect_bayes_factor`] math; carries
/// the posterior-maturity `armed` flag and `last_emit` step for cooldown,
/// both serialised so a daemon checkpointing in BF mode round-trips.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BfRuleState {
    pub threshold: f64,
    pub short_horizon: usize,
    pub cooldown: usize,
    pub armed: bool,
    pub last_emit: Option<usize>,
    /// Quantile-based dynamic threshold. `Some(q)` ⇒ once `recent_bf`
    /// has filled `quantile_window`, fire when current BF exceeds the
    /// `q`-th quantile of the ring. `None` ⇒ fixed `threshold`.
    /// Forward-compatibility: snapshots without this field default to None.
    #[serde(default)]
    pub quantile: Option<f64>,
    /// Sliding window over which the empirical-BF quantile is computed.
    #[serde(default = "default_bf_quantile_window")]
    pub quantile_window: usize,
    /// Sliding ring of recent BF values. Updated AFTER the per-step
    /// fire decision so the current BF never contaminates its own
    /// reference distribution (mirrors the rank-transform pattern in
    /// `multistream::hc::RankWindow`).
    ///
    /// Cold-spot caveat: the dispatch in [`crate::streaming::StreamingDetector::step`]
    /// gates the entire BF block on `!in_cooldown`, so during the
    /// post-fire cooldown window (default 15 steps) the ring is not
    /// updated. Omitting cooldown samples can affect calibration;
    /// evaluate the quantile, window and cooldown together on
    /// representative streams.
    #[serde(default)]
    pub recent_bf: std::collections::VecDeque<f64>,
}

fn default_bf_quantile_window() -> usize {
    500
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
    /// `true` if pushed by the BF rule. On drain failure (shift_sigma <
    /// 1e-9), the BF detector re-arms so the offline-matching "small
    /// shift doesn't burn arm" property is preserved.
    is_bf: bool,
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
    /// Bayes-factor rule state. `None` ⇒ MAP-drop trigger (default).
    bf_rule: Option<BfRuleState>,
    /// Pre-detector observation transform. `Identity` by default;
    /// `Logit` for bounded-data callers (for example, probabilities).
    transform: PriceTransform,
}

const SHIFT_WINDOW: usize = 20;

/// Empirical `q`-quantile of a small ring (linear interpolation between
/// adjacent ranks, mirroring NumPy/Polars `quantile(method='linear')`).
/// Caller guarantees `ring` non-empty and `q ∈ (0, 1)`. NaN values are
/// dropped (BF should always be finite, but the recursion can produce
/// NaN under degenerate inputs).
fn quantile_of(ring: &std::collections::VecDeque<f64>, q: f64) -> f64 {
    let mut v: Vec<f64> = ring.iter().copied().filter(|x| x.is_finite()).collect();
    if v.is_empty() {
        return f64::INFINITY;
    }
    v.sort_by(|a, b| a.total_cmp(b));
    let n = v.len();
    let pos = q * (n - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi {
        v[lo]
    } else {
        let frac = pos - lo as f64;
        v[lo] * (1.0 - frac) + v[hi] * frac
    }
}

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
            bf_rule: None,
            transform: PriceTransform::Identity,
        }
    }

    /// Set the pre-detector observation transform.
    ///
    /// `Identity` (default) preserves byte-identical behaviour with
    /// earlier callers. `Logit` unbounds prices in (0, 1) for use
    /// against the NIG Gaussian likelihood (for example, probabilities).
    pub fn with_transform(mut self, transform: PriceTransform) -> Self {
        self.transform = transform;
        self
    }

    /// Switch the streaming detector from MAP-drop (default) to the
    /// Bayes-factor decision rule on aggregated short-run-length mass.
    /// Mirrors the offline [`crate::BocpdDetector::detect_bayes_factor`].
    ///
    /// `threshold`: BF cutoff (e.g. `2.0`).
    /// `short_horizon`: K -- mass over r ∈ {0..=K} forms the short side.
    /// `cooldown`: minimum steps between consecutive emissions.
    ///
    /// The rule arms only after BF first drops below 1.0 (posterior
    /// maturity), avoiding the all-mass-at-zero startup spike. Disarms
    /// after each emission so the next fire requires the new regime's
    /// posterior to mature again.
    ///
    /// # Panics
    /// Panics if `threshold <= 0` or `short_horizon >= max_rl`.
    pub fn with_bayes_factor_rule(
        mut self,
        threshold: f64,
        short_horizon: usize,
        cooldown: usize,
    ) -> Self {
        assert!(threshold > 0.0, "threshold must be > 0, got {threshold}");
        assert!(
            short_horizon < self.max_rl,
            "short_horizon {short_horizon} must be < max_rl {}",
            self.max_rl
        );
        self.bf_rule = Some(BfRuleState {
            threshold,
            short_horizon,
            cooldown,
            armed: false,
            last_emit: None,
            quantile: None,
            quantile_window: default_bf_quantile_window(),
            recent_bf: std::collections::VecDeque::new(),
        });
        self
    }

    /// Switch the streaming detector to a quantile-based dynamic-threshold
    /// BF rule. Once a sliding window of `window` BF values is full,
    /// fires when current BF exceeds the `quantile`-th quantile of that
    /// window. During warmup (window not yet full), falls back to the
    /// fixed `threshold` argument so the detector still emits if the
    /// stream produces a real outlier early.
    ///
    /// The window updates *after* the fire decision so the current BF
    /// never contaminates its own reference distribution.
    ///
    /// # Panics
    /// Panics if `threshold <= 0`, `short_horizon >= max_rl`,
    /// `quantile` ∉ (0, 1), or `window == 0`.
    pub fn with_bayes_factor_quantile_rule(
        mut self,
        threshold: f64,
        short_horizon: usize,
        cooldown: usize,
        quantile: f64,
        window: usize,
    ) -> Self {
        assert!(threshold > 0.0, "threshold must be > 0, got {threshold}");
        assert!(
            short_horizon < self.max_rl,
            "short_horizon {short_horizon} must be < max_rl {}",
            self.max_rl
        );
        assert!(
            quantile > 0.0 && quantile < 1.0,
            "quantile must be in (0, 1), got {quantile}"
        );
        assert!(window > 0, "window must be > 0");
        self.bf_rule = Some(BfRuleState {
            threshold,
            short_horizon,
            cooldown,
            armed: false,
            last_emit: None,
            quantile: Some(quantile),
            quantile_window: window,
            recent_bf: std::collections::VecDeque::with_capacity(window),
        });
        self
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
            // Pre-detector transform. `Identity` is a no-op pass-through
            // (byte-identical to earlier behaviour); `Logit` unbounds
            // x ∈ (0, 1) for the Gaussian likelihood.
            let x = self.transform.apply(x);
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
            // Borrow-split: collect drained candidates first, then mutate
            // `self.bf_rule` for any failed BF drains so the BF rule can
            // re-arm (offline parity: shift_sigma < 1e-9 must not burn arm).
            let mut drained_failed_bf = false;
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
                    } else if p.is_bf {
                        // Offline `detect_bayes_factor` continues without
                        // disarming when shift_sigma < 1e-9. Mirror that.
                        drained_failed_bf = true;
                    }
                    false
                } else {
                    true
                }
            });
            if drained_failed_bf {
                if let Some(bf) = self.bf_rule.as_mut() {
                    bf.armed = true;
                }
            }

            // Change-point detection. Dispatch on the configured decision
            // rule: BF (opt-in via with_bayes_factor_rule) or MAP-drop.
            let i = self.total_steps - 1;

            if let Some(bf) = self.bf_rule.as_mut() {
                // Offline-vs-streaming divergence note: offline
                // `detect_bayes_factor` does not commit `last_emit` /
                // `armed` when shift_sigma < 1e-9 (i.e. neither cooldown
                // nor arm is "burned"). Streaming commits both at trigger
                // to prevent multi-trigger pile-up during the post-window;
                // failed-drain re-arming above restores the arm so the
                // detector can fire again on the next posterior-maturity
                // cycle. Cooldown remains burned on a failed drain,
                // which can affect the next possible emission. Compare
                // both paths on representative application fixtures.
                let in_cooldown = matches!(
                    bf.last_emit,
                    Some(le) if i.saturating_sub(le) <= bf.cooldown
                );
                if !in_cooldown {
                    // short_mass = Σ_{r ∈ 0..=K} P(r_t = r | y_{1:t}).
                    // rl_log is the just-swapped, normalised posterior.
                    let cap = bf.short_horizon.min(self.max_rl);
                    let mut sm_log = f64::NEG_INFINITY;
                    for r in 0..=cap {
                        let v = self.rl_log[r];
                        if v.is_finite() {
                            sm_log = log_add_exp(sm_log, v);
                        }
                    }
                    let sm = if sm_log.is_finite() { sm_log.exp() } else { 0.0 };
                    let long = (1.0 - sm).max(1e-12);
                    let bf_val = sm / long;

                    if !bf.armed {
                        if bf_val < 1.0 {
                            bf.armed = true;
                        }
                    } else {
                        // Resolve the active threshold: dynamic quantile
                        // once the window is full; fixed `threshold`
                        // during warmup or in fixed mode.
                        let active_threshold = match bf.quantile {
                            Some(q) if bf.recent_bf.len() >= bf.quantile_window => {
                                quantile_of(&bf.recent_bf, q)
                            }
                            _ => bf.threshold,
                        };
                        if bf_val > active_threshold {
                            let look_back = bf.cooldown.min(i);
                            let confidence = self.cp_probs[i.saturating_sub(look_back)..=i]
                                .iter()
                                .copied()
                                .fold(0.0_f64, f64::max)
                                .clamp(0.0, 1.0);
                            bf.last_emit = Some(i);
                            bf.armed = false;
                            self.pending.push(PendingCp {
                                raw_index: self.raw_index_map[i],
                                fire_step: self.total_steps,
                                confidence,
                                is_bf: true,
                            });
                        }
                    }
                    // Update the recent-BF ring AFTER the fire decision
                    // so the current BF never participates in its own
                    // reference distribution. The ring is updated only
                    // when quantile mode is enabled; fixed mode leaves
                    // the ring unchanged.
                    if bf.quantile.is_some() {
                        bf.recent_bf.push_back(bf_val);
                        while bf.recent_bf.len() > bf.quantile_window {
                            bf.recent_bf.pop_front();
                        }
                    }
                }
            } else {
                let drop_to = 3;
                let min_prev_rl = 30;
                let cooldown = 15;
                if i >= min_prev_rl
                    && self.map_rls[i] <= drop_to
                    && i - self.last_detection >= cooldown
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
                            is_bf: false,
                        });
                    }
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
            bf_rule: self.bf_rule.clone(),
            transform: self.transform,
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
            bf_rule: state.bf_rule,
            transform: state.transform,
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

    /// Per-step change-evidence score, derived from current state.
    /// In BF mode: `bf_val = short_mass / (1 - short_mass)`.
    /// In MAP-drop mode: `cp_probs[total_steps - 1]` -- the BOCPD
    /// posterior P(r=0).
    /// Returns 0 before any observations have been processed.
    /// **Convention: higher = more change-evidence.** Used by
    /// `multistream::ScoreStream`.
    pub fn last_score(&self) -> f64 {
        if self.total_steps == 0 {
            return 0.0;
        }
        let i = self.total_steps - 1;
        match self.bf_rule.as_ref() {
            Some(bf) => {
                let cap = bf.short_horizon.min(self.max_rl);
                let mut sm_log = f64::NEG_INFINITY;
                for r in 0..=cap {
                    let v = self.rl_log[r];
                    if v.is_finite() {
                        sm_log = log_add_exp(sm_log, v);
                    }
                }
                let sm = if sm_log.is_finite() { sm_log.exp() } else { 0.0 };
                let long = (1.0 - sm).max(1e-12);
                sm / long
            }
            None => self.cp_probs[i],
        }
    }
}

impl crate::multistream::ScoreStream for StreamingDetector {
    fn step_score(&mut self, x: f64) -> f64 {
        // Run the recursion on this single observation; ignore any
        // CPs the detector itself emits (the aggregator owns the
        // multi-stream trigger). Then read the per-step score.
        let _ = self.step(&[x]);
        self.last_score()
    }

    fn score_kind(&self) -> crate::multistream::ScoreKind {
        if self.bf_rule.is_some() {
            crate::multistream::ScoreKind::BayesFactor
        } else {
            crate::multistream::ScoreKind::CpProbability
        }
    }

    fn step_count(&self) -> usize {
        self.total_steps
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
        // Synthetic latency budget for the test profile; hardware affects elapsed time.
        assert!(
            elapsed.as_millis() < 60,
            "streaming tick took {}ms, need <60ms (debug) / <10ms (release)",
            elapsed.as_millis()
        );
    }

    #[test]
    fn streaming_bf_detects_clean_shift() {
        let mut det =
            StreamingDetector::new(200.0, 250).with_bayes_factor_rule(2.0, 3, 15);
        let data: Vec<f64> = std::iter::repeat_n(0.0, 100)
            .chain(std::iter::repeat_n(5.0, 100))
            .collect();
        let cps = det.step(&data);
        assert!(!cps.is_empty(), "BF rule should detect the mean shift");
        assert!(
            (cps[0].index as i64 - 100).abs() < 25,
            "BF CP near 100, got {}",
            cps[0].index
        );
    }

    #[test]
    fn streaming_bf_no_startup_fire() {
        // Posterior-maturity guard: with only stationary data (no real shift)
        // BF must not emit. The all-mass-at-r=0 startup spike is what the
        // armed flag specifically defends against.
        let mut det =
            StreamingDetector::new(200.0, 250).with_bayes_factor_rule(2.0, 3, 15);
        let mut rng = crate::eval::Rng::new(7);
        let data: Vec<f64> = (0..200).map(|_| rng.normal(0.0, 1.0)).collect();
        let cps = det.step(&data);
        assert!(
            cps.is_empty(),
            "stationary data → 0 CPs, got {} ({:?})",
            cps.len(),
            cps.iter().map(|c| c.index).collect::<Vec<_>>()
        );
    }

    #[test]
    fn streaming_bf_state_save_restore_roundtrips() {
        // Run partial sequence under BF rule, save+restore, finish on both;
        // detections must match identically.
        let mut det =
            StreamingDetector::new(200.0, 250).with_bayes_factor_rule(2.0, 3, 15);
        let phase1: Vec<f64> = std::iter::repeat_n(0.0, 80).collect();
        det.step(&phase1);

        let json = serde_json::to_string(&det.save_state()).unwrap();
        let restored: DetectorState = serde_json::from_str(&json).unwrap();
        // Round-trip must preserve the BF rule (else continuation silently
        // falls back to MAP-drop with a different `last_detection` bookkeeping).
        assert!(
            restored.bf_rule.is_some(),
            "bf_rule must survive serde round-trip"
        );
        let mut det2 = StreamingDetector::restore(restored).unwrap();

        let phase2: Vec<f64> = std::iter::repeat_n(5.0, 120).collect();
        let cps1 = det.step(&phase2);
        let cps2 = det2.step(&phase2);
        assert_eq!(cps1.len(), cps2.len(), "BF restore drift in CP count");
        for (a, b) in cps1.iter().zip(cps2.iter()) {
            assert_eq!(a.index, b.index, "BF restore drift in CP index");
            assert!((a.confidence - b.confidence).abs() < 1e-9);
        }
    }

    #[test]
    fn streaming_bf_quantile_fires_on_real_outlier() {
        // 200 stationary steps where BF stays small, then a clean mean
        // shift large enough that BF spikes well above the q99 of the
        // post-warmup window. Quantile rule (q=0.99, window=100) and
        // fixed rule (τ=2.0) should both fire near the shift.
        use crate::eval::Rng;
        let mut rng = Rng::new(7);
        let mut data: Vec<f64> = (0..200).map(|_| rng.normal(0.0, 1.0)).collect();
        data.extend((0..100).map(|_| 5.0 + rng.normal(0.0, 1.0)));

        let mut q = StreamingDetector::new(200.0, 250)
            .with_bayes_factor_quantile_rule(2.0, 3, 15, 0.99, 100);
        let cps_q = q.step(&data);
        assert!(
            !cps_q.is_empty(),
            "quantile rule should fire on the post-200 shift"
        );
        let near_shift = cps_q.iter().any(|c| (c.index as i64 - 200).abs() < 40);
        assert!(near_shift, "quantile fire should land near the shift, got {cps_q:?}");

        let mut f = StreamingDetector::new(200.0, 250)
            .with_bayes_factor_rule(2.0, 3, 15);
        let cps_f = f.step(&data);
        assert!(!cps_f.is_empty(), "fixed rule should also fire on this shift");
    }

    #[test]
    fn streaming_bf_quantile_far_tracks_quantile_level() {
        // Calibration property: on stationary data the quantile rule's
        // FAR is governed by (1 - q) modulo armed/cooldown gating. A
        // higher q ⇒ stricter threshold ⇒ fewer fires. This is the knob
        // the user actually has: pick `q` to target a FAR, not pick a
        // BF magnitude they can't predict from the stream's noise.
        use crate::eval::Rng;
        let n = 1500;

        let counts: Vec<usize> = [0.90_f64, 0.99_f64]
            .iter()
            .enumerate()
            .map(|(i, &q)| {
                let mut rng = Rng::new(11 + i as u64);
                let data: Vec<f64> = (0..n).map(|_| rng.normal(0.0, 1.0)).collect();
                let mut det = StreamingDetector::new(200.0, 250)
                    .with_bayes_factor_quantile_rule(2.0, 3, 15, q, 300);
                det.step(&data).len()
            })
            .collect();
        assert!(
            counts[0] >= counts[1],
            "stricter quantile must fire ≤ permissive: q=.90 → {}, q=.99 → {}",
            counts[0],
            counts[1]
        );
    }

    #[test]
    fn streaming_bf_quantile_warmup_fills_ring_to_cap() {
        // Pre-warmup: window not yet full ⇒ fallback (fixed τ=2.0) is
        // the active threshold. After enough steps the ring saturates
        // at `quantile_window` and the quantile threshold takes over.
        // This pins (a) the ring growth + cap behaviour, (b) that
        // fires can land under the fallback regime, and (c) that the
        // armed/cooldown gating still allows the ring to populate
        // through cooldown windows enough to reach the cap.
        use crate::eval::Rng;
        let mut rng = Rng::new(13);
        let mut data: Vec<f64> = (0..30).map(|_| rng.normal(0.0, 1.0)).collect();
        data.extend((0..50).map(|_| 5.0 + rng.normal(0.0, 1.0)));
        data.extend((0..400).map(|_| rng.normal(5.0, 1.0)));

        let mut det = StreamingDetector::new(200.0, 250)
            .with_bayes_factor_quantile_rule(2.0, 3, 15, 0.99, 200);
        let _ = det.step(&data);
        let bf = det.save_state().bf_rule.expect("bf_rule present");
        assert_eq!(
            bf.recent_bf.len(),
            bf.quantile_window,
            "ring should saturate at the configured window cap"
        );
        assert!(
            bf.recent_bf.iter().all(|x| x.is_finite()),
            "ring contents must all be finite"
        );
    }

    #[test]
    fn quantile_of_drops_nan_values() {
        // The dispatch's recent_bf push only stores finite values in
        // practice, but the helper must still handle NaN defensively
        // because restored snapshots may contain non-finite values.
        use std::collections::VecDeque;
        let mut ring: VecDeque<f64> = VecDeque::new();
        ring.push_back(0.1);
        ring.push_back(f64::NAN);
        ring.push_back(0.5);
        ring.push_back(f64::NAN);
        ring.push_back(1.0);
        // q-th quantile of {0.1, 0.5, 1.0} (NaN dropped). q=0.5 ⇒ 0.5.
        let q50 = super::quantile_of(&ring, 0.5);
        assert!((q50 - 0.5).abs() < 1e-9, "expected 0.5, got {q50}");
        // Empty / all-NaN ring ⇒ +inf so the rule never fires (safe
        // degenerate behaviour rather than a panic).
        let mut ring2: VecDeque<f64> = VecDeque::new();
        ring2.push_back(f64::NAN);
        ring2.push_back(f64::NAN);
        let q = super::quantile_of(&ring2, 0.99);
        assert_eq!(q, f64::INFINITY, "all-NaN ring must return +inf");
    }

    #[test]
    fn bf_rule_state_serde_round_trip_with_quantile() {
        // Snapshot without quantile support: no `quantile`, `quantile_window`, or
        // `recent_bf` fields. Forward-compat via #[serde(default)].
        let mut state_json: serde_json::Value = serde_json::json!({
            "threshold": 2.0,
            "short_horizon": 3,
            "cooldown": 15,
            "armed": true,
            "last_emit": null,
        });
        let s: BfRuleState = serde_json::from_value(state_json.clone()).unwrap();
        assert_eq!(s.quantile, None);
        assert_eq!(s.quantile_window, default_bf_quantile_window());
        assert!(s.recent_bf.is_empty());

        // Forward direction: a quantile-enabled state round-trips with
        // its full ring intact.
        state_json["quantile"] = serde_json::json!(0.99);
        state_json["quantile_window"] = serde_json::json!(50);
        state_json["recent_bf"] = serde_json::json!([0.1, 0.2, 0.5, 1.5]);
        let s2: BfRuleState = serde_json::from_value(state_json).unwrap();
        assert_eq!(s2.quantile, Some(0.99));
        assert_eq!(s2.quantile_window, 50);
        assert_eq!(s2.recent_bf.len(), 4);
        let json = serde_json::to_string(&s2).unwrap();
        let s3: BfRuleState = serde_json::from_str(&json).unwrap();
        assert_eq!(s2.recent_bf.len(), s3.recent_bf.len());
        assert_eq!(s2.quantile, s3.quantile);
    }

    #[test]
    fn streaming_bf_quantile_save_restore_round_trips() {
        // Run a partial stream under quantile mode, save+restore, finish
        // on both; detections + recent_bf ring must match.
        let mut det = StreamingDetector::new(200.0, 250)
            .with_bayes_factor_quantile_rule(2.0, 3, 15, 0.99, 100);
        let phase1: Vec<f64> = std::iter::repeat_n(0.0, 150).collect();
        det.step(&phase1);

        let json = serde_json::to_string(&det.save_state()).unwrap();
        let restored: DetectorState = serde_json::from_str(&json).unwrap();
        let bf = restored.bf_rule.as_ref().expect("bf_rule survives roundtrip");
        assert_eq!(bf.quantile, Some(0.99));
        assert_eq!(bf.quantile_window, 100);
        assert!(
            !bf.recent_bf.is_empty(),
            "recent_bf ring must persist; got empty after 150 steps"
        );
        let len_before = bf.recent_bf.len();

        let mut det2 = StreamingDetector::restore(restored).unwrap();
        // The deserialized restore preserves the ring (verified above);
        // running phase2 on both must produce identical CPs.
        let phase2: Vec<f64> = std::iter::repeat_n(5.0, 100).collect();
        let cps1 = det.step(&phase2);
        let cps2 = det2.step(&phase2);
        assert_eq!(cps1.len(), cps2.len(), "quantile restore drift in CP count");
        for (a, b) in cps1.iter().zip(cps2.iter()) {
            assert_eq!(a.index, b.index, "quantile restore drift in CP index");
        }
        // Ring should grow further over phase2.
        let len_after = det2.bf_rule.as_ref().unwrap().recent_bf.len();
        assert!(
            len_after >= len_before,
            "ring shrank from {len_before} to {len_after}"
        );
    }

    #[test]
    fn streaming_bf_old_state_without_bf_rule_loads() {
        // Forward-compat: a v0.8 snapshot has no `bf_rule` field. After
        // round-trip the restored detector should default to MAP-drop.
        let mut det = StreamingDetector::new(200.0, 250);
        det.step(&vec![0.0; 80]);
        let mut state_json = serde_json::to_value(det.save_state()).unwrap();
        // Drop the bf_rule field to simulate a pre-0.9 snapshot.
        state_json.as_object_mut().unwrap().remove("bf_rule");
        let restored: DetectorState = serde_json::from_value(state_json).unwrap();
        assert!(restored.bf_rule.is_none());
        let _det = StreamingDetector::restore(restored).unwrap();
    }

    // PriceTransform tests --------------------------------------------

    #[test]
    fn transform_identity_is_byte_identical_to_default() {
        // Hard invariant: explicit Identity produces the exact same map_rls
        // and cp_probs as a detector constructed without with_transform.
        // This pins byte-identical behavior on the synthetic mean-shift fixture.
        let data: Vec<f64> = (0..300)
            .map(|i| if i < 150 { 0.0_f64 } else { 5.0_f64 })
            .collect();

        let mut default_det = StreamingDetector::new(200.0, 250);
        let default_cps = default_det.step(&data);

        let mut identity_det =
            StreamingDetector::new(200.0, 250).with_transform(PriceTransform::Identity);
        let identity_cps = identity_det.step(&data);

        assert_eq!(
            default_cps.len(),
            identity_cps.len(),
            "Identity must produce same number of change points as default"
        );
        for (d, i) in default_cps.iter().zip(identity_cps.iter()) {
            assert_eq!(d.index, i.index, "Identity cp index must match default");
            assert_eq!(
                d.confidence, i.confidence,
                "Identity cp confidence must match default"
            );
            assert_eq!(
                d.shift_sigma, i.shift_sigma,
                "Identity shift_sigma must match default"
            );
        }
        assert_eq!(
            default_det.save_state().map_rls,
            identity_det.save_state().map_rls,
            "Identity map_rls must match default"
        );
    }

    #[test]
    fn transform_logit_operates_on_different_scale_than_identity() {
        // Logit maps sub-cent prices into log-odds space. After feeding the
        // same sub-cent prices through both transforms, the Welford running
        // mean must differ dramatically -- Identity tracks the raw price
        // (~0.0075) while Logit tracks the log-odds (~-4.95).
        let data: Vec<f64> = (0..300)
            .map(|i| if i < 150 { 0.005_f64 } else { 0.010_f64 })
            .collect();

        let mut identity_det = StreamingDetector::new(200.0, 250);
        identity_det.step(&data);
        let id_mean = identity_det.save_state().welford.mean;

        let mut logit_det =
            StreamingDetector::new(200.0, 250).with_transform(PriceTransform::Logit);
        logit_det.step(&data);
        let lg_mean = logit_det.save_state().welford.mean;

        // Identity mean ≈ 0.0075 (raw price); Logit mean ≈ -4.95 (log-odds).
        assert!(
            (id_mean - lg_mean).abs() > 1.0,
            "Welford means must differ by >1.0 (identity≈0.0075 vs logit≈-4.95), got {id_mean} vs {lg_mean}"
        );
    }

    #[test]
    fn transform_logit_round_trips_through_save_restore() {
        let data = vec![0.3_f64; 80];
        let mut det =
            StreamingDetector::new(200.0, 250).with_transform(PriceTransform::Logit);
        det.step(&data);

        let state = det.save_state();
        assert_eq!(state.transform, PriceTransform::Logit, "state must carry Logit");

        let restored = StreamingDetector::restore(state).unwrap();
        let mut state_json = serde_json::to_value(restored.save_state()).unwrap();

        // Forward-compat: a earlier snapshot has no `transform` field.
        // After dropping it the detector should restore as Identity.
        state_json.as_object_mut().unwrap().remove("transform");
        let compat: DetectorState = serde_json::from_value(state_json).unwrap();
        assert_eq!(
            compat.transform,
            PriceTransform::Identity,
            "missing transform field must default to Identity"
        );
    }

    #[test]
    fn logit_apply_is_monotone_and_finite_on_interior() {
        // Basic sanity on PriceTransform::Logit.apply: monotone, finite,
        // symmetric around 0.5, and no ±inf on near-boundary inputs.
        let t = PriceTransform::Logit;
        assert!(t.apply(0.0).is_finite(), "logit(0) must be clamped finite");
        assert!(t.apply(1.0).is_finite(), "logit(1) must be clamped finite");
        assert_eq!(t.apply(0.5), 0.0, "logit(0.5) = 0");
        assert!(t.apply(0.3) < 0.0, "logit(0.3) < 0");
        assert!(t.apply(0.7) > 0.0, "logit(0.7) > 0");
        // Monotone: logit(0.3) < logit(0.5) < logit(0.7)
        assert!(t.apply(0.3) < t.apply(0.5));
        assert!(t.apply(0.5) < t.apply(0.7));
    }
}
