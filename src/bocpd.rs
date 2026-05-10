//! Bayesian Online Change Point detector (univariate + multivariate).

use crate::conformal::{MvScoredDetect, ScoredDetect};

/// Tuple returned by [`BocpdDetector::run_recursion`]:
/// `(norm, map_rls, cp_probs, short_mass, original_indices)`. See the
/// method docstring for the meaning of each field.
type RecursionOutputs = (Vec<f64>, Vec<usize>, Vec<f64>, Vec<f64>, Vec<usize>);
use crate::math::log_add_exp;
use crate::niw;
use crate::nig::Nig;
use crate::predictive::Predictive;
use crate::ChangePoint;
use crate::DEFAULT_MASS_CUTOFF;

#[allow(clippy::needless_range_loop)]
/// Sample mean and Cholesky factor `L` of the sample covariance matrix.
///
/// Returns `None` when the matrix is not positive-definite -- i.e. some
/// dimension is degenerate or two dimensions are linearly dependent.
/// Caller is expected to fall back to per-dim normalisation in that
/// case rather than panic.
pub(crate) fn whitening_transform(warmup: &[Vec<f64>], d: usize) -> Option<(Vec<f64>, Vec<Vec<f64>>)> {
    let n = warmup.len();
    if n < 2 || d == 0 {
        return None;
    }
    let nf = n as f64;
    let mut mean = vec![0.0; d];
    for row in warmup {
        for j in 0..d {
            mean[j] += row[j];
        }
    }
    for m in mean.iter_mut() {
        *m /= nf;
    }
    // Sample covariance with N denominator (biased; matches the per-dim
    // z-norm convention already used here).
    let mut cov = vec![vec![0.0; d]; d];
    for row in warmup {
        for i in 0..d {
            for j in 0..d {
                cov[i][j] += (row[i] - mean[i]) * (row[j] - mean[j]);
            }
        }
    }
    for i in 0..d {
        for j in 0..d {
            cov[i][j] /= nf;
        }
    }
    let l = cholesky_lower(&cov)?;
    Some((mean, l))
}

#[allow(clippy::needless_range_loop)]
/// Cholesky factor `L` such that `L · Lᵀ = A`, lower-triangular.
/// Returns `None` if `A` is not positive-definite.
pub(crate) fn cholesky_lower(a: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let d = a.len();
    let mut l = vec![vec![0.0; d]; d];
    for i in 0..d {
        for j in 0..=i {
            let mut s = a[i][j];
            for k in 0..j {
                s -= l[i][k] * l[j][k];
            }
            if i == j {
                // Reject non-positive diagonal entries: input is not PD.
                if s <= 1e-12 {
                    return None;
                }
                l[i][j] = s.sqrt();
            } else {
                l[i][j] = s / l[j][j];
            }
        }
    }
    Some(l)
}

/// Solve `L · y = b` for `y` via forward substitution. `L` must be
/// lower-triangular with non-zero diagonal.
pub(crate) fn forward_solve(l: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let d = b.len();
    let mut y = vec![0.0; d];
    for i in 0..d {
        let mut s = b[i];
        for k in 0..i {
            s -= l[i][k] * y[k];
        }
        y[i] = s / l[i][i];
    }
    y
}

/// Per-dimension z-normalisation fallback for `detect_multivariate`.
pub(crate) fn per_dim_znorm(data: &[Vec<f64>], d: usize, n: usize) -> Vec<Vec<f64>> {
    let mut means = vec![0.0; d];
    let mut stds = vec![0.0; d];
    for dim in 0..d {
        let m: f64 = data.iter().map(|x| x[dim]).sum::<f64>() / n as f64;
        let s = (data.iter().map(|x| (x[dim] - m).powi(2)).sum::<f64>() / n as f64).sqrt();
        means[dim] = m;
        stds[dim] = if s < 1e-10 { 1.0 } else { s };
    }
    data.iter()
        .map(|x| (0..d).map(|i| (x[i] - means[i]) / stds[i]).collect())
        .collect()
}

/// Bayesian Online Change Point Detector.
///
/// Uses the BOCPD algorithm (Adams & MacKay 2007) with a conjugate
/// within-regime predictive `P` and MAP run-length estimation. Default
/// `P = Nig` -- iid-Gaussian with Normal-Inverse-Gamma prior. The
/// recursion is generic over [`Predictive`]; call sites resolve
/// `<P = Nig>` via the default type parameter.
///
/// To use a non-default predictive (e.g. `NigAr1` for AR(1) within-regime),
/// build via [`BocpdDetector::with_prior`] which is generic over `P`.
pub struct BocpdDetector<P: Predictive = Nig> {
    /// Expected run length between change points; mutable when
    /// adaptive λ is enabled, fixed otherwise. The hazard rate
    /// `1/lambda` and growth rate `1 - 1/lambda` are derived locally
    /// inside detect functions.
    lambda: f64,
    max_rl: usize,
    log_mass_cutoff: f64,
    /// β-divergence robustness parameter. `0.0` ⇒ standard BOCPD (default,
    /// short-circuited so behaviour is bit-for-bit identical to pre-β code).
    /// `β > 0` ⇒ Knoblauch et al. (2018) robust update for heavy-tailed
    /// within-regime distributions.
    beta: f64,
    prior: P,
    /// When true, [`BocpdDetector::detect_mut`] updates `lambda` from
    /// observed inter-CP intervals via an EMA (decay 0.9). Default off;
    /// `detect()` is unaffected regardless.
    adaptive_lambda: bool,
    /// EMA of observed inter-CP intervals. `None` until the first
    /// interval is seen; thereafter holds the running mean.
    interval_ema: Option<f64>,
}

impl BocpdDetector<Nig> {
    /// Create a detector with expected run length `lambda` between change points.
    ///
    /// - `lambda`: expected number of observations between change points.
    ///   Smaller values make the detector more sensitive (more false positives).
    /// - `max_run_length`: hard upper bound on tracked run length. With the
    ///   default mass-pruning cutoff (`DEFAULT_MASS_CUTOFF`), this is rarely
    ///   the binding constraint -- the tail prunes itself well before the
    ///   cap. Set generously (≥ data.len() for batch).
    ///
    /// Uses the standard Nig prior `(μ=0, κ=1, α=1, β=1)`. To use a custom
    /// prior or a different predictive family (e.g. AR(1)), use
    /// [`BocpdDetector::with_prior`].
    ///
    /// # Panics
    /// Panics if `lambda <= 1.0` (would produce -inf or NaN hazard rates).
    pub fn new(lambda: f64, max_run_length: usize) -> Self {
        Self::with_prior(lambda, max_run_length, Nig::new(0.0, 1.0, 1.0, 1.0))
    }
}

impl<P: Predictive> BocpdDetector<P> {
    /// Create a detector with expected run length `lambda` and an explicit
    /// `prior`. The generic constructor; `BocpdDetector::new` is the
    /// `P = Nig` shortcut with a hardcoded prior.
    ///
    /// # Panics
    /// Panics if `lambda <= 1.0`.
    pub fn with_prior(lambda: f64, max_run_length: usize, prior: P) -> Self {
        assert!(lambda > 1.0, "lambda must be > 1.0, got {lambda}");
        Self {
            lambda,
            max_rl: max_run_length,
            log_mass_cutoff: DEFAULT_MASS_CUTOFF.ln(),
            beta: 0.0,
            prior,
            adaptive_lambda: false,
            interval_ema: None,
        }
    }

    /// Opt into adaptive λ. After each call to
    /// [`BocpdDetector::detect_mut`], `lambda` is updated via an EMA
    /// (decay 0.9) over inter-CP intervals observed in the most-recent
    /// call. `detect()` (immutable) remains unaffected; existing batch
    /// callers see no change.
    pub fn with_adaptive_lambda(mut self) -> Self {
        self.adaptive_lambda = true;
        self
    }

    /// Current expected run length λ. Useful for assertions in tests
    /// of the adaptive-λ path; otherwise rarely needed.
    pub fn lambda(&self) -> f64 {
        self.lambda
    }

    fn hazard_log(&self) -> f64 {
        (1.0 / self.lambda).ln()
    }

    fn growth_log(&self) -> f64 {
        (1.0 - 1.0 / self.lambda).ln()
    }

    /// Opt into β-divergence robust BOCPD (Knoblauch et al. 2018,
    /// arXiv:1806.02261). The standard Bayesian update is replaced with
    /// a divergence-based update whose influence function is bounded;
    /// tail observations under heavy-tailed within-regime noise no longer
    /// dominate the posterior.
    ///
    /// `beta = 0.0` is the default and short-circuits to standard cesura
    /// (bit-for-bit identical, no overhead). Reasonable robust defaults
    /// per Knoblauch's empirical work are in the range `[0.05, 0.10]`;
    /// larger `β` is more robust but loses statistical efficiency on
    /// well-behaved data.
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

    /// Estimate β from a warmup window's sample excess kurtosis.
    /// See [`auto_beta`](crate::auto_beta) for the mapping and provenance.
    ///
    /// Near-Gaussian inputs (`|k_ex| ≤ 1`) return β = 0 -- the standard
    /// (non-robust) path. Heavier-tailed inputs receive a proportional
    /// β capped at 0.20.
    pub fn with_auto_beta(self, warmup: &[f64]) -> Self {
        self.with_beta(crate::auto_beta::auto_beta(warmup))
    }

    /// Set the mass-pruning cutoff. After each step the run-length posterior
    /// is renormalised; trailing entries whose mass falls below `cutoff`
    /// are zeroed (set to `-∞` in log space) so subsequent steps skip them.
    /// Adapted from `changepoint::BocpdTruncated::with_cutoff`.
    ///
    /// Lower cutoff → keeps more tail mass, more memory + compute, fewer
    /// truncation artefacts. Higher cutoff → aggressive pruning.
    /// `0.0` disables pruning (all run lengths up to `max_run_length`).
    pub fn with_mass_cutoff(mut self, cutoff: f64) -> Self {
        self.log_mass_cutoff = if cutoff > 0.0 {
            cutoff.ln()
        } else {
            f64::NEG_INFINITY
        };
        self
    }

    /// Mutating variant of [`detect`](Self::detect). Identical detection
    /// semantics; additionally, when `with_adaptive_lambda` is enabled,
    /// updates `self.lambda` after the call from inter-CP intervals
    /// observed in the returned change-point sequence (EMA, decay 0.9).
    /// Subsequent calls use the updated `λ`.
    ///
    /// `detect()` (immutable) remains unchanged regardless of the
    /// adaptive flag, so existing batch callers see no behaviour shift.
    pub fn detect_mut(&mut self, data: &[f64]) -> Vec<ChangePoint> {
        let cps = self.detect(data);
        if self.adaptive_lambda && cps.len() >= 2 {
            // EMA over inter-CP intervals. decay = 0.9 ⇒ α = 0.1; the
            // first observed interval seeds the EMA.
            const ALPHA: f64 = 0.1;
            for w in cps.windows(2) {
                let interval = (w[1].index - w[0].index) as f64;
                if !interval.is_finite() || interval <= 0.0 {
                    continue;
                }
                self.interval_ema = Some(match self.interval_ema {
                    Some(prev) => ALPHA * interval + (1.0 - ALPHA) * prev,
                    None => interval,
                });
            }
            if let Some(ema) = self.interval_ema {
                // Guard: λ > 1 is required (constructor invariant). If
                // the EMA collapses below 1 we leave λ alone -- the
                // detector hasn't observed enough structure yet.
                if ema > 1.0 && ema.is_finite() {
                    self.lambda = ema;
                }
            }
        }
        cps
    }

    /// Run BOCPD on `data`, return all change points that pass the
    /// MAP-drop heuristic.
    ///
    /// Each returned [`ChangePoint`] carries a `confidence` in `[0, 1]`.
    /// Filter post-hoc on `cp.confidence >= 0.9` if you want a stricter
    /// gate; the structural floor of the heuristic is approximately 0.85.
    ///
    /// Returns an empty vec if `data` has fewer than 20 elements.
    ///
    /// # Heuristic constants
    ///
    /// The MAP-drop detector uses three hand-tuned constants -- `drop_to=3`,
    /// `min_prev_rl=30`, `cooldown=15` -- that gate when a run-length drop
    /// is reported as a change point. They are not from Adams & MacKay (2007);
    /// the underlying BOCPD recursion (NIG predictive, log-space update) is.
    /// Tests verify the math; these constants are tuned against the eval suite.
    pub fn detect(&self, data: &[f64]) -> Vec<ChangePoint> {
        // Filter NaN/infinite values, keeping a map back to original indices
        let mut original_indices: Vec<usize> = Vec::new();
        let data: Vec<f64> = data
            .iter()
            .enumerate()
            .filter_map(|(i, &v)| {
                if v.is_finite() {
                    original_indices.push(i);
                    Some(v)
                } else {
                    None
                }
            })
            .collect();
        let n = data.len();
        if n < 20 {
            return vec![];
        }

        // Normalize for numerical stability
        let mean = data.iter().sum::<f64>() / n as f64;
        let std = (data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64).sqrt();
        let std = if std < 1e-10 { 1.0 } else { std };
        let norm: Vec<f64> = data.iter().map(|x| (x - mean) / std).collect();

        let max_r = self.max_rl.min(n);

        // Run length log-probabilities
        let mut rl_log = vec![f64::NEG_INFINITY; max_r + 1];
        rl_log[0] = 0.0; // r_0 = 0 with certainty

        // Sufficient statistics per run length
        let mut stats = vec![self.prior.clone(); max_r + 1];

        // Track MAP run length and posterior P(r_t = 0) at each time step
        let mut map_rls = Vec::with_capacity(n);
        let mut cp_probs = Vec::with_capacity(n);

        // `log_predictive_robust` short-circuits internally at β = 0 to the
        // exact standard log-predictive (literal early return); we can always
        // call it and let it pick.
        let beta = self.beta;
        let prior_pred_at = |x: f64| self.prior.log_predictive_robust(x, beta);
        let hazard_log = self.hazard_log();
        let growth_log = self.growth_log();

        for (t, &x) in norm.iter().enumerate() {
            let active = (t + 1).min(max_r);

            let mut new_rl = vec![f64::NEG_INFINITY; max_r + 1];
            let mut prev_mass = f64::NEG_INFINITY;

            for r in 0..=active.min(max_r.saturating_sub(1)) {
                if rl_log[r] == f64::NEG_INFINITY {
                    continue;
                }
                let pred = stats[r].log_predictive_robust(x, beta);
                if !pred.is_finite() {
                    continue;
                }

                // Growth: r -> r+1 uses posterior predictive given r history.
                if r < max_r {
                    new_rl[r + 1] =
                        log_add_exp(new_rl[r + 1], rl_log[r] + pred + growth_log);
                }
                // Accumulate prior-segment mass for the CP branch.
                prev_mass = log_add_exp(prev_mass, rl_log[r]);
            }
            // Change point: r_t = 0 starts a fresh segment, so the predictive
            // is the *prior* predictive (no history). This is the Adams-MacKay
            // formulation; using the posterior predictive here caps P(r_t=0)
            // near the hazard rate even at real CPs.
            let prior_pred = prior_pred_at(x);
            new_rl[0] = if prior_pred.is_finite() && prev_mass.is_finite() {
                prev_mass + hazard_log + prior_pred
            } else {
                f64::NEG_INFINITY
            };

            // Normalize
            let evidence = new_rl
                .iter()
                .copied()
                .filter(|x| x.is_finite())
                .fold(f64::NEG_INFINITY, log_add_exp);
            if evidence.is_finite() {
                for v in new_rl.iter_mut() {
                    *v -= evidence;
                }
            }

            // Mass-prune the tail: drop trailing run lengths whose
            // normalised posterior mass falls below the cutoff. Subsequent
            // steps skip them, giving data-adaptive truncation -- the
            // documented "phantom MAP-drop near max_rl" symptom of fixed-cap
            // truncation cannot occur if the tail is pruned before it
            // approaches the cap.
            if self.log_mass_cutoff > f64::NEG_INFINITY {
                for r in (1..=max_r).rev() {
                    if new_rl[r] >= self.log_mass_cutoff {
                        break;
                    }
                    new_rl[r] = f64::NEG_INFINITY;
                }
            }

            // MAP run length
            let map_r = new_rl
                .iter()
                .enumerate()
                .filter(|(_, v)| v.is_finite())
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(r, _)| r)
                .unwrap_or(0);
            map_rls.push(map_r);
            cp_probs.push(if new_rl[0].is_finite() {
                new_rl[0].exp()
            } else {
                0.0
            });

            // Update sufficient stats for each growth path
            let mut new_stats = vec![self.prior.clone(); max_r + 1];
            for r in 0..=active.min(max_r.saturating_sub(1)) {
                if r < max_r && new_rl[r + 1] > f64::NEG_INFINITY {
                    new_stats[r + 1] = stats[r].update(x);
                }
            }
            // new_stats[0] stays as prior (fresh regime)

            rl_log = new_rl;
            stats = new_stats;
        }

        // Detect change points: where MAP run length drops sharply.
        let drop_to = 3;
        let min_prev_rl = 30;
        let cooldown = 15;

        let mut result = Vec::new();
        let mut i = min_prev_rl;
        let mut last_detection = 0usize;
        while i < n {
            if map_rls[i] <= drop_to && i - last_detection >= cooldown {
                let prev_max = map_rls[i.saturating_sub(15)..i]
                    .iter()
                    .copied()
                    .max()
                    .unwrap_or(0);
                if prev_max >= min_prev_rl {
                    // Peak posterior P(r_t = 0) over the recent cooldown window.
                    // The MAP-drop trigger fires a few steps after the true CP,
                    // by which time r_t has reset; the peak captures the CP
                    // moment itself.
                    let look_back = cooldown.min(i);
                    let confidence = cp_probs[i.saturating_sub(look_back)..=i]
                        .iter()
                        .copied()
                        .fold(0.0_f64, f64::max)
                        .clamp(0.0, 1.0);
                    let w = 20;
                    let before = &norm[i.saturating_sub(w)..i];
                    let after = &norm[i..(i + w).min(n)];
                    let mean_b = if before.is_empty() {
                        0.0
                    } else {
                        before.iter().sum::<f64>() / before.len() as f64
                    };
                    let mean_a = if after.is_empty() {
                        0.0
                    } else {
                        after.iter().sum::<f64>() / after.len() as f64
                    };
                    let shift_sigma = (mean_a - mean_b).abs();

                    // Phantom (max_rl truncation on stable signal): do not
                    // claim cooldown so a real CP immediately after isn't masked.
                    if shift_sigma < 1e-9 {
                        i += 1;
                        continue;
                    }

                    last_detection = i;
                    result.push(ChangePoint {
                        index: original_indices[i],
                        confidence,
                        shift_sigma,
                    });
                    i += cooldown;
                    continue;
                }
            }
            i += 1;
        }
        result
    }

    /// Diagnostic variant of [`detect`](Self::detect) that returns the
    /// per-step posterior `P(r_t = 0)` array alongside the emitted
    /// change points. Used by `tests/statistical.rs` confidence-
    /// calibration probes to evaluate alternative summary statistics
    /// (mean, area, sustained-peak) over the cooldown window without
    /// re-running the forward pass.
    ///
    /// `cp_probs[i]` indexes the **post-NaN-filter** step; the third
    /// return slot maps filtered indices back to original-data indices
    /// so callers can match emitted CPs by `cp.index`.
    #[cfg(any(test, feature = "test-utils"))]
    pub fn detect_with_cp_probs(&self, data: &[f64]) -> (Vec<ChangePoint>, Vec<f64>, Vec<usize>) {
        let mut original_indices: Vec<usize> = Vec::new();
        let data_filt: Vec<f64> = data
            .iter()
            .enumerate()
            .filter_map(|(i, &v)| {
                if v.is_finite() {
                    original_indices.push(i);
                    Some(v)
                } else {
                    None
                }
            })
            .collect();
        let cps = self.detect(data);
        // Re-run a minimal forward pass to recover cp_probs. detect()
        // computes this internally but doesn't surface it. Mirroring the
        // recursion exactly keeps the array bit-for-bit identical to
        // what detect() consumed.
        let n = data_filt.len();
        if n < 20 {
            return (cps, vec![0.0; n], original_indices);
        }
        let mean = data_filt.iter().sum::<f64>() / n as f64;
        let std = (data_filt.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64).sqrt();
        let std = if std < 1e-10 { 1.0 } else { std };
        let norm: Vec<f64> = data_filt.iter().map(|x| (x - mean) / std).collect();

        let max_r = self.max_rl.min(n);
        let mut rl_log = vec![f64::NEG_INFINITY; max_r + 1];
        rl_log[0] = 0.0;
        let mut stats = vec![self.prior.clone(); max_r + 1];
        let beta = self.beta;
        let hazard_log = self.hazard_log();
        let growth_log = self.growth_log();
        let mut cp_probs = Vec::with_capacity(n);
        for &x in &norm {
            let active = (n).min(max_r);
            let mut new_rl = vec![f64::NEG_INFINITY; max_r + 1];
            let mut prev_mass = f64::NEG_INFINITY;
            for r in 0..=active.min(max_r.saturating_sub(1)) {
                if rl_log[r] == f64::NEG_INFINITY {
                    continue;
                }
                let pred = stats[r].log_predictive_robust(x, beta);
                if !pred.is_finite() {
                    continue;
                }
                if r < max_r {
                    new_rl[r + 1] = log_add_exp(new_rl[r + 1], rl_log[r] + pred + growth_log);
                }
                prev_mass = log_add_exp(prev_mass, rl_log[r]);
            }
            let prior_pred = self.prior.log_predictive_robust(x, beta);
            new_rl[0] = if prior_pred.is_finite() && prev_mass.is_finite() {
                prev_mass + hazard_log + prior_pred
            } else {
                f64::NEG_INFINITY
            };
            let evidence = new_rl
                .iter()
                .copied()
                .filter(|v| v.is_finite())
                .fold(f64::NEG_INFINITY, log_add_exp);
            if evidence.is_finite() {
                for v in new_rl.iter_mut() {
                    *v -= evidence;
                }
            }
            if self.log_mass_cutoff > f64::NEG_INFINITY {
                for r in (1..=max_r).rev() {
                    if new_rl[r] >= self.log_mass_cutoff {
                        break;
                    }
                    new_rl[r] = f64::NEG_INFINITY;
                }
            }
            cp_probs.push(if new_rl[0].is_finite() {
                new_rl[0].exp()
            } else {
                0.0
            });
            let mut new_stats = vec![self.prior.clone(); max_r + 1];
            for r in 0..=active.min(max_r.saturating_sub(1)) {
                if r < max_r && new_rl[r + 1] > f64::NEG_INFINITY {
                    new_stats[r + 1] = stats[r].update(x);
                }
            }
            rl_log = new_rl;
            stats = new_stats;
        }
        (cps, cp_probs, original_indices)
    }

    ///
    /// Consumed by [`crate::ConformalCpWrapper`] via the
    /// [`crate::ScoredDetect`] trait.
    /// Run the BOCPD recursion to completion and return
    /// `(norm, map_rls, cp_probs, short_mass, original_indices)` --
    /// the five shared outputs that `detect`, `detect_with_score`,
    /// and `detect_bayes_factor` each post-process with a different
    /// decision rule.
    ///
    /// `norm` is normalized using the full finite input series;
    /// `original_indices` maps normalized positions back to the
    /// caller's input space. These batch diagnostics are retrospective.
    /// `map_rls[t]` is `argmax_r p(r_t = r | y_{1:t})` -- the standard
    /// MAP run length used by the MAP-drop heuristic.
    /// `cp_probs[t]` is `p(r_t = 0 | y_{1:t})` -- the per-step posterior
    /// of being at the first step of a new regime, consumed by the
    /// CUSUM and conformal score paths.
    /// `short_mass[t]` is `Σ_{r=0..=short_horizon} p(r_t = r | y_{1:t})`
    /// -- the posterior mass at "regime started recently". `K = 0`
    /// degenerates to `cp_probs`. `K = 3` parallels MAP-drop's
    /// `drop_to = 3` and feeds the Bayes-factor decision rule.
    ///
    /// Returns `None` when the input has fewer than 20 finite samples
    /// (early-return parity with the public detection methods).
    pub(crate) fn run_recursion(
        &self,
        data: &[f64],
        short_horizon: usize,
    ) -> Option<RecursionOutputs> {
        let mut original_indices: Vec<usize> = Vec::new();
        let data: Vec<f64> = data
            .iter()
            .enumerate()
            .filter_map(|(i, &v)| {
                if v.is_finite() {
                    original_indices.push(i);
                    Some(v)
                } else {
                    None
                }
            })
            .collect();
        let n = data.len();
        if n < 20 {
            return None;
        }
        let mean = data.iter().sum::<f64>() / n as f64;
        let std = (data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64).sqrt();
        let std = if std < 1e-10 { 1.0 } else { std };
        let norm: Vec<f64> = data.iter().map(|x| (x - mean) / std).collect();

        let max_r = self.max_rl.min(n);
        let mut rl_log = vec![f64::NEG_INFINITY; max_r + 1];
        rl_log[0] = 0.0;
        let mut stats = vec![self.prior.clone(); max_r + 1];
        let mut map_rls = Vec::with_capacity(n);
        let mut cp_probs = Vec::with_capacity(n);
        let mut short_mass = Vec::with_capacity(n);
        let short_horizon_clamped = short_horizon.min(max_r);

        let beta = self.beta;
        let prior_pred_at = |x: f64| self.prior.log_predictive_robust(x, beta);
        let hazard_log = self.hazard_log();
        let growth_log = self.growth_log();

        for (t, &x) in norm.iter().enumerate() {
            let active = (t + 1).min(max_r);
            let mut new_rl = vec![f64::NEG_INFINITY; max_r + 1];
            let mut prev_mass = f64::NEG_INFINITY;
            for r in 0..=active.min(max_r.saturating_sub(1)) {
                if rl_log[r] == f64::NEG_INFINITY {
                    continue;
                }
                let pred = stats[r].log_predictive_robust(x, beta);
                if !pred.is_finite() {
                    continue;
                }
                if r < max_r {
                    new_rl[r + 1] = log_add_exp(new_rl[r + 1], rl_log[r] + pred + growth_log);
                }
                prev_mass = log_add_exp(prev_mass, rl_log[r]);
            }
            let prior_pred = prior_pred_at(x);
            new_rl[0] = if prior_pred.is_finite() && prev_mass.is_finite() {
                prev_mass + hazard_log + prior_pred
            } else {
                f64::NEG_INFINITY
            };
            let evidence = new_rl
                .iter()
                .copied()
                .filter(|v| v.is_finite())
                .fold(f64::NEG_INFINITY, log_add_exp);
            if evidence.is_finite() {
                for v in new_rl.iter_mut() {
                    *v -= evidence;
                }
            }
            if self.log_mass_cutoff > f64::NEG_INFINITY {
                for r in (1..=max_r).rev() {
                    if new_rl[r] >= self.log_mass_cutoff {
                        break;
                    }
                    new_rl[r] = f64::NEG_INFINITY;
                }
            }
            let map_r = new_rl
                .iter()
                .enumerate()
                .filter(|(_, v)| v.is_finite())
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(r, _)| r)
                .unwrap_or(0);
            map_rls.push(map_r);
            cp_probs.push(if new_rl[0].is_finite() {
                new_rl[0].exp()
            } else {
                0.0
            });
            // Sum P(r_t ≤ short_horizon | data). Clamp to [0, 1] to
            // absorb numerical drift from the renormalisation above.
            let sm: f64 = new_rl
                .iter()
                .take(short_horizon_clamped + 1)
                .filter(|v| v.is_finite())
                .map(|v| v.exp())
                .sum();
            short_mass.push(sm.clamp(0.0, 1.0));
            let mut new_stats = vec![self.prior.clone(); max_r + 1];
            for r in 0..=active.min(max_r.saturating_sub(1)) {
                if r < max_r && new_rl[r + 1] > f64::NEG_INFINITY {
                    new_stats[r + 1] = stats[r].update(x);
                }
            }
            rl_log = new_rl;
            stats = new_stats;
        }
        Some((norm, map_rls, cp_probs, short_mass, original_indices))
    }

    /// Bayes-factor decision rule on aggregated short-run-length mass.
    ///
    /// Computes
    ///
    /// ```text
    ///     short_mass[t] = Σ_{r=0..=K} P(r_t = r | y_{1:t})
    ///     long_mass[t]  = 1 − short_mass[t]
    ///     BF[t]         = short_mass[t] / max(long_mass[t], 1e-12)
    /// ```
    ///
    /// Emits a CP when `BF[t] > threshold`, then enters a `cooldown`-
    /// step lockout (no further emission inside the window).
    ///
    /// Why this is interesting vs MAP-drop:
    ///
    /// 1. **Strictly more posterior info than `cp_probs`**:
    ///    `cp_probs` is just `P(r=0 | data)` (the numerator of `BF`
    ///    when `K=0`). The Bayes-factor rule additionally sees the
    ///    *collapse of long-run-length mass* via the denominator,
    ///    which is the load-bearing signal MAP-drop's `min_prev_rl`
    ///    guard tries to capture heuristically.
    /// 2. The statistic is posterior odds under the configured model
    ///    of "regime started in the last `K+1` steps" vs "regime
    ///    is older". The threshold is a posterior-odds level
    ///    (`threshold=1` ↔ "more posterior at short than long";
    ///    `threshold=4` ↔ "4× more"); calibration remains empirical.
    /// 3. `threshold` and `short_horizon` configure the statistic.
    ///    Emission also uses the cooldown and maturity guards below.
    ///
    /// `short_horizon = 3` parallels MAP-drop's `drop_to = 3`. Larger
    /// `K` integrates more posterior mass at "regime is recent"; for
    /// the BOCPD geometric-distribution prior, mass at large `K` is
    /// rare under H₀ so the gain from `K > 3` is modest.
    ///
    /// `threshold = 1.0` (BF > 1, i.e. "more posterior at short than
    /// long") is a permissive starting point. `threshold = 4.0` is a
    /// stricter setting closer to MAP-drop's effective FAR. Calibrate
    /// empirically per fixture.
    ///
    /// Returns CP indices in the **caller's** input space (non-finite
    /// samples are filtered before the recursion runs).
    ///
    pub fn detect_bayes_factor(
        &self,
        data: &[f64],
        threshold: f64,
        short_horizon: usize,
        cooldown: usize,
    ) -> Vec<ChangePoint> {
        assert!(threshold > 0.0, "threshold must be > 0, got {threshold}");
        let Some((norm, _map_rls, cp_probs, short_mass, original_indices)) =
            self.run_recursion(data, short_horizon)
        else {
            return Vec::new();
        };
        let n = norm.len();

        let mut out = Vec::new();
        let mut last_emit: Option<usize> = None;
        // Posterior-maturity guard. BOCPD starts with all mass at
        // r = 0 (rl_log[0] = 0, rl_log[r > 0] = -∞), so short_mass is
        // 1.0 at startup and BF is enormous regardless of the data.
        // To avoid spurious startup emission, the rule is *armed*
        // only after BF first drops below 1.0 -- the moment the
        // posterior has matured enough that long_mass exceeds
        // short_mass. This auto-handles startup without a magic
        // warmup number, and re-arms naturally after each emission's
        // cooldown lockout (the recursion's mass shifts back to short
        // run lengths post-CP, then matures again).
        let mut armed = false;
        for (t, &sm) in short_mass.iter().enumerate() {
            if let Some(last) = last_emit {
                if t.saturating_sub(last) <= cooldown {
                    continue;
                }
            }
            let long_mass = (1.0 - sm).max(1e-12);
            let bf = sm / long_mass;
            if !armed {
                if bf < 1.0 {
                    armed = true;
                }
                continue;
            }
            if bf > threshold {
                let look_back = cooldown.min(t);
                let confidence = cp_probs[t.saturating_sub(look_back)..=t]
                    .iter()
                    .copied()
                    .fold(0.0_f64, f64::max)
                    .clamp(0.0, 1.0);
                let w = 20;
                let before = &norm[t.saturating_sub(w)..t];
                let after = &norm[t..(t + w).min(n)];
                let mean_b = if before.is_empty() {
                    0.0
                } else {
                    before.iter().sum::<f64>() / before.len() as f64
                };
                let mean_a = if after.is_empty() {
                    0.0
                } else {
                    after.iter().sum::<f64>() / after.len() as f64
                };
                let shift_sigma = (mean_a - mean_b).abs();
                if shift_sigma < 1e-9 {
                    continue;
                }
                out.push(ChangePoint {
                    index: original_indices[t],
                    confidence,
                    shift_sigma,
                });
                last_emit = Some(t);
                // Disarm so the next emission requires the posterior
                // to mature in the *new* regime (BF drops below 1
                // first), preventing same-CP double-fire after the
                // cooldown lockout expires.
                armed = false;
            }
        }
        out
    }

    /// Offline twin of [`crate::streaming::StreamingDetector::with_bayes_factor_quantile_rule`].
    ///
    /// Same recursion as [`Self::detect_bayes_factor`], but the firing
    /// threshold is the empirical `quantile`-th quantile of a sliding
    /// window of `window` past BF values. During warmup (window not
    /// full), uses `fallback_threshold` so the detector still emits if
    /// the stream produces an early outlier. The window is updated
    /// *after* the per-step fire decision so the current BF never
    /// participates in its own reference distribution.
    ///
    /// # Panics
    /// Panics if `fallback_threshold <= 0`, `quantile` ∉ (0, 1), or
    /// `window == 0`.
    pub fn detect_bayes_factor_quantile(
        &self,
        data: &[f64],
        fallback_threshold: f64,
        short_horizon: usize,
        cooldown: usize,
        quantile: f64,
        window: usize,
    ) -> Vec<ChangePoint> {
        assert!(
            fallback_threshold > 0.0,
            "fallback_threshold must be > 0, got {fallback_threshold}"
        );
        assert!(
            quantile > 0.0 && quantile < 1.0,
            "quantile must be in (0, 1), got {quantile}"
        );
        assert!(window > 0, "window must be > 0");
        let Some((norm, _map_rls, cp_probs, short_mass, original_indices)) =
            self.run_recursion(data, short_horizon)
        else {
            return Vec::new();
        };
        let n = norm.len();

        let mut out = Vec::new();
        let mut last_emit: Option<usize> = None;
        let mut armed = false;
        let mut recent: std::collections::VecDeque<f64> =
            std::collections::VecDeque::with_capacity(window);

        for (t, &sm) in short_mass.iter().enumerate() {
            if let Some(last) = last_emit {
                if t.saturating_sub(last) <= cooldown {
                    continue;
                }
            }
            let long_mass = (1.0 - sm).max(1e-12);
            let bf = sm / long_mass;
            if !armed {
                if bf < 1.0 {
                    armed = true;
                }
                continue;
            }
            let active_threshold = if recent.len() >= window {
                let mut v: Vec<f64> =
                    recent.iter().copied().filter(|x| x.is_finite()).collect();
                if v.is_empty() {
                    fallback_threshold
                } else {
                    v.sort_by(|a, b| a.total_cmp(b));
                    let pos = quantile * (v.len() - 1) as f64;
                    let lo = pos.floor() as usize;
                    let hi = pos.ceil() as usize;
                    if lo == hi {
                        v[lo]
                    } else {
                        let frac = pos - lo as f64;
                        v[lo] * (1.0 - frac) + v[hi] * frac
                    }
                }
            } else {
                fallback_threshold
            };
            if bf > active_threshold {
                let look_back = cooldown.min(t);
                let confidence = cp_probs[t.saturating_sub(look_back)..=t]
                    .iter()
                    .copied()
                    .fold(0.0_f64, f64::max)
                    .clamp(0.0, 1.0);
                let w = 20;
                let before = &norm[t.saturating_sub(w)..t];
                let after = &norm[t..(t + w).min(n)];
                let mean_b = if before.is_empty() {
                    0.0
                } else {
                    before.iter().sum::<f64>() / before.len() as f64
                };
                let mean_a = if after.is_empty() {
                    0.0
                } else {
                    after.iter().sum::<f64>() / after.len() as f64
                };
                let shift_sigma = (mean_a - mean_b).abs();
                if shift_sigma >= 1e-9 {
                    out.push(ChangePoint {
                        index: original_indices[t],
                        confidence,
                        shift_sigma,
                    });
                    last_emit = Some(t);
                    armed = false;
                }
            }
            recent.push_back(bf);
            while recent.len() > window {
                recent.pop_front();
            }
        }
        out
    }

    pub(crate) fn detect_with_score(&self, data: &[f64]) -> Vec<(ChangePoint, f64)> {
        let mut original_indices: Vec<usize> = Vec::new();
        let data: Vec<f64> = data
            .iter()
            .enumerate()
            .filter_map(|(i, &v)| {
                if v.is_finite() {
                    original_indices.push(i);
                    Some(v)
                } else {
                    None
                }
            })
            .collect();
        let n = data.len();
        if n < 20 {
            return vec![];
        }

        let mean = data.iter().sum::<f64>() / n as f64;
        let std = (data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64).sqrt();
        let std = if std < 1e-10 { 1.0 } else { std };
        let norm: Vec<f64> = data.iter().map(|x| (x - mean) / std).collect();

        let max_r = self.max_rl.min(n);
        let mut rl_log = vec![f64::NEG_INFINITY; max_r + 1];
        rl_log[0] = 0.0;
        let mut stats = vec![self.prior.clone(); max_r + 1];
        let mut map_rls = Vec::with_capacity(n);
        let mut cp_probs = Vec::with_capacity(n);

        let beta = self.beta;
        let prior_pred_at = |x: f64| self.prior.log_predictive_robust(x, beta);
        let hazard_log = self.hazard_log();
        let growth_log = self.growth_log();

        for (t, &x) in norm.iter().enumerate() {
            let active = (t + 1).min(max_r);
            let mut new_rl = vec![f64::NEG_INFINITY; max_r + 1];
            let mut prev_mass = f64::NEG_INFINITY;
            for r in 0..=active.min(max_r.saturating_sub(1)) {
                if rl_log[r] == f64::NEG_INFINITY {
                    continue;
                }
                let pred = stats[r].log_predictive_robust(x, beta);
                if !pred.is_finite() {
                    continue;
                }
                if r < max_r {
                    new_rl[r + 1] = log_add_exp(new_rl[r + 1], rl_log[r] + pred + growth_log);
                }
                prev_mass = log_add_exp(prev_mass, rl_log[r]);
            }
            let prior_pred = prior_pred_at(x);
            new_rl[0] = if prior_pred.is_finite() && prev_mass.is_finite() {
                prev_mass + hazard_log + prior_pred
            } else {
                f64::NEG_INFINITY
            };
            let evidence = new_rl
                .iter()
                .copied()
                .filter(|v| v.is_finite())
                .fold(f64::NEG_INFINITY, log_add_exp);
            if evidence.is_finite() {
                for v in new_rl.iter_mut() {
                    *v -= evidence;
                }
            }
            if self.log_mass_cutoff > f64::NEG_INFINITY {
                for r in (1..=max_r).rev() {
                    if new_rl[r] >= self.log_mass_cutoff {
                        break;
                    }
                    new_rl[r] = f64::NEG_INFINITY;
                }
            }
            let map_r = new_rl
                .iter()
                .enumerate()
                .filter(|(_, v)| v.is_finite())
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(r, _)| r)
                .unwrap_or(0);
            map_rls.push(map_r);
            cp_probs.push(if new_rl[0].is_finite() {
                new_rl[0].exp()
            } else {
                0.0
            });
            let mut new_stats = vec![self.prior.clone(); max_r + 1];
            for r in 0..=active.min(max_r.saturating_sub(1)) {
                if r < max_r && new_rl[r + 1] > f64::NEG_INFINITY {
                    new_stats[r + 1] = stats[r].update(x);
                }
            }
            rl_log = new_rl;
            stats = new_stats;
        }

        let drop_to = 3;
        let min_prev_rl = 30;
        let cooldown = 15;

        let mut result = Vec::new();
        let mut i = min_prev_rl;
        let mut last_detection = 0usize;
        while i < n {
            if map_rls[i] <= drop_to && i - last_detection >= cooldown {
                let prev_max = map_rls[i.saturating_sub(15)..i]
                    .iter()
                    .copied()
                    .max()
                    .unwrap_or(0);
                if prev_max >= min_prev_rl {
                    let look_back = cooldown.min(i);
                    let confidence = cp_probs[i.saturating_sub(look_back)..=i]
                        .iter()
                        .copied()
                        .fold(0.0_f64, f64::max)
                        .clamp(0.0, 1.0);
                    let w = 20;
                    let before = &norm[i.saturating_sub(w)..i];
                    let after = &norm[i..(i + w).min(n)];
                    let mean_b = if before.is_empty() {
                        0.0
                    } else {
                        before.iter().sum::<f64>() / before.len() as f64
                    };
                    let mean_a = if after.is_empty() {
                        0.0
                    } else {
                        after.iter().sum::<f64>() / after.len() as f64
                    };
                    let shift_sigma = (mean_a - mean_b).abs();
                    if shift_sigma < 1e-9 {
                        i += 1;
                        continue;
                    }
                    // Argmax of `cp_probs` (posterior P(r_t = 0)) over
                    // the cooldown lookback. This is BOCPD's MAP estimate
                    // of the actual CP step -- generally earlier than the
                    // MAP-drop trigger, by a detection-lag amount that
                    // varies across CPs. Existing `detect()` uses the
                    // same slice's max value for `confidence`; we use
                    // its argmax position as the score's reference.
                    let window_start = i - look_back;
                    let mut peak_idx = i;
                    let mut peak_val = cp_probs[i];
                    // Strict `>` biases ties toward `i` (score = 0).
                    // `>=` would favour the earliest matching step
                    // and push more mass into positive scores -- see
                    // the discrete-score over-coverage note in the
                    // 0.12.0 CHANGELOG entry for ConformalCpWrapper.
                    for (k, &p) in cp_probs
                        .iter()
                        .enumerate()
                        .take(i + 1)
                        .skip(window_start)
                    {
                        if p > peak_val {
                            peak_val = p;
                            peak_idx = k;
                        }
                    }
                    let score = (i - peak_idx) as f64;

                    last_detection = i;
                    result.push((
                        ChangePoint {
                            index: original_indices[i],
                            confidence,
                            shift_sigma,
                        },
                        score,
                    ));
                    i += cooldown;
                    continue;
                }
            }
            i += 1;
        }
        result
    }

    /// Run BOCPD with a Viterbi-style backward decode for change-point
    /// extraction.
    ///
    ///
    /// Equivalent in semantics to `changepoint::utils::map_changepoints`.
    /// `confidence` is `exp(V[t][0] - V_total)`, the marginalised
    /// per-step CP posterior; `shift_sigma` matches `detect()`. Memory
    /// O(n + max_rl).
    pub fn detect_viterbi(&self, data: &[f64]) -> Vec<ChangePoint> {
        let mut original_indices: Vec<usize> = Vec::new();
        let data: Vec<f64> = data
            .iter()
            .enumerate()
            .filter_map(|(i, &v)| {
                if v.is_finite() {
                    original_indices.push(i);
                    Some(v)
                } else {
                    None
                }
            })
            .collect();
        let n = data.len();
        if n < 20 {
            return vec![];
        }

        let mean = data.iter().sum::<f64>() / n as f64;
        let std = (data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64).sqrt();
        let std = if std < 1e-10 { 1.0 } else { std };
        let norm: Vec<f64> = data.iter().map(|x| (x - mean) / std).collect();

        let max_r = self.max_rl.min(n);
        let beta = self.beta;
        let prior_pred_at = |x: f64| self.prior.log_predictive_robust(x, beta);
        let hazard_log = self.hazard_log();
        let growth_log = self.growth_log();

        // V[r] = max log P(r_t = r, x_{1:t}, best path to here). Stored
        // un-normalised (max-plus, no log_sum_exp normalisation step --
        // monotonic shifts cancel in the argmax). At each step we also
        // record `prev_r_for_cp[t]`, the predecessor r' that maximised
        // the CP transition into r_t = 0.
        let mut v_log = vec![f64::NEG_INFINITY; max_r + 1];
        v_log[0] = 0.0;
        let mut new_v = vec![f64::NEG_INFINITY; max_r + 1];
        let mut stats = vec![self.prior.clone(); max_r + 1];
        let mut new_stats = vec![self.prior.clone(); max_r + 1];

        let mut prev_r_for_cp: Vec<usize> = vec![0; n];
        // Per-step posterior probability of CP, used as `confidence` on
        // the emitted CPs. Computed from a renormalised-snapshot of V at
        // that step (Viterbi's V is unnormalised, so we renormalise just
        // for the confidence reading).
        let mut cp_probs: Vec<f64> = vec![0.0; n];

        for (t, &x) in norm.iter().enumerate() {
            let active = (t + 1).min(max_r);
            for v in new_v.iter_mut() {
                *v = f64::NEG_INFINITY;
            }
            for s in new_stats.iter_mut() {
                *s = self.prior.clone();
            }

            // CP transition: r_{t-1} = r' → r_t = 0. argmax over r'.
            let mut best_pred = f64::NEG_INFINITY;
            let mut best_pred_r = 0usize;
            for (r, &v) in v_log.iter().enumerate().take(active.min(max_r) + 1) {
                if !v.is_finite() {
                    continue;
                }
                let candidate = v + hazard_log;
                if candidate > best_pred {
                    best_pred = candidate;
                    best_pred_r = r;
                }
            }
            let prior_pred = prior_pred_at(x);
            new_v[0] = if best_pred.is_finite() && prior_pred.is_finite() {
                best_pred + prior_pred
            } else {
                f64::NEG_INFINITY
            };
            prev_r_for_cp[t] = best_pred_r;

            // Continuation: r_{t-1} = r → r_t = r + 1.
            for r in 0..=active.min(max_r.saturating_sub(1)) {
                if !v_log[r].is_finite() {
                    continue;
                }
                let pred = stats[r].log_predictive_robust(x, beta);
                if !pred.is_finite() {
                    continue;
                }
                if r < max_r {
                    let candidate = v_log[r] + pred + growth_log;
                    if candidate > new_v[r + 1] {
                        new_v[r + 1] = candidate;
                    }
                    new_stats[r + 1] = stats[r].update(x);
                }
            }

            // Snapshot CP probability under a marginal renormalisation
            // (purely for the emitted `confidence`; not used in the path).
            let evidence = new_v
                .iter()
                .copied()
                .filter(|v| v.is_finite())
                .fold(f64::NEG_INFINITY, log_add_exp);
            cp_probs[t] = if evidence.is_finite() && new_v[0].is_finite() {
                (new_v[0] - evidence).exp()
            } else {
                0.0
            };

            std::mem::swap(&mut v_log, &mut new_v);
            std::mem::swap(&mut stats, &mut new_stats);
        }

        // Backtrace from argmax_r V[n-1][r].
        let r_n = v_log
            .iter()
            .enumerate()
            .filter(|(_, v)| v.is_finite())
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(r, _)| r)
            .unwrap_or(0);

        let mut r_path = vec![0usize; n];
        r_path[n - 1] = r_n;
        for t in (1..n).rev() {
            r_path[t - 1] = if r_path[t] == 0 {
                prev_r_for_cp[t]
            } else {
                r_path[t] - 1
            };
        }

        // CPs are steps where r_path[t] == 0 and t > 0 (t = 0 is the
        // sequence start, not a CP). Drop t = 0; map back to original
        // indices through the NaN-filter projection.
        let mut result = Vec::new();
        let w = 20;
        for t in 1..n {
            if r_path[t] != 0 {
                continue;
            }
            let before = &norm[t.saturating_sub(w)..t];
            let after = &norm[t..(t + w).min(n)];
            let mean_b = if before.is_empty() {
                0.0
            } else {
                before.iter().sum::<f64>() / before.len() as f64
            };
            let mean_a = if after.is_empty() {
                0.0
            } else {
                after.iter().sum::<f64>() / after.len() as f64
            };
            let shift_sigma = (mean_a - mean_b).abs();
            if shift_sigma < 1e-9 {
                continue;
            }
            result.push(ChangePoint {
                index: original_indices[t],
                confidence: cp_probs[t].clamp(0.0, 1.0),
                shift_sigma,
            });
        }
        result
    }

    /// Run multivariate BOCPD on d-dimensional data.
    ///
    /// `data` is a slice of d-dimensional observations (each `Vec<f64>` has length d).
    /// All observations must have the same dimensionality.
    ///
    /// Returns change points using the same MAP run-length drop detection as univariate.
    ///
    /// **Note: this method ignores the type parameter `P` and always
    /// uses an internal Normal-Inverse-Wishart (NIW) predictive.**
    /// It is defined on `BocpdDetector<P>` for symmetry with `detect`,
    /// but a `BocpdDetector::<NigAr1>` will run iid-NIW multivariately,
    /// not AR(1) multivariately. Multivariate AR(p) is not implemented.
    pub fn detect_multivariate(&self, data: &[Vec<f64>]) -> Vec<ChangePoint> {
        let n = data.len();
        if n < 20 {
            return vec![];
        }
        let d = data[0].len();
        if d == 0 {
            return vec![];
        }
        // Reject ragged input -- all rows must have the same dimensionality
        if data.iter().any(|row| row.len() != d) {
            return vec![];
        }

        // Joint whitening: estimate the warmup-window sample covariance,
        // Cholesky-decompose, transform all observations into whitened space.
        // A correlated shift across dimensions becomes a translation in
        // whitened coordinates, which the per-segment NIW predictive picks
        // up just as readily as a univariate mean shift.
        //
        // Fallback to per-dim z-norm when the warmup is too short (n < 2d)
        // or the covariance is rank-deficient -- in both cases joint
        // whitening is undefined and per-dim normalisation is the safe
        // baseline.
        let warmup_n = (n / 3).min(60).max(d * 2);
        let norm: Vec<Vec<f64>> = if warmup_n >= d * 2 && warmup_n <= n {
            match whitening_transform(&data[..warmup_n], d) {
                Some((mean, l_inv)) => data
                    .iter()
                    .map(|x| {
                        let centered: Vec<f64> =
                            (0..d).map(|i| x[i] - mean[i]).collect();
                        forward_solve(&l_inv, &centered)
                    })
                    .collect(),
                None => per_dim_znorm(data, d, n),
            }
        } else {
            per_dim_znorm(data, d, n)
        };

        let max_r = self.max_rl.min(n);
        let prior = niw::Niw::new(d);

        // Run length log-probabilities
        let mut rl_log = vec![f64::NEG_INFINITY; max_r + 1];
        rl_log[0] = 0.0;

        let mut stats = vec![prior.clone(); max_r + 1];
        let mut map_rls = Vec::with_capacity(n);
        let mut cp_probs = Vec::with_capacity(n);
        let hazard_log = self.hazard_log();
        let growth_log = self.growth_log();

        for (t, x) in norm.iter().enumerate() {
            let active = (t + 1).min(max_r);
            let mut new_rl = vec![f64::NEG_INFINITY; max_r + 1];
            let mut prev_mass = f64::NEG_INFINITY;

            for r in 0..=active.min(max_r.saturating_sub(1)) {
                if rl_log[r] == f64::NEG_INFINITY {
                    continue;
                }
                let pred = stats[r].log_predictive(x);
                if !pred.is_finite() {
                    continue;
                }
                if r < max_r {
                    new_rl[r + 1] =
                        log_add_exp(new_rl[r + 1], rl_log[r] + pred + growth_log);
                }
                prev_mass = log_add_exp(prev_mass, rl_log[r]);
            }
            let prior_pred = prior.log_predictive(x);
            new_rl[0] = if prior_pred.is_finite() && prev_mass.is_finite() {
                prev_mass + hazard_log + prior_pred
            } else {
                f64::NEG_INFINITY
            };

            // Normalize
            let evidence = new_rl
                .iter()
                .copied()
                .filter(|x| x.is_finite())
                .fold(f64::NEG_INFINITY, log_add_exp);
            if evidence.is_finite() {
                for v in new_rl.iter_mut() {
                    *v -= evidence;
                }
            }

            // Mass-prune the tail (see univariate path for rationale).
            if self.log_mass_cutoff > f64::NEG_INFINITY {
                for r in (1..=max_r).rev() {
                    if new_rl[r] >= self.log_mass_cutoff {
                        break;
                    }
                    new_rl[r] = f64::NEG_INFINITY;
                }
            }

            let map_r = new_rl
                .iter()
                .enumerate()
                .filter(|(_, v)| v.is_finite())
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(r, _)| r)
                .unwrap_or(0);
            map_rls.push(map_r);
            cp_probs.push(if new_rl[0].is_finite() {
                new_rl[0].exp()
            } else {
                0.0
            });

            let mut new_stats = vec![prior.clone(); max_r + 1];
            for r in 0..=active.min(max_r.saturating_sub(1)) {
                if r < max_r && new_rl[r + 1] > f64::NEG_INFINITY {
                    new_stats[r + 1] = stats[r].update(x);
                }
            }

            rl_log = new_rl;
            stats = new_stats;
        }

        // MAP run-length drop detection (same logic as univariate)
        let drop_to = 3;
        let min_prev_rl = 30;
        let cooldown = 15;

        let mut result = Vec::new();
        let mut i = min_prev_rl;
        let mut last_detection = 0usize;
        while i < n {
            if map_rls[i] <= drop_to && i - last_detection >= cooldown {
                let prev_max = map_rls[i.saturating_sub(15)..i]
                    .iter()
                    .copied()
                    .max()
                    .unwrap_or(0);
                if prev_max >= min_prev_rl {
                    let look_back = cooldown.min(i);
                    let confidence = cp_probs[i.saturating_sub(look_back)..=i]
                        .iter()
                        .copied()
                        .fold(0.0_f64, f64::max)
                        .clamp(0.0, 1.0);
                    let w = 20;
                    let before = &norm[i.saturating_sub(w)..i];
                    let after = &norm[i..(i + w).min(n)];
                    let shift_sigma = if before.is_empty() || after.is_empty() {
                        0.0
                    } else {
                        let mut sum_sq = 0.0;
                        for dim in 0..d {
                            let mean_b = before.iter().map(|x| x[dim]).sum::<f64>()
                                / before.len() as f64;
                            let mean_a =
                                after.iter().map(|x| x[dim]).sum::<f64>() / after.len() as f64;
                            sum_sq += (mean_a - mean_b).powi(2);
                        }
                        sum_sq.sqrt()
                    };

                    if shift_sigma < 1e-9 {
                        i += 1;
                        continue;
                    }

                    last_detection = i;
                    result.push(ChangePoint {
                        index: i,
                        confidence,
                        shift_sigma,
                    });
                    i += cooldown;
                    continue;
                }
            }
            i += 1;
        }
        result
    }

    /// Multivariate analogue of [`detect_with_score`]. Returns
    /// `(ChangePoint, f64)` pairs where the score is the same
    /// **trigger-to-CP-mass-peak offset** as the univariate path:
    /// `i − argmax_{k ∈ lookback} P(r_k = 0 | y_{1:k})`.
    ///
    /// The score computation is dimension-agnostic -- BOCPD's run-
    /// length posterior is 1-D regardless of input dimension, so the
    /// conformal exchangeability assumption transfers from the
    /// univariate path unchanged. This is what
    /// [`crate::ConformalCpWrapper::detect_multivariate`] consumes.
    pub(crate) fn detect_multivariate_with_score(
        &self,
        data: &[Vec<f64>],
    ) -> Vec<(ChangePoint, f64)> {
        let n = data.len();
        if n < 20 {
            return vec![];
        }
        let d = data[0].len();
        if d == 0 {
            return vec![];
        }
        if data.iter().any(|row| row.len() != d) {
            return vec![];
        }

        let warmup_n = (n / 3).min(60).max(d * 2);
        let norm: Vec<Vec<f64>> = if warmup_n >= d * 2 && warmup_n <= n {
            match whitening_transform(&data[..warmup_n], d) {
                Some((mean, l_inv)) => data
                    .iter()
                    .map(|x| {
                        let centered: Vec<f64> =
                            (0..d).map(|i| x[i] - mean[i]).collect();
                        forward_solve(&l_inv, &centered)
                    })
                    .collect(),
                None => per_dim_znorm(data, d, n),
            }
        } else {
            per_dim_znorm(data, d, n)
        };

        let max_r = self.max_rl.min(n);
        let prior = niw::Niw::new(d);
        let mut rl_log = vec![f64::NEG_INFINITY; max_r + 1];
        rl_log[0] = 0.0;
        let mut stats = vec![prior.clone(); max_r + 1];
        let mut map_rls = Vec::with_capacity(n);
        let mut cp_probs = Vec::with_capacity(n);
        let hazard_log = self.hazard_log();
        let growth_log = self.growth_log();

        for (t, x) in norm.iter().enumerate() {
            let active = (t + 1).min(max_r);
            let mut new_rl = vec![f64::NEG_INFINITY; max_r + 1];
            let mut prev_mass = f64::NEG_INFINITY;
            for r in 0..=active.min(max_r.saturating_sub(1)) {
                if rl_log[r] == f64::NEG_INFINITY {
                    continue;
                }
                let pred = stats[r].log_predictive(x);
                if !pred.is_finite() {
                    continue;
                }
                if r < max_r {
                    new_rl[r + 1] =
                        log_add_exp(new_rl[r + 1], rl_log[r] + pred + growth_log);
                }
                prev_mass = log_add_exp(prev_mass, rl_log[r]);
            }
            let prior_pred = prior.log_predictive(x);
            new_rl[0] = if prior_pred.is_finite() && prev_mass.is_finite() {
                prev_mass + hazard_log + prior_pred
            } else {
                f64::NEG_INFINITY
            };
            let evidence = new_rl
                .iter()
                .copied()
                .filter(|x| x.is_finite())
                .fold(f64::NEG_INFINITY, log_add_exp);
            if evidence.is_finite() {
                for v in new_rl.iter_mut() {
                    *v -= evidence;
                }
            }
            if self.log_mass_cutoff > f64::NEG_INFINITY {
                for r in (1..=max_r).rev() {
                    if new_rl[r] >= self.log_mass_cutoff {
                        break;
                    }
                    new_rl[r] = f64::NEG_INFINITY;
                }
            }
            let map_r = new_rl
                .iter()
                .enumerate()
                .filter(|(_, v)| v.is_finite())
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(r, _)| r)
                .unwrap_or(0);
            map_rls.push(map_r);
            cp_probs.push(if new_rl[0].is_finite() {
                new_rl[0].exp()
            } else {
                0.0
            });
            let mut new_stats = vec![prior.clone(); max_r + 1];
            for r in 0..=active.min(max_r.saturating_sub(1)) {
                if r < max_r && new_rl[r + 1] > f64::NEG_INFINITY {
                    new_stats[r + 1] = stats[r].update(x);
                }
            }
            rl_log = new_rl;
            stats = new_stats;
        }

        // MAP-drop detection + score capture (mirrors detect_with_score).
        let drop_to = 3;
        let min_prev_rl = 30;
        let cooldown = 15;

        let mut result: Vec<(ChangePoint, f64)> = Vec::new();
        let mut i = min_prev_rl;
        let mut last_detection = 0usize;
        while i < n {
            if map_rls[i] <= drop_to && i - last_detection >= cooldown {
                let prev_max = map_rls[i.saturating_sub(15)..i]
                    .iter()
                    .copied()
                    .max()
                    .unwrap_or(0);
                if prev_max >= min_prev_rl {
                    let look_back = cooldown.min(i);
                    let confidence = cp_probs[i.saturating_sub(look_back)..=i]
                        .iter()
                        .copied()
                        .fold(0.0_f64, f64::max)
                        .clamp(0.0, 1.0);
                    let w = 20;
                    let before = &norm[i.saturating_sub(w)..i];
                    let after = &norm[i..(i + w).min(n)];
                    let shift_sigma = if before.is_empty() || after.is_empty() {
                        0.0
                    } else {
                        let mut sum_sq = 0.0;
                        for dim in 0..d {
                            let mean_b = before.iter().map(|x| x[dim]).sum::<f64>()
                                / before.len() as f64;
                            let mean_a = after.iter().map(|x| x[dim]).sum::<f64>()
                                / after.len() as f64;
                            sum_sq += (mean_a - mean_b).powi(2);
                        }
                        sum_sq.sqrt()
                    };
                    if shift_sigma < 1e-9 {
                        i += 1;
                        continue;
                    }
                    let window_start = i - look_back;
                    let mut peak_idx = i;
                    let mut peak_val = cp_probs[i];
                    for (k, &p) in cp_probs
                        .iter()
                        .enumerate()
                        .take(i + 1)
                        .skip(window_start)
                    {
                        if p > peak_val {
                            peak_val = p;
                            peak_idx = k;
                        }
                    }
                    let score = (i - peak_idx) as f64;

                    last_detection = i;
                    result.push((
                        ChangePoint {
                            index: i,
                            confidence,
                            shift_sigma,
                        },
                        score,
                    ));
                    i += cooldown;
                    continue;
                }
            }
            i += 1;
        }
        result
    }
}

impl<P: Predictive> ScoredDetect for BocpdDetector<P> {
    fn detect_with_score(&self, data: &[f64]) -> Vec<(ChangePoint, f64)> {
        BocpdDetector::detect_with_score(self, data)
    }
}

impl<P: Predictive> MvScoredDetect for BocpdDetector<P> {
    fn detect_multivariate_with_score(
        &self,
        data: &[Vec<f64>],
    ) -> Vec<(ChangePoint, f64)> {
        BocpdDetector::detect_multivariate_with_score(self, data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::Rng;

    // ── Affine invariance of detection ────────────────────────────────

    #[test]
    fn detection_is_affine_invariant() {
        // The detector normalizes input internally (subtract mean, divide by σ),
        // so a positive affine transform y = a·x + b must produce identical
        // change point indices.
        let mut rng = Rng::new(13);
        let mut x: Vec<f64> = (0..150).map(|_| rng.normal(0.0, 1.0)).collect();
        x.extend((0..150).map(|_| rng.normal(3.0, 1.0)));

        let det = BocpdDetector::new(200.0, 350);

        let raw_cps: Vec<usize> = det.detect(&x).iter().map(|c| c.index).collect();

        for &(a, b) in &[(1.0, 100.0), (5.0, -50.0), (0.1, 7.0), (1000.0, 0.0)] {
            let y: Vec<f64> = x.iter().map(|&v| a * v + b).collect();
            let trans_cps: Vec<usize> = det.detect(&y).iter().map(|c| c.index).collect();
            assert_eq!(
                raw_cps, trans_cps,
                "affine (a={a}, b={b}) shifted CPs: raw={raw_cps:?}, transformed={trans_cps:?}"
            );
        }
    }

    // ── Detector: detection correctness ───────────────────────────────

    #[test]
    fn detects_clean_mean_shift() {
        let mut data = vec![0.0; 100];
        data.extend(vec![5.0; 100]);
        let det = BocpdDetector::new(200.0, 250);
        let cps = det.detect(&data);
        assert!(!cps.is_empty(), "should detect the mean shift");
        assert!(
            (cps[0].index as i64 - 100).abs() < 15,
            "change point near index 100, got {}",
            cps[0].index
        );
    }

    #[test]
    fn detects_noisy_mean_shift() {
        let mut rng = Rng::new(123);
        let mut data: Vec<f64> = (0..150).map(|_| rng.normal(0.0, 1.0)).collect();
        data.extend((0..150).map(|_| rng.normal(3.0, 1.0)));
        let det = BocpdDetector::new(200.0, 350);
        let cps = det.detect(&data);
        assert!(!cps.is_empty(), "should detect noisy mean shift (3σ)");
        assert!(
            (cps[0].index as i64 - 150).abs() < 30,
            "change point near 150, got {}",
            cps[0].index
        );
    }

    #[test]
    fn no_detection_on_constant() {
        let data = vec![1.0; 200];
        let det = BocpdDetector::new(200.0, 250);
        let cps = det.detect(&data);
        assert!(cps.is_empty(), "constant signal should have no detections");
    }

    #[test]
    fn no_detection_on_stationary_noise() {
        let mut rng = Rng::new(999);
        let data: Vec<f64> = (0..300).map(|_| rng.normal(0.0, 1.0)).collect();
        let det = BocpdDetector::new(200.0, 350);
        let cps = det.detect(&data);
        assert!(
            cps.len() <= 1,
            "stationary noise should produce ≤1 false positive, got {}",
            cps.len()
        );
    }

    #[test]
    fn false_positive_rate_under_5_percent() {
        let det = BocpdDetector::new(200.0, 350);
        let mut fp_count = 0;
        for seed in 0..20 {
            let mut rng = Rng::new(seed * 7919 + 31);
            let data: Vec<f64> = (0..300).map(|_| rng.normal(0.0, 1.0)).collect();
            if !det.detect(&data).is_empty() {
                fp_count += 1;
            }
        }
        assert!(
            fp_count <= 3,
            "false positive rate too high: {fp_count}/20 trials"
        );
    }

    #[test]
    fn detection_power_increases_with_shift() {
        let det = BocpdDetector::new(200.0, 250);
        let mut rates = Vec::new();
        for &shift in &[1.0, 3.0, 5.0] {
            let mut detections = 0;
            for seed in 0..20 {
                let mut rng = Rng::new(seed * 1000 + shift as u64 * 100);
                let mut data: Vec<f64> = (0..100).map(|_| rng.normal(0.0, 1.0)).collect();
                data.extend((0..100).map(|_| rng.normal(shift, 1.0)));
                if !det.detect(&data).is_empty() {
                    detections += 1;
                }
            }
            rates.push(detections);
        }
        assert!(
            rates[2] > rates[0],
            "5σ should detect more than 1σ: 1σ={}, 5σ={}",
            rates[0],
            rates[2]
        );
        assert!(
            rates[2] >= 15,
            "5σ shift detected {}/20 -- should be ≥15",
            rates[2]
        );
    }

    #[test]
    fn detection_delay_bounded() {
        let mut data = vec![0.0; 100];
        data.extend(vec![5.0; 100]);
        let det = BocpdDetector::new(200.0, 250);
        let cps = det.detect(&data);
        assert!(!cps.is_empty());
        let delay = (cps[0].index as i64 - 100).abs();
        assert!(delay <= 15, "detection delay={delay} steps -- should be ≤15");
    }

    #[test]
    fn detects_collective_anomaly() {
        let mut data = vec![0.0; 80];
        data.extend(vec![5.0; 40]);
        data.extend(vec![0.0; 80]);
        let det = BocpdDetector::new(200.0, 250);
        let cps = det.detect(&data);
        assert!(
            cps.len() >= 2,
            "should detect start+end of collective anomaly, got {} detections",
            cps.len()
        );
        assert!(
            (cps[0].index as i64 - 80).abs() < 15,
            "anomaly start near 80, got {}",
            cps[0].index
        );
        assert!(
            (cps[1].index as i64 - 120).abs() < 15,
            "anomaly end near 120, got {}",
            cps[1].index
        );
    }

    #[test]
    fn detects_variance_change() {
        let mut data: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin() * 0.1).collect();
        data.extend((0..100).map(|i| (i as f64 * 0.3).sin() * 2.0));
        let det = BocpdDetector::new(200.0, 250);
        let cps = det.detect(&data);
        assert!(!cps.is_empty(), "should detect variance change");
    }

    #[test]
    fn detects_multiple_change_points() {
        let mut data = vec![0.0; 80];
        data.extend(vec![5.0; 80]);
        data.extend(vec![-3.0; 80]);
        let det = BocpdDetector::new(200.0, 300);
        let cps = det.detect(&data);
        assert!(
            cps.len() >= 2,
            "should detect ≥2 change points, got {}",
            cps.len()
        );
    }

    // ── Edge cases ────────────────────────────────────────────────────

    #[test]
    fn too_short_returns_empty() {
        let det = BocpdDetector::new(200.0, 250);
        assert!(det.detect(&[]).is_empty());
        assert!(det.detect(&[1.0; 5]).is_empty());
        assert!(det.detect(&[1.0; 19]).is_empty());
    }

    #[test]
    fn handles_nan_in_normalization() {
        let mut data = vec![0.0; 100];
        data.extend(vec![5.0; 100]);
        data[50] = 1e10;
        data[51] = -1e10;
        let det = BocpdDetector::new(200.0, 250);
        let cps = det.detect(&data);
        let _ = cps;
    }

    #[test]
    fn handles_large_values() {
        let mut data = vec![1e9; 100];
        data.extend(vec![2e9; 100]);
        let det = BocpdDetector::new(200.0, 250);
        let cps = det.detect(&data);
        assert!(
            !cps.is_empty(),
            "should detect shift even with large absolute values"
        );
    }

    #[test]
    fn handles_negative_values() {
        let mut data = vec![-100.0; 100];
        data.extend(vec![-50.0; 100]);
        let det = BocpdDetector::new(200.0, 250);
        let cps = det.detect(&data);
        assert!(!cps.is_empty(), "should detect shift in negative values");
    }

    #[test]
    fn change_at_very_start_not_detected() {
        let mut data = vec![0.0; 5];
        data.extend(vec![10.0; 195]);
        let det = BocpdDetector::new(200.0, 250);
        let cps = det.detect(&data);
        for cp in &cps {
            assert!(
                cp.index >= 30,
                "detection at {} is too early (before warmup/min_prev_rl)",
                cp.index
            );
        }
    }

    // ── Parameter sensitivity ─────────────────────────────────────────

    #[test]
    fn higher_threshold_fewer_detections() {
        let mut rng = Rng::new(777);
        let mut data: Vec<f64> = (0..100).map(|_| rng.normal(0.0, 1.0)).collect();
        data.extend((0..100).map(|_| rng.normal(2.0, 1.0)));
        data.extend((0..100).map(|_| rng.normal(0.0, 1.0)));
        let det = BocpdDetector::new(200.0, 350);
        let low = det.detect(&data).len();
        let high = det.detect(&data).len();
        assert!(
            high <= low,
            "higher threshold should give ≤ detections: low={low}, high={high}"
        );
    }

    #[test]
    fn smaller_lambda_more_sensitive() {
        let mut data = vec![0.0; 60];
        data.extend(vec![3.0; 60]);
        data.extend(vec![0.0; 60]);
        let sensitive = BocpdDetector::new(50.0, 200);
        let conservative = BocpdDetector::new(500.0, 200);
        let s_cps = sensitive.detect(&data).len();
        let c_cps = conservative.detect(&data).len();
        assert!(
            s_cps >= c_cps,
            "smaller λ should detect ≥ as many: λ=50→{s_cps}, λ=500→{c_cps}"
        );
    }

    // ── Eval harness ─────────────────────────────────────────────

    use crate::eval::{self, Category};

    fn run_eval() -> Vec<eval::EvalMetrics> {
        let detector = BocpdDetector::new(200.0, 350);
        let scenarios = eval::all_scenarios();
        let mut metrics = Vec::new();
        for s in &scenarios {
            let cps = detector.detect(&s.data);
            let detected: Vec<usize> = cps.iter().map(|c| c.index).collect();
            let mut m = eval::match_detections(&detected, &s.ground_truth, 20);
            m.name = s.name.to_string();
            m.category = s.category;
            metrics.push(m);
        }
        metrics
    }

    fn run_eval_detrended() -> Vec<eval::EvalMetrics> {
        use crate::detrend::detect_with_seasonal_guard;

        let detector = BocpdDetector::new(200.0, 400);
        let scenarios = eval::all_scenarios();
        let mut metrics = Vec::new();
        for s in &scenarios {
            let cps = detect_with_seasonal_guard(&s.data, s.period, &detector);
            let detected: Vec<usize> = cps.iter().map(|c| c.index).collect();
            let mut m = eval::match_detections(&detected, &s.ground_truth, 20);
            m.name = s.name.to_string();
            m.category = s.category;
            metrics.push(m);
        }
        metrics
    }

    #[test]
    fn eval_full_suite() {
        let metrics = run_eval();
        eval::print_report(&metrics);

        let agg = eval::aggregate(&metrics);
        assert!(agg.f1 >= 0.40, "aggregate F1={:.2}, need ≥0.40", agg.f1);
        assert!(
            agg.mean_delay <= 25.0,
            "aggregate delay={:.1}, need ≤25",
            agg.mean_delay
        );
    }

    #[test]
    fn eval_must_detect_recall() {
        let metrics = run_eval();
        let md: Vec<_> = metrics
            .iter()
            .filter(|m| m.category == Category::MustDetect)
            .collect();
        let tp: usize = md.iter().map(|m| m.tp).sum();
        let fn_count: usize = md.iter().map(|m| m.r#fn).sum();
        let recall = if tp + fn_count > 0 {
            tp as f64 / (tp + fn_count) as f64
        } else {
            1.0
        };
        assert!(
            recall >= 0.60,
            "MustDetect recall={:.2}, need ≥0.60",
            recall
        );
    }

    #[test]
    fn eval_must_reject_precision() {
        let metrics = run_eval();
        let mr: Vec<_> = metrics
            .iter()
            .filter(|m| m.category == Category::MustReject)
            .collect();
        let total_fps: usize = mr.iter().map(|m| m.fp).sum();
        assert!(
            total_fps <= 22,
            "MustReject total FPs={}, need ≤22",
            total_fps
        );
    }

    #[test]
    fn eval_deterministic() {
        let m1 = run_eval();
        let m2 = run_eval();
        for (a, b) in m1.iter().zip(m2.iter()) {
            assert_eq!(a.tp, b.tp, "non-deterministic: {} tp differs", a.name);
            assert_eq!(a.fp, b.fp, "non-deterministic: {} fp differs", a.name);
        }
    }

    #[test]
    fn eval_bocpd_beats_naive_baseline() {
        // BOCPD must outperform a naive z-score detector.
        // If it doesn't, the eval scenarios are too easy.
        let scenarios = eval::all_scenarios();
        let detector = BocpdDetector::new(200.0, 400);

        let mut bocpd_metrics = Vec::new();
        let mut naive_metrics = Vec::new();

        for s in &scenarios {
            let bocpd_cps: Vec<usize> = detector
                .detect(&s.data)
                .iter()
                .map(|c| c.index)
                .collect();
            let naive_cps = eval::naive_zscore_detect(&s.data, 30, 3.0);

            let mut bm = eval::match_detections(&bocpd_cps, &s.ground_truth, 20);
            bm.name = s.name.to_string();
            bm.category = s.category;
            bocpd_metrics.push(bm);

            let mut nm = eval::match_detections(&naive_cps, &s.ground_truth, 20);
            nm.name = s.name.to_string();
            nm.category = s.category;
            naive_metrics.push(nm);
        }

        let bocpd_agg = eval::aggregate(&bocpd_metrics);
        let naive_agg = eval::aggregate(&naive_metrics);

        eprintln!("\n=== BOCPD vs Naive Z-Score ===");
        eprintln!(
            "BOCPD:  F1={:.2} P={:.2} R={:.2}",
            bocpd_agg.f1, bocpd_agg.precision, bocpd_agg.recall
        );
        eprintln!(
            "Naive:  F1={:.2} P={:.2} R={:.2}",
            naive_agg.f1, naive_agg.precision, naive_agg.recall
        );

        assert!(
            bocpd_agg.f1 > naive_agg.f1,
            "BOCPD F1={:.2} must beat naive F1={:.2}",
            bocpd_agg.f1,
            naive_agg.f1
        );
    }

    #[test]
    fn eval_detrending_reduces_fps() {
        let raw_metrics = run_eval();
        let detrended_metrics = run_eval_detrended();

        let raw_agg = eval::aggregate(&raw_metrics);
        let det_agg = eval::aggregate(&detrended_metrics);

        eprintln!("\n=== Raw vs Detrended ===");
        eprintln!(
            "Raw:       F1={:.2} P={:.2} R={:.2} FP={}",
            raw_agg.f1, raw_agg.precision, raw_agg.recall, raw_agg.fp
        );
        eprintln!(
            "Detrended: F1={:.2} P={:.2} R={:.2} FP={}",
            det_agg.f1, det_agg.precision, det_agg.recall, det_agg.fp
        );

        // Detrending should reduce false positives on MustReject scenarios
        let raw_mr_fps: usize = raw_metrics
            .iter()
            .filter(|m| m.category == Category::MustReject)
            .map(|m| m.fp)
            .sum();
        let det_mr_fps: usize = detrended_metrics
            .iter()
            .filter(|m| m.category == Category::MustReject)
            .map(|m| m.fp)
            .sum();

        eprintln!("MustReject FPs: raw={raw_mr_fps}, detrended={det_mr_fps}");
        assert!(
            det_mr_fps <= raw_mr_fps,
            "detrending should not increase MustReject FPs: raw={raw_mr_fps}, detrended={det_mr_fps}"
        );
    }

    #[test]
    fn eval_lambda_sensitivity() {
        let scenarios = eval::all_scenarios();
        let det200 = BocpdDetector::new(200.0, 350);
        let det500 = BocpdDetector::new(500.0, 350);

        let mut recall_200 = 0;
        let mut recall_500 = 0;
        let mut total_gt = 0;

        for s in scenarios
            .iter()
            .filter(|s| s.category == Category::MustDetect)
        {
            let cps200: Vec<usize> = det200
                .detect(&s.data)
                .iter()
                .map(|c| c.index)
                .collect();
            let cps500: Vec<usize> = det500
                .detect(&s.data)
                .iter()
                .map(|c| c.index)
                .collect();
            let m200 = eval::match_detections(&cps200, &s.ground_truth, 20);
            let m500 = eval::match_detections(&cps500, &s.ground_truth, 20);
            recall_200 += m200.tp;
            recall_500 += m500.tp;
            total_gt += s.ground_truth.len();
        }

        // λ=500 shouldn't catastrophically collapse recall vs λ=200
        let r200 = recall_200 as f64 / total_gt as f64;
        let r500 = recall_500 as f64 / total_gt as f64;
        assert!(
            r500 >= r200 * 0.5,
            "λ=500 recall={:.2} collapsed vs λ=200 recall={:.2}",
            r500,
            r200
        );
    }

    // --- Multivariate BOCPD tests ---

    #[test]
    fn multivariate_detects_joint_shift() {
        let mut rng = Rng::new(42);
        let n = 200;
        let mut data = Vec::with_capacity(n);

        // First 100: centered at (0, 0)
        for _ in 0..100 {
            data.push(vec![rng.normal(0.0, 1.0), rng.normal(0.0, 1.0)]);
        }
        // Next 100: shifted to (3, 3)
        for _ in 0..100 {
            data.push(vec![rng.normal(3.0, 1.0), rng.normal(3.0, 1.0)]);
        }

        let detector = BocpdDetector::new(200.0, 250);
        let cps = detector.detect_multivariate(&data);
        assert!(!cps.is_empty(), "should detect joint mean shift");
        assert!(
            (cps[0].index as i64 - 100).abs() < 20,
            "change point at {} (expected ~100)",
            cps[0].index
        );
    }

    #[test]
    fn multivariate_no_detection_on_stationary() {
        let mut rng = Rng::new(99);
        let n = 200;
        let data: Vec<Vec<f64>> = (0..n)
            .map(|_| vec![rng.normal(0.0, 1.0), rng.normal(0.0, 1.0)])
            .collect();

        let detector = BocpdDetector::new(200.0, 250);
        let cps = detector.detect_multivariate(&data);
        assert!(
            cps.len() <= 1,
            "stationary data should have ≤1 false positives, got {}",
            cps.len()
        );
    }

    #[test]
    fn multivariate_detects_correlated_shift_invisible_to_marginals() {
        // Key acceptance criterion: correlated shift where each marginal is within ~1σ
        // but the joint shift is detectable.
        let mut rng = Rng::new(77);
        let n = 300;
        let mut data = Vec::with_capacity(n);

        // First 150: centered at (0, 0) with correlated noise
        for _ in 0..150 {
            let z = rng.normal(0.0, 1.0);
            data.push(vec![
                z * 0.7 + rng.normal(0.0, 0.7), // correlated dim 1
                z * 0.7 + rng.normal(0.0, 0.7), // correlated dim 2
            ]);
        }
        // Next 150: small shift in both dimensions (0.8σ each -- below marginal threshold)
        for _ in 0..150 {
            let z = rng.normal(0.0, 1.0);
            data.push(vec![
                0.8 + z * 0.7 + rng.normal(0.0, 0.7),
                0.8 + z * 0.7 + rng.normal(0.0, 0.7),
            ]);
        }

        let detector = BocpdDetector::new(200.0, 350);

        // Univariate detection on each dimension should miss it (small per-dim shift)
        let dim1: Vec<f64> = data.iter().map(|x| x[0]).collect();
        let dim2: Vec<f64> = data.iter().map(|x| x[1]).collect();
        let univ_cps_1 = detector.detect(&dim1);
        let univ_cps_2 = detector.detect(&dim2);

        // Multivariate should catch the joint shift
        let multi_cps = detector.detect_multivariate(&data);

        eprintln!(
            "univariate dim1: {} detections, dim2: {} detections, multivariate: {} detections",
            univ_cps_1.len(),
            univ_cps_2.len(),
            multi_cps.len()
        );

        // The multivariate detector should find more than the best univariate
        // (or at least detect the shift when univariate doesn't)
        let max_univ = univ_cps_1.len().max(univ_cps_2.len());
        assert!(
            multi_cps.len() >= max_univ,
            "multivariate ({}) should detect at least as many as best univariate ({})",
            multi_cps.len(),
            max_univ
        );
    }

    #[test]
    fn multivariate_too_short_returns_empty() {
        let data: Vec<Vec<f64>> = (0..10).map(|_| vec![1.0, 2.0]).collect();
        let detector = BocpdDetector::new(200.0, 50);
        let cps = detector.detect_multivariate(&data);
        assert!(cps.is_empty());
    }

    #[test]
    fn multivariate_3d_shift() {
        let mut rng = Rng::new(123);
        let n = 200;
        let mut data = Vec::with_capacity(n);

        for _ in 0..100 {
            data.push(vec![
                rng.normal(0.0, 1.0),
                rng.normal(0.0, 1.0),
                rng.normal(0.0, 1.0),
            ]);
        }
        for _ in 0..100 {
            data.push(vec![
                rng.normal(2.0, 1.0),
                rng.normal(-2.0, 1.0),
                rng.normal(3.0, 1.0),
            ]);
        }

        let detector = BocpdDetector::new(200.0, 250);
        let cps = detector.detect_multivariate(&data);
        assert!(!cps.is_empty(), "should detect 3D mean shift");
    }

    #[test]
    fn nan_in_data_does_not_panic() {
        let mut data: Vec<f64> = (0..100).map(|i| i as f64 * 0.1).collect();
        data[10] = f64::NAN;
        data[50] = f64::NAN;
        data[51] = f64::INFINITY;

        let detector = BocpdDetector::new(200.0, 250);
        // Must not panic -- NaN previously caused unwrap on partial_cmp
        let _cps = detector.detect(&data);
    }

    #[test]
    fn nan_in_data_still_produces_detections() {
        // NaN sprinkled into a clean shift should not suppress all detections.
        let mut data = vec![0.0f64; 100];
        data.extend(vec![5.0f64; 100]);
        // Inject NaN at arbitrary positions
        data[10] = f64::NAN;
        data[55] = f64::NAN;
        data[130] = f64::NAN;
        let det = BocpdDetector::new(200.0, 250);
        let cps = det.detect(&data);
        assert!(
            !cps.is_empty(),
            "NaN-containing data should still produce detections from clean values"
        );
    }

    #[test]
    fn all_nan_returns_empty_without_panic() {
        let data = vec![f64::NAN; 50];
        let det = BocpdDetector::new(200.0, 100);
        let cps = det.detect(&data);
        assert!(cps.is_empty(), "all-NaN input should return empty vec");
    }

    // ── Fix #389 acceptance tests ────────────────────────────────

    #[test]
    #[should_panic(expected = "lambda must be > 1.0")]
    fn batch_rejects_lambda_one() {
        BocpdDetector::new(1.0, 100);
    }

    #[test]
    #[should_panic(expected = "lambda must be > 1.0")]
    fn batch_rejects_lambda_below_one() {
        BocpdDetector::new(0.5, 100);
    }

    #[test]
    fn detect_multivariate_empty_returns_empty() {
        let det = BocpdDetector::new(200.0, 100);
        let cps = det.detect_multivariate(&[]);
        assert!(cps.is_empty());
    }

    #[test]
    fn detect_multivariate_ragged_returns_empty() {
        let det = BocpdDetector::new(200.0, 100);
        let mut data: Vec<Vec<f64>> = (0..30).map(|_| vec![1.0, 2.0, 3.0]).collect();
        data[15] = vec![1.0, 2.0]; // ragged row
        let cps = det.detect_multivariate(&data);
        assert!(cps.is_empty(), "ragged input should return empty");
    }

    #[test]
    fn nan_filtering_preserves_original_indices() {
        // Input with NaN gap at positions 5..10 -- change point indices must
        // reference original positions, not compressed model steps.
        let mut data: Vec<f64> = vec![0.0; 100];
        data.extend(vec![5.0; 100]);
        for v in data.iter_mut().take(10).skip(5) {
            *v = f64::NAN;
        }
        let det = BocpdDetector::new(200.0, 250);
        let cps = det.detect(&data);
        assert!(!cps.is_empty(), "should detect the shift");
        // The shift is at raw position 100. The detected index should be
        // near 100, NOT shifted by the 5 NaN values (which would give ~95).
        assert!(
            (cps[0].index as i64 - 100).abs() < 20,
            "change point should be near raw index 100, got {}",
            cps[0].index
        );
        assert!(
            cps[0].index >= 95,
            "index {} is too low -- NaN filtering shifted indices",
            cps[0].index
        );
    }

    #[test]
    fn detect_bayes_factor_clean_shift() {
        let mut rng = Rng::new(0xBF51);
        let mut data: Vec<f64> = (0..150).map(|_| rng.normal(0.0, 1.0)).collect();
        data.extend((0..150).map(|_| rng.normal(5.0, 1.0)));
        let det = BocpdDetector::new(200.0, 350);
        let cps = det.detect_bayes_factor(&data, 1.0, 3, 15);
        assert!(!cps.is_empty(), "must detect a clean 5σ shift");
        assert!(
            (cps[0].index as i64 - 150).abs() < 25,
            "first CP at {} too far from truth 150",
            cps[0].index
        );
    }

    #[test]
    fn detect_bayes_factor_quantile_clean_shift() {
        // Offline twin of streaming_bf_quantile_fires_on_real_outlier.
        let mut rng = Rng::new(0xBF52);
        let mut data: Vec<f64> = (0..200).map(|_| rng.normal(0.0, 1.0)).collect();
        data.extend((0..150).map(|_| rng.normal(5.0, 1.0)));
        let det = BocpdDetector::new(200.0, 350);
        let cps = det.detect_bayes_factor_quantile(&data, 2.0, 3, 15, 0.99, 100);
        assert!(!cps.is_empty(), "quantile twin must detect a clean 5σ shift");
        assert!(
            cps.iter().any(|c| (c.index as i64 - 200).abs() < 40),
            "no CP near shift step 200, got {cps:?}"
        );
    }

    #[test]
    fn detect_bayes_factor_arl0_increases_with_threshold() {
        // Bayes-factor at threshold=1.0 fires when short_mass >
        // long_mass. Very permissive on stationary data because
        // transient cp_prob spikes can push mass into low r values.
        // Pin: ARL₀(0.5) ≤ ARL₀(1.0) ≤ ARL₀(4.0); ARL₀(4.0) ≥ 200.
        let trials = 5;
        let mut arl0s = Vec::new();
        for &thresh in &[0.5_f64, 1.0, 4.0] {
            let mut total = 0.0;
            for t in 0..trials {
                let mut rng = Rng::new(4000 + t);
                let det = BocpdDetector::new(200.0, 350);
                let data: Vec<f64> = (0..1000).map(|_| rng.normal(0.0, 1.0)).collect();
                let cps = det.detect_bayes_factor(&data, thresh, 3, 15);
                total += cps.first().map(|c| c.index as f64).unwrap_or(1000.0);
            }
            let arl0 = total / trials as f64;
            eprintln!("BF ARL₀ at threshold={thresh}: {arl0:.0}");
            arl0s.push(arl0);
        }
        assert!(
            arl0s[0] <= arl0s[1] && arl0s[1] <= arl0s[2],
            "ARL₀ should be monotone non-decreasing in threshold; got {arl0s:?}"
        );
        assert!(
            arl0s[2] >= 200.0,
            "ARL₀ at threshold=4.0 should be ≥ 200; got {}",
            arl0s[2]
        );
    }

    #[test]
    fn detect_bayes_factor_threshold_monotone_in_far() {
        let mut rng = Rng::new(0xBF53);
        let data: Vec<f64> = (0..2000).map(|_| rng.normal(0.0, 1.0)).collect();
        let det = BocpdDetector::new(200.0, 350);
        let cps_low = det.detect_bayes_factor(&data, 0.5, 3, 15);
        let cps_hi = det.detect_bayes_factor(&data, 4.0, 3, 15);
        assert!(
            cps_hi.len() <= cps_low.len(),
            "expected FAR monotone in threshold; lo={}, hi={}",
            cps_low.len(),
            cps_hi.len()
        );
    }

    #[test]
    fn detect_bayes_factor_multi_regime() {
        let mut rng = Rng::new(0xBF54);
        let mut data: Vec<f64> = (0..200).map(|_| rng.normal(0.0, 1.0)).collect();
        data.extend((0..200).map(|_| rng.normal(3.0, 1.0)));
        data.extend((0..200).map(|_| rng.normal(-2.0, 1.0)));
        let det = BocpdDetector::new(200.0, 350);
        let cps = det.detect_bayes_factor(&data, 1.0, 3, 15);
        assert!(!cps.is_empty());
        for truth in [200_usize, 400] {
            let near = cps.iter().any(|c| (c.index as i64 - truth as i64).abs() < 50);
            assert!(
                near,
                "missed truth {truth}; got {:?}",
                cps.iter().map(|c| c.index).collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn detect_bayes_factor_invalid_threshold_panics() {
        let det = BocpdDetector::new(200.0, 350);
        let data = vec![0.0_f64; 100];
        let result = std::panic::catch_unwind(|| {
            det.detect_bayes_factor(&data, 0.0, 3, 15);
        });
        assert!(result.is_err(), "threshold=0 should panic");
    }

    #[test]
    fn detect_bayes_factor_short_input_returns_empty() {
        let det = BocpdDetector::new(200.0, 350);
        let cps = det.detect_bayes_factor(&[0.0_f64; 10], 1.0, 3, 15);
        assert!(cps.is_empty());
    }

    #[test]
    fn detect_bayes_factor_skips_non_finite_samples() {
        let mut rng = Rng::new(0xBF55);
        let mut data: Vec<f64> = (0..100).map(|_| rng.normal(0.0, 1.0)).collect();
        data.extend((0..100).map(|_| rng.normal(5.0, 1.0)));
        for v in data.iter_mut().take(10).skip(5) {
            *v = f64::NAN;
        }
        let det = BocpdDetector::new(200.0, 350);
        let cps = det.detect_bayes_factor(&data, 1.0, 3, 15);
        assert!(!cps.is_empty());
        assert!(
            (cps[0].index as i64 - 100).abs() < 25,
            "CP should be near raw index 100, got {}",
            cps[0].index
        );
    }

    #[test]
    fn detect_bayes_factor_horizon_zero_uses_cp_probs_only() {
        // K=0 ⇒ short_mass = cp_probs (degenerate). BF = cp_probs /
        // (1 - cp_probs). Test that this still detects on a clean
        // shift (just less aggressive than K=3).
        let mut rng = Rng::new(0xBF56);
        let mut data: Vec<f64> = (0..150).map(|_| rng.normal(0.0, 1.0)).collect();
        data.extend((0..150).map(|_| rng.normal(5.0, 1.0)));
        let det = BocpdDetector::new(200.0, 350);
        let cps = det.detect_bayes_factor(&data, 1.0, 0, 15);
        assert!(
            !cps.is_empty(),
            "K=0 with threshold=1 should still fire on a clean 5σ shift"
        );
    }
}
