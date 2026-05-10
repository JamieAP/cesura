//! Diffusion-Score-Matching BOCD (Altamirano-Briol-Knoblauch ICML 2023,
//! arXiv:2302.04759). Multivariate, mean-only "Path A_min" first cut.
//!
//! ## What this ships
//!
//! Closed-form generalised-Bayes-via-DSM posterior on the Gaussian mean
//! parameter, with constant `m = I_d` weight (no Prop 3.2 robustness).
//! Observation noise covariance is estimated from a warmup window and
//! held fixed -- observations are Cholesky-whitened so the Dm streaming
//! update of Appendix B.3 reduces to:
//!
//! ```text
//! Σ_{T+1}⁻¹ = Σ_T⁻¹ + 2ω · I
//! μ_{T+1}   = Σ_{T+1} · (Σ_T⁻¹ μ_T + 2ω · x_{T+1})
//! ```
//!
//! Run-length recursion + MAP-drop trigger mirror
//! [`BocpdDetector::detect_multivariate`]; the only difference is the
//! per-segment predictive (Dm-posterior here vs NIW posterior there).
//!

use crate::bocpd::{cholesky_lower, forward_solve, per_dim_znorm, whitening_transform};
use crate::conformal::MvScoredDetect;
use crate::log_add_exp;
use crate::ChangePoint;
use crate::DEFAULT_MASS_CUTOFF;

const TWO_PI: f64 = 2.0 * std::f64::consts::PI;

/// Default generalised-Bayes weight. Matches the paper's d=2 synthetic
/// experiment (`synthetic.ipynb` cell 9). Tunable via [`DmBocdDetector::with_omega`].
pub const DEFAULT_OMEGA: f64 = 0.1;

// ── MWeight trait + canonical impls ───────────────────────────────────

/// Run-length-conditional predictive moments at the time of an
/// `m`-function call. Lets [`MWeight`] impls condition their
/// behaviour on the per-segment posterior state -- e.g.
/// [`AdaptiveImqM`]'s ST-RCGP centering uses `mu`/`sigma_inv` to
/// derive bandwidth and centering at each step.
pub struct PredictiveCtx<'a> {
    /// Posterior mean of θ for this run-length.
    pub mu: &'a [f64],
    /// Posterior precision Σ⁻¹ for this run-length (`d × d`,
    /// row-major).
    pub sigma_inv: &'a [Vec<f64>],
}

/// Pluggable `m`-function for the Dm-posterior streaming update
/// (Altamirano et al. 2023, Prop 3.1 / 3.2).
///
/// The Dm-update with a generic constant-in-θ `m(x)` is:
/// ```text
/// A(x) = m(x) m(x)^T                              # d×d
/// b(x) = ∇·(m m^T)(x)                             # d, divergence of mm^T
/// Σ⁻¹_new = Σ⁻¹ + 2ω·A(x)
/// μ_new   = Σ_new · (Σ⁻¹ μ + 2ω·(A(x) x − 0.5·b(x)))
/// ```
/// [`IdentityM`] (`m = I_d`) reduces this to the identity-weight form
/// bit-for-bit (regression-pinned). [`ImqM`] uses the inverse
/// multi-quadric kernel `m(x) = (c² + ‖x‖²)^(-1/2)·I_d`, which has
/// bounded influence and is therefore Prop 3.2 globally bias-robust.
///
/// Implementations must be deterministic and side-effect-free; the
/// detector calls `m` and `divergence_mmt` repeatedly per step per
/// active run-length. Heap-allocating per call is acceptable for
/// small `d` (≤ ~8); flagged for optimisation if `d` grows.
pub trait MWeight: Send + Sync {
    /// `m(x)`: d×d matrix. Row-major. Context-free fallback used
    /// when no [`PredictiveCtx`] is available (e.g. unit tests).
    fn m(&self, x: &[f64]) -> Vec<Vec<f64>>;
    /// `∇·(m m^T)(x)`: divergence of the d×d field `m m^T`.
    /// Component i = `Σ_j ∂_j (m m^T)_{ij}`. Length d. Context-free.
    fn divergence_mmt(&self, x: &[f64]) -> Vec<f64>;
    /// Context-aware `m`. Default: forward to [`Self::m`].
    /// Override to make the kernel adaptive to the run-length-
    /// conditional predictive moments (ST-RCGP / [`AdaptiveImqM`]).
    fn m_ctx(&self, x: &[f64], _ctx: &PredictiveCtx<'_>) -> Vec<Vec<f64>> {
        self.m(x)
    }
    /// Context-aware divergence. Default: forward to
    /// [`Self::divergence_mmt`]. Override alongside [`Self::m_ctx`].
    fn divergence_mmt_ctx(&self, x: &[f64], _ctx: &PredictiveCtx<'_>) -> Vec<f64> {
        self.divergence_mmt(x)
    }
    /// Marker: when true, the detector skips the generic matmul path
    /// and uses the bit-identical `m = I_d` fast path. Default
    /// `false`. [`IdentityM`] overrides to `true`.
    ///
    /// `#[inline]` ensures the trivial-bool body is visible to LLVM
    /// in the caller's CGU pre-LTO, so the branch on `is_identity()`
    /// can fold to a constant after monomorphisation. Without it,
    /// the generic body sits in a separate generics CGU and is only
    /// inlined by post-link LTO -- which many cesura consumers
    /// don't enable. (See rust-lang/rust#102539.)
    #[inline]
    fn is_identity(&self) -> bool {
        false
    }
}

/// `m(x) = I_d`, the default identity weight. Activates the
/// fast path in [`DmStats::update`]; regression tests pin its
/// numerical equivalence to the identity-weight update.
pub struct IdentityM;

impl MWeight for IdentityM {
    fn m(&self, x: &[f64]) -> Vec<Vec<f64>> {
        identity_matrix(x.len())
    }
    fn divergence_mmt(&self, x: &[f64]) -> Vec<f64> {
        vec![0.0; x.len()]
    }
    /// `#[inline]` mirrors the trait-method default. Inline is not
    /// transitive: both the trait method and the impl override need
    /// the attribute for the call chain `DmStats::update →
    /// m_weight.is_identity()` to fold cleanly under monomorphisation.
    #[inline]
    fn is_identity(&self) -> bool {
        true
    }
}

/// Inverse multi-quadric `m`-function:
/// `m(x) = (c² + ‖x‖²)^(-1/2) · I_d`.
///
/// `m m^T = (c² + ‖x‖²)^(-1) · I_d`, so each diagonal entry of the
/// matrix field is `w(x) = (c² + ‖x‖²)^(-1)` and off-diagonals are
/// zero. The divergence is then
/// `(∇·(m m^T))_i = ∂_i w(x) = -2·x_i / (c² + ‖x‖²)²`.
///
/// Bounded influence (`‖m(x) x‖ → 1` as `‖x‖ → ∞`) makes this a
/// Prop 3.2 globally bias-robust choice. `c` controls bandwidth:
/// smaller `c` ⇒ more aggressive downweighting of large `‖x‖`.
///
/// # Known failure mode (centering / bandwidth selection)
///
///
///
pub struct ImqM {
    /// Bandwidth. Smaller ⇒ more aggressive downweighting.
    pub c: f64,
}

impl ImqM {
    /// Construct an IMQ m-function with bandwidth `c > 0`.
    ///
    /// # Panics
    /// - `c <= 0.0` or `c` non-finite.
    pub fn new(c: f64) -> Self {
        assert!(c > 0.0 && c.is_finite(), "ImqM bandwidth c must be > 0, got {c}");
        Self { c }
    }
}

impl MWeight for ImqM {
    fn m(&self, x: &[f64]) -> Vec<Vec<f64>> {
        let d = x.len();
        let r2 = self.c * self.c + x.iter().map(|v| v * v).sum::<f64>();
        let w = r2.sqrt().recip();
        let mut out = vec![vec![0.0; d]; d];
        for (i, row) in out.iter_mut().enumerate() {
            row[i] = w;
        }
        out
    }
    fn divergence_mmt(&self, x: &[f64]) -> Vec<f64> {
        let r2 = self.c * self.c + x.iter().map(|v| v * v).sum::<f64>();
        let denom = r2 * r2;
        x.iter().map(|xi| -2.0 * xi / denom).collect()
    }
}

/// Adaptive IMQ kernel implementing the ST-RCGP centering /
/// bandwidth recipe (Laplante-Altamirano-Duncan-Knoblauch-Briol,
/// ICML 2025, [arXiv:2502.02450](https://arxiv.org/abs/2502.02450)
/// §3) on the BOCD posterior:
///
/// ```text
/// γ_t = μ_r          (run-length-conditional posterior mean of θ)
/// c_t = α · √mean(diag(Σ_r) + 1)   (predictive std proxy in d dims)
/// m(x) = (c_t² + ‖x − γ_t‖²)^(-1/2) · I_d
/// ```
///
/// `α` is a user-supplied multiplier on the predictive std; defaults
/// to `1.0` (matches ST-RCGP's variance-scaled choice). `Σ_r` is
/// recovered from `Σ_r⁻¹` via diagonal inversion as a cheap
/// approximation -- the full `invert_pd` would be O(d³) per call
/// per active run-length, and BOCD only needs scale.
///
/// Closed forms used in `divergence_mmt_ctx`:
/// `(m m^T)(x) = (c_t² + ‖x − γ_t‖²)^(-1) · I_d` ⇒
/// `(∇·(m m^T))_i = -2·(x_i − γ_{t,i}) / (c_t² + ‖x − γ_t‖²)²`.
///
/// Context-free `m`/`divergence_mmt` (called when no
/// [`PredictiveCtx`] is available, e.g. unit tests) fall back to
/// `c = α`, `γ = 0` -- equivalent to [`ImqM::new(α)`].
pub struct AdaptiveImqM {
    /// Multiplier on the predictive std for `c_t`. Default 1.0.
    pub alpha: f64,
}

impl AdaptiveImqM {
    /// Construct with bandwidth multiplier `alpha > 0`.
    ///
    /// # Panics
    /// - `alpha <= 0.0` or non-finite.
    pub fn new(alpha: f64) -> Self {
        assert!(
            alpha > 0.0 && alpha.is_finite(),
            "AdaptiveImqM alpha must be > 0, got {alpha}"
        );
        Self { alpha }
    }

    /// Derive `(c_t, γ_t)` from the predictive context. Pure helper
    /// so unit tests can pin the closed-form derivation
    /// independently of the detector loop.
    pub fn params(&self, ctx: &PredictiveCtx<'_>) -> (f64, Vec<f64>) {
        let d = ctx.mu.len();
        // Σ_ii ≈ 1 / (Σ⁻¹)_ii (exact for diagonal Σ⁻¹). The mean-only
        // path keeps Σ⁻¹ near-diagonal (updates only add diagonal +
        // outer-product perturbations), so the approx is tight in
        // practice and avoids the O(d³) full invert.
        let mut sum_pred_var = 0.0;
        for i in 0..d {
            let inv_diag = ctx.sigma_inv[i][i];
            let prior_var = if inv_diag > 0.0 { 1.0 / inv_diag } else { 1.0 };
            sum_pred_var += prior_var + 1.0; // + I_d (whitened obs noise)
        }
        let avg_pred_var = if d == 0 { 1.0 } else { sum_pred_var / d as f64 };
        let c_t = (self.alpha * avg_pred_var.sqrt()).max(1e-6);
        (c_t, ctx.mu.to_vec())
    }
}

impl MWeight for AdaptiveImqM {
    fn m(&self, x: &[f64]) -> Vec<Vec<f64>> {
        // Context-free fallback: γ=0, c=α.
        let r2 = self.alpha * self.alpha + x.iter().map(|v| v * v).sum::<f64>();
        let w = r2.sqrt().recip();
        let mut out = vec![vec![0.0; x.len()]; x.len()];
        for (i, row) in out.iter_mut().enumerate() {
            row[i] = w;
        }
        out
    }
    fn divergence_mmt(&self, x: &[f64]) -> Vec<f64> {
        let r2 = self.alpha * self.alpha + x.iter().map(|v| v * v).sum::<f64>();
        let denom = r2 * r2;
        x.iter().map(|xi| -2.0 * xi / denom).collect()
    }
    fn m_ctx(&self, x: &[f64], ctx: &PredictiveCtx<'_>) -> Vec<Vec<f64>> {
        let (c_t, gamma) = self.params(ctx);
        let r2 = c_t * c_t
            + x.iter()
                .zip(gamma.iter())
                .map(|(xi, gi)| (xi - gi).powi(2))
                .sum::<f64>();
        let w = r2.sqrt().recip();
        let mut out = vec![vec![0.0; x.len()]; x.len()];
        for (i, row) in out.iter_mut().enumerate() {
            row[i] = w;
        }
        out
    }
    fn divergence_mmt_ctx(&self, x: &[f64], ctx: &PredictiveCtx<'_>) -> Vec<f64> {
        let (c_t, gamma) = self.params(ctx);
        let r2 = c_t * c_t
            + x.iter()
                .zip(gamma.iter())
                .map(|(xi, gi)| (xi - gi).powi(2))
                .sum::<f64>();
        let denom = r2 * r2;
        x.iter()
            .zip(gamma.iter())
            .map(|(xi, gi)| -2.0 * (xi - gi) / denom)
            .collect()
    }
}

/// Multivariate Dm-BOCD detector (mean-only, configurable `m`).
///
/// `M` is the m-function type. Defaults to [`IdentityM`] -- the
/// regression-safe Path A_min. Use [`Self::with_m_weight`] to swap
/// in a robust kernel like [`ImqM`]; this rebuilds the detector
/// with the new generic parameter (returns `DmBocdDetector<M2>`).
/// Generic dispatch lets the compiler inline the m-function call
/// into the per-run-length update loop.
///
/// # Default-type-param ergonomics
///
/// Construction via [`Self::new`] needs no type annotations:
///
///
/// This works because `new` lives on the inherent impl
/// `impl DmBocdDetector<IdentityM>`, pinning `M = IdentityM` at
/// path resolution. It does **not** rely on RFC 213 type-default
/// fallback (which is unimplemented; see rust-lang/rust#27336).
/// Pinned by this doc-test so a future generic-`new` refactor
/// can't silently break the no-annotation construction.
pub struct DmBocdDetector<M: MWeight = IdentityM> {
    d: usize,
    hazard_log: f64,
    growth_log: f64,
    max_rl: usize,
    log_mass_cutoff: f64,
    omega: f64,
    /// Squared-exponential prior mean over θ. Defaults to zero (matches
    /// whitened observations centred at the warmup mean).
    prior_mu: Vec<f64>,
    /// Squared-exponential prior precision (Σ⁻¹). Defaults to I_d.
    prior_sigma_inv: Vec<Vec<f64>>,
    /// MAP-drop trigger constants. Defaults `(3, 30, 15)` match
    /// [`crate::BocpdDetector::detect_multivariate`]. Override via
    /// [`DmBocdDetector::with_map_drop_trigger`] for
    /// trigger-sensitivity evaluation.
    drop_to: usize,
    min_prev_rl: usize,
    cooldown: usize,
    m_weight: M,
}

impl DmBocdDetector<IdentityM> {
    /// Construct a detector for `d`-dimensional observations with
    /// expected run length `lambda` and posterior tracked up to
    /// `max_run_length` steps back.
    ///
    /// Defaults: `omega = DEFAULT_OMEGA`, prior `θ ~ N(0, I_d)`, mass
    /// cutoff `DEFAULT_MASS_CUTOFF` (matches `BocpdDetector`).
    ///
    /// # Panics
    /// - `d == 0`
    /// - `lambda <= 1.0`
    pub fn new(d: usize, lambda: f64, max_run_length: usize) -> Self {
        assert!(d > 0, "d must be > 0");
        assert!(lambda > 1.0, "lambda must be > 1.0, got {lambda}");
        let h = 1.0 / lambda;
        let prior_sigma_inv = identity_matrix(d);
        Self {
            d,
            hazard_log: h.ln(),
            growth_log: (1.0 - h).ln(),
            max_rl: max_run_length,
            log_mass_cutoff: DEFAULT_MASS_CUTOFF.ln(),
            omega: DEFAULT_OMEGA,
            prior_mu: vec![0.0; d],
            prior_sigma_inv,
            drop_to: 3,
            min_prev_rl: 30,
            cooldown: 15,
            m_weight: IdentityM,
        }
    }
}

impl<M: MWeight> DmBocdDetector<M> {
    /// Override the `m`-function used in the Dm streaming update.
    /// Default is [`IdentityM`] (bit-identical to the `m = I_d`
    /// numerics). Use [`ImqM`] for the Prop 3.2 bounded-influence
    /// posterior, or supply a custom [`MWeight`] for any constant-in-θ
    /// reweighting.
    ///
    /// Returns `DmBocdDetector<M2>`: the detector's generic parameter
    /// changes to the new m-function type so the call can be inlined
    /// at the update site. Chains type-swap, e.g.
    /// `DmBocdDetector::new(...).with_m_weight(ImqM::new(1.0))` is a
    /// `DmBocdDetector<ImqM>`.
    pub fn with_m_weight<M2: MWeight>(self, m: M2) -> DmBocdDetector<M2> {
        DmBocdDetector {
            d: self.d,
            hazard_log: self.hazard_log,
            growth_log: self.growth_log,
            max_rl: self.max_rl,
            log_mass_cutoff: self.log_mass_cutoff,
            omega: self.omega,
            prior_mu: self.prior_mu,
            prior_sigma_inv: self.prior_sigma_inv,
            drop_to: self.drop_to,
            min_prev_rl: self.min_prev_rl,
            cooldown: self.cooldown,
            m_weight: m,
        }
    }

    ///
    /// # Panics
    /// - `min_prev_rl == 0`
    /// - `drop_to >= min_prev_rl`
    pub fn with_map_drop_trigger(
        mut self,
        drop_to: usize,
        min_prev_rl: usize,
        cooldown: usize,
    ) -> Self {
        assert!(min_prev_rl > 0, "min_prev_rl must be > 0");
        assert!(
            drop_to < min_prev_rl,
            "drop_to ({drop_to}) must be < min_prev_rl ({min_prev_rl})",
        );
        self.drop_to = drop_to;
        self.min_prev_rl = min_prev_rl;
        self.cooldown = cooldown;
        self
    }

    pub fn with_omega(mut self, omega: f64) -> Self {
        assert!(omega > 0.0, "omega must be > 0, got {omega}");
        self.omega = omega;
        self
    }

    /// Override the prior. `mu` length must equal `d`; `sigma_inv` must
    /// be a `d × d` symmetric positive-definite matrix.
    pub fn with_prior(mut self, mu: Vec<f64>, sigma_inv: Vec<Vec<f64>>) -> Self {
        assert_eq!(mu.len(), self.d, "prior mu has wrong length");
        assert_eq!(sigma_inv.len(), self.d, "prior sigma_inv has wrong row count");
        for row in &sigma_inv {
            assert_eq!(row.len(), self.d, "prior sigma_inv has ragged rows");
        }
        self.prior_mu = mu;
        self.prior_sigma_inv = sigma_inv;
        self
    }

    /// Mass-pruning cutoff (matches [`crate::BocpdDetector::with_mass_cutoff`]).
    pub fn with_mass_cutoff(mut self, cutoff: f64) -> Self {
        self.log_mass_cutoff = if cutoff > 0.0 { cutoff.ln() } else { f64::NEG_INFINITY };
        self
    }

    /// Run Dm-BOCD on `data`, return change points via the same
    /// MAP-drop heuristic [`BocpdDetector::detect_multivariate`] uses.
    ///
    /// Returns an empty vec for `n < 20`, `d == 0`, ragged rows, or
    /// observation dimensionality mismatching the constructor's `d`.
    pub fn detect_multivariate(&self, data: &[Vec<f64>]) -> Vec<ChangePoint> {
        self.detect_multivariate_with_score(data)
            .into_iter()
            .map(|(cp, _score)| cp)
            .collect()
    }

    /// Same as [`Self::detect_multivariate`] but each emission carries
    /// a scalar score for [`crate::ConformalCpWrapper`]. Score is the
    /// trigger-step displacement from BOCPD's MAP-collapse argmax in
    /// the cooldown lookback window: `score = (i - peak_idx) as f64`.
    /// Same step-unit convention as `BocpdDetector::detect_multivariate_with_score`.
    pub fn detect_multivariate_with_score(&self, data: &[Vec<f64>]) -> Vec<(ChangePoint, f64)> {
        let n = data.len();
        if n < 20 {
            return vec![];
        }
        let d = self.d;
        if data.iter().any(|row| row.len() != d) {
            return vec![];
        }

        // Whitening preamble -- same as BocpdDetector::detect_multivariate.
        // After whitening the effective observation noise is I_d, so the
        // Dm update with constant m = I_d uses formulas above the file.
        let warmup_n = (n / 3).min(60).max(d * 2);
        let norm: Vec<Vec<f64>> = if warmup_n >= d * 2 && warmup_n <= n {
            match whitening_transform(&data[..warmup_n], d) {
                Some((mean, l)) => data
                    .iter()
                    .map(|x| {
                        let centered: Vec<f64> = (0..d).map(|i| x[i] - mean[i]).collect();
                        forward_solve(&l, &centered)
                    })
                    .collect(),
                None => per_dim_znorm(data, d, n),
            }
        } else {
            per_dim_znorm(data, d, n)
        };

        let max_r = self.max_rl.min(n);
        let prior = DmStats::from_prior(&self.prior_mu, &self.prior_sigma_inv);

        let mut rl_log = vec![f64::NEG_INFINITY; max_r + 1];
        rl_log[0] = 0.0;
        let mut stats: Vec<DmStats> = vec![prior.clone(); max_r + 1];
        let mut map_rls = Vec::with_capacity(n);
        let mut cp_probs = Vec::with_capacity(n);

        for x in norm.iter() {
            let active = (rl_log.iter().filter(|v| v.is_finite()).count() + 1).min(max_r);
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
                        log_add_exp(new_rl[r + 1], rl_log[r] + pred + self.growth_log);
                }
                prev_mass = log_add_exp(prev_mass, rl_log[r]);
            }
            let prior_pred = prior.log_predictive(x);
            new_rl[0] = if prior_pred.is_finite() && prev_mass.is_finite() {
                prev_mass + self.hazard_log + prior_pred
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
            cp_probs.push(if new_rl[0].is_finite() { new_rl[0].exp() } else { 0.0 });

            let mut new_stats: Vec<DmStats> = vec![prior.clone(); max_r + 1];
            for r in 0..=active.min(max_r.saturating_sub(1)) {
                if r < max_r && new_rl[r + 1] > f64::NEG_INFINITY {
                    new_stats[r + 1] = stats[r].update(x, self.omega, &self.m_weight);
                }
            }

            rl_log = new_rl;
            stats = new_stats;
        }

        // MAP-drop trigger -- same constants as BocpdDetector::detect_multivariate
        // by default; configurable via with_map_drop_trigger.
        let drop_to = self.drop_to;
        let min_prev_rl = self.min_prev_rl;
        let cooldown = self.cooldown;

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
                    let window_start = i - look_back;
                    let confidence = cp_probs[window_start..=i]
                        .iter()
                        .copied()
                        .fold(0.0_f64, f64::max)
                        .clamp(0.0, 1.0);
                    let mut peak_idx = i;
                    let mut peak_val = cp_probs[i];
                    for (k, &p) in cp_probs.iter().enumerate().take(i + 1).skip(window_start) {
                        if p > peak_val {
                            peak_val = p;
                            peak_idx = k;
                        }
                    }
                    let score = (i - peak_idx) as f64;
                    let w = 20;
                    let before = &norm[i.saturating_sub(w)..i];
                    let after = &norm[i..(i + w).min(n)];
                    let shift_sigma = if before.is_empty() || after.is_empty() {
                        0.0
                    } else {
                        let mut sum_sq = 0.0;
                        for dim in 0..d {
                            let mean_b =
                                before.iter().map(|x| x[dim]).sum::<f64>() / before.len() as f64;
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
                    result.push((ChangePoint { index: i, confidence, shift_sigma }, score));
                    i += cooldown;
                    continue;
                }
            }
            i += 1;
        }
        result
    }

    ///
    /// `cp_probs` and `map_rls` have length `n` (or `0` when input is
    /// rejected via the same checks `detect_multivariate` applies).
    ///
    #[cfg(any(test, feature = "test-utils"))]
    pub fn detect_multivariate_with_diagnostics(
        &self,
        data: &[Vec<f64>],
    ) -> (Vec<ChangePoint>, Vec<f64>, Vec<usize>) {
        let n = data.len();
        if n < 20 {
            return (vec![], vec![], vec![]);
        }
        let d = self.d;
        if data.iter().any(|row| row.len() != d) {
            return (vec![], vec![], vec![]);
        }

        let warmup_n = (n / 3).min(60).max(d * 2);
        let norm: Vec<Vec<f64>> = if warmup_n >= d * 2 && warmup_n <= n {
            match whitening_transform(&data[..warmup_n], d) {
                Some((mean, l)) => data
                    .iter()
                    .map(|x| {
                        let centered: Vec<f64> = (0..d).map(|i| x[i] - mean[i]).collect();
                        forward_solve(&l, &centered)
                    })
                    .collect(),
                None => per_dim_znorm(data, d, n),
            }
        } else {
            per_dim_znorm(data, d, n)
        };

        let max_r = self.max_rl.min(n);
        let prior = DmStats::from_prior(&self.prior_mu, &self.prior_sigma_inv);

        let mut rl_log = vec![f64::NEG_INFINITY; max_r + 1];
        rl_log[0] = 0.0;
        let mut stats: Vec<DmStats> = vec![prior.clone(); max_r + 1];
        let mut map_rls: Vec<usize> = Vec::with_capacity(n);
        let mut cp_probs: Vec<f64> = Vec::with_capacity(n);

        for x in norm.iter() {
            let active = (rl_log.iter().filter(|v| v.is_finite()).count() + 1).min(max_r);
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
                        log_add_exp(new_rl[r + 1], rl_log[r] + pred + self.growth_log);
                }
                prev_mass = log_add_exp(prev_mass, rl_log[r]);
            }
            let prior_pred = prior.log_predictive(x);
            new_rl[0] = if prior_pred.is_finite() && prev_mass.is_finite() {
                prev_mass + self.hazard_log + prior_pred
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
            cp_probs.push(if new_rl[0].is_finite() { new_rl[0].exp() } else { 0.0 });

            let mut new_stats: Vec<DmStats> = vec![prior.clone(); max_r + 1];
            for r in 0..=active.min(max_r.saturating_sub(1)) {
                if r < max_r && new_rl[r + 1] > f64::NEG_INFINITY {
                    new_stats[r + 1] = stats[r].update(x, self.omega, &self.m_weight);
                }
            }

            rl_log = new_rl;
            stats = new_stats;
        }

        let drop_to = self.drop_to;
        let min_prev_rl = self.min_prev_rl;
        let cooldown = self.cooldown;

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
                            let mean_b =
                                before.iter().map(|x| x[dim]).sum::<f64>() / before.len() as f64;
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
                    result.push(ChangePoint { index: i, confidence, shift_sigma });
                    i += cooldown;
                    continue;
                }
            }
            i += 1;
        }
        (result, cp_probs, map_rls)
    }
}

impl MvScoredDetect for DmBocdDetector {
    fn detect_multivariate_with_score(
        &self,
        data: &[Vec<f64>],
    ) -> Vec<(ChangePoint, f64)> {
        DmBocdDetector::detect_multivariate_with_score(self, data)
    }
}

//
// Mirrors `crate::streaming::StreamingDetector` for the BOCPD path.
// Owns one batch-equivalent forward sweep + a 20-bar lookahead buffer
// so trigger decisions at index `i` see the same `before/after` window
// the batch detector uses (`i.saturating_sub(20)..i` and
// `i..(i+20).min(n)`). Net latency: 20 bars between observation and
// CP emission for that bar.
//
// Whitening: the streaming detector buffers the first
// `WARMUP_N_DEFAULT=60` raw observations, computes (mean, L) once, and
// freezes them. For `n >= 180` and `d <= 30`, the batch warmup
// formula `(n/3).min(60).max(d*2)` also selects 60. This alone does
// not establish agreement: transforms, fallback statistics, detector
// settings and available lookahead must also match. Other dimensions
// or calibration windows can differ.
//
// Save/restore state and `with_bayes_factor_rule` are not implemented
// in this wrapper; it exposes `step()` only.

const STREAMING_WARMUP_N: usize = 60;
const STREAMING_LOOKAHEAD: usize = 20;
const STREAMING_PREV_MAX_LB: usize = 15;

/// Snapshot of `StreamingDmBocd`'s history-buffer occupancy. Returned
/// by [`StreamingDmBocd::history_buffer_report`] for tests asserting
/// the ring buffer keeps memory bounded under long streams.
#[derive(Debug, Clone)]
pub struct HistoryBufferReport {
    pub norm_history_len: usize,
    pub map_rls_len: usize,
    pub cp_probs_len: usize,
    pub history_offset: usize,
}

/// Online Dm-BOCD wrapper around the same forward recursion as
/// [`DmBocdDetector::detect_multivariate_with_score`]. Each call to
/// [`step`](Self::step) feeds one observation; CPs are emitted with
/// `STREAMING_LOOKAHEAD=20` bars of latency (the trigger uses a 20-bar
/// after-window so the streaming detector waits for that data).
///
/// Offline-agreement tests cover fixtures with matching calibration
/// and detector settings and a full 20-bar lookahead. General agreement
/// is not guaranteed: the batch warmup depends on `n` and `d`, and its
/// fallback may use the full series. Streaming leaves tail triggers
/// pending until more observations arrive; batch scans a truncated tail.
pub struct StreamingDmBocd<M: MWeight = IdentityM> {
    d: usize,
    hazard_log: f64,
    growth_log: f64,
    max_rl: usize,
    log_mass_cutoff: f64,
    omega: f64,
    prior_mu: Vec<f64>,
    prior_sigma_inv: Vec<Vec<f64>>,
    drop_to: usize,
    min_prev_rl: usize,
    cooldown: usize,
    m_weight: M,

    // ── Whitening state ─────────────────────────────────────────────
    /// Raw observations buffered until warmup completes.
    raw_warmup_buf: Vec<Vec<f64>>,
    /// (mean, lower-Cholesky-L) from `whitening_transform`. Frozen
    /// after warmup. `None` while in warmup.
    whitening: Option<(Vec<f64>, Vec<Vec<f64>>)>,
    /// Whitening fallback: if the warmup window's Cholesky fails (rank
    /// deficient), fall back to `per_dim_znorm` per-dimension stats.
    znorm_fallback: Option<(Vec<f64>, Vec<f64>)>,

    // ── Forward recursion state ─────────────────────────────────────
    rl_log: Vec<f64>,
    stats: Vec<DmStats>,
    /// All processed (whitened) observations. Used by the trigger pass
    /// for the `before/after` shift_sigma window.
    norm_history: Vec<Vec<f64>>,
    /// Per-step MAP run-length and `P(r_t = 0)` (run-length 0 = "fresh
    /// segment", i.e., the change-point posterior). Both grow with
    /// `total_steps`.
    map_rls: Vec<usize>,
    cp_probs: Vec<f64>,

    // ── Trigger state ───────────────────────────────────────────────
    /// Next index in `map_rls` / `cp_probs` to evaluate the trigger at.
    /// Trigger eval is deferred until the index has its full 20-bar
    /// after-window of normalised observations.
    next_eval_i: usize,
    /// Last index where a CP fired (initialised to 0 to match batch).
    last_detection: usize,
    /// Total observations passed to `step` (including warmup).
    total_steps: usize,
    /// Absolute index of `map_rls[0]` / `cp_probs[0]` / `norm_history[0]`.
    /// Periodically advanced by `prune_history` to bound the buffers.
    /// Without pruning these vectors grow with `total_steps`; with
    /// pruning they stay at `max(lookback, cooldown) + 2·lookahead`
    /// entries.
    history_offset: usize,
}

impl StreamingDmBocd<IdentityM> {
    /// Construct with default `IdentityM` (matches
    /// [`DmBocdDetector::new`]).
    ///
    /// # Panics
    /// - `d == 0`
    /// - `lambda <= 1.0`
    pub fn new(d: usize, lambda: f64, max_run_length: usize) -> Self {
        assert!(d > 0, "d must be > 0");
        assert!(lambda > 1.0, "lambda must be > 1.0, got {lambda}");
        let h = 1.0 / lambda;
        Self {
            d,
            hazard_log: h.ln(),
            growth_log: (1.0 - h).ln(),
            max_rl: max_run_length,
            log_mass_cutoff: DEFAULT_MASS_CUTOFF.ln(),
            omega: DEFAULT_OMEGA,
            prior_mu: vec![0.0; d],
            prior_sigma_inv: identity_matrix(d),
            drop_to: 3,
            min_prev_rl: 30,
            cooldown: 15,
            m_weight: IdentityM,
            raw_warmup_buf: Vec::with_capacity(STREAMING_WARMUP_N),
            whitening: None,
            znorm_fallback: None,
            rl_log: Vec::new(),
            stats: Vec::new(),
            norm_history: Vec::new(),
            map_rls: Vec::new(),
            cp_probs: Vec::new(),
            next_eval_i: 0,
            last_detection: 0,
            total_steps: 0,
            history_offset: 0,
        }
    }
}

impl<M: MWeight> StreamingDmBocd<M> {
    /// Swap in a custom m-function. Mirrors
    /// [`DmBocdDetector::with_m_weight`]; identical semantic.
    pub fn with_m_weight<M2: MWeight>(self, m: M2) -> StreamingDmBocd<M2> {
        StreamingDmBocd {
            d: self.d,
            hazard_log: self.hazard_log,
            growth_log: self.growth_log,
            max_rl: self.max_rl,
            log_mass_cutoff: self.log_mass_cutoff,
            omega: self.omega,
            prior_mu: self.prior_mu,
            prior_sigma_inv: self.prior_sigma_inv,
            drop_to: self.drop_to,
            min_prev_rl: self.min_prev_rl,
            cooldown: self.cooldown,
            m_weight: m,
            raw_warmup_buf: self.raw_warmup_buf,
            whitening: self.whitening,
            znorm_fallback: self.znorm_fallback,
            rl_log: self.rl_log,
            stats: self.stats,
            norm_history: self.norm_history,
            map_rls: self.map_rls,
            cp_probs: self.cp_probs,
            next_eval_i: self.next_eval_i,
            last_detection: self.last_detection,
            total_steps: self.total_steps,
            history_offset: self.history_offset,
        }
    }

    /// Override the MAP-drop trigger constants. Same semantics as
    /// [`DmBocdDetector::with_map_drop_trigger`].
    ///
    /// # Panics
    /// - `min_prev_rl == 0`
    /// - `drop_to >= min_prev_rl`
    pub fn with_map_drop_trigger(
        mut self,
        drop_to: usize,
        min_prev_rl: usize,
        cooldown: usize,
    ) -> Self {
        assert!(min_prev_rl > 0, "min_prev_rl must be > 0");
        assert!(
            drop_to < min_prev_rl,
            "drop_to ({drop_to}) must be < min_prev_rl ({min_prev_rl})",
        );
        self.drop_to = drop_to;
        self.min_prev_rl = min_prev_rl;
        self.cooldown = cooldown;
        self
    }

    /// Total observations seen across all `step` calls.
    pub fn total_steps(&self) -> usize {
        self.total_steps
    }

    /// History-buffer occupancy snapshot, exposed for tests asserting
    /// the bounded ring-buffer behavior. Callers
    /// don't need this; the lengths grow to a steady-state cap of
    /// `max(prev_max_lb, cooldown, lookahead) + lookahead` and stay
    /// there.
    pub fn history_buffer_report(&self) -> HistoryBufferReport {
        HistoryBufferReport {
            norm_history_len: self.norm_history.len(),
            map_rls_len: self.map_rls.len(),
            cp_probs_len: self.cp_probs.len(),
            history_offset: self.history_offset,
        }
    }

    /// Feed one observation. Returns any CPs that triggered during the
    /// trigger-pass scan that runs after this step. CPs are emitted at
    /// indices `i` with `i + STREAMING_LOOKAHEAD <= total_steps`.
    pub fn step(&mut self, observation: &[f64]) -> Vec<ChangePoint> {
        if observation.len() != self.d {
            return vec![];
        }
        self.total_steps += 1;

        // ── Whitening preamble ─────────────────────────────────────
        if self.whitening.is_none() && self.znorm_fallback.is_none() {
            self.raw_warmup_buf.push(observation.to_vec());
            if self.raw_warmup_buf.len() < STREAMING_WARMUP_N {
                return vec![];
            }
            // Window full -- calibrate.
            match whitening_transform(&self.raw_warmup_buf, self.d) {
                Some((mean, l)) => self.whitening = Some((mean, l)),
                None => {
                    // Mirror `per_dim_znorm` calibration on the warmup
                    // window so `process_one` can apply the same
                    // per-dim z-norm to all subsequent observations.
                    let n = self.raw_warmup_buf.len();
                    let mut mean = vec![0.0; self.d];
                    let mut std = vec![0.0; self.d];
                    for row in &self.raw_warmup_buf {
                        for (m, x) in mean.iter_mut().zip(row.iter()) {
                            *m += *x;
                        }
                    }
                    for m in mean.iter_mut() {
                        *m /= n as f64;
                    }
                    for row in &self.raw_warmup_buf {
                        for (s, (m, x)) in
                            std.iter_mut().zip(mean.iter().zip(row.iter()))
                        {
                            let dx = *x - *m;
                            *s += dx * dx;
                        }
                    }
                    for s in std.iter_mut() {
                        *s = (*s / n as f64).sqrt().max(1e-12);
                    }
                    self.znorm_fallback = Some((mean, std));
                }
            }
            self.init_recursion();
            // Process all buffered observations through the recursion.
            let buf = std::mem::take(&mut self.raw_warmup_buf);
            for x in &buf {
                self.process_one(x);
            }
            return self.scan_trigger();
        }

        // Past warmup: process this observation directly.
        self.process_one(observation);
        self.scan_trigger()
    }

    fn init_recursion(&mut self) {
        let prior = DmStats::from_prior(&self.prior_mu, &self.prior_sigma_inv);
        self.rl_log = vec![f64::NEG_INFINITY; self.max_rl + 1];
        self.rl_log[0] = 0.0;
        self.stats = vec![prior; self.max_rl + 1];
    }

    fn whiten(&self, x: &[f64]) -> Vec<f64> {
        if let Some((mean, l)) = self.whitening.as_ref() {
            let centered: Vec<f64> = (0..self.d).map(|i| x[i] - mean[i]).collect();
            forward_solve(l, &centered)
        } else if let Some((mean, std)) = self.znorm_fallback.as_ref() {
            (0..self.d).map(|i| (x[i] - mean[i]) / std[i]).collect()
        } else {
            unreachable!("process_one called before whitening calibrated");
        }
    }

    fn process_one(&mut self, raw: &[f64]) {
        let x = self.whiten(raw);
        let max_r = self.max_rl;
        let prior = DmStats::from_prior(&self.prior_mu, &self.prior_sigma_inv);
        let active = (self.rl_log.iter().filter(|v| v.is_finite()).count() + 1).min(max_r);
        let mut new_rl = vec![f64::NEG_INFINITY; max_r + 1];
        let mut prev_mass = f64::NEG_INFINITY;

        for r in 0..=active.min(max_r.saturating_sub(1)) {
            if self.rl_log[r] == f64::NEG_INFINITY {
                continue;
            }
            let pred = self.stats[r].log_predictive(&x);
            if !pred.is_finite() {
                continue;
            }
            if r < max_r {
                new_rl[r + 1] =
                    log_add_exp(new_rl[r + 1], self.rl_log[r] + pred + self.growth_log);
            }
            prev_mass = log_add_exp(prev_mass, self.rl_log[r]);
        }
        let prior_pred = prior.log_predictive(&x);
        new_rl[0] = if prior_pred.is_finite() && prev_mass.is_finite() {
            prev_mass + self.hazard_log + prior_pred
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

        let mut new_stats: Vec<DmStats> = vec![prior; max_r + 1];
        for r in 0..=active.min(max_r.saturating_sub(1)) {
            if r < max_r && new_rl[r + 1] > f64::NEG_INFINITY {
                new_stats[r + 1] = self.stats[r].update(&x, self.omega, &self.m_weight);
            }
        }

        self.cp_probs.push(if new_rl[0].is_finite() { new_rl[0].exp() } else { 0.0 });
        self.map_rls.push(map_r);
        self.norm_history.push(x);
        self.rl_log = new_rl;
        self.stats = new_stats;
    }

    /// Trigger pass over indices that now have their 20-bar after-window
    /// fully buffered. Mirrors [`DmBocdDetector::detect_multivariate`]'s
    /// post-loop trigger code exactly. Emits 0 or more CPs in index
    /// order. Each emitted CP advances `last_detection` and skips
    /// `cooldown` indices forward, matching batch's `i += cooldown`.
    fn scan_trigger(&mut self) -> Vec<ChangePoint> {
        // `n_processed` = absolute index past the last buffered bar.
        let n_processed = self.history_offset + self.map_rls.len();
        let evaluable = n_processed.saturating_sub(STREAMING_LOOKAHEAD - 1);
        // Stay >= min_prev_rl (matches batch's `let mut i = min_prev_rl`).
        if self.next_eval_i < self.min_prev_rl {
            self.next_eval_i = self.min_prev_rl;
        }

        let mut out = Vec::new();
        while self.next_eval_i < evaluable {
            let i = self.next_eval_i;
            let fired = self.try_fire(i);
            match fired {
                Some(cp) => {
                    self.last_detection = i;
                    out.push(cp);
                    self.next_eval_i = i + self.cooldown;
                }
                None => {
                    self.next_eval_i = i + 1;
                }
            }
        }
        self.prune_history();
        out
    }

    /// Drop history entries that no future trigger evaluation can read.
    /// After scan, the lowest absolute index any future `try_fire(i)` for
    /// `i >= next_eval_i` will read is `next_eval_i - max_lookback`.
    /// Drop everything below that. Keeps buffers bounded at
    /// `max_lookback + STREAMING_LOOKAHEAD` entries instead of growing
    /// with `total_steps`.
    fn prune_history(&mut self) {
        let max_lookback = STREAMING_PREV_MAX_LB
            .max(self.cooldown)
            .max(STREAMING_LOOKAHEAD);
        let lowest_needed = self.next_eval_i.saturating_sub(max_lookback);
        if lowest_needed > self.history_offset {
            let drop = lowest_needed - self.history_offset;
            // Cap drain at current buffer size to avoid out-of-range
            // panics if the trigger has not yet visited the front.
            let drop = drop.min(self.map_rls.len());
            self.map_rls.drain(0..drop);
            self.cp_probs.drain(0..drop);
            self.norm_history.drain(0..drop);
            self.history_offset += drop;
        }
    }

    /// Translate absolute bar index `i` to a buffer offset, clamped at
    /// the current buffer length. Caller must ensure
    /// `i >= self.history_offset` and `i < history_offset + len`; the
    /// trigger evaluation loop guarantees this via `next_eval_i` ≥
    /// `min_prev_rl` ≥ `history_offset` after the first fire-and-prune
    /// cycle.
    fn idx(&self, i: usize) -> usize {
        i - self.history_offset
    }

    fn try_fire(&self, i: usize) -> Option<ChangePoint> {
        if self.map_rls[self.idx(i)] > self.drop_to {
            return None;
        }
        if i.saturating_sub(self.last_detection) < self.cooldown
            && self.last_detection != 0
        {
            return None;
        }
        let prev_lo_abs = i.saturating_sub(STREAMING_PREV_MAX_LB);
        let prev_lo = prev_lo_abs.saturating_sub(self.history_offset);
        let prev_hi = self.idx(i);
        let prev_max = self.map_rls[prev_lo..prev_hi]
            .iter()
            .copied()
            .max()
            .unwrap_or(0);
        if prev_max < self.min_prev_rl {
            return None;
        }
        let look_back = self.cooldown.min(i);
        let window_start_abs = i - look_back;
        let window_start = window_start_abs.saturating_sub(self.history_offset);
        let i_buf = self.idx(i);
        let confidence = self.cp_probs[window_start..=i_buf]
            .iter()
            .copied()
            .fold(0.0_f64, f64::max)
            .clamp(0.0, 1.0);
        let mut peak_idx_abs = i;
        let mut peak_val = self.cp_probs[i_buf];
        for (k, &p) in self
            .cp_probs
            .iter()
            .enumerate()
            .take(i_buf + 1)
            .skip(window_start)
        {
            if p > peak_val {
                peak_val = p;
                peak_idx_abs = k + self.history_offset;
            }
        }
        let _score = (i - peak_idx_abs) as f64;
        let w = STREAMING_LOOKAHEAD;
        let lo_abs = i.saturating_sub(w);
        let lo = lo_abs.saturating_sub(self.history_offset);
        let hi_abs = (i + w).min(self.history_offset + self.norm_history.len());
        let hi = hi_abs.saturating_sub(self.history_offset);
        let before = &self.norm_history[lo..i_buf];
        let after = &self.norm_history[i_buf..hi];
        let shift_sigma = if before.is_empty() || after.is_empty() {
            0.0
        } else {
            let mut sum_sq = 0.0;
            for dim in 0..self.d {
                let mean_b =
                    before.iter().map(|x| x[dim]).sum::<f64>() / before.len() as f64;
                let mean_a =
                    after.iter().map(|x| x[dim]).sum::<f64>() / after.len() as f64;
                sum_sq += (mean_a - mean_b).powi(2);
            }
            sum_sq.sqrt()
        };
        if shift_sigma < 1e-9 {
            return None;
        }
        Some(ChangePoint { index: i, confidence, shift_sigma })
    }
}

// ── Per-segment Dm posterior state ────────────────────────────────────

#[derive(Clone)]
struct DmStats {
    mu: Vec<f64>,
    sigma_inv: Vec<Vec<f64>>,
}

impl DmStats {
    fn from_prior(mu: &[f64], sigma_inv: &[Vec<f64>]) -> Self {
        Self { mu: mu.to_vec(), sigma_inv: sigma_inv.to_vec() }
    }

    fn update<M: MWeight + ?Sized>(&self, x: &[f64], omega: f64, m_weight: &M) -> Self {
        let d = self.mu.len();
        let two_omega = 2.0 * omega;

        if m_weight.is_identity() {
            // Fast path: Σ⁻¹_new = Σ⁻¹ + 2ω·I, rhs = Σ⁻¹ μ + 2ω x.
            let mut sigma_inv = self.sigma_inv.clone();
            for (i, row) in sigma_inv.iter_mut().enumerate() {
                row[i] += two_omega;
            }
            let mut rhs = mat_vec(&self.sigma_inv, &self.mu);
            for i in 0..d {
                rhs[i] += two_omega * x[i];
            }
            let mu = match solve_pd(&sigma_inv, &rhs) {
                Some(v) => v,
                None => x.to_vec(),
            };
            return Self { mu, sigma_inv };
        }

        // Generic path: A = m m^T, b = ∇·(m m^T). Pass run-length-
        // conditional predictive moments so context-aware kernels
        // (ST-RCGP / AdaptiveImqM) can derive bandwidth/centering.
        let ctx = PredictiveCtx { mu: &self.mu, sigma_inv: &self.sigma_inv };
        let m_mat = m_weight.m_ctx(x, &ctx);
        let a = mat_mat_mt(&m_mat);
        let b = m_weight.divergence_mmt_ctx(x, &ctx);

        // Σ⁻¹_new = Σ⁻¹ + 2ω·A
        let mut sigma_inv = self.sigma_inv.clone();
        for i in 0..d {
            for j in 0..d {
                sigma_inv[i][j] += two_omega * a[i][j];
            }
        }
        // rhs = Σ⁻¹ μ + 2ω (A x − 0.5 b)
        let ax = mat_vec(&a, x);
        let mut rhs = mat_vec(&self.sigma_inv, &self.mu);
        for i in 0..d {
            rhs[i] += two_omega * (ax[i] - 0.5 * b[i]);
        }
        let mu = match solve_pd(&sigma_inv, &rhs) {
            Some(v) => v,
            None => x.to_vec(),
        };
        Self { mu, sigma_inv }
    }

    /// Log of the posterior predictive p(x | data_{1:T}).
    /// With θ ~ N(μ, Σ) and x|θ ~ N(θ, I_d) (whitened), x ~ N(μ, Σ + I_d).
    fn log_predictive(&self, x: &[f64]) -> f64 {
        let d = self.mu.len();
        // Σ = inv(Σ⁻¹). For small d this is fine; if d grows large this
        // is the obvious O(d³) hot loop to optimise.
        let sigma = match invert_pd(&self.sigma_inv) {
            Some(s) => s,
            None => return f64::NEG_INFINITY,
        };
        let mut covar = sigma;
        for (i, row) in covar.iter_mut().enumerate() {
            row[i] += 1.0;
        }
        let l = match cholesky_lower(&covar) {
            Some(l) => l,
            None => return f64::NEG_INFINITY,
        };
        let centered: Vec<f64> = (0..d).map(|i| x[i] - self.mu[i]).collect();
        let y = forward_solve(&l, &centered);
        let quad = y.iter().map(|v| v * v).sum::<f64>();
        let log_det = 2.0 * l.iter().enumerate().map(|(i, row)| row[i].ln()).sum::<f64>();
        -0.5 * (d as f64 * TWO_PI.ln() + log_det + quad)
    }
}

// ── Local linalg helpers (PD-only; small d) ───────────────────────────

fn identity_matrix(d: usize) -> Vec<Vec<f64>> {
    (0..d)
        .map(|i| (0..d).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
        .collect()
}

/// `M Mᵀ` for square `d×d` `M`. Returns the symmetric outer-product
/// matrix used in the Dm update's `A(x) = m(x) m(x)^T`.
fn mat_mat_mt(m: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let d = m.len();
    let mut out = vec![vec![0.0; d]; d];
    for (i, row_i) in m.iter().enumerate() {
        for (j, row_j) in m.iter().enumerate() {
            out[i][j] = row_i.iter().zip(row_j.iter()).map(|(a, b)| a * b).sum();
        }
    }
    out
}

fn mat_vec(a: &[Vec<f64>], v: &[f64]) -> Vec<f64> {
    a.iter()
        .map(|row| row.iter().zip(v.iter()).map(|(x, y)| x * y).sum())
        .collect()
}

/// Solve `A x = b` for symmetric PD `A` via Cholesky. `None` if not PD.
fn solve_pd(a: &[Vec<f64>], b: &[f64]) -> Option<Vec<f64>> {
    let l = cholesky_lower(a)?;
    let y = forward_solve(&l, b);
    // Back-solve Lᵀ x = y.
    let d = y.len();
    let mut x = vec![0.0; d];
    for i in (0..d).rev() {
        let mut s = y[i];
        for k in (i + 1)..d {
            s -= l[k][i] * x[k];
        }
        x[i] = s / l[i][i];
    }
    Some(x)
}

/// Invert symmetric PD `A` via Cholesky. `None` if not PD.
fn invert_pd(a: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let d = a.len();
    let mut inv = vec![vec![0.0; d]; d];
    for j in 0..d {
        let mut e = vec![0.0; d];
        e[j] = 1.0;
        let col = solve_pd(a, &e)?;
        for i in 0..d {
            inv[i][j] = col[i];
        }
    }
    Some(inv)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_then_solve() {
        let a = identity_matrix(3);
        let x = solve_pd(&a, &[1.0, 2.0, 3.0]).unwrap();
        assert_eq!(x, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn invert_diagonal() {
        let mut a = identity_matrix(2);
        a[0][0] = 4.0;
        a[1][1] = 9.0;
        let inv = invert_pd(&a).unwrap();
        assert!((inv[0][0] - 0.25).abs() < 1e-12);
        assert!((inv[1][1] - 1.0 / 9.0).abs() < 1e-12);
    }

    #[test]
    fn streaming_update_shrinks_posterior_variance() {
        // Σ⁻¹ starts at I; after one update with ω=0.5, Σ⁻¹ = I + I = 2·I,
        // so Σ shrinks from I to 0.5·I -- the posterior gets sharper.
        let prior = DmStats::from_prior(&[0.0, 0.0], &identity_matrix(2));
        let post = prior.update(&[1.0, 2.0], 0.5, &IdentityM);
        assert!((post.sigma_inv[0][0] - 2.0).abs() < 1e-12);
        assert!((post.sigma_inv[1][1] - 2.0).abs() < 1e-12);
        assert!((post.sigma_inv[0][1]).abs() < 1e-12);
        // μ_new = (2I)⁻¹ · (I·0 + 1·x) = 0.5·x
        assert!((post.mu[0] - 0.5).abs() < 1e-12);
        assert!((post.mu[1] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn first_step_pinned_for_default_prior() {
        // Default prior μ=0, Σ⁻¹=I. First update with ω=DEFAULT_OMEGA=0.1
        // gives Σ⁻¹_new = I + 0.2·I = 1.2·I, so each μ_new component is
        // (1/1.2) · (0 + 0.2·x) = (0.2/1.2)·x = 0.1666…·x. Pin the
        // closed-form so any future tweak to the streaming update stays
        // honest.
        let prior = DmStats::from_prior(&[0.0; 3], &identity_matrix(3));
        let post = prior.update(&[3.0, 6.0, -9.0], DEFAULT_OMEGA, &IdentityM);
        let factor = 0.2 / 1.2;
        for (got, want) in post.mu.iter().zip([3.0, 6.0, -9.0].iter()) {
            assert!((got - factor * want).abs() < 1e-12, "first-step μ drift");
        }
        for i in 0..3 {
            assert!((post.sigma_inv[i][i] - 1.2).abs() < 1e-12);
            for j in 0..3 {
                if i != j {
                    assert!(post.sigma_inv[i][j].abs() < 1e-12);
                }
            }
        }
    }

    #[test]
    fn long_run_streaming_stays_finite() {
        // Σ⁻¹ accumulates 2ω·I per step; after T=2000 steps the
        // diagonal is 1 + 0.2·2000 = 401. Σ becomes ~1/401·I, well
        // away from numerical underflow. Predictive `Σ + I` is ~I,
        // so the log_predictive at any reasonable x stays finite.
        // Pin: no NaN / -inf after 2000 i.i.d. draws.
        let mut state = DmStats::from_prior(&[0.0, 0.0, 0.0], &identity_matrix(3));
        let mut rng = crate::eval::Rng::new(0xCAFE_F00D);
        for _ in 0..2_000 {
            let x: Vec<f64> = (0..3).map(|_| rng.normal(0.0, 1.0)).collect();
            let lp = state.log_predictive(&x);
            assert!(lp.is_finite(), "log_predictive went non-finite mid-run");
            state = state.update(&x, DEFAULT_OMEGA, &IdentityM);
        }
        // Sanity: diagonal grew as expected.
        assert!(
            (state.sigma_inv[0][0] - 401.0).abs() < 1e-6,
            "Σ⁻¹ diag drifted: {}",
            state.sigma_inv[0][0]
        );
        // μ stays bounded (data has unit variance, so posterior mean
        // shouldn't blow up).
        for v in &state.mu {
            assert!(v.abs() < 1.0, "μ blew up: {v}");
        }
    }

    #[test]
    fn imq_m_closed_form_matches_paper_definition() {
        // m(x) = (c² + ‖x‖²)^(-1/2) · I_d. For c=1, x=[3,4]:
        //   ‖x‖² = 25, c² + ‖x‖² = 26, w = 1/√26.
        let imq = ImqM::new(1.0);
        let m = imq.m(&[3.0, 4.0]);
        let expected = (26.0_f64).sqrt().recip();
        assert!((m[0][0] - expected).abs() < 1e-15);
        assert!((m[1][1] - expected).abs() < 1e-15);
        assert_eq!(m[0][1], 0.0);
        assert_eq!(m[1][0], 0.0);
    }

    #[test]
    fn imq_divergence_mmt_closed_form() {
        // (∇·(m m^T))_i = -2·x_i / (c² + ‖x‖²)². For c=1, x=[3,4]:
        //   denom = 26² = 676; expected = [-6/676, -8/676].
        let imq = ImqM::new(1.0);
        let b = imq.divergence_mmt(&[3.0, 4.0]);
        assert!((b[0] - (-6.0 / 676.0)).abs() < 1e-15);
        assert!((b[1] - (-8.0 / 676.0)).abs() < 1e-15);
    }

    #[test]
    fn imq_at_origin_is_constant() {
        // x = 0 ⇒ ‖x‖² = 0, w = 1/c, divergence = 0.
        let imq = ImqM::new(2.0);
        let m = imq.m(&[0.0, 0.0, 0.0]);
        assert!((m[0][0] - 0.5).abs() < 1e-15);
        let b = imq.divergence_mmt(&[0.0, 0.0, 0.0]);
        assert_eq!(b, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn imq_bounded_influence_far_from_origin() {
        // ‖m(x)·x‖ → 1 as ‖x‖ → ∞, a bounded limit (Prop 3.2).
        let imq = ImqM::new(1.0);
        let m_far = imq.m(&[1000.0, 0.0]);
        let influence = m_far[0][0] * 1000.0;
        // (1 + 10^6)^(-1/2) · 10^3 ≈ 1.0; bounded, not blowing up.
        assert!(influence < 1.001);
        assert!(influence > 0.999);
        // Check a much larger x: influence should stay near 1, not grow.
        let m_huge = imq.m(&[1e9, 0.0]);
        let influence_huge = m_huge[0][0] * 1e9;
        assert!(influence_huge.is_finite());
        assert!(influence_huge < 1.0001);
    }

    #[test]
    #[should_panic(expected = "ImqM bandwidth c must be > 0")]
    fn imq_rejects_zero_c() {
        let _ = ImqM::new(0.0);
    }

    #[test]
    fn imq_dm_update_runs_to_completion() {
        // Smoke-test the generic update path with IMQ: posterior stays
        // PD and µ stays finite after a handful of steps.
        let mut state = DmStats::from_prior(&[0.0, 0.0], &identity_matrix(2));
        let imq = ImqM::new(1.0);
        for x in &[[0.5, -0.3], [1.2, 0.7], [-2.0, 1.5], [0.1, 0.4]] {
            state = state.update(x, DEFAULT_OMEGA, &imq);
            assert!(state.mu.iter().all(|v| v.is_finite()));
            assert!(state.sigma_inv[0][0] > 0.0 && state.sigma_inv[1][1] > 0.0);
        }
    }

    #[test]
    fn adaptive_imq_params_at_prior() {
        // Prior state: μ=0, Σ⁻¹=I. Predictive var per dim = 1/1 + 1 = 2.
        // c_t = α · √2 ≈ 1.4142… for α=1.0; γ_t = 0.
        let mu = vec![0.0; 3];
        let sigma_inv = identity_matrix(3);
        let ctx = PredictiveCtx { mu: &mu, sigma_inv: &sigma_inv };
        let aimq = AdaptiveImqM::new(1.0);
        let (c, gamma) = aimq.params(&ctx);
        assert!((c - 2.0_f64.sqrt()).abs() < 1e-12);
        assert_eq!(gamma, vec![0.0; 3]);
    }

    #[test]
    fn adaptive_imq_params_after_concentration() {
        // Σ⁻¹ = 5·I ⇒ Σ_ii = 0.2 ⇒ predictive var = 0.2 + 1 = 1.2 per dim.
        // c_t = α · √1.2; γ_t = μ.
        let mu = vec![0.5, -0.3];
        let mut sigma_inv = identity_matrix(2);
        for (i, row) in sigma_inv.iter_mut().enumerate() {
            row[i] = 5.0;
        }
        let ctx = PredictiveCtx { mu: &mu, sigma_inv: &sigma_inv };
        let aimq = AdaptiveImqM::new(2.0);
        let (c, gamma) = aimq.params(&ctx);
        assert!((c - 2.0 * 1.2_f64.sqrt()).abs() < 1e-12);
        assert_eq!(gamma, vec![0.5, -0.3]);
    }

    #[test]
    fn adaptive_imq_centers_on_predictive_mean() {
        // The load-bearing claim: when γ_t = μ, x = μ ⇒ ‖x − γ‖ = 0
        // and m has its maximum norm. Pin: m at x = μ_r is bigger
        // than m at x far from μ_r.
        let mu = vec![3.0, 4.0];
        let sigma_inv = identity_matrix(2);
        let ctx = PredictiveCtx { mu: &mu, sigma_inv: &sigma_inv };
        let aimq = AdaptiveImqM::new(1.0);
        let m_at_mean = aimq.m_ctx(&[3.0, 4.0], &ctx);
        let m_far = aimq.m_ctx(&[100.0, -100.0], &ctx);
        assert!(m_at_mean[0][0] > m_far[0][0]);
        // Divergence at γ_t (centring) is exactly zero per closed form.
        let div_at_mean = aimq.divergence_mmt_ctx(&[3.0, 4.0], &ctx);
        assert!(div_at_mean.iter().all(|v| v.abs() < 1e-15));
    }

    #[test]
    fn adaptive_imq_falls_back_to_imq_without_ctx() {
        // Context-free `m`/`divergence_mmt` should match ImqM::new(α)
        // exactly. Pinned so future adaptive impls don't accidentally
        // diverge in the no-ctx fallback path.
        let aimq = AdaptiveImqM::new(0.7);
        let imq = ImqM::new(0.7);
        let x = [1.5, -0.8, 2.1];
        assert_eq!(aimq.m(&x), imq.m(&x));
        assert_eq!(aimq.divergence_mmt(&x), imq.divergence_mmt(&x));
    }

    #[test]
    #[should_panic(expected = "AdaptiveImqM alpha must be > 0")]
    fn adaptive_imq_rejects_zero_alpha() {
        let _ = AdaptiveImqM::new(0.0);
    }

    #[test]
    fn detect_multivariate_returns_empty_for_short_input() {
        let det = DmBocdDetector::new(2, 100.0, 200);
        assert!(det.detect_multivariate(&[]).is_empty());
        let short: Vec<Vec<f64>> = (0..5).map(|_| vec![0.0, 0.0]).collect();
        assert!(det.detect_multivariate(&short).is_empty());
    }

    #[test]
    fn detect_multivariate_rejects_ragged_input() {
        let det = DmBocdDetector::new(2, 100.0, 200);
        let ragged: Vec<Vec<f64>> = (0..30)
            .map(|i| if i % 2 == 0 { vec![0.0, 0.0] } else { vec![0.0] })
            .collect();
        assert!(det.detect_multivariate(&ragged).is_empty());
    }
}
