//! AR(1) within-regime predictive with joint NIG conjugate prior.
//!
//! Implements the Bayesian linear regression posterior
//! over `(β, σ²) = ([intercept, AR(1) coefficient], σ²)` with regressor
//! `φ_t = [1, x_{t-1}]`, NIG conjugate prior, closed-form recursion.
//!
//! References:
//! - Prado, Ferreira & West (2021), *Time Series: Modeling, Computation,
//!   and Inference* 2nd ed, CRC, §1.16-1.17 (canonical AR(p) NIG
//!   conjugate analysis -- specialised to AR design from the start).
//! - Murphy 2023, *Probabilistic Machine Learning Vol. 2: Advanced
//!   Topics*, MIT Press, §15.2.2 eqs. 15.55-15.63 (modern
//!   general-regression NIG, cross-reference).
//! - Tsaknaki, Lillo & Mazzarisi 2025, *Comm. Nonlinear Sci. Numer.
//!   Simul.* (arXiv:2407.16376v2) -- direct BOCPD-AR(p) prior art with
//!   Normal-on-mean-only conjugacy + Score-Driven updates for σ², ρ.
//!   The joint NIG over `(β, σ²)` here is *stronger* conjugacy than
//!   their published form.
//!
//! ## Streaming update (one observation per call)
//!
//! Given prior `(β | σ²) ~ N(m_{n-1}, σ² Λ_{n-1}⁻¹)`, `σ² ~ IG(a_{n-1}, b_{n-1})`
//! and a fresh observation `x_n` with regressor `φ_n = [1, x_{n-1}]ᵀ`:
//!
//! ```text
//! Λ_n   = Λ_{n-1} + φ_n φ_nᵀ
//! m_n   = Λ_n⁻¹ (Λ_{n-1} m_{n-1} + φ_n x_n)
//! a_n   = a_{n-1} + 1/2
//! b_n   = b_{n-1} + ½ (m_{n-1}ᵀ Λ_{n-1} m_{n-1} + x_n²
//!                                       − m_nᵀ Λ_n m_n)
//! ```
//!
//! ## Predictive (marginal Student-t after integrating out (β, σ²))
//!
//! ```text
//! p(x_n | history, φ_n)
//!   = t_{2 a_{n-1}}(x_n;
//!         loc   = φ_nᵀ m_{n-1},
//!         scale²= (b_{n-1}/a_{n-1}) (1 + φ_nᵀ Λ_{n-1}⁻¹ φ_n))
//! ```
//!
//! ## First-observation handling (prev = None)
//!
//! At the first observation in a regime, no `x_{t-1}` is available, so
//! the AR coefficient is unobserved -- `φ` collapses to the intercept
//! `[1, 0]` and the AR-coefficient prior is integrated out trivially
//! (drops out of the regression). The predictive becomes a Student-t
//! over the intercept-only sub-model:
//!
//! ```text
//! loc   = m_0[0]
//! scale²= (b_0 / a_0) (1 + Λ_0⁻¹[0,0])
//! ```
//!
//! The corresponding `update` only stores `prev = Some(x)` and leaves
//! `(m, Λ, a, b)` unchanged: with no AR pair yet there is no regression
//! observation to absorb. This makes the iterative `update` chain
//! *exactly* equivalent to feeding the closed-form recursion the
//! `(φ_t, x_t)` pairs for `t = 1..n` (skipping the t=0 unpaired obs).
//! Pinned by `nig_ar1_iterative_matches_closed_form`.

use crate::math::student_t_lpdf;
use crate::predictive::Predictive;

/// Joint NIG sufficient statistics over `([intercept, AR(1) coef], σ²)`.
///
/// Construct via [`NigAr1::default_prior`] for the weakly-informative
/// `m=[0,0]`, `Λ=I_2`, `a=b=1` choice, or via [`NigAr1::new`] to
/// specify all four hyperparameters.
#[derive(Clone, Debug)]
pub struct NigAr1 {
    /// Posterior mean of `[intercept, AR(1) coefficient]`.
    pub(crate) m: [f64; 2],
    /// Posterior precision matrix `Λ` (NOT its inverse). 2×2 symmetric
    /// positive-definite. Stored in dense form so `update` accumulates
    /// without repeated inversions.
    pub(crate) lambda: [[f64; 2]; 2],
    /// IG shape parameter on `σ²`. After `n` paired updates from prior
    /// `a_0`: `a_n = a_0 + n/2`.
    pub(crate) a: f64,
    /// IG rate parameter on `σ²`.
    pub(crate) b: f64,
    /// Lagged observation `x_{t-1}` from the within-regime stream.
    /// `None` until the regime's first observation is consumed.
    pub(crate) prev: Option<f64>,
}

impl NigAr1 {
    /// Construct with explicit hyperparameters. `lambda` must be
    /// symmetric positive-definite; `a` and `b` must be strictly
    /// positive.
    pub fn new(m: [f64; 2], lambda: [[f64; 2]; 2], a: f64, b: f64) -> Self {
        assert!(a > 0.0, "a must be > 0, got {a}");
        assert!(b > 0.0, "b must be > 0, got {b}");
        assert!(
            (lambda[0][1] - lambda[1][0]).abs() < 1e-12,
            "lambda must be symmetric"
        );
        let det = lambda[0][0] * lambda[1][1] - lambda[0][1] * lambda[1][0];
        assert!(det > 0.0, "lambda must be positive-definite (det > 0)");
        assert!(lambda[0][0] > 0.0, "lambda[0][0] must be > 0");
        Self {
            m,
            lambda,
            a,
            b,
            prev: None,
        }
    }

    /// Weakly-informative default prior: `m = [0, 0]`, `Λ = I_2`,
    /// `a = b = 1`. Suitable when the AR(1) regime is unknown a priori.
    pub fn default_prior() -> Self {
        Self::new([0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]], 1.0, 1.0)
    }
}

/// Inverse of a 2×2 symmetric positive-definite matrix. Returns `None`
/// if `det <= 0` or non-finite (caller falls back to `-∞` log-pred).
fn invert_2x2(m: [[f64; 2]; 2]) -> Option<([[f64; 2]; 2], f64)> {
    let det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
    if det <= 0.0 || !det.is_finite() {
        return None;
    }
    let inv = [
        [m[1][1] / det, -m[0][1] / det],
        [-m[1][0] / det, m[0][0] / det],
    ];
    Some((inv, det))
}

impl Predictive for NigAr1 {
    fn update(&self, x: f64) -> Self {
        // First observation in regime: no x_{t-1} → no AR pair. Just
        // record it; (m, Λ, a, b) absorb the next obs.
        let prev = match self.prev {
            Some(p) => p,
            None => {
                return Self {
                    prev: Some(x),
                    ..self.clone()
                };
            }
        };

        // Regressor φ = [1, x_{t-1}]ᵀ.
        let phi = [1.0_f64, prev];

        // Λ_new = Λ_old + φ φᵀ.
        let phi_outer = [
            [phi[0] * phi[0], phi[0] * phi[1]],
            [phi[1] * phi[0], phi[1] * phi[1]],
        ];
        let lambda_new = [
            [
                self.lambda[0][0] + phi_outer[0][0],
                self.lambda[0][1] + phi_outer[0][1],
            ],
            [
                self.lambda[1][0] + phi_outer[1][0],
                self.lambda[1][1] + phi_outer[1][1],
            ],
        ];

        // Λ_old m_old (vector).
        let lm_old = [
            self.lambda[0][0] * self.m[0] + self.lambda[0][1] * self.m[1],
            self.lambda[1][0] * self.m[0] + self.lambda[1][1] * self.m[1],
        ];
        // Λ_old m_old + φ x.
        let rhs = [lm_old[0] + phi[0] * x, lm_old[1] + phi[1] * x];

        // m_new = Λ_new⁻¹ (Λ_old m_old + φ x).
        let (lambda_new_inv, _) = match invert_2x2(lambda_new) {
            Some(v) => v,
            None => {
                // Degenerate Λ_new: leave state unchanged (best-effort
                // graceful degradation; should never happen with a
                // well-conditioned prior + finite x).
                return self.clone();
            }
        };
        let m_new = [
            lambda_new_inv[0][0] * rhs[0] + lambda_new_inv[0][1] * rhs[1],
            lambda_new_inv[1][0] * rhs[0] + lambda_new_inv[1][1] * rhs[1],
        ];

        // b_new = b_old + ½ (m_oldᵀ Λ_old m_old + x² − m_newᵀ Λ_new m_new).
        let m_old_q = self.m[0] * lm_old[0] + self.m[1] * lm_old[1];
        let lm_new = [
            lambda_new[0][0] * m_new[0] + lambda_new[0][1] * m_new[1],
            lambda_new[1][0] * m_new[0] + lambda_new[1][1] * m_new[1],
        ];
        let m_new_q = m_new[0] * lm_new[0] + m_new[1] * lm_new[1];
        let b_new = self.b + 0.5 * (m_old_q + x * x - m_new_q);

        Self {
            m: m_new,
            lambda: lambda_new,
            a: self.a + 0.5,
            b: b_new,
            prev: Some(x),
        }
    }

    fn log_predictive(&self, x: f64) -> f64 {
        // Regressor: full AR(1) form when prev is known; intercept-only
        // when this is the regime's first observation.
        let phi: [f64; 2] = match self.prev {
            Some(p) => [1.0, p],
            None => [1.0, 0.0],
        };

        let (lambda_inv, _det) = match invert_2x2(self.lambda) {
            Some(v) => v,
            None => return f64::NEG_INFINITY,
        };

        // φᵀ Λ⁻¹ φ.
        let l_inv_phi = [
            lambda_inv[0][0] * phi[0] + lambda_inv[0][1] * phi[1],
            lambda_inv[1][0] * phi[0] + lambda_inv[1][1] * phi[1],
        ];
        let phi_l_inv_phi = phi[0] * l_inv_phi[0] + phi[1] * l_inv_phi[1];

        let scale_sq = (self.b / self.a) * (1.0 + phi_l_inv_phi);
        if scale_sq <= 0.0 || !scale_sq.is_finite() {
            return f64::NEG_INFINITY;
        }

        let loc = phi[0] * self.m[0] + phi[1] * self.m[1];
        let df = 2.0 * self.a;
        student_t_lpdf(x, df, loc, scale_sq.sqrt())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::Rng;

    /// Closed-form NIG posterior given a prior and a sequence of paired
    /// regressors `(φ_t, x_t)`. Reference: Prado-West 2021 §1.17, equivalent
    /// to Murphy 2023 §15.2.2 eqs. 15.58-15.61.
    fn closed_form(prior: &NigAr1, pairs: &[([f64; 2], f64)]) -> NigAr1 {
        let n = pairs.len() as f64;
        // Λ_n = Λ_0 + Σ φ_t φ_tᵀ
        let mut lambda = prior.lambda;
        for (phi, _) in pairs {
            lambda[0][0] += phi[0] * phi[0];
            lambda[0][1] += phi[0] * phi[1];
            lambda[1][0] += phi[1] * phi[0];
            lambda[1][1] += phi[1] * phi[1];
        }
        // m_n = Λ_n⁻¹ (Λ_0 m_0 + Σ φ_t x_t)
        let lm0 = [
            prior.lambda[0][0] * prior.m[0] + prior.lambda[0][1] * prior.m[1],
            prior.lambda[1][0] * prior.m[0] + prior.lambda[1][1] * prior.m[1],
        ];
        let mut rhs = lm0;
        for (phi, x) in pairs {
            rhs[0] += phi[0] * x;
            rhs[1] += phi[1] * x;
        }
        let (l_inv, _) = invert_2x2(lambda).expect("posterior Λ must be PD");
        let m = [
            l_inv[0][0] * rhs[0] + l_inv[0][1] * rhs[1],
            l_inv[1][0] * rhs[0] + l_inv[1][1] * rhs[1],
        ];
        // a_n = a_0 + n/2
        let a = prior.a + 0.5 * n;
        // b_n = b_0 + ½ (m_0ᵀ Λ_0 m_0 + Σ x² − m_nᵀ Λ_n m_n)
        let m0_q = prior.m[0] * lm0[0] + prior.m[1] * lm0[1];
        let lm_n = [
            lambda[0][0] * m[0] + lambda[0][1] * m[1],
            lambda[1][0] * m[0] + lambda[1][1] * m[1],
        ];
        let m_n_q = m[0] * lm_n[0] + m[1] * lm_n[1];
        let sum_x_sq: f64 = pairs.iter().map(|(_, x)| x * x).sum();
        let b = prior.b + 0.5 * (m0_q + sum_x_sq - m_n_q);
        NigAr1 {
            m,
            lambda,
            a,
            b,
            prev: pairs.last().map(|(phi, _)| phi[1]),
        }
    }

    #[test]
    fn nig_ar1_iterative_matches_closed_form() {
        // Feed an AR(1)-like sequence through both the iterative `update`
        // chain and the closed-form posterior. They must agree to
        // O(rounding error) on every hyperparameter.
        let prior = NigAr1::new([0.5, -0.2], [[1.5, 0.0], [0.0, 1.5]], 2.0, 1.0);
        let mut rng = Rng::new(20251);
        let n = 200;
        let mut data = Vec::with_capacity(n);
        let mut x_prev = 0.0_f64;
        for _ in 0..n {
            let x = 0.4 + 0.5 * x_prev + rng.normal(0.0, 1.0);
            data.push(x);
            x_prev = x;
        }

        // Iterative.
        let mut state = prior.clone();
        for &x in &data {
            state = state.update(x);
        }

        // Closed form (skipping the unpaired first obs, whose iterative
        // path leaves state unchanged except for prev).
        let pairs: Vec<([f64; 2], f64)> = data
            .windows(2)
            .map(|w| ([1.0, w[0]], w[1]))
            .collect();
        let closed = closed_form(&prior, &pairs);

        for i in 0..2 {
            assert!(
                (state.m[i] - closed.m[i]).abs() < 1e-9,
                "m[{i}]: iterative={}, closed={}",
                state.m[i],
                closed.m[i]
            );
            for j in 0..2 {
                assert!(
                    (state.lambda[i][j] - closed.lambda[i][j]).abs() < 1e-9,
                    "Λ[{i}][{j}]: iterative={}, closed={}",
                    state.lambda[i][j],
                    closed.lambda[i][j]
                );
            }
        }
        assert!((state.a - closed.a).abs() < 1e-9, "a: it={}, cf={}", state.a, closed.a);
        // β accumulates rounding; tolerance scales with n.
        assert!(
            (state.b - closed.b).abs() < 1e-6,
            "b: iterative={}, closed={}",
            state.b,
            closed.b
        );
    }

    #[test]
    fn nig_ar1_first_obs_skips_predictive_emits_no_update() {
        let prior = NigAr1::default_prior();
        let after_one = prior.update(1.5);
        // (m, Λ, a, b) must be unchanged; only `prev` advances.
        assert_eq!(after_one.m, prior.m);
        assert_eq!(after_one.lambda, prior.lambda);
        assert_eq!(after_one.a, prior.a);
        assert_eq!(after_one.b, prior.b);
        assert_eq!(after_one.prev, Some(1.5));
    }

    #[test]
    fn nig_ar1_predictive_is_finite_after_warmup() {
        let mut state = NigAr1::default_prior();
        let mut rng = Rng::new(42);
        let mut prev = 0.0_f64;
        for _ in 0..50 {
            let x = 0.6 * prev + rng.normal(0.0, 1.0);
            state = state.update(x);
            prev = x;
        }
        let lp = state.log_predictive(prev);
        assert!(lp.is_finite(), "log-pred must be finite, got {lp}");
    }

    #[test]
    fn nig_ar1_predictive_is_a_proper_density_at_prior() {
        // Wide-grid trapezoidal integral of exp(log_predictive) should
        // be ≈ 1. At the prior with prev=None, predictive is a
        // Student-t(2a, m[0], (b/a)(1 + Λ⁻¹[0,0])).
        let prior = NigAr1::new([0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]], 2.0, 1.0);
        let n_grid = 50_000;
        let (a, b) = (-200.0_f64, 200.0_f64);
        let dx = (b - a) / n_grid as f64;
        let mut mass = 0.0;
        for i in 0..=n_grid {
            let x = a + i as f64 * dx;
            let w = if i == 0 || i == n_grid { 0.5 } else { 1.0 };
            mass += w * prior.log_predictive(x).exp() * dx;
        }
        assert!(
            (mass - 1.0).abs() < 0.01,
            "predictive mass = {mass}, expected ≈ 1.0"
        );
    }
}
