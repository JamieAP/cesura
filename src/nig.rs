//! Normal-Inverse-Gamma conjugate prior for the BOCPD recursion.
//!
//! Implements the iid-Gaussian within-regime predictive used by
//! `BocpdDetector::<Nig>` (the default). This public module allows
//! external callers to spell `BocpdDetector<Nig>` and supplies its
//! `Predictive` implementation.

use std::f64::consts::PI;

use crate::math::student_t_lpdf;
use crate::predictive::Predictive;

/// Normal-Inverse-Gamma sufficient statistics. Conjugate prior for the
/// iid-Gaussian within-regime model used by [`crate::BocpdDetector`].
///
/// The struct is `pub` (so `BocpdDetector<Nig>` resolves outside the
/// crate); fields stay `pub(crate)` -- construct via [`Nig::new`].
#[derive(Clone)]
pub struct Nig {
    pub(crate) mu: f64,
    pub(crate) kappa: f64,
    pub(crate) alpha: f64,
    pub(crate) beta: f64,
}

impl Nig {
    /// Construct sufficient statistics with `(μ, κ, α, β)`. The default
    /// prior used by `BocpdDetector::new(...)` is `Nig::new(0, 1, 1, 1)`.
    pub fn new(mu: f64, kappa: f64, alpha: f64, beta: f64) -> Self {
        Self {
            mu,
            kappa,
            alpha,
            beta,
        }
    }

    pub(crate) fn update(&self, x: f64) -> Self {
        let kappa = self.kappa + 1.0;
        let mu = (self.kappa * self.mu + x) / kappa;
        let alpha = self.alpha + 0.5;
        let beta = self.beta + 0.5 * self.kappa * (x - self.mu).powi(2) / kappa;
        Self {
            mu,
            kappa,
            alpha,
            beta,
        }
    }

    pub(crate) fn log_predictive(&self, x: f64) -> f64 {
        let df = 2.0 * self.alpha;
        let scale_sq = self.beta * (self.kappa + 1.0) / (self.alpha * self.kappa);
        if scale_sq <= 0.0 || !scale_sq.is_finite() {
            return f64::NEG_INFINITY;
        }
        student_t_lpdf(x, df, self.mu, scale_sq.sqrt())
    }

    /// β-divergence "log-likelihood" for the BOCPD recursion.
    ///
    /// Knoblauch et al. (2018, arXiv:1806.02261) replace the standard
    /// Bayesian update `p(θ|x) ∝ p(x|θ)·p(θ)` with a divergence-based
    /// update `p_β(θ|x) ∝ exp(L_β(x;θ))·p(θ)`, where
    ///
    ///   L_β(x;θ) = (1/β) f(x;θ)^β − (1/(β+1)) ∫ f(z;θ)^{β+1} dz.
    ///
    /// As `β → 0` this reduces to the log-likelihood. As `β` grows the
    /// loss becomes bounded -- a single tail observation cannot dominate
    /// the posterior. We approximate the Student-t predictive by a
    /// Gaussian at the predictive's mean and variance; the β-power
    /// integral is then closed-form. The approximation tightens as the
    /// run length grows (df = 2α → ∞).
    ///
    /// Returns the L_β score, used as an unnormalised log-weight in the
    /// run-length recursion. The recursion's per-step renormalisation
    /// makes the absolute scale irrelevant; only relative weights across
    /// run lengths matter.
    pub(crate) fn log_predictive_robust(&self, x: f64, beta: f64) -> f64 {
        // β = 0 ⇒ standard log-density. Strict short-circuit: callers that
        // build with `with_beta(0.0)` get bit-for-bit standard behaviour.
        if beta == 0.0 {
            return self.log_predictive(x);
        }
        let var = self.beta * (self.kappa + 1.0) / (self.alpha * self.kappa);
        if var <= 0.0 || !var.is_finite() {
            return f64::NEG_INFINITY;
        }
        // Gaussian approximation of the predictive's log-density.
        let log_g = -0.5 * ((2.0 * PI * var).ln() + (x - self.mu).powi(2) / var);
        // f(x;θ)^β = exp(β · log_g)
        let f_pow_beta = (beta * log_g).exp();
        // ∫ f(z;θ)^{β+1} dz = (2π σ²)^{-β/2} / √(β+1)
        let int_log = -0.5 * beta * (2.0 * PI * var).ln() - 0.5 * (beta + 1.0).ln();
        let int_term = int_log.exp();
        f_pow_beta / beta - int_term / (beta + 1.0)
    }
}

impl Predictive for Nig {
    fn update(&self, x: f64) -> Self {
        Nig::update(self, x)
    }

    fn log_predictive(&self, x: f64) -> f64 {
        Nig::log_predictive(self, x)
    }

    fn log_predictive_robust(&self, x: f64, beta: f64) -> f64 {
        Nig::log_predictive_robust(self, x, beta)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::Rng;

    #[test]
    fn nig_posterior_mean_converges() {
        let mut nig = Nig {
            mu: 0.0,
            kappa: 1.0,
            alpha: 1.0,
            beta: 1.0,
        };
        for _ in 0..1000 {
            nig = nig.update(3.0);
        }
        assert!(
            (nig.mu - 3.0).abs() < 0.01,
            "posterior mean should converge to 3.0, got {}",
            nig.mu
        );
    }

    #[test]
    fn nig_posterior_precision_grows() {
        let nig = Nig {
            mu: 0.0,
            kappa: 1.0,
            alpha: 1.0,
            beta: 1.0,
        };
        let nig50 = (0..50).fold(nig, |acc, _| acc.update(1.0));
        assert_eq!(nig50.kappa, 51.0);
        assert_eq!(nig50.alpha, 26.0);
    }

    #[test]
    fn nig_predictive_peaked_at_data() {
        let mut nig = Nig {
            mu: 0.0,
            kappa: 1.0,
            alpha: 1.0,
            beta: 1.0,
        };
        let mut rng = Rng::new(42);
        for _ in 0..200 {
            nig = nig.update(rng.normal(5.0, 0.5));
        }
        let at_5 = nig.log_predictive(5.0);
        let at_0 = nig.log_predictive(0.0);
        let at_10 = nig.log_predictive(10.0);
        assert!(at_5 > at_0);
        assert!(at_5 > at_10);
    }

    /// Closed-form NIG posterior given prior (μ₀, κ₀, α₀, β₀) and data {x_i}.
    ///
    /// Reference: Murphy (2012) §3.3 / §4.6, conjugate analysis for normal
    /// with unknown mean and variance.
    fn closed_form_nig(prior: &Nig, data: &[f64]) -> Nig {
        let n = data.len() as f64;
        let kappa_n = prior.kappa + n;
        let mean = data.iter().sum::<f64>() / n;
        let mu_n = (prior.kappa * prior.mu + n * mean) / kappa_n;
        let alpha_n = prior.alpha + n / 2.0;
        let ssq = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>();
        let beta_n = prior.beta
            + 0.5 * ssq
            + prior.kappa * n * (mean - prior.mu).powi(2) / (2.0 * kappa_n);
        Nig {
            mu: mu_n,
            kappa: kappa_n,
            alpha: alpha_n,
            beta: beta_n,
        }
    }

    #[test]
    fn nig_iterative_update_matches_closed_form() {
        let prior = Nig {
            mu: 0.5,
            kappa: 2.0,
            alpha: 3.0,
            beta: 4.0,
        };
        let mut rng = Rng::new(20251);
        let data: Vec<f64> = (0..500).map(|_| rng.normal(2.0, 1.5)).collect();

        let online = data.iter().fold(prior.clone(), |acc, &x| acc.update(x));
        let offline = closed_form_nig(&prior, &data);

        // Tight: this is pure floating-point algebra, no Monte-Carlo.
        assert!(
            (online.mu - offline.mu).abs() < 1e-10,
            "μ: online={}, closed-form={}",
            online.mu,
            offline.mu
        );
        assert!((online.kappa - offline.kappa).abs() < 1e-10);
        assert!((online.alpha - offline.alpha).abs() < 1e-10);
        // β accumulates rounding error -- looser bound, still ≪ value scale.
        assert!(
            (online.beta - offline.beta).abs() < 1e-6,
            "β: online={}, closed-form={}",
            online.beta,
            offline.beta
        );
    }

    /// Numerical integration of exp(NIG.log_predictive) over a wide grid.
    /// The posterior predictive is a Student-t; integral over ℝ should be 1.
    fn integrate_predictive(nig: &Nig, a: f64, b: f64, n: usize) -> f64 {
        let dx = (b - a) / n as f64;
        let mut s = 0.0;
        for i in 0..=n {
            let x = a + i as f64 * dx;
            let w = if i == 0 || i == n { 0.5 } else { 1.0 };
            s += w * nig.log_predictive(x).exp() * dx;
        }
        s
    }

    #[test]
    fn nig_predictive_is_a_proper_density_at_prior() {
        let prior = Nig {
            mu: 0.0,
            kappa: 1.0,
            alpha: 2.0, // df = 4 → finite mean and variance
            beta: 1.0,
        };
        let mass = integrate_predictive(&prior, -100.0, 100.0, 50_000);
        assert!(
            (mass - 1.0).abs() < 0.01,
            "prior predictive mass = {mass}, expected ≈ 1.0"
        );
    }

    #[test]
    fn nig_predictive_variance_matches_closed_form() {
        // Catches a wrong (df, scale) in log_predictive that the normalization
        // test cannot. Posterior predictive is Student-t(df=2α, μ, scale²);
        // its variance is β(κ+1)/(κ(α-1)) for α > 1.
        for &(kappa, alpha, beta) in &[(2.0, 3.0, 1.0), (5.0, 10.0, 4.0), (1.0, 2.5, 0.7)] {
            let nig = Nig {
                mu: 0.5,
                kappa,
                alpha,
                beta,
            };
            let expected_var = beta * (kappa + 1.0) / (kappa * (alpha - 1.0));

            // ∫ (x − μ)² p(x) dx via wide trapezoidal grid.
            let n = 100_000;
            let (a, b) = (-200.0_f64, 200.0_f64);
            let dx = (b - a) / n as f64;
            let mut second_moment = 0.0;
            for i in 0..=n {
                let x = a + i as f64 * dx;
                let w = if i == 0 || i == n { 0.5 } else { 1.0 };
                second_moment += w * (x - nig.mu).powi(2) * nig.log_predictive(x).exp() * dx;
            }
            // Heavy-tail truncation costs us a few percent -- but a wrong (df, scale)
            // would be off by an O(1) factor.
            let rel_err = (second_moment - expected_var).abs() / expected_var;
            assert!(
                rel_err < 0.05,
                "predictive variance: got {second_moment}, expected {expected_var} (κ={kappa}, α={alpha}, β={beta})"
            );
        }
    }

    #[test]
    fn nig_predictive_is_a_proper_density_post_update() {
        // After 200 observations from N(5, 0.5²), the predictive should still
        // integrate to 1 (now centered near 5, much narrower).
        let mut nig = Nig {
            mu: 0.0,
            kappa: 1.0,
            alpha: 1.0,
            beta: 1.0,
        };
        let mut rng = Rng::new(404);
        for _ in 0..200 {
            nig = nig.update(rng.normal(5.0, 0.5));
        }
        let mass = integrate_predictive(&nig, -50.0, 60.0, 50_000);
        assert!(
            (mass - 1.0).abs() < 0.01,
            "posterior predictive mass = {mass}, expected ≈ 1.0"
        );
    }
}
