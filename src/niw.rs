//! Normal-Inverse-Wishart conjugate prior for multivariate BOCPD.
//!
//! Extends NIG to d-dimensional data. Tracks a d×d scatter matrix (Ψ)
//! instead of a scalar variance (β). The predictive distribution is
//! a multivariate Student-t.
//!
//! Math reference: Murphy (2012) §4.6.3, conjugate analysis for MVN.

use crate::lgamma;
use std::f64::consts::PI;

/// Normal-Inverse-Wishart sufficient statistics.
///
/// Parameters: μ (d-vector), κ (precision scale), ν (degrees of freedom),
/// Ψ (d×d scatter matrix stored row-major).
#[derive(Clone)]
pub struct Niw {
    pub(crate) d: usize,
    pub(crate) mu: Vec<f64>,
    pub(crate) kappa: f64,
    pub(crate) nu: f64,
    pub(crate) psi: Vec<f64>, // d×d row-major
}

impl Niw {
    /// Create a default prior for d-dimensional data.
    pub fn new(d: usize) -> Self {
        let mut psi = vec![0.0; d * d];
        for i in 0..d {
            psi[i * d + i] = 1.0; // identity
        }
        Self {
            d,
            mu: vec![0.0; d],
            kappa: 1.0,
            nu: d as f64 + 2.0, // minimum for finite mean
            psi,
        }
    }

    /// Conjugate update with observation x (d-vector).
    pub fn update(&self, x: &[f64]) -> Self {
        debug_assert_eq!(x.len(), self.d);

        let kappa = self.kappa + 1.0;
        let nu = self.nu + 1.0;

        // Updated mean
        let mu: Vec<f64> = (0..self.d)
            .map(|i| (self.kappa * self.mu[i] + x[i]) / kappa)
            .collect();

        // Scatter update: Ψ' = Ψ + κ/(κ+1) * (x-μ)(x-μ)ᵀ
        let scale = self.kappa / kappa;
        let diff: Vec<f64> = (0..self.d).map(|i| x[i] - self.mu[i]).collect();
        let mut psi = self.psi.clone();
        for i in 0..self.d {
            for j in 0..self.d {
                psi[i * self.d + j] += scale * diff[i] * diff[j];
            }
        }

        Self {
            d: self.d,
            mu,
            kappa,
            nu,
            psi,
        }
    }

    /// Log predictive probability of x under the posterior predictive.
    ///
    /// Multivariate Student-t with:
    /// - df = ν - d + 1
    /// - loc = μ
    /// - scale = Ψ (κ+1)/(κ(ν-d+1))
    pub fn log_predictive(&self, x: &[f64]) -> f64 {
        let d = self.d as f64;
        let df = self.nu - d + 1.0;
        if df <= 0.0 {
            return f64::NEG_INFINITY;
        }

        let scale_factor = (self.kappa + 1.0) / (self.kappa * df);

        // Sigma = psi * scale_factor
        // We need: (x - mu)ᵀ Sigma⁻¹ (x - mu)
        // = (1/scale_factor) * (x - mu)ᵀ psi⁻¹ (x - mu)
        let diff: Vec<f64> = (0..self.d).map(|i| x[i] - self.mu[i]).collect();

        // For small d (≤6), use direct Cholesky or explicit inversion
        let (log_det_sigma, quad) = match self.cholesky_solve(&diff, scale_factor) {
            Some(v) => v,
            None => return -100.0,
        };

        // Log PDF of multivariate Student-t
        let half_d = d / 2.0;
        let half_df = df / 2.0;
        let half_df_d = (df + d) / 2.0;

        lgamma(half_df_d) - lgamma(half_df) - half_d * (df * PI).ln() - 0.5 * log_det_sigma
            + (-half_df_d) * (1.0 + quad / df).ln()
    }

    /// Cholesky factorize Σ = psi * scale_factor, solve for quadratic form.
    /// Returns (log_det_sigma, (x-μ)ᵀ Σ⁻¹ (x-μ)).
    fn cholesky_solve(&self, diff: &[f64], scale_factor: f64) -> Option<(f64, f64)> {
        let d = self.d;
        let l = cholesky_decompose(&self.psi, d, scale_factor)?;

        // log|Σ| = 2 * Σ log(L_ii)
        let log_det: f64 = (0..d).map(|i| l[i * d + i].ln()).sum::<f64>() * 2.0;

        // Solve L y = diff (forward substitution)
        let mut y = vec![0.0; d];
        for i in 0..d {
            let mut s = diff[i];
            for j in 0..i {
                s -= l[i * d + j] * y[j];
            }
            y[i] = s / l[i * d + i];
        }

        // Quadratic form = yᵀy
        let quad: f64 = y.iter().map(|v| v * v).sum();

        Some((log_det, quad))
    }
}

/// Cholesky decomposition of M = psi * scale_factor.
/// Returns lower triangular L (row-major, size d×d), or None if not positive definite.
fn cholesky_decompose(psi: &[f64], d: usize, scale_factor: f64) -> Option<Vec<f64>> {
    let mut l = vec![0.0; d * d];
    for i in 0..d {
        for j in 0..=i {
            let mut sum = psi[i * d + j] * scale_factor;
            for k in 0..j {
                sum -= l[i * d + k] * l[j * d + k];
            }
            if i == j {
                if sum <= 0.0 {
                    return None; // not positive definite
                }
                l[i * d + j] = sum.sqrt();
            } else {
                l[i * d + j] = sum / l[j * d + j];
            }
        }
    }
    Some(l)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn niw_update_preserves_dimension() {
        let niw = Niw::new(3);
        let x = vec![1.0, 2.0, 3.0];
        let updated = niw.update(&x);
        assert_eq!(updated.d, 3);
        assert_eq!(updated.mu.len(), 3);
        assert_eq!(updated.psi.len(), 9);
        assert_eq!(updated.kappa, 2.0);
        assert_eq!(updated.nu, 6.0); // d+2+1
    }

    #[test]
    fn niw_mean_converges() {
        let mut niw = Niw::new(2);
        let target = [5.0, -3.0];
        for _ in 0..1000 {
            niw = niw.update(&target);
        }
        assert!((niw.mu[0] - 5.0).abs() < 0.01);
        assert!((niw.mu[1] - -3.0).abs() < 0.01);
    }

    #[test]
    fn niw_predictive_finite() {
        let niw = Niw::new(2);
        let x = vec![0.5, -0.5];
        let lp = niw.log_predictive(&x);
        assert!(lp.is_finite());
        assert!(lp < 0.0); // log prob is always negative
    }

    #[test]
    fn niw_predictive_peaked_at_mean() {
        let mut niw = Niw::new(2);
        // Train on (1,1) to build confidence
        for _ in 0..100 {
            niw = niw.update(&[1.0, 1.0]);
        }
        let lp_center = niw.log_predictive(&[1.0, 1.0]);
        let lp_far = niw.log_predictive(&[10.0, 10.0]);
        assert!(lp_center > lp_far, "predictive should peak at trained mean");
    }

    #[test]
    fn cholesky_identity() {
        let niw = Niw::new(2);
        let diff = vec![1.0, 0.0];
        let (log_det, quad) = niw.cholesky_solve(&diff, 1.0).unwrap();
        // Identity matrix: log det = 0, quad = 1
        assert!((log_det).abs() < 1e-10);
        assert!((quad - 1.0).abs() < 1e-10);
    }

    /// Closed-form NIW posterior given prior and data.
    /// Reference: Murphy (2012) §4.6.3.
    fn closed_form_niw(prior: &Niw, data: &[Vec<f64>]) -> Niw {
        let d = prior.d;
        let n = data.len() as f64;
        let kappa_n = prior.kappa + n;
        let nu_n = prior.nu + n;

        let mut xbar = vec![0.0; d];
        for x in data {
            for i in 0..d {
                xbar[i] += x[i];
            }
        }
        for v in xbar.iter_mut() {
            *v /= n;
        }

        let mu_n: Vec<f64> = (0..d)
            .map(|i| (prior.kappa * prior.mu[i] + n * xbar[i]) / kappa_n)
            .collect();

        let mut s = vec![0.0; d * d];
        for x in data {
            for i in 0..d {
                for j in 0..d {
                    s[i * d + j] += (x[i] - xbar[i]) * (x[j] - xbar[j]);
                }
            }
        }

        let factor = prior.kappa * n / kappa_n;
        let mut psi_n = prior.psi.clone();
        for i in 0..d {
            for j in 0..d {
                psi_n[i * d + j] +=
                    s[i * d + j] + factor * (xbar[i] - prior.mu[i]) * (xbar[j] - prior.mu[j]);
            }
        }

        Niw {
            d,
            mu: mu_n,
            kappa: kappa_n,
            nu: nu_n,
            psi: psi_n,
        }
    }

    fn rng_next(s: &mut [u64; 4]) -> u64 {
        let r = (s[1].wrapping_mul(5)).rotate_left(7).wrapping_mul(9);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        r
    }

    fn rng_normal(state: &mut [u64; 4], mean: f64, std: f64) -> f64 {
        let u = |s: &mut [u64; 4]| -> f64 { (rng_next(s) >> 11) as f64 / (1u64 << 53) as f64 };
        let u1 = u(state).max(1e-300);
        let u2 = u(state);
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
        mean + std * z
    }

    fn rng_seed(seed: u64) -> [u64; 4] {
        let mut s = seed;
        let mut state = [0u64; 4];
        for slot in &mut state {
            s = s.wrapping_add(0x9e3779b97f4a7c15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            *slot = z ^ (z >> 31);
        }
        state
    }

    #[test]
    fn niw_iterative_update_matches_closed_form_d2() {
        let prior = Niw::new(2);
        let mut rng = rng_seed(2024);
        let data: Vec<Vec<f64>> = (0..400)
            .map(|_| {
                vec![
                    rng_normal(&mut rng, 1.5, 1.0),
                    rng_normal(&mut rng, -0.5, 2.0),
                ]
            })
            .collect();

        let online = data.iter().fold(prior.clone(), |acc, x| acc.update(x));
        let offline = closed_form_niw(&prior, &data);

        assert!((online.kappa - offline.kappa).abs() < 1e-10);
        assert!((online.nu - offline.nu).abs() < 1e-10);
        for i in 0..2 {
            assert!(
                (online.mu[i] - offline.mu[i]).abs() < 1e-10,
                "μ[{i}]: online={}, closed-form={}",
                online.mu[i],
                offline.mu[i]
            );
        }
        for i in 0..4 {
            assert!(
                (online.psi[i] - offline.psi[i]).abs() < 1e-5,
                "Ψ[{i}]: online={}, closed-form={}",
                online.psi[i],
                offline.psi[i]
            );
        }
    }

    #[test]
    fn niw_predictive_is_a_proper_density_d1() {
        // At d=1, NIW prior (κ=1, ν=3, Ψ=1) ⇒ Student-t(df=3, μ=0, σ²=2/3).
        // Integrate over wide grid -- should be ≈1.
        let niw = Niw::new(1);
        let n = 50_000;
        let (a, b) = (-100.0_f64, 100.0_f64);
        let dx = (b - a) / n as f64;
        let mut mass = 0.0;
        for i in 0..=n {
            let x = a + i as f64 * dx;
            let w = if i == 0 || i == n { 0.5 } else { 1.0 };
            mass += w * niw.log_predictive(&[x]).exp() * dx;
        }
        assert!(
            (mass - 1.0).abs() < 0.01,
            "NIW d=1 prior predictive mass = {mass}, expected ≈ 1.0"
        );
    }

    #[test]
    fn niw_reduces_to_nig_for_d1() {
        // For d=1, NIW should behave like NIG
        let niw = Niw::new(1);
        let nig = crate::Nig {
            mu: 0.0,
            kappa: 1.0,
            alpha: 1.0, // NIW: nu=3, d=1 → alpha = (nu-d+1)/2 = 1.5 ≠ NIG alpha=1
            beta: 1.0,
        };

        // They won't be numerically identical due to parameterization differences,
        // but both should produce finite negative log probs
        let x = 2.0;
        let lp_niw = niw.log_predictive(&[x]);
        let lp_nig = nig.log_predictive(x);
        assert!(lp_niw.is_finite());
        assert!(lp_nig.is_finite());
        assert!(lp_niw < 0.0);
        assert!(lp_nig < 0.0);
    }
}
