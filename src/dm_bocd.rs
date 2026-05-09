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
use crate::log_add_exp;
use crate::ChangePoint;
use crate::DEFAULT_MASS_CUTOFF;

const TWO_PI: f64 = 2.0 * std::f64::consts::PI;

/// Default generalised-Bayes weight. Matches the paper's d=2 synthetic
/// experiment (`synthetic.ipynb` cell 9). Tunable via [`DmBocdDetector::with_omega`].
pub const DEFAULT_OMEGA: f64 = 0.1;

/// Multivariate Dm-BOCD detector (mean-only, identity weight).
pub struct DmBocdDetector {
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
}

impl DmBocdDetector {
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
        }
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
                    new_stats[r + 1] = stats[r].update(x, self.omega);
                }
            }

            rl_log = new_rl;
            stats = new_stats;
        }

        // MAP-drop trigger -- same constants as BocpdDetector::detect_multivariate.
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
        result
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

    /// Streaming Dm update with constant m = I_d on whitened obs:
    ///   Σ⁻¹_new = Σ⁻¹ + 2ω·I
    ///   μ_new   = Σ_new · (Σ⁻¹ μ + 2ω·x)
    fn update(&self, x: &[f64], omega: f64) -> Self {
        let d = self.mu.len();
        let two_omega = 2.0 * omega;
        // Σ⁻¹_new = Σ⁻¹ + 2ω·I
        let mut sigma_inv = self.sigma_inv.clone();
        for (i, row) in sigma_inv.iter_mut().enumerate() {
            row[i] += two_omega;
        }
        // rhs = Σ⁻¹ μ + 2ω x
        let mut rhs = mat_vec(&self.sigma_inv, &self.mu);
        for i in 0..d {
            rhs[i] += two_omega * x[i];
        }
        // μ_new = Σ_new · rhs = solve(Σ⁻¹_new, rhs)
        let mu = match solve_pd(&sigma_inv, &rhs) {
            Some(v) => v,
            // Numerically singular Σ⁻¹_new ⇒ fall back to prior-style identity update.
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
        let post = prior.update(&[1.0, 2.0], 0.5);
        assert!((post.sigma_inv[0][0] - 2.0).abs() < 1e-12);
        assert!((post.sigma_inv[1][1] - 2.0).abs() < 1e-12);
        assert!((post.sigma_inv[0][1]).abs() < 1e-12);
        // μ_new = (2I)⁻¹ · (I·0 + 1·x) = 0.5·x
        assert!((post.mu[0] - 0.5).abs() < 1e-12);
        assert!((post.mu[1] - 1.0).abs() < 1e-12);
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
