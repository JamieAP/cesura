//! BOCD recursion integration for PrO-BOCD.
//!
//! Replaces the standard Bayesian per-segment update with the
//! Wasserstein-projected predictive `P_Q` from
//! [`crate::pro_bocd::posterior::PrOPosterior`] and
//! [`crate::pro_bocd::langevin::langevin_step`]. Mirrors the
//! `BocpdDetector::detect_multivariate` recursion shell (whitening
//! preamble → run-length recursion → MAP-RL drop CP emission) so
//! cross-detector comparisons isolate the predictive-update change.

use crate::bocpd::{forward_solve, per_dim_znorm, whitening_transform};
use crate::math::log_add_exp;
use crate::pro_bocd::langevin::{lambda_n, langevin_step};
use crate::pro_bocd::posterior::PrOPosterior;
use crate::pro_bocd::PrOBocpdDetector;
use crate::ChangePoint;

/// log N(x; μ, σ² I) for a diagonal-isotropic Gaussian.
fn log_iso_gaussian(x: &[f64], mean: &[f64], var: f64) -> f64 {
    debug_assert_eq!(x.len(), mean.len());
    let d = x.len() as f64;
    let mut sq_dist = 0.0;
    for (xi, mi) in x.iter().zip(mean.iter()) {
        let dz = xi - mi;
        sq_dist += dz * dz;
    }
    -0.5 * sq_dist / var
        - 0.5 * d * (2.0 * std::f64::consts::PI * var).ln()
}

/// log P_Q(x_t) under the particle approximation
/// `P_Q(x) ≈ (1/p) Σ_j N(x; ϑ^(j), I_d)`. Computed
/// in log-space via log-sum-exp for numerical stability.
fn log_mixture_density(x: &[f64], particles: &[Vec<f64>]) -> f64 {
    let p = particles.len();
    if p == 0 {
        return f64::NEG_INFINITY;
    }
    let mut log_terms = Vec::with_capacity(p);
    let mut max_log = f64::NEG_INFINITY;
    for ϑ in particles.iter() {
        let lg = log_iso_gaussian(x, ϑ, 1.0);
        if lg > max_log {
            max_log = lg;
        }
        log_terms.push(lg);
    }
    if !max_log.is_finite() {
        return f64::NEG_INFINITY;
    }
    let mut sum_exp = 0.0;
    for v in &log_terms {
        sum_exp += (v - max_log).exp();
    }
    if sum_exp <= 0.0 {
        return f64::NEG_INFINITY;
    }
    max_log + sum_exp.ln() - (p as f64).ln()
}

/// Closed-form r=0 prior predictive `N(x; μ_0, Σ_0 + I)`. Cesura's
/// defaults are `μ_0 = 0`, `Σ_0 = I` ⇒ predictive is `N(0, 2I)`.
fn log_prior_predictive(x: &[f64]) -> f64 {
    let d = x.len();
    let mu = vec![0.0; d];
    log_iso_gaussian(x, &mu, 2.0)
}

/// Deterministic approximately normal RNG using twelve uniforms minus
/// six, as in the langevin module's test RNG. Its output is bounded;
/// it supplies particle initialisation and Langevin noise.
struct DetRng {
    state: u64,
}
impl DetRng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407),
        }
    }
    fn next_normal(&mut self) -> f64 {
        let mut acc = 0.0;
        for _ in 0..12 {
            self.state = self
                .state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = ((self.state >> 32) as u32 as f64) / (u32::MAX as f64);
            acc += u;
        }
        acc - 6.0
    }
}

impl PrOBocpdDetector {
    /// PrO-BOCD multivariate detect with a deterministic Brownian
    /// noise sequence seeded by `seed` for repeatable evaluation.
    ///
    /// Standard BOCD run-length recursion shell with
    /// the per-segment predictive replaced by `P_Q`. One
    /// Langevin step per (run-length, time-step) cell amortises burn-in
    /// across the segment.
    ///
    /// Trigger semantics mirror `BocpdDetector::detect_multivariate`
    /// (MAP-RL drop + cooldown) so PrO-vs-NIW comparisons isolate the
    /// posterior-change axis from the trigger axis.
    pub fn detect_multivariate_seeded(&self, data: &[Vec<f64>], seed: u64) -> Vec<ChangePoint> {
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

        // Whitening preamble (mirrors BocpdDetector::detect_multivariate).
        let warmup_n = (n / 3).min(60).max(d * 2);
        let norm: Vec<Vec<f64>> = if warmup_n >= d * 2 && warmup_n <= n {
            match whitening_transform(&data[..warmup_n], d) {
                Some((mean, l_inv)) => data
                    .iter()
                    .map(|x| {
                        let centered: Vec<f64> = (0..d).map(|i| x[i] - mean[i]).collect();
                        forward_solve(&l_inv, &centered)
                    })
                    .collect(),
                None => per_dim_znorm(data, d, n),
            }
        } else {
            per_dim_znorm(data, d, n)
        };

        let max_r = self.max_rl.min(n);
        let prior_mean = vec![0.0; d];
        let prior_inv_var = vec![1.0; d];

        let mut rng_state = DetRng::new(seed);
        let mut rng = || rng_state.next_normal();

        // Per-run-length particle states. posteriors[r] holds the
        // per-segment posterior for run-length r at the current step.
        let mut posteriors: Vec<PrOPosterior> = (0..=max_r)
            .map(|_| PrOPosterior::new(d, self.n_particles))
            .collect();
        // r=0 starts from prior.
        posteriors[0].sample_from_prior(&mut rng);

        // Hazard / growth (lambda is mean run length; hazard = 1/λ).
        // Use the same closed forms as BocpdDetector: hazard_log =
        // -log(λ), growth_log = log(1 - 1/λ) = log((λ-1)/λ).
        let hazard_log = -(self.lambda).ln();
        let growth_log = ((self.lambda - 1.0) / self.lambda).ln();

        let mut rl_log = vec![f64::NEG_INFINITY; max_r + 1];
        rl_log[0] = 0.0;
        let mut map_rls = Vec::with_capacity(n);
        let mut cp_probs = Vec::with_capacity(n);

        for (t, x) in norm.iter().enumerate() {
            let active = (t + 1).min(max_r);
            let mut new_rl = vec![f64::NEG_INFINITY; max_r + 1];
            let mut prev_mass = f64::NEG_INFINITY;

            // Continuation transitions r → r+1 with PrO predictive.
            for r in 0..=active.min(max_r.saturating_sub(1)) {
                if rl_log[r] == f64::NEG_INFINITY {
                    continue;
                }
                let pred = log_mixture_density(x, &posteriors[r].particles);
                if !pred.is_finite() {
                    continue;
                }
                if r < max_r {
                    new_rl[r + 1] =
                        log_add_exp(new_rl[r + 1], rl_log[r] + pred + growth_log);
                }
                prev_mass = log_add_exp(prev_mass, rl_log[r]);
            }

            // r=0 transition: closed-form prior predictive N(x; 0, 2I).
            let prior_pred = log_prior_predictive(x);
            new_rl[0] = if prior_pred.is_finite() && prev_mass.is_finite() {
                prev_mass + hazard_log + prior_pred
            } else {
                f64::NEG_INFINITY
            };

            // Normalise.
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

            // Particle update: ϑ_{r+1} ← Langevin(ϑ_r) using segment
            // x_{t-r..=t}. Build new posteriors in a separate buffer so
            // this step's reads are consistent.
            let mut new_posteriors: Vec<PrOPosterior> = (0..=max_r)
                .map(|_| PrOPosterior::new(d, self.n_particles))
                .collect();
            new_posteriors[0].sample_from_prior(&mut rng);
            for r in 0..=active.min(max_r.saturating_sub(1)) {
                if r >= max_r {
                    continue;
                }
                let lambda_factor = lambda_n(r + 1, 1.0)
                    / (((r + 1) as f64).sqrt() / (((r + 1) as f64).ln().max(1e-9)).powi(2))
                        .max(1e-12);
                // Copy old r posterior to new r+1 then take one step.
                new_posteriors[r + 1].particles = posteriors[r].particles.clone();
                let seg_start = t.saturating_sub(r);
                let segment = &norm[seg_start..=t];
                langevin_step(
                    &mut new_posteriors[r + 1],
                    segment,
                    self.langevin_step,
                    lambda_factor,
                    &prior_mean,
                    &prior_inv_var,
                    &mut rng,
                );
            }

            rl_log = new_rl;
            posteriors = new_posteriors;
        }

        // CP emission via MAP-RL drop (mirrors detect_multivariate
        // exactly so PrO and NIW share the same trigger geometry).
        let drop_to: usize = 3;
        let min_prev_rl: usize = 30;
        let cooldown: usize = 15;

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

    /// Convenience wrapper for `detect_multivariate_seeded(data, 0xCESURA42)`.
    /// Callers can pass an explicit seed for reproducible comparisons.
    pub fn detect_multivariate(&self, data: &[Vec<f64>]) -> Vec<ChangePoint> {
        self.detect_multivariate_seeded(data, 0x_CE_5A_4A_42_u64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Smoke: detector runs on a clean shift fixture without panicking
    /// and produces ≥ 1 CP near the true shift point. Note: PrO is
    /// strictly an iso-Gaussian-mean model; this is the well-spec
    /// regime, so it should detect cleanly.
    #[test]
    fn pro_bocd_detect_clean_shift_smoke() {
        // 2-channel clean 5σ shift at t=150.
        let mut data: Vec<Vec<f64>> = Vec::with_capacity(300);
        // Use deterministic noise via DetRng.
        let mut rng = DetRng::new(0xDEADBEEF);
        for t in 0..300 {
            let s = if t < 150 { 0.0 } else { 5.0 };
            data.push(vec![s + rng.next_normal(), s + rng.next_normal()]);
        }

        let det = PrOBocpdDetector::new(200.0, 250).with_n_particles(8);
        let cps = det.detect_multivariate_seeded(&data, 0xC0FFEE);
        assert!(
            !cps.is_empty(),
            "PrO-BOCD produced 0 CPs on a 5σ-shift fixture"
        );
        let near_shift = cps.iter().any(|c| (c.index as i64 - 150).abs() <= 30);
        assert!(
            near_shift,
            "no CP within ±30 of GT=150; cps={cps:?}"
        );
    }

    /// Determinism: same seed produces identical CPs.
    #[test]
    fn pro_bocd_deterministic_on_seed() {
        let mut data: Vec<Vec<f64>> = Vec::with_capacity(200);
        let mut rng = DetRng::new(0xABCD);
        for t in 0..200 {
            let s = if t < 100 { 0.0 } else { 4.0 };
            data.push(vec![s + rng.next_normal(), s + rng.next_normal()]);
        }
        let det = PrOBocpdDetector::new(200.0, 250).with_n_particles(8);
        let cps_a = det.detect_multivariate_seeded(&data, 0x1234);
        let cps_b = det.detect_multivariate_seeded(&data, 0x1234);
        assert_eq!(
            cps_a.len(),
            cps_b.len(),
            "same seed, different CP counts: {} vs {}",
            cps_a.len(),
            cps_b.len()
        );
        for (a, b) in cps_a.iter().zip(cps_b.iter()) {
            assert_eq!(a.index, b.index);
        }
    }
}
