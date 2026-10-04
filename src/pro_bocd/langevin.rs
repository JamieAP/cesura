//! Mean-field Langevin sampler for the PrO per-segment posterior.
//!
//! Implements an importance-weighted Wasserstein gradient and a
//! leave-one-out discretised Langevin step. Cost per step:
//! `O(p² · r · d)` for the leave-one-out
//! mixture density evaluation.
//!
//! For numerical stability the leave-one-out mixture density is
//! computed in log-space via log-sum-exp, then importance weights
//! `w_i^(¬j) = exp(log N(x_i; ϑ_j, I) − log P_Q^(¬j)(x_i))`.

use crate::pro_bocd::posterior::PrOPosterior;

/// Log of the standard-normal density `N(x; μ, I_d)` at `x`. The
/// `(2π)^(d/2)` normalisation factor cancels out of every importance
/// weight `w_i^(¬j)` (numerator and denominator both N(·; ·, I_d)),
/// so it would be safe to drop -- but we keep it so this helper is
/// usable on its own for BOCD predictive integration.
fn log_gaussian_iid(x: &[f64], mean: &[f64]) -> f64 {
    debug_assert_eq!(x.len(), mean.len());
    let d = x.len() as f64;
    let mut sq_dist = 0.0;
    for (xi, mi) in x.iter().zip(mean.iter()) {
        let dz = xi - mi;
        sq_dist += dz * dz;
    }
    -0.5 * sq_dist - 0.5 * d * (2.0 * std::f64::consts::PI).ln()
}

/// `λ_r = √r / log(r)²`, the run-length-dependent regularisation rate.
/// `lambda_factor` is a multiplicative tunable; `1.0` matches the
/// unscaled rate.
pub fn lambda_n(r: usize, lambda_factor: f64) -> f64 {
    if r < 2 {
        return 0.0;
    }
    let r = r as f64;
    let log_r = r.ln().max(1e-9);
    lambda_factor * r.sqrt() / (log_r * log_r)
}

/// One step of the leave-one-out mean-field Langevin update.
/// Mutates `posterior.particles` in place.
///
/// Arguments:
/// - `data`: segment observations `x_{1..r}`, each of length `p_dim`.
///   May be empty (degenerate r=0 case) -- in that case the data term
///   is zero and only the prior term + Brownian noise drive particles.
/// - `step_size`: discretisation step `η > 0`.
/// - `lambda_factor`: tunable for `λ_r` (default 1.0).
/// - `prior_mean` / `prior_inv_var`: Gaussian prior `N(μ_0, Σ_0)` with
///   diagonal `Σ_0`. `prior_inv_var[i] = 1/(Σ_0)_{ii}`.
/// - `rng`: closure returning iid `N(0, 1)` samples (Brownian noise).
///
/// **Numerical stability**: the leave-one-out mixture density is
/// computed in log-space via log-sum-exp; for `r = 0` or `p < 2` the
/// data term contribution is set to zero (gradient ill-defined; only
/// the prior term + Brownian noise drive particles).
pub fn langevin_step(
    posterior: &mut PrOPosterior,
    data: &[Vec<f64>],
    step_size: f64,
    lambda_factor: f64,
    prior_mean: &[f64],
    prior_inv_var: &[f64],
    rng: &mut impl FnMut() -> f64,
) {
    let p = posterior.n_particles();
    let d = posterior.p_dim;
    let r = data.len();

    if p == 0 || d == 0 {
        return;
    }
    debug_assert_eq!(prior_mean.len(), d);
    debug_assert_eq!(prior_inv_var.len(), d);

    let lambda_r = lambda_n(r, lambda_factor);
    let sqrt_2_eta = (2.0 * step_size).sqrt();

    // Pre-compute log N(x_i; ϑ_ℓ, I) for all (i, ℓ). Cost O(p · r · d);
    // amortises across the per-particle gradient evaluation below.
    // Indexing: log_ng[i * p + ℓ] = log N(x_i; ϑ_ℓ, I).
    let mut log_ng: Vec<f64> = Vec::with_capacity(r * p);
    if r > 0 {
        for x in data.iter() {
            for ϑ_l in posterior.particles.iter() {
                log_ng.push(log_gaussian_iid(x, ϑ_l));
            }
        }
    }

    // Accumulate updates into a separate buffer so the per-particle
    // gradient sees the K-th iteration's particle locations consistently
    // (Euler-Maruyama; explicit step).
    let mut new_particles: Vec<Vec<f64>> = posterior.particles.clone();

    for j in 0..p {
        let ϑ_j = &posterior.particles[j];

        // Data-term gradient: λ_r/r Σ_i (x_i - ϑ_j) w_i^(¬j)(ϑ_j),
        // with leave-one-out mixture density in the denominator.
        // Skip when r = 0 (no data) or p < 2 (no leave-one-out
        // possible).
        let mut data_grad = vec![0.0_f64; d];
        if r > 0 && p >= 2 {
            let inv_p_minus_1 = 1.0 / (p as f64 - 1.0);
            let inv_r = 1.0 / r as f64;
            for (i, x_i) in data.iter().enumerate() {
                // log P_Q^(¬j)(x_i) = -log(p-1) + logsumexp_{ℓ≠j} log_ng[i*p+ℓ]
                let mut max_log = f64::NEG_INFINITY;
                for ℓ in 0..p {
                    if ℓ == j {
                        continue;
                    }
                    let v = log_ng[i * p + ℓ];
                    if v > max_log {
                        max_log = v;
                    }
                }
                if !max_log.is_finite() {
                    // Degenerate row; skip this observation's contribution.
                    continue;
                }
                let mut sum_exp = 0.0;
                for ℓ in 0..p {
                    if ℓ == j {
                        continue;
                    }
                    sum_exp += (log_ng[i * p + ℓ] - max_log).exp();
                }
                if sum_exp <= 0.0 || !sum_exp.is_finite() {
                    continue;
                }
                let log_p_q_loo = max_log + sum_exp.ln() + inv_p_minus_1.ln();
                // w_i^(¬j)(ϑ_j) = exp(log N(x_i; ϑ_j, I) − log P_Q^(¬j)(x_i))
                let log_w = log_ng[i * p + j] - log_p_q_loo;
                let w = log_w.exp();
                if !w.is_finite() {
                    continue;
                }
                for k in 0..d {
                    data_grad[k] += (x_i[k] - ϑ_j[k]) * w;
                }
            }
            let scale = lambda_r * inv_r;
            for k in 0..d {
                data_grad[k] *= scale;
            }
        }

        // Prior gradient: -Σ_0^{-1} (ϑ_j - μ_0). For diagonal Σ_0,
        // per-dimension `prior_inv_var[k] * (μ_0[k] - ϑ_j[k])`.
        // Note: paper's eq (12) has +∇log dΠ which equals
        // -Σ_0^{-1}(ϑ_j - μ_0); we add it (not subtract).
        for k in 0..d {
            let prior_term = prior_inv_var[k] * (prior_mean[k] - ϑ_j[k]);
            // Drift = data_grad + prior_term (note: data_grad has the
            // sign and λ_r factor baked in; prior_term is the raw
            // ∇log dΠ. Both enter eq (12) with positive sign.)
            new_particles[j][k] = ϑ_j[k]
                + step_size * (data_grad[k] + prior_term)
                + sqrt_2_eta * rng();
        }
    }

    posterior.particles = new_particles;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic Brownian RNG: returns the elements of a fixed
    /// pre-seeded sequence. Used by the smoke tests below to compare
    /// trajectories deterministically.
    fn fixed_noise(seed: u64) -> impl FnMut() -> f64 {
        // Deterministic LCG uniforms: sum twelve uniforms and subtract
        // six for a bounded approximation to N(0,1) in this smoke test.
        let mut state = seed.wrapping_mul(6364136223846793005);
        move || {
            let mut acc = 0.0;
            for _ in 0..12 {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let u = ((state >> 32) as u32 as f64) / (u32::MAX as f64);
                acc += u;
            }
            acc - 6.0
        }
    }

    #[test]
    fn lambda_n_basic_properties() {
        // λ_n = √n / log(n)² is non-monotone in finite r: it dips
        // around r ∈ [20, 60] before climbing again as √r dominates.
        // Test the load-bearing properties only:
        //   (a) finite + positive for r ≥ 2;
        //   (b) zero at the degenerate r < 2;
        //   (c) asymptotic growth: λ(10000) > λ(100).
        let small = lambda_n(20, 1.0);
        let large = lambda_n(10000, 1.0);
        assert!(small.is_finite() && small > 0.0);
        assert!(large.is_finite() && large > 0.0);
        assert!(large > small, "λ(10000) should exceed λ(20) (asymptotic √r growth dominates log² damping)");
        // r < 2 returns 0 (degenerate).
        assert_eq!(lambda_n(0, 1.0), 0.0);
        assert_eq!(lambda_n(1, 1.0), 0.0);
    }

    #[test]
    fn log_gaussian_at_mean_is_just_normaliser() {
        // log N(0; 0, I_d) = -d/2 · log(2π).
        let d = 3;
        let x = vec![0.0; d];
        let m = vec![0.0; d];
        let lg = log_gaussian_iid(&x, &m);
        let expected = -0.5 * d as f64 * (2.0 * std::f64::consts::PI).ln();
        assert!((lg - expected).abs() < 1e-12);
    }

    #[test]
    fn langevin_step_zero_data_only_prior_drives() {
        // r = 0: data term is zero. Particles drift toward μ_0 = 0
        // under prior. Verify that with zero step + zero noise the
        // step is a no-op.
        let mut posterior = PrOPosterior::new(2, 4);
        for (i, ϑ) in posterior.particles.iter_mut().enumerate() {
            ϑ[0] = i as f64;
            ϑ[1] = -(i as f64);
        }
        let prior_mean = vec![0.0, 0.0];
        let prior_inv_var = vec![1.0, 1.0];
        let mut rng = fixed_noise(0xDEADBEEF);
        let snapshot: Vec<Vec<f64>> = posterior.particles.clone();
        langevin_step(
            &mut posterior,
            &[],
            0.0, // step_size = 0
            1.0,
            &prior_mean,
            &prior_inv_var,
            &mut rng,
        );
        for (a, b) in posterior.particles.iter().zip(snapshot.iter()) {
            for (av, bv) in a.iter().zip(b.iter()) {
                assert!(
                    (av - bv).abs() < 1e-12,
                    "step_size=0 should be no-op; got {av} vs {bv}"
                );
            }
        }
    }

    #[test]
    fn langevin_step_concentrates_under_data() {
        // r segments of N(μ*, I) data should pull particles toward μ*.
        // Initialise particles from prior; run a few steps; check that
        // empirical mean has moved toward μ* and variance hasn't blown
        // up.
        let d = 2;
        let n_particles = 16;
        let mu_star = vec![3.0, -2.0];
        let r = 50;
        let mut data: Vec<Vec<f64>> = Vec::with_capacity(r);
        let mut data_rng = fixed_noise(0xC0FFEE);
        for _ in 0..r {
            data.push(vec![mu_star[0] + 0.1 * data_rng(), mu_star[1] + 0.1 * data_rng()]);
        }

        let mut posterior = PrOPosterior::new(d, n_particles);
        let mut init_rng = fixed_noise(0xF00DBABE);
        posterior.sample_from_prior(&mut init_rng);

        let initial_mean = posterior.empirical_mean();
        let dist_initial = ((initial_mean[0] - mu_star[0]).powi(2)
            + (initial_mean[1] - mu_star[1]).powi(2))
        .sqrt();

        let prior_mean = vec![0.0, 0.0];
        let prior_inv_var = vec![1.0, 1.0];
        let mut step_rng = fixed_noise(0xCAFEBABE);
        for _ in 0..200 {
            langevin_step(
                &mut posterior,
                &data,
                1e-2,
                1.0,
                &prior_mean,
                &prior_inv_var,
                &mut step_rng,
            );
        }

        let final_mean = posterior.empirical_mean();
        let dist_final = ((final_mean[0] - mu_star[0]).powi(2)
            + (final_mean[1] - mu_star[1]).powi(2))
        .sqrt();

        assert!(
            dist_final < dist_initial,
            "particles should move closer to data centre; initial dist={dist_initial}, final={dist_final}, final_mean={final_mean:?}"
        );
        let var = posterior.empirical_variance();
        assert!(
            var > 1e-6 && var.is_finite(),
            "particle variance collapsed or diverged: {var}"
        );
    }
}
