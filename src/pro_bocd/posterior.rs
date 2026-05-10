//! PrO per-segment posterior -- particle representation.
//!
//! Implements a Gaussian-mean specialization:
//! post-whitening model `P_θ(x) = N(x; θ, I_d)`, prior `Π = N(μ_0, Σ_0)`.
//! Particle approximation:
//!
//! ```text
//! P_Q(x) ≈ (1/p) Σ_j N(x; ϑ^(j), I_d)
//! ```

/// Per-segment Predictively-Oriented posterior `Q_r` over the latent
/// regime mean `θ ∈ R^d`. Represented as `n_particles` weighted samples
/// in parameter space; weights are uniform with the default sampler
/// (mean-field Langevin, no resampling).
pub struct PrOPosterior {
    /// Latent parameter dim (= whitened observation dim `d`).
    pub p_dim: usize,
    /// Particle locations: `n_particles` rows × `p_dim` cols.
    pub particles: Vec<Vec<f64>>,
}

impl PrOPosterior {
    /// Construct an empty posterior with `n_particles` particles in
    /// `R^{p_dim}`, all initialised to the origin (≡ prior mean for the
    /// `μ_0 = 0` default). Caller should typically follow with
    /// [`Self::sample_from_prior`] before running Langevin.
    pub fn new(p_dim: usize, n_particles: usize) -> Self {
        Self {
            p_dim,
            particles: vec![vec![0.0; p_dim]; n_particles],
        }
    }

    /// Number of particles `p`.
    pub fn n_particles(&self) -> usize {
        self.particles.len()
    }

    /// Sample particles from a standard-normal prior (`μ_0 = 0`,
    /// `Σ_0 = I`). The supplied closure must return iid `N(0, 1)`
    /// samples; callers seed it to reproduce particle trajectories.
    pub fn sample_from_prior(&mut self, rng: &mut impl FnMut() -> f64) {
        for ϑ in self.particles.iter_mut() {
            for v in ϑ.iter_mut() {
                *v = rng();
            }
        }
    }

    /// Empirical particle-cloud mean (per-dimension average). Used by
    /// the BOCD recursion's predictive integration.
    pub fn empirical_mean(&self) -> Vec<f64> {
        let p = self.n_particles();
        if p == 0 {
            return vec![0.0; self.p_dim];
        }
        let mut mean = vec![0.0; self.p_dim];
        for ϑ in self.particles.iter() {
            for (i, v) in ϑ.iter().enumerate() {
                mean[i] += v;
            }
        }
        let inv_p = 1.0 / p as f64;
        for v in mean.iter_mut() {
            *v *= inv_p;
        }
        mean
    }

    /// Empirical per-particle variance averaged across dimensions.
    /// Regression tests use this to flag particle
    /// collapse (variance → 0 ⇒ posterior degraded to MLE).
    pub fn empirical_variance(&self) -> f64 {
        let p = self.n_particles();
        if p < 2 {
            return 0.0;
        }
        let mean = self.empirical_mean();
        let mut sum_sq = 0.0;
        for ϑ in self.particles.iter() {
            for (i, v) in ϑ.iter().enumerate() {
                let d = v - mean[i];
                sum_sq += d * d;
            }
        }
        sum_sq / (p as f64 * self.p_dim as f64)
    }
}
