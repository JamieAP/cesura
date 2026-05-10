//! PrO per-segment posterior representation.
//!

/// Per-segment Predictively-Oriented posterior `Q_r` over the latent
/// regime parameter `θ`.
///
/// Represented as `n_particles` weighted samples in parameter space.
/// The intended Wasserstein gradient updates particle locations.
/// Importance weights are uniform for mean-field Langevin without
/// resampling.
pub struct PrOPosterior {
    /// Latent parameter dim (e.g. `d` for whitened-Gaussian-mean).
    pub p_dim: usize,
    /// Particle locations: `n_particles × p_dim`, initially all zero.
    pub particles: Vec<Vec<f64>>,
}

impl PrOPosterior {
    pub fn new(p_dim: usize, n_particles: usize) -> Self {
        Self {
            p_dim,
            particles: vec![vec![0.0; p_dim]; n_particles],
        }
    }
}
