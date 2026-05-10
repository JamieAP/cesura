//! Mean-field Langevin sampler for the PrO per-segment posterior.
//!

use crate::pro_bocd::posterior::PrOPosterior;

/// One Langevin step over the per-segment posterior.
///
/// Placeholder implementation: returns the posterior unchanged.
/// The intended update is:
///
/// ```text
/// θ_i_{t+1} = θ_i_t − step · ∇_θ logp̄(D_r | θ) + noise · √(2 step) · ξ_i
/// ```
///
/// where `logp̄` is the log Wasserstein-projected predictive
/// and `ξ_i` is per-particle Brownian noise. Brownian seeding via a
/// per-step deterministic RNG allows repeatable trajectories given
/// the same noise sequence.
pub fn langevin_step(
    posterior: &mut PrOPosterior,
    _data: &[Vec<f64>],
    _step_size: f64,
    _noise_scale: f64,
) {
    // Placeholder: leaves the posterior unchanged.
    let _ = posterior;
}
