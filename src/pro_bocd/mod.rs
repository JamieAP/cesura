//! Predictively-Oriented BOCD (Knoblauch et al., arXiv:2510.01915).
//!
//! Particle posterior, Langevin sampling, and batch detection.
//! Sampling introduces Monte Carlo variability; seeded APIs support repeatable
//! evaluation. Hyperparameters should be calibrated for each application.

pub mod detect;
pub mod langevin;
pub mod posterior;

/// Predictively-Oriented BOCD detector.
///
/// Holds run-length, particle-count, and Langevin hyperparameters.
pub struct PrOBocpdDetector {
    pub lambda: f64,
    pub max_rl: usize,
    pub n_particles: usize,
    pub langevin_step: f64,
    pub langevin_noise: f64,
}

impl PrOBocpdDetector {
    /// Construct with 32 particles, Langevin step 0.001, and unit noise.
    pub fn new(lambda: f64, max_rl: usize) -> Self {
        Self {
            lambda,
            max_rl,
            n_particles: 32,
            langevin_step: 1e-3,
            langevin_noise: 1.0,
        }
    }

    pub fn with_n_particles(mut self, n: usize) -> Self {
        self.n_particles = n;
        self
    }

    pub fn with_langevin(mut self, step: f64, noise: f64) -> Self {
        self.langevin_step = step;
        self.langevin_noise = noise;
        self
    }
}
