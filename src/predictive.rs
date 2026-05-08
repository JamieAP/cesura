//! Conjugate predictive abstraction for the BOCPD recursion.
//!
//! `Predictive` lets `BocpdDetector` work over any within-regime
//! conjugate model: i.i.d.-Gaussian (`Nig`), AR(1) (`NigAr1`), and
//! future families (Poisson, NP, etc.) without forking the recursion.
//! The trait surface is the minimum BOCPD needs:
//!
//! - `update(x) -> Self` -- conjugate posterior step.
//! - `log_predictive(x)` -- log of the marginal predictive at `x`.
//! - `log_predictive_robust(x, beta)` -- β-divergence robust variant
//!   (Knoblauch et al. 2018), with a default impl that falls back to
//!   `log_predictive` for any predictive that hasn't derived its own
//!   β-power closed form yet.
//!
//! The default `log_predictive_robust` matches the `Nig` short-circuit
//! at `beta == 0` (bit-for-bit identical to standard BOCPD), so a new
//! predictive can be slotted in without touching the robust path.

/// Conjugate within-regime predictive used by `BocpdDetector`.
///
/// Implementors maintain sufficient statistics; `update` returns a new
/// state advanced by one observation, `log_predictive` returns the log
/// of the marginal predictive density.
pub trait Predictive: Clone + Send {
    /// Conjugate posterior update: returns a new state that has absorbed
    /// observation `x`.
    fn update(&self, x: f64) -> Self;

    /// Log of the marginal predictive density at `x` under the current
    /// posterior.
    fn log_predictive(&self, x: f64) -> f64;

    /// β-divergence robust log-predictive (Knoblauch et al. 2018,
    /// arXiv:1806.02261). At `beta == 0` this collapses to the standard
    /// `log_predictive` (bit-for-bit identical) and that is the default
    /// impl. Predictives that ship a robust-update closed form override
    /// this method; predictives that don't get standard behaviour for
    /// free at β = 0 and the same standard-density fallback at β > 0.
    fn log_predictive_robust(&self, x: f64, _beta: f64) -> f64 {
        self.log_predictive(x)
    }
}
