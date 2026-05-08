//! Auto-tune β-divergence robustness from a warmup window's kurtosis.
//!

/// Sample excess kurtosis (`E[(X-μ)⁴]/σ⁴ − 3`). Returns `0.0` for n < 4
/// or zero variance (degenerate input → no signal, no β).
///
/// Plain biased moment estimator. The bias correction matters less than
/// the variability of fourth-moment estimates on short windows, so we
/// keep the simpler form.
pub fn excess_kurtosis(data: &[f64]) -> f64 {
    let n = data.len();
    if n < 4 {
        return 0.0;
    }
    let nf = n as f64;
    let mean = data.iter().sum::<f64>() / nf;
    let mut m2 = 0.0;
    let mut m4 = 0.0;
    for &x in data {
        let d = x - mean;
        let d2 = d * d;
        m2 += d2;
        m4 += d2 * d2;
    }
    m2 /= nf;
    m4 /= nf;
    // Treat near-constant input as zero variance: floating-point cancellation
    // leaves m2 of order ε relative to the input scale, which would otherwise
    // produce a degenerate kurt ≈ 1 (excess −2) signal that's pure noise.
    let scale_sq = (mean * mean).max(1.0);
    if m2 <= 1e-24 * scale_sq || !m2.is_finite() {
        return 0.0;
    }
    let kurt = m4 / (m2 * m2);
    kurt - 3.0
}

/// Map excess kurtosis to a β-divergence parameter.
///
/// `β = clamp(SLOPE · max(0, k_ex − DEAD_ZONE), 0.0, BETA_MAX)`.
///
///
///
///
pub fn beta_from_excess_kurtosis(k_ex: f64) -> f64 {
    const SLOPE: f64 = 0.030;
    const DEAD_ZONE: f64 = 1.0;
    const BETA_MAX: f64 = 0.20;
    if !k_ex.is_finite() {
        return BETA_MAX;
    }
    let raw = SLOPE * (k_ex - DEAD_ZONE).max(0.0);
    raw.clamp(0.0, BETA_MAX)
}

/// One-shot convenience: kurtosis → β. Always returns a finite value
/// in `[0.0, BETA_MAX]`.
pub fn auto_beta(warmup: &[f64]) -> f64 {
    beta_from_excess_kurtosis(excess_kurtosis(warmup))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gaussian_window_maps_to_zero_or_near_zero_beta() {
        // 1000 N(0,1) samples; excess kurtosis ≈ 0 → β = 0 (dead zone).
        let mut rng = crate::eval::Rng::new(1);
        let data: Vec<f64> = (0..1000).map(|_| rng.normal(0.0, 1.0)).collect();
        let b = auto_beta(&data);
        assert_eq!(b, 0.0, "Gaussian should map to β=0, got {b}");
    }

    #[test]
    fn excess_kurtosis_zero_for_constant_input() {
        let data = vec![3.7; 50];
        assert_eq!(excess_kurtosis(&data), 0.0);
    }

    #[test]
    fn excess_kurtosis_short_input_returns_zero() {
        assert_eq!(excess_kurtosis(&[]), 0.0);
        assert_eq!(excess_kurtosis(&[1.0, 2.0, 3.0]), 0.0);
    }

    #[test]
    fn beta_clamps_to_max() {
        // Massive kurtosis must not produce β > 0.20.
        assert_eq!(beta_from_excess_kurtosis(1e9), 0.20);
        // Infinite (Cauchy-style) collapses to BETA_MAX cleanly.
        assert_eq!(beta_from_excess_kurtosis(f64::INFINITY), 0.20);
    }
}
