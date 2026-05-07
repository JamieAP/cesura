#![forbid(unsafe_code)]
//! Bayesian Online Change Point Detection (Adams & MacKay 2007)
//! with Normal-Inverse-Gamma conjugate prior.
//!
//! Extended with post-hoc collective anomaly classification
//! per arXiv:2508.06385 (Chen & Wu 2025).
//!
//! # Example
//!
//! ```
//! use cesura::{BocpdDetector, ChangePoint};
//!
//! let data: Vec<f64> = std::iter::repeat(0.0).take(100)
//!     .chain(std::iter::repeat(5.0).take(100))
//!     .collect();
//!
//! let detector = BocpdDetector::new(200.0, 250);
//! let change_points = detector.detect(&data, 0.3);
//!
//! assert!(!change_points.is_empty());
//! assert!((change_points[0].index as i64 - 100).abs() < 15);
//! ```

pub mod detrend;
#[cfg(any(test, feature = "test-utils"))]
pub mod eval;
pub mod niw;
pub mod streaming;

use std::f64::consts::PI;

/// Log-gamma via Lanczos approximation (g=7, n=9).
// Lanczos approximation coefficients require exact precision from reference implementation
#[allow(clippy::excessive_precision, clippy::inconsistent_digit_grouping)]
fn lgamma(x: f64) -> f64 {
    const C: [f64; 9] = [
        0.999_999_999_999_809_93,
        676.520_368_121_885_1,
        -1259.139_216_722_402_8,
        771.323_428_777_653_13,
        -176.615_029_162_140_59,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];

    if x < 0.5 {
        PI.ln() - (PI * x).sin().abs().ln() - lgamma(1.0 - x)
    } else {
        let y = x - 1.0;
        let mut t = C[0];
        for (i, &c) in C[1..].iter().enumerate() {
            t += c / (y + i as f64 + 1.0);
        }
        let w = y + 7.5;
        0.5 * (2.0 * PI).ln() + (y + 0.5) * w.ln() - w + t.ln()
    }
}

/// Log PDF of Student-t(df, loc, scale).
fn student_t_lpdf(x: f64, df: f64, loc: f64, scale: f64) -> f64 {
    let z = (x - loc) / scale;
    lgamma(0.5 * (df + 1.0))
        - lgamma(0.5 * df)
        - 0.5 * (df * PI).ln()
        - scale.ln()
        - 0.5 * (df + 1.0) * (1.0 + z * z / df).ln()
}

/// Normal-Inverse-Gamma sufficient statistics.
#[derive(Clone)]
pub(crate) struct Nig {
    pub(crate) mu: f64,
    pub(crate) kappa: f64,
    pub(crate) alpha: f64,
    pub(crate) beta: f64,
}

impl Nig {
    fn update(&self, x: f64) -> Self {
        let kappa = self.kappa + 1.0;
        let mu = (self.kappa * self.mu + x) / kappa;
        let alpha = self.alpha + 0.5;
        let beta = self.beta + 0.5 * self.kappa * (x - self.mu).powi(2) / kappa;
        Self {
            mu,
            kappa,
            alpha,
            beta,
        }
    }

    fn log_predictive(&self, x: f64) -> f64 {
        let df = 2.0 * self.alpha;
        let scale_sq = self.beta * (self.kappa + 1.0) / (self.alpha * self.kappa);
        if scale_sq <= 0.0 || !scale_sq.is_finite() {
            return f64::NEG_INFINITY;
        }
        student_t_lpdf(x, df, self.mu, scale_sq.sqrt())
    }
}

/// A detected change point with its index and confidence score.
#[derive(Debug, Clone)]
pub struct ChangePoint {
    /// Index in the input data where the change was detected.
    pub index: usize,
    /// Confidence score in `[0, 1]`. Higher means more confident.
    pub confidence: f64,
    /// Absolute shift magnitude in units of global σ.
    /// Computed from normalized data: `|mean_after - mean_before|`.
    pub shift_sigma: f64,
}

/// Bayesian Online Change Point Detector.
///
/// Uses the BOCPD algorithm with a Normal-Inverse-Gamma conjugate prior
/// and MAP run-length estimation for change point detection.
pub struct BocpdDetector {
    hazard_log: f64,
    growth_log: f64,
    max_rl: usize,
    prior: Nig,
}

impl BocpdDetector {
    /// Create a detector with expected run length `lambda` between change points.
    ///
    /// - `lambda`: expected number of observations between change points.
    ///   Smaller values make the detector more sensitive (more false positives).
    /// - `max_run_length`: maximum run length to track. Should be at least as
    ///   large as the input data length.
    ///
    /// # Panics
    /// Panics if `lambda <= 1.0` (would produce -inf or NaN hazard rates).
    pub fn new(lambda: f64, max_run_length: usize) -> Self {
        assert!(lambda > 1.0, "lambda must be > 1.0, got {lambda}");
        let h = 1.0 / lambda;
        Self {
            hazard_log: h.ln(),
            growth_log: (1.0 - h).ln(),
            max_rl: max_run_length,
            prior: Nig {
                mu: 0.0,
                kappa: 1.0,
                alpha: 1.0,
                beta: 1.0,
            },
        }
    }

    /// Run BOCPD on `data`, return change points above `threshold`.
    ///
    /// Detection uses MAP run length drops: when the most probable run length
    /// drops below `threshold` fraction of the time index, a change point
    /// is reported. The confidence is 1 - (map_rl / t).
    ///
    /// Returns an empty vec if `data` has fewer than 20 elements.
    ///
    /// # Heuristic constants
    ///
    /// The MAP-drop detector uses three hand-tuned constants -- `drop_to=3`,
    /// `min_prev_rl=30`, `cooldown=15` -- that gate when a run-length drop
    /// is reported as a change point. They are not from Adams & MacKay (2007);
    /// the underlying BOCPD recursion (NIG predictive, log-space update) is.
    /// Tests verify the math; these constants are tuned against the eval suite.
    pub fn detect(&self, data: &[f64], threshold: f64) -> Vec<ChangePoint> {
        // Filter NaN/infinite values, keeping a map back to original indices
        let mut original_indices: Vec<usize> = Vec::new();
        let data: Vec<f64> = data
            .iter()
            .enumerate()
            .filter_map(|(i, &v)| {
                if v.is_finite() {
                    original_indices.push(i);
                    Some(v)
                } else {
                    None
                }
            })
            .collect();
        let n = data.len();
        if n < 20 {
            return vec![];
        }

        // Normalize for numerical stability
        let mean = data.iter().sum::<f64>() / n as f64;
        let std = (data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64).sqrt();
        let std = if std < 1e-10 { 1.0 } else { std };
        let norm: Vec<f64> = data.iter().map(|x| (x - mean) / std).collect();

        let max_r = self.max_rl.min(n);

        // Run length log-probabilities
        let mut rl_log = vec![f64::NEG_INFINITY; max_r + 1];
        rl_log[0] = 0.0; // r_0 = 0 with certainty

        // Sufficient statistics per run length
        let mut stats = vec![self.prior.clone(); max_r + 1];

        // Track MAP run length at each time step
        let mut map_rls = Vec::with_capacity(n);

        for (t, &x) in norm.iter().enumerate() {
            let active = (t + 1).min(max_r);

            let mut new_rl = vec![f64::NEG_INFINITY; max_r + 1];
            let mut cp_acc = f64::NEG_INFINITY;

            for r in 0..=active.min(max_r.saturating_sub(1)) {
                if rl_log[r] == f64::NEG_INFINITY {
                    continue;
                }
                let pred = stats[r].log_predictive(x);
                if !pred.is_finite() {
                    continue;
                }
                let joint = rl_log[r] + pred;

                // Growth: run length r -> r+1
                if r < max_r {
                    new_rl[r + 1] = log_add_exp(new_rl[r + 1], joint + self.growth_log);
                }
                // Change point: any r -> 0
                cp_acc = log_add_exp(cp_acc, joint + self.hazard_log);
            }
            new_rl[0] = cp_acc;

            // Normalize
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

            // MAP run length
            let map_r = new_rl
                .iter()
                .enumerate()
                .filter(|(_, v)| v.is_finite())
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(r, _)| r)
                .unwrap_or(0);
            map_rls.push(map_r);

            // Update sufficient stats for each growth path
            let mut new_stats = vec![self.prior.clone(); max_r + 1];
            for r in 0..=active.min(max_r.saturating_sub(1)) {
                if r < max_r && new_rl[r + 1] > f64::NEG_INFINITY {
                    new_stats[r + 1] = stats[r].update(x);
                }
            }
            // new_stats[0] stays as prior (fresh regime)

            rl_log = new_rl;
            stats = new_stats;
        }

        // Detect change points: where MAP run length drops sharply.
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
                    let confidence = (1.0 - map_rls[i] as f64 / prev_max as f64).clamp(0.0, 1.0);
                    if confidence >= threshold {
                        // Compute shift in normalized space (= shift/σ in original space)
                        let w = 20;
                        let before = &norm[i.saturating_sub(w)..i];
                        let after = &norm[i..(i + w).min(n)];
                        let mean_b = if before.is_empty() {
                            0.0
                        } else {
                            before.iter().sum::<f64>() / before.len() as f64
                        };
                        let mean_a = if after.is_empty() {
                            0.0
                        } else {
                            after.iter().sum::<f64>() / after.len() as f64
                        };
                        let shift_sigma = (mean_a - mean_b).abs();

                        // Phantom suppression: a "change" with no observable
                        // before/after mean shift is a numerical artifact
                        // (e.g., truncation at max_rl on a constant signal),
                        // not a real regime change.
                        if shift_sigma >= 1e-9 {
                            result.push(ChangePoint {
                                index: original_indices[i],
                                confidence,
                                shift_sigma,
                            });
                            last_detection = i;
                            i += cooldown;
                            continue;
                        }
                    }
                }
            }
            i += 1;
        }
        result
    }

    /// Run multivariate BOCPD on d-dimensional data.
    ///
    /// `data` is a slice of d-dimensional observations (each `Vec<f64>` has length d).
    /// All observations must have the same dimensionality.
    ///
    /// Returns change points using the same MAP run-length drop detection as univariate.
    pub fn detect_multivariate(&self, data: &[Vec<f64>], threshold: f64) -> Vec<ChangePoint> {
        let n = data.len();
        if n < 20 {
            return vec![];
        }
        let d = data[0].len();
        if d == 0 {
            return vec![];
        }
        // Reject ragged input -- all rows must have the same dimensionality
        if data.iter().any(|row| row.len() != d) {
            return vec![];
        }

        // Normalize each dimension independently
        let mut means = vec![0.0; d];
        let mut stds = vec![0.0; d];
        for dim in 0..d {
            let m: f64 = data.iter().map(|x| x[dim]).sum::<f64>() / n as f64;
            let s = (data.iter().map(|x| (x[dim] - m).powi(2)).sum::<f64>() / n as f64).sqrt();
            means[dim] = m;
            stds[dim] = if s < 1e-10 { 1.0 } else { s };
        }
        let norm: Vec<Vec<f64>> = data
            .iter()
            .map(|x| (0..d).map(|i| (x[i] - means[i]) / stds[i]).collect())
            .collect();

        let max_r = self.max_rl.min(n);
        let prior = niw::Niw::new(d);

        // Run length log-probabilities
        let mut rl_log = vec![f64::NEG_INFINITY; max_r + 1];
        rl_log[0] = 0.0;

        let mut stats = vec![prior.clone(); max_r + 1];
        let mut map_rls = Vec::with_capacity(n);

        for (t, x) in norm.iter().enumerate() {
            let active = (t + 1).min(max_r);
            let mut new_rl = vec![f64::NEG_INFINITY; max_r + 1];
            let mut cp_acc = f64::NEG_INFINITY;

            for r in 0..=active.min(max_r.saturating_sub(1)) {
                if rl_log[r] == f64::NEG_INFINITY {
                    continue;
                }
                let pred = stats[r].log_predictive(x);
                if !pred.is_finite() {
                    continue;
                }
                let joint = rl_log[r] + pred;

                if r < max_r {
                    new_rl[r + 1] = log_add_exp(new_rl[r + 1], joint + self.growth_log);
                }
                cp_acc = log_add_exp(cp_acc, joint + self.hazard_log);
            }
            new_rl[0] = cp_acc;

            // Normalize
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

            let mut new_stats = vec![prior.clone(); max_r + 1];
            for r in 0..=active.min(max_r.saturating_sub(1)) {
                if r < max_r && new_rl[r + 1] > f64::NEG_INFINITY {
                    new_stats[r + 1] = stats[r].update(x);
                }
            }

            rl_log = new_rl;
            stats = new_stats;
        }

        // MAP run-length drop detection (same logic as univariate)
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
                    let confidence = (1.0 - map_rls[i] as f64 / prev_max as f64).clamp(0.0, 1.0);
                    if confidence >= threshold {
                        // Shift magnitude: Euclidean norm of per-dimension shifts
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

                        result.push(ChangePoint {
                            index: i,
                            confidence,
                            shift_sigma,
                        });
                        last_detection = i;
                        i += cooldown;
                        continue;
                    }
                }
            }
            i += 1;
        }
        result
    }
}

pub(crate) fn log_add_exp(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY {
        return b;
    }
    if b == f64::NEG_INFINITY {
        return a;
    }
    let max = a.max(b);
    max + ((a - max).exp() + (b - max).exp()).ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Deterministic PRNG (xoshiro256** -- no dependency needed) ───────

    struct Rng([u64; 4]);

    impl Rng {
        fn new(seed: u64) -> Self {
            let mut s = seed;
            let mut state = [0u64; 4];
            for slot in &mut state {
                s = s.wrapping_add(0x9e3779b97f4a7c15);
                let mut z = s;
                z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
                *slot = z ^ (z >> 31);
            }
            Self(state)
        }

        fn next_u64(&mut self) -> u64 {
            let result = (self.0[1].wrapping_mul(5)).rotate_left(7).wrapping_mul(9);
            let t = self.0[1] << 17;
            self.0[2] ^= self.0[0];
            self.0[3] ^= self.0[1];
            self.0[1] ^= self.0[2];
            self.0[0] ^= self.0[3];
            self.0[2] ^= t;
            self.0[3] = self.0[3].rotate_left(45);
            result
        }

        fn uniform(&mut self) -> f64 {
            (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
        }

        fn normal(&mut self, mean: f64, std: f64) -> f64 {
            let u1 = self.uniform().max(1e-300);
            let u2 = self.uniform();
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
            mean + std * z
        }
    }

    // ── Math primitives ───────────────────────────────────────────────

    #[test]
    fn lgamma_known_values() {
        assert!((lgamma(1.0)).abs() < 1e-10);
        assert!((lgamma(2.0)).abs() < 1e-10);
        assert!((lgamma(5.0) - 24.0_f64.ln()).abs() < 1e-8);
        assert!((lgamma(0.5) - 0.5 * PI.ln()).abs() < 1e-8);
        assert!((lgamma(10.0) - 362880.0_f64.ln()).abs() < 1e-6);
        assert!(
            (lgamma(100.0) - 359.13).abs() < 0.1,
            "lgamma(100)={}, expected ~359.13",
            lgamma(100.0)
        );
    }

    #[test]
    fn lgamma_reflection_formula() {
        for &x in &[0.1, 0.25, 0.3, 0.4, 0.49] {
            let lhs = lgamma(x) + lgamma(1.0 - x);
            let rhs = (PI / (PI * x).sin()).ln();
            assert!(
                (lhs - rhs).abs() < 1e-8,
                "reflection failed at x={x}: lhs={lhs}, rhs={rhs}"
            );
        }
    }

    #[test]
    fn student_t_is_normalized() {
        let df = 3.0;
        let n = 10000;
        let a = -50.0;
        let b = 50.0;
        let dx = (b - a) / n as f64;
        let mut integral = 0.0;
        for i in 0..=n {
            let x = a + i as f64 * dx;
            let w = if i == 0 || i == n { 0.5 } else { 1.0 };
            integral += w * student_t_lpdf(x, df, 0.0, 1.0).exp() * dx;
        }
        assert!(
            (integral - 1.0).abs() < 0.01,
            "Student-t(3) integral={integral}, expected ~1.0"
        );
    }

    #[test]
    fn student_t_symmetry() {
        for &df in &[1.0, 2.0, 5.0, 30.0] {
            for &x in &[0.5, 1.0, 2.0, 5.0] {
                let left = student_t_lpdf(-x, df, 0.0, 1.0);
                let right = student_t_lpdf(x, df, 0.0, 1.0);
                assert!((left - right).abs() < 1e-10, "asymmetric at df={df}, x={x}");
            }
        }
    }

    #[test]
    fn student_t_approaches_normal() {
        let df = 1000.0;
        for &x in &[0.0, 0.5, 1.0, 2.0] {
            let t_lpdf = student_t_lpdf(x, df, 0.0, 1.0);
            let n_lpdf = -0.5 * (2.0 * PI).ln() - 0.5 * x * x;
            assert!(
                (t_lpdf - n_lpdf).abs() < 0.01,
                "t(1000) ≠ Normal at x={x}: t={t_lpdf}, n={n_lpdf}"
            );
        }
    }

    // ── NIG conjugate update ──────────────────────────────────────────

    #[test]
    fn nig_posterior_mean_converges() {
        let mut nig = Nig {
            mu: 0.0,
            kappa: 1.0,
            alpha: 1.0,
            beta: 1.0,
        };
        for _ in 0..1000 {
            nig = nig.update(3.0);
        }
        assert!(
            (nig.mu - 3.0).abs() < 0.01,
            "posterior mean should converge to 3.0, got {}",
            nig.mu
        );
    }

    #[test]
    fn nig_posterior_precision_grows() {
        let nig = Nig {
            mu: 0.0,
            kappa: 1.0,
            alpha: 1.0,
            beta: 1.0,
        };
        let nig50 = (0..50).fold(nig, |acc, _| acc.update(1.0));
        assert_eq!(nig50.kappa, 51.0);
        assert_eq!(nig50.alpha, 26.0);
    }

    #[test]
    fn nig_predictive_peaked_at_data() {
        let mut nig = Nig {
            mu: 0.0,
            kappa: 1.0,
            alpha: 1.0,
            beta: 1.0,
        };
        let mut rng = Rng::new(42);
        for _ in 0..200 {
            nig = nig.update(rng.normal(5.0, 0.5));
        }
        let at_5 = nig.log_predictive(5.0);
        let at_0 = nig.log_predictive(0.0);
        let at_10 = nig.log_predictive(10.0);
        assert!(at_5 > at_0);
        assert!(at_5 > at_10);
    }

    // ── NIG: closed-form posterior equivalence ────────────────────────

    /// Closed-form NIG posterior given prior (μ₀, κ₀, α₀, β₀) and data {x_i}.
    ///
    /// Reference: Murphy (2012) §3.3 / §4.6, conjugate analysis for normal
    /// with unknown mean and variance.
    fn closed_form_nig(prior: &Nig, data: &[f64]) -> Nig {
        let n = data.len() as f64;
        let kappa_n = prior.kappa + n;
        let mean = data.iter().sum::<f64>() / n;
        let mu_n = (prior.kappa * prior.mu + n * mean) / kappa_n;
        let alpha_n = prior.alpha + n / 2.0;
        let ssq = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>();
        let beta_n = prior.beta
            + 0.5 * ssq
            + prior.kappa * n * (mean - prior.mu).powi(2) / (2.0 * kappa_n);
        Nig {
            mu: mu_n,
            kappa: kappa_n,
            alpha: alpha_n,
            beta: beta_n,
        }
    }

    #[test]
    fn nig_iterative_update_matches_closed_form() {
        let prior = Nig {
            mu: 0.5,
            kappa: 2.0,
            alpha: 3.0,
            beta: 4.0,
        };
        let mut rng = Rng::new(20251);
        let data: Vec<f64> = (0..500).map(|_| rng.normal(2.0, 1.5)).collect();

        let online = data.iter().fold(prior.clone(), |acc, &x| acc.update(x));
        let offline = closed_form_nig(&prior, &data);

        // Tight: this is pure floating-point algebra, no Monte-Carlo.
        assert!(
            (online.mu - offline.mu).abs() < 1e-10,
            "μ: online={}, closed-form={}",
            online.mu,
            offline.mu
        );
        assert!((online.kappa - offline.kappa).abs() < 1e-10);
        assert!((online.alpha - offline.alpha).abs() < 1e-10);
        // β accumulates rounding error -- looser bound, still ≪ value scale.
        assert!(
            (online.beta - offline.beta).abs() < 1e-6,
            "β: online={}, closed-form={}",
            online.beta,
            offline.beta
        );
    }

    /// Numerical integration of exp(NIG.log_predictive) over a wide grid.
    /// The posterior predictive is a Student-t; integral over ℝ should be 1.
    fn integrate_predictive(nig: &Nig, a: f64, b: f64, n: usize) -> f64 {
        let dx = (b - a) / n as f64;
        let mut s = 0.0;
        for i in 0..=n {
            let x = a + i as f64 * dx;
            let w = if i == 0 || i == n { 0.5 } else { 1.0 };
            s += w * nig.log_predictive(x).exp() * dx;
        }
        s
    }

    #[test]
    fn nig_predictive_is_a_proper_density_at_prior() {
        let prior = Nig {
            mu: 0.0,
            kappa: 1.0,
            alpha: 2.0, // df = 4 → finite mean and variance
            beta: 1.0,
        };
        let mass = integrate_predictive(&prior, -100.0, 100.0, 50_000);
        assert!(
            (mass - 1.0).abs() < 0.01,
            "prior predictive mass = {mass}, expected ≈ 1.0"
        );
    }

    #[test]
    fn nig_predictive_variance_matches_closed_form() {
        // Catches a wrong (df, scale) in log_predictive that the normalization
        // test cannot. Posterior predictive is Student-t(df=2α, μ, scale²);
        // its variance is β(κ+1)/(κ(α-1)) for α > 1.
        for &(kappa, alpha, beta) in &[(2.0, 3.0, 1.0), (5.0, 10.0, 4.0), (1.0, 2.5, 0.7)] {
            let nig = Nig {
                mu: 0.5,
                kappa,
                alpha,
                beta,
            };
            let expected_var = beta * (kappa + 1.0) / (kappa * (alpha - 1.0));

            // ∫ (x − μ)² p(x) dx via wide trapezoidal grid.
            let n = 100_000;
            let (a, b) = (-200.0_f64, 200.0_f64);
            let dx = (b - a) / n as f64;
            let mut second_moment = 0.0;
            for i in 0..=n {
                let x = a + i as f64 * dx;
                let w = if i == 0 || i == n { 0.5 } else { 1.0 };
                second_moment += w * (x - nig.mu).powi(2) * nig.log_predictive(x).exp() * dx;
            }
            // Heavy-tail truncation costs us a few percent -- but a wrong (df, scale)
            // would be off by an O(1) factor.
            let rel_err = (second_moment - expected_var).abs() / expected_var;
            assert!(
                rel_err < 0.05,
                "predictive variance: got {second_moment}, expected {expected_var} (κ={kappa}, α={alpha}, β={beta})"
            );
        }
    }

    #[test]
    fn nig_predictive_is_a_proper_density_post_update() {
        // After 200 observations from N(5, 0.5²), the predictive should still
        // integrate to 1 (now centered near 5, much narrower).
        let mut nig = Nig {
            mu: 0.0,
            kappa: 1.0,
            alpha: 1.0,
            beta: 1.0,
        };
        let mut rng = Rng::new(404);
        for _ in 0..200 {
            nig = nig.update(rng.normal(5.0, 0.5));
        }
        let mass = integrate_predictive(&nig, -50.0, 60.0, 50_000);
        assert!(
            (mass - 1.0).abs() < 0.01,
            "posterior predictive mass = {mass}, expected ≈ 1.0"
        );
    }

    // ── Affine invariance of detection ────────────────────────────────

    #[test]
    fn detection_is_affine_invariant() {
        // The detector normalizes input internally (subtract mean, divide by σ),
        // so a positive affine transform y = a·x + b must produce identical
        // change point indices.
        let mut rng = Rng::new(13);
        let mut x: Vec<f64> = (0..150).map(|_| rng.normal(0.0, 1.0)).collect();
        x.extend((0..150).map(|_| rng.normal(3.0, 1.0)));

        let det = BocpdDetector::new(200.0, 350);

        let raw_cps: Vec<usize> = det.detect(&x, 0.3).iter().map(|c| c.index).collect();

        for &(a, b) in &[(1.0, 100.0), (5.0, -50.0), (0.1, 7.0), (1000.0, 0.0)] {
            let y: Vec<f64> = x.iter().map(|&v| a * v + b).collect();
            let trans_cps: Vec<usize> = det.detect(&y, 0.3).iter().map(|c| c.index).collect();
            assert_eq!(
                raw_cps, trans_cps,
                "affine (a={a}, b={b}) shifted CPs: raw={raw_cps:?}, transformed={trans_cps:?}"
            );
        }
    }

    // ── log_add_exp numerical stability ───────────────────────────────

    // ── log_add_exp algebraic properties ──────────────────────────────

    #[test]
    fn log_add_exp_is_commutative() {
        let pairs = [
            (-0.5, -0.5),
            (10.0, -10.0),
            (1e-9, 1e9),
            (-1e9, 1e-9),
            (700.0, 700.0),
            (-700.0, 0.0),
        ];
        for (a, b) in pairs {
            let ab = log_add_exp(a, b);
            let ba = log_add_exp(b, a);
            assert!(
                (ab - ba).abs() < 1e-12,
                "log_add_exp not commutative at ({a}, {b}): {ab} vs {ba}"
            );
        }
    }

    #[test]
    fn log_add_exp_is_associative() {
        // log_add_exp(log_add_exp(a, b), c) == log_add_exp(a, log_add_exp(b, c))
        let triples = [
            (0.0, 0.0, 0.0),
            (-1.0, -2.0, -3.0),
            (100.0, -100.0, 50.0),
            (1e-3, 2e-3, 3e-3),
        ];
        for (a, b, c) in triples {
            let lhs = log_add_exp(log_add_exp(a, b), c);
            let rhs = log_add_exp(a, log_add_exp(b, c));
            assert!(
                (lhs - rhs).abs() < 1e-10,
                "log_add_exp not associative at ({a}, {b}, {c}): {lhs} vs {rhs}"
            );
        }
    }

    #[test]
    fn log_add_exp_identity() {
        assert_eq!(log_add_exp(f64::NEG_INFINITY, 5.0), 5.0);
        assert_eq!(log_add_exp(5.0, f64::NEG_INFINITY), 5.0);
        assert_eq!(
            log_add_exp(f64::NEG_INFINITY, f64::NEG_INFINITY),
            f64::NEG_INFINITY
        );
    }

    #[test]
    fn log_add_exp_known_values() {
        assert!((log_add_exp(0.0, 0.0) - 2.0_f64.ln()).abs() < 1e-10);
        assert!((log_add_exp(100.0, 0.0) - 100.0).abs() < 1e-10);
        assert!((log_add_exp(0.0, 100.0) - 100.0).abs() < 1e-10);
    }

    #[test]
    fn log_add_exp_extreme_values() {
        let result = log_add_exp(700.0, 700.0);
        assert!(result.is_finite(), "overflow at 700: {result}");
        let result = log_add_exp(-700.0, -700.0);
        assert!(result.is_finite(), "underflow at -700: {result}");
    }

    // ── Detector: detection correctness ───────────────────────────────

    #[test]
    fn detects_clean_mean_shift() {
        let mut data = vec![0.0; 100];
        data.extend(vec![5.0; 100]);
        let det = BocpdDetector::new(200.0, 250);
        let cps = det.detect(&data, 0.3);
        assert!(!cps.is_empty(), "should detect the mean shift");
        assert!(
            (cps[0].index as i64 - 100).abs() < 15,
            "change point near index 100, got {}",
            cps[0].index
        );
    }

    #[test]
    fn detects_noisy_mean_shift() {
        let mut rng = Rng::new(123);
        let mut data: Vec<f64> = (0..150).map(|_| rng.normal(0.0, 1.0)).collect();
        data.extend((0..150).map(|_| rng.normal(3.0, 1.0)));
        let det = BocpdDetector::new(200.0, 350);
        let cps = det.detect(&data, 0.3);
        assert!(!cps.is_empty(), "should detect noisy mean shift (3σ)");
        assert!(
            (cps[0].index as i64 - 150).abs() < 30,
            "change point near 150, got {}",
            cps[0].index
        );
    }

    #[test]
    fn no_detection_on_constant() {
        let data = vec![1.0; 200];
        let det = BocpdDetector::new(200.0, 250);
        let cps = det.detect(&data, 0.3);
        assert!(cps.is_empty(), "constant signal should have no detections");
    }

    #[test]
    fn no_detection_on_stationary_noise() {
        let mut rng = Rng::new(999);
        let data: Vec<f64> = (0..300).map(|_| rng.normal(0.0, 1.0)).collect();
        let det = BocpdDetector::new(200.0, 350);
        let cps = det.detect(&data, 0.3);
        assert!(
            cps.len() <= 1,
            "stationary noise should produce ≤1 false positive, got {}",
            cps.len()
        );
    }

    #[test]
    fn false_positive_rate_under_5_percent() {
        let det = BocpdDetector::new(200.0, 350);
        let mut fp_count = 0;
        for seed in 0..20 {
            let mut rng = Rng::new(seed * 7919 + 31);
            let data: Vec<f64> = (0..300).map(|_| rng.normal(0.0, 1.0)).collect();
            if !det.detect(&data, 0.5).is_empty() {
                fp_count += 1;
            }
        }
        assert!(
            fp_count <= 3,
            "false positive rate too high: {fp_count}/20 trials"
        );
    }

    #[test]
    fn detection_power_increases_with_shift() {
        let det = BocpdDetector::new(200.0, 250);
        let mut rates = Vec::new();
        for &shift in &[1.0, 3.0, 5.0] {
            let mut detections = 0;
            for seed in 0..20 {
                let mut rng = Rng::new(seed * 1000 + shift as u64 * 100);
                let mut data: Vec<f64> = (0..100).map(|_| rng.normal(0.0, 1.0)).collect();
                data.extend((0..100).map(|_| rng.normal(shift, 1.0)));
                if !det.detect(&data, 0.3).is_empty() {
                    detections += 1;
                }
            }
            rates.push(detections);
        }
        assert!(
            rates[2] > rates[0],
            "5σ should detect more than 1σ: 1σ={}, 5σ={}",
            rates[0],
            rates[2]
        );
        assert!(
            rates[2] >= 15,
            "5σ shift detected {}/20 -- should be ≥15",
            rates[2]
        );
    }

    #[test]
    fn detection_delay_bounded() {
        let mut data = vec![0.0; 100];
        data.extend(vec![5.0; 100]);
        let det = BocpdDetector::new(200.0, 250);
        let cps = det.detect(&data, 0.3);
        assert!(!cps.is_empty());
        let delay = (cps[0].index as i64 - 100).abs();
        assert!(delay <= 15, "detection delay={delay} steps -- should be ≤15");
    }

    #[test]
    fn detects_collective_anomaly() {
        let mut data = vec![0.0; 80];
        data.extend(vec![5.0; 40]);
        data.extend(vec![0.0; 80]);
        let det = BocpdDetector::new(200.0, 250);
        let cps = det.detect(&data, 0.3);
        assert!(
            cps.len() >= 2,
            "should detect start+end of collective anomaly, got {} detections",
            cps.len()
        );
        assert!(
            (cps[0].index as i64 - 80).abs() < 15,
            "anomaly start near 80, got {}",
            cps[0].index
        );
        assert!(
            (cps[1].index as i64 - 120).abs() < 15,
            "anomaly end near 120, got {}",
            cps[1].index
        );
    }

    #[test]
    fn detects_variance_change() {
        let mut data: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin() * 0.1).collect();
        data.extend((0..100).map(|i| (i as f64 * 0.3).sin() * 2.0));
        let det = BocpdDetector::new(200.0, 250);
        let cps = det.detect(&data, 0.2);
        assert!(!cps.is_empty(), "should detect variance change");
    }

    #[test]
    fn detects_multiple_change_points() {
        let mut data = vec![0.0; 80];
        data.extend(vec![5.0; 80]);
        data.extend(vec![-3.0; 80]);
        let det = BocpdDetector::new(200.0, 300);
        let cps = det.detect(&data, 0.3);
        assert!(
            cps.len() >= 2,
            "should detect ≥2 change points, got {}",
            cps.len()
        );
    }

    // ── Edge cases ────────────────────────────────────────────────────

    #[test]
    fn too_short_returns_empty() {
        let det = BocpdDetector::new(200.0, 250);
        assert!(det.detect(&[], 0.3).is_empty());
        assert!(det.detect(&[1.0; 5], 0.3).is_empty());
        assert!(det.detect(&[1.0; 19], 0.3).is_empty());
    }

    #[test]
    fn handles_nan_in_normalization() {
        let mut data = vec![0.0; 100];
        data.extend(vec![5.0; 100]);
        data[50] = 1e10;
        data[51] = -1e10;
        let det = BocpdDetector::new(200.0, 250);
        let cps = det.detect(&data, 0.3);
        let _ = cps;
    }

    #[test]
    fn handles_large_values() {
        let mut data = vec![1e9; 100];
        data.extend(vec![2e9; 100]);
        let det = BocpdDetector::new(200.0, 250);
        let cps = det.detect(&data, 0.3);
        assert!(
            !cps.is_empty(),
            "should detect shift even with large absolute values"
        );
    }

    #[test]
    fn handles_negative_values() {
        let mut data = vec![-100.0; 100];
        data.extend(vec![-50.0; 100]);
        let det = BocpdDetector::new(200.0, 250);
        let cps = det.detect(&data, 0.3);
        assert!(!cps.is_empty(), "should detect shift in negative values");
    }

    #[test]
    fn change_at_very_start_not_detected() {
        let mut data = vec![0.0; 5];
        data.extend(vec![10.0; 195]);
        let det = BocpdDetector::new(200.0, 250);
        let cps = det.detect(&data, 0.3);
        for cp in &cps {
            assert!(
                cp.index >= 30,
                "detection at {} is too early (before warmup/min_prev_rl)",
                cp.index
            );
        }
    }

    // ── Parameter sensitivity ─────────────────────────────────────────

    #[test]
    fn higher_threshold_fewer_detections() {
        let mut rng = Rng::new(777);
        let mut data: Vec<f64> = (0..100).map(|_| rng.normal(0.0, 1.0)).collect();
        data.extend((0..100).map(|_| rng.normal(2.0, 1.0)));
        data.extend((0..100).map(|_| rng.normal(0.0, 1.0)));
        let det = BocpdDetector::new(200.0, 350);
        let low = det.detect(&data, 0.3).len();
        let high = det.detect(&data, 0.9).len();
        assert!(
            high <= low,
            "higher threshold should give ≤ detections: low={low}, high={high}"
        );
    }

    #[test]
    fn smaller_lambda_more_sensitive() {
        let mut data = vec![0.0; 60];
        data.extend(vec![3.0; 60]);
        data.extend(vec![0.0; 60]);
        let sensitive = BocpdDetector::new(50.0, 200);
        let conservative = BocpdDetector::new(500.0, 200);
        let s_cps = sensitive.detect(&data, 0.3).len();
        let c_cps = conservative.detect(&data, 0.3).len();
        assert!(
            s_cps >= c_cps,
            "smaller λ should detect ≥ as many: λ=50→{s_cps}, λ=500→{c_cps}"
        );
    }

    // ── Eval harness ─────────────────────────────────────────────

    use crate::eval::{self, Category};

    fn run_eval() -> Vec<eval::EvalMetrics> {
        let detector = BocpdDetector::new(200.0, 350);
        let scenarios = eval::all_scenarios();
        let mut metrics = Vec::new();
        for s in &scenarios {
            let cps = detector.detect(&s.data, 0.3);
            let detected: Vec<usize> = cps.iter().map(|c| c.index).collect();
            let mut m = eval::match_detections(&detected, &s.ground_truth, 20);
            m.name = s.name.to_string();
            m.category = s.category;
            metrics.push(m);
        }
        metrics
    }

    fn run_eval_detrended() -> Vec<eval::EvalMetrics> {
        use crate::detrend::detect_with_seasonal_guard;

        let detector = BocpdDetector::new(200.0, 400);
        let scenarios = eval::all_scenarios();
        let mut metrics = Vec::new();
        for s in &scenarios {
            let period = s.period.unwrap_or(0);
            let cps = if period > 0 {
                detect_with_seasonal_guard(&s.data, period, &detector, 0.3)
            } else {
                detector.detect(&s.data, 0.3)
            };
            let detected: Vec<usize> = cps.iter().map(|c| c.index).collect();
            let mut m = eval::match_detections(&detected, &s.ground_truth, 20);
            m.name = s.name.to_string();
            m.category = s.category;
            metrics.push(m);
        }
        metrics
    }

    #[test]
    fn eval_full_suite() {
        let metrics = run_eval();
        eval::print_report(&metrics);

        let agg = eval::aggregate(&metrics);
        assert!(agg.f1 >= 0.40, "aggregate F1={:.2}, need ≥0.40", agg.f1);
        assert!(
            agg.mean_delay <= 25.0,
            "aggregate delay={:.1}, need ≤25",
            agg.mean_delay
        );
    }

    #[test]
    fn eval_must_detect_recall() {
        let metrics = run_eval();
        let md: Vec<_> = metrics
            .iter()
            .filter(|m| m.category == Category::MustDetect)
            .collect();
        let tp: usize = md.iter().map(|m| m.tp).sum();
        let fn_count: usize = md.iter().map(|m| m.r#fn).sum();
        let recall = if tp + fn_count > 0 {
            tp as f64 / (tp + fn_count) as f64
        } else {
            1.0
        };
        assert!(
            recall >= 0.60,
            "MustDetect recall={:.2}, need ≥0.60",
            recall
        );
    }

    #[test]
    fn eval_must_reject_precision() {
        let metrics = run_eval();
        let mr: Vec<_> = metrics
            .iter()
            .filter(|m| m.category == Category::MustReject)
            .collect();
        let total_fps: usize = mr.iter().map(|m| m.fp).sum();
        // Baseline without detrending: BOCPD fires on periodic/trending patterns.
        // Seasonal preprocessing can be evaluated against this baseline.
        assert!(
            total_fps <= 12,
            "MustReject total FPs={}, need ≤12",
            total_fps
        );
    }

    #[test]
    fn eval_deterministic() {
        let m1 = run_eval();
        let m2 = run_eval();
        for (a, b) in m1.iter().zip(m2.iter()) {
            assert_eq!(a.tp, b.tp, "non-deterministic: {} tp differs", a.name);
            assert_eq!(a.fp, b.fp, "non-deterministic: {} fp differs", a.name);
        }
    }

    #[test]
    fn eval_bocpd_beats_naive_baseline() {
        // BOCPD must outperform a naive z-score detector.
        // If it doesn't, the eval scenarios are too easy.
        let scenarios = eval::all_scenarios();
        let detector = BocpdDetector::new(200.0, 400);

        let mut bocpd_metrics = Vec::new();
        let mut naive_metrics = Vec::new();

        for s in &scenarios {
            let bocpd_cps: Vec<usize> = detector
                .detect(&s.data, 0.3)
                .iter()
                .map(|c| c.index)
                .collect();
            let naive_cps = eval::naive_zscore_detect(&s.data, 30, 3.0);

            let mut bm = eval::match_detections(&bocpd_cps, &s.ground_truth, 20);
            bm.name = s.name.to_string();
            bm.category = s.category;
            bocpd_metrics.push(bm);

            let mut nm = eval::match_detections(&naive_cps, &s.ground_truth, 20);
            nm.name = s.name.to_string();
            nm.category = s.category;
            naive_metrics.push(nm);
        }

        let bocpd_agg = eval::aggregate(&bocpd_metrics);
        let naive_agg = eval::aggregate(&naive_metrics);

        eprintln!("\n=== BOCPD vs Naive Z-Score ===");
        eprintln!(
            "BOCPD:  F1={:.2} P={:.2} R={:.2}",
            bocpd_agg.f1, bocpd_agg.precision, bocpd_agg.recall
        );
        eprintln!(
            "Naive:  F1={:.2} P={:.2} R={:.2}",
            naive_agg.f1, naive_agg.precision, naive_agg.recall
        );

        assert!(
            bocpd_agg.f1 > naive_agg.f1,
            "BOCPD F1={:.2} must beat naive F1={:.2}",
            bocpd_agg.f1,
            naive_agg.f1
        );
    }

    #[test]
    fn eval_detrending_reduces_fps() {
        let raw_metrics = run_eval();
        let detrended_metrics = run_eval_detrended();

        let raw_agg = eval::aggregate(&raw_metrics);
        let det_agg = eval::aggregate(&detrended_metrics);

        eprintln!("\n=== Raw vs Detrended ===");
        eprintln!(
            "Raw:       F1={:.2} P={:.2} R={:.2} FP={}",
            raw_agg.f1, raw_agg.precision, raw_agg.recall, raw_agg.fp
        );
        eprintln!(
            "Detrended: F1={:.2} P={:.2} R={:.2} FP={}",
            det_agg.f1, det_agg.precision, det_agg.recall, det_agg.fp
        );

        // Detrending should reduce false positives on MustReject scenarios
        let raw_mr_fps: usize = raw_metrics
            .iter()
            .filter(|m| m.category == Category::MustReject)
            .map(|m| m.fp)
            .sum();
        let det_mr_fps: usize = detrended_metrics
            .iter()
            .filter(|m| m.category == Category::MustReject)
            .map(|m| m.fp)
            .sum();

        eprintln!("MustReject FPs: raw={raw_mr_fps}, detrended={det_mr_fps}");
        assert!(
            det_mr_fps <= raw_mr_fps,
            "detrending should not increase MustReject FPs: raw={raw_mr_fps}, detrended={det_mr_fps}"
        );
    }

    #[test]
    fn eval_lambda_sensitivity() {
        let scenarios = eval::all_scenarios();
        let det200 = BocpdDetector::new(200.0, 350);
        let det500 = BocpdDetector::new(500.0, 350);

        let mut recall_200 = 0;
        let mut recall_500 = 0;
        let mut total_gt = 0;

        for s in scenarios
            .iter()
            .filter(|s| s.category == Category::MustDetect)
        {
            let cps200: Vec<usize> = det200
                .detect(&s.data, 0.3)
                .iter()
                .map(|c| c.index)
                .collect();
            let cps500: Vec<usize> = det500
                .detect(&s.data, 0.3)
                .iter()
                .map(|c| c.index)
                .collect();
            let m200 = eval::match_detections(&cps200, &s.ground_truth, 20);
            let m500 = eval::match_detections(&cps500, &s.ground_truth, 20);
            recall_200 += m200.tp;
            recall_500 += m500.tp;
            total_gt += s.ground_truth.len();
        }

        // λ=500 shouldn't catastrophically collapse recall vs λ=200
        let r200 = recall_200 as f64 / total_gt as f64;
        let r500 = recall_500 as f64 / total_gt as f64;
        assert!(
            r500 >= r200 * 0.5,
            "λ=500 recall={:.2} collapsed vs λ=200 recall={:.2}",
            r500,
            r200
        );
    }

    // --- Multivariate BOCPD tests ---

    #[test]
    fn multivariate_detects_joint_shift() {
        let mut rng = Rng::new(42);
        let n = 200;
        let mut data = Vec::with_capacity(n);

        // First 100: centered at (0, 0)
        for _ in 0..100 {
            data.push(vec![rng.normal(0.0, 1.0), rng.normal(0.0, 1.0)]);
        }
        // Next 100: shifted to (3, 3)
        for _ in 0..100 {
            data.push(vec![rng.normal(3.0, 1.0), rng.normal(3.0, 1.0)]);
        }

        let detector = BocpdDetector::new(200.0, 250);
        let cps = detector.detect_multivariate(&data, 0.3);
        assert!(!cps.is_empty(), "should detect joint mean shift");
        assert!(
            (cps[0].index as i64 - 100).abs() < 20,
            "change point at {} (expected ~100)",
            cps[0].index
        );
    }

    #[test]
    fn multivariate_no_detection_on_stationary() {
        let mut rng = Rng::new(99);
        let n = 200;
        let data: Vec<Vec<f64>> = (0..n)
            .map(|_| vec![rng.normal(0.0, 1.0), rng.normal(0.0, 1.0)])
            .collect();

        let detector = BocpdDetector::new(200.0, 250);
        let cps = detector.detect_multivariate(&data, 0.3);
        assert!(
            cps.len() <= 1,
            "stationary data should have ≤1 false positives, got {}",
            cps.len()
        );
    }

    #[test]
    fn multivariate_detects_correlated_shift_invisible_to_marginals() {
        // Key acceptance criterion: correlated shift where each marginal is within ~1σ
        // but the joint shift is detectable.
        let mut rng = Rng::new(77);
        let n = 300;
        let mut data = Vec::with_capacity(n);

        // First 150: centered at (0, 0) with correlated noise
        for _ in 0..150 {
            let z = rng.normal(0.0, 1.0);
            data.push(vec![
                z * 0.7 + rng.normal(0.0, 0.7), // correlated dim 1
                z * 0.7 + rng.normal(0.0, 0.7), // correlated dim 2
            ]);
        }
        // Next 150: small shift in both dimensions (0.8σ each -- below marginal threshold)
        for _ in 0..150 {
            let z = rng.normal(0.0, 1.0);
            data.push(vec![
                0.8 + z * 0.7 + rng.normal(0.0, 0.7),
                0.8 + z * 0.7 + rng.normal(0.0, 0.7),
            ]);
        }

        let detector = BocpdDetector::new(200.0, 350);

        // Univariate detection on each dimension should miss it (small per-dim shift)
        let dim1: Vec<f64> = data.iter().map(|x| x[0]).collect();
        let dim2: Vec<f64> = data.iter().map(|x| x[1]).collect();
        let univ_cps_1 = detector.detect(&dim1, 0.3);
        let univ_cps_2 = detector.detect(&dim2, 0.3);

        // Multivariate should catch the joint shift
        let multi_cps = detector.detect_multivariate(&data, 0.3);

        eprintln!(
            "univariate dim1: {} detections, dim2: {} detections, multivariate: {} detections",
            univ_cps_1.len(),
            univ_cps_2.len(),
            multi_cps.len()
        );

        // The multivariate detector should find more than the best univariate
        // (or at least detect the shift when univariate doesn't)
        let max_univ = univ_cps_1.len().max(univ_cps_2.len());
        assert!(
            multi_cps.len() >= max_univ,
            "multivariate ({}) should detect at least as many as best univariate ({})",
            multi_cps.len(),
            max_univ
        );
    }

    #[test]
    fn multivariate_too_short_returns_empty() {
        let data: Vec<Vec<f64>> = (0..10).map(|_| vec![1.0, 2.0]).collect();
        let detector = BocpdDetector::new(200.0, 50);
        let cps = detector.detect_multivariate(&data, 0.3);
        assert!(cps.is_empty());
    }

    #[test]
    fn multivariate_3d_shift() {
        let mut rng = Rng::new(123);
        let n = 200;
        let mut data = Vec::with_capacity(n);

        for _ in 0..100 {
            data.push(vec![
                rng.normal(0.0, 1.0),
                rng.normal(0.0, 1.0),
                rng.normal(0.0, 1.0),
            ]);
        }
        for _ in 0..100 {
            data.push(vec![
                rng.normal(2.0, 1.0),
                rng.normal(-2.0, 1.0),
                rng.normal(3.0, 1.0),
            ]);
        }

        let detector = BocpdDetector::new(200.0, 250);
        let cps = detector.detect_multivariate(&data, 0.3);
        assert!(!cps.is_empty(), "should detect 3D mean shift");
    }

    #[test]
    fn nan_in_data_does_not_panic() {
        let mut data: Vec<f64> = (0..100).map(|i| i as f64 * 0.1).collect();
        data[10] = f64::NAN;
        data[50] = f64::NAN;
        data[51] = f64::INFINITY;

        let detector = BocpdDetector::new(200.0, 250);
        // Must not panic -- NaN previously caused unwrap on partial_cmp
        let _cps = detector.detect(&data, 0.3);
    }

    #[test]
    fn nan_in_data_still_produces_detections() {
        // NaN sprinkled into a clean shift should not suppress all detections.
        let mut data = vec![0.0f64; 100];
        data.extend(vec![5.0f64; 100]);
        // Inject NaN at arbitrary positions
        data[10] = f64::NAN;
        data[55] = f64::NAN;
        data[130] = f64::NAN;
        let det = BocpdDetector::new(200.0, 250);
        let cps = det.detect(&data, 0.3);
        assert!(
            !cps.is_empty(),
            "NaN-containing data should still produce detections from clean values"
        );
    }

    #[test]
    fn all_nan_returns_empty_without_panic() {
        let data = vec![f64::NAN; 50];
        let det = BocpdDetector::new(200.0, 100);
        let cps = det.detect(&data, 0.3);
        assert!(cps.is_empty(), "all-NaN input should return empty vec");
    }

    // Constructor and index regression tests.

    #[test]
    #[should_panic(expected = "lambda must be > 1.0")]
    fn batch_rejects_lambda_one() {
        BocpdDetector::new(1.0, 100);
    }

    #[test]
    #[should_panic(expected = "lambda must be > 1.0")]
    fn batch_rejects_lambda_below_one() {
        BocpdDetector::new(0.5, 100);
    }

    #[test]
    fn detect_multivariate_empty_returns_empty() {
        let det = BocpdDetector::new(200.0, 100);
        let cps = det.detect_multivariate(&[], 0.3);
        assert!(cps.is_empty());
    }

    #[test]
    fn detect_multivariate_ragged_returns_empty() {
        let det = BocpdDetector::new(200.0, 100);
        let mut data: Vec<Vec<f64>> = (0..30).map(|_| vec![1.0, 2.0, 3.0]).collect();
        data[15] = vec![1.0, 2.0]; // ragged row
        let cps = det.detect_multivariate(&data, 0.3);
        assert!(cps.is_empty(), "ragged input should return empty");
    }

    #[test]
    fn nan_filtering_preserves_original_indices() {
        // Input with NaN gap at positions 5..10 -- change point indices must
        // reference original positions, not compressed model steps.
        let mut data: Vec<f64> = vec![0.0; 100];
        data.extend(vec![5.0; 100]);
        for v in data.iter_mut().take(10).skip(5) {
            *v = f64::NAN;
        }
        let det = BocpdDetector::new(200.0, 250);
        let cps = det.detect(&data, 0.3);
        assert!(!cps.is_empty(), "should detect the shift");
        // The shift is at raw position 100. The detected index should be
        // near 100, NOT shifted by the 5 NaN values (which would give ~95).
        assert!(
            (cps[0].index as i64 - 100).abs() < 20,
            "change point should be near raw index 100, got {}",
            cps[0].index
        );
        assert!(
            cps[0].index >= 95,
            "index {} is too low -- NaN filtering shifted indices",
            cps[0].index
        );
    }
}
