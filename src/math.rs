//! Math primitives shared across detector modules.

use std::f64::consts::PI;

/// Log-gamma via Lanczos approximation (g=7, n=9).
// Lanczos approximation coefficients require exact precision from reference implementation
#[allow(clippy::excessive_precision, clippy::inconsistent_digit_grouping)]
pub(crate) fn lgamma(x: f64) -> f64 {
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
pub(crate) fn student_t_lpdf(x: f64, df: f64, loc: f64, scale: f64) -> f64 {
    let z = (x - loc) / scale;
    lgamma(0.5 * (df + 1.0))
        - lgamma(0.5 * df)
        - 0.5 * (df * PI).ln()
        - scale.ln()
        - 0.5 * (df + 1.0) * (1.0 + z * z / df).ln()
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
}
