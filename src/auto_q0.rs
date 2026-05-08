//! Auto-tune the Chen & Wu (2025) anomaly-end prior `q_0`.
//!
//! Paper § 5 / eq. 14: for the joint detector to avoid the "frequent
//! spurious anomaly alarms" failure mode the user picks `q_0` such that
//!
//! ```text
//!   p_0 · q_0 · Σ_{i=0}^{Δt-1} (1-q_0)^i · (1-p_0)^{Δt-1-i}
//!   ────────────────────────────────────────────────────── < λ_a
//!   p_0 · q_0 · Σ ...                +   p_0 · (1-q_0)^{Δt}
//! ```
//!
//! The LHS is monotonically increasing in `q_0` (paper supplementary),
//! so the feasible set is `[0, q_0^*)` with the upper bound `q_0^*`
//! found by line search (root of LHS = λ_a). No closed-form solution;
//! bisection converges in `~64` steps to machine precision. cesura
//! exposes the picker as `q0_upper_bound` and a public convenience
//! `auto_q0` paralleling [`crate::auto_beta::auto_beta`].
//!
//! `eq. 14`'s constraint is on the *hyperparameter triple*
//! `(p_0, λ_a, Δt)` -- no data warmup needed. A data-driven flavour
//! `with_auto_q0_from_warmup(&[f64])` is deliberately deferred until
//! empirical signal demands it.

/// LHS of eq. 14 evaluated at `(p_0, q_0, Δt)`. The `p_0` factor in
/// numerator and one denominator term algebraically cancels; this
/// implementation uses the simplified form
/// `q · S / (q · S + (1-q)^Δt)` where
/// `S = Σ_{i=0}^{Δt-1} (1-q)^i · (1-p_0)^{Δt-1-i}`. Result is well-
/// defined for all `(p_0, q_0) ∈ [0, 1) × [0, 1)` and any `Δt ≥ 1`.
fn lhs_eq14(p0: f64, q0: f64, delta_t: usize) -> f64 {
    let one_minus_q = 1.0 - q0;
    let one_minus_p = 1.0 - p0;
    let mut sum = 0.0_f64;
    let mut q_pow = 1.0_f64;
    let mut p_pow = one_minus_p.powi((delta_t - 1) as i32);
    let p_inv = if one_minus_p > 0.0 { 1.0 / one_minus_p } else { 0.0 };
    for _ in 0..delta_t {
        sum += q_pow * p_pow;
        q_pow *= one_minus_q;
        p_pow *= p_inv;
    }
    let num = q0 * sum;
    let denom = num + one_minus_q.powi(delta_t as i32);
    if denom <= 0.0 {
        return 0.0;
    }
    num / denom
}

/// Largest `q_0 ∈ [0, 1)` such that paper eq. 14's LHS is strictly
/// less than `λ_a` for the given `(p_0, Δt)`. Bisection over `[0, 1)`.
///
/// Edge cases:
/// - `λ_a ≤ 0`: no `q_0` strictly satisfies the inequality. Returns
///   `0.0` (the closest feasible value -- corresponds to "never emit
///   collective anomalies").
/// - `λ_a ≥ 1`: trivially satisfied. Returns `1.0 - ε` (just inside
///   the constructor's `[0, 1)` assertion).
/// - `p_0 ∉ [0, 1)` or `Δt < 1`: out-of-range inputs return `0.0`.
pub(crate) fn q0_upper_bound(p0: f64, lambda_a: f64, delta_t: usize) -> f64 {
    if !(0.0..1.0).contains(&p0) || delta_t < 1 {
        return 0.0;
    }
    if !lambda_a.is_finite() || lambda_a <= 0.0 {
        return 0.0;
    }
    if lambda_a >= 1.0 {
        return 1.0 - f64::EPSILON;
    }
    let mut lo = 0.0_f64;
    let mut hi = 1.0_f64 - f64::EPSILON;
    if lhs_eq14(p0, hi, delta_t) < lambda_a {
        return hi;
    }
    for _ in 0..64 {
        let mid = 0.5 * (lo + hi);
        let val = lhs_eq14(p0, mid, delta_t);
        if val < lambda_a {
            lo = mid;
        } else {
            hi = mid;
        }
        if (hi - lo) < 1e-12 {
            break;
        }
    }
    lo
}

/// One-shot convenience: pick the largest feasible `q_0` for the
/// hyperparameter triple `(p_0, λ_a, Δt)` per paper eq. 14. Mirrors
/// the shape of [`crate::auto_beta::auto_beta`].
pub fn auto_q0(p0: f64, lambda_a: f64, delta_t: usize) -> f64 {
    q0_upper_bound(p0, lambda_a, delta_t)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lhs_monotone_in_q0() {
        // Paper supplementary: LHS increases monotonically in q_0.
        // Sample at fixed (p_0=0.1, Δt=4) and verify across a grid.
        let prev = lhs_eq14(0.1, 0.0, 4);
        assert!((prev - 0.0).abs() < 1e-12, "LHS(q=0) should be 0, got {prev}");
        let mut prev = 0.0;
        for k in 1..=99 {
            let q = k as f64 / 100.0;
            let v = lhs_eq14(0.1, q, 4);
            assert!(
                v > prev - 1e-12,
                "LHS not monotone at q={q}: prev={prev} v={v}"
            );
            prev = v;
        }
    }

    #[test]
    fn lhs_at_endpoints() {
        assert!(lhs_eq14(0.1, 0.0, 4).abs() < 1e-15);
        // q → 1 should drive LHS → 1 (anomaly always closes ⇒ posterior 1).
        let near_one = lhs_eq14(0.1, 1.0 - 1e-9, 4);
        assert!(near_one > 0.999, "LHS(q≈1) = {near_one}, expected ≈ 1");
    }

    #[test]
    fn upper_bound_at_paper_section_6_1_settings() {
        // Paper § 6.1 hyperparameters: p_0 = 0.1, λ_a = 0.5, Δt = 4.
        // Hand-computed bound q_0^* ≈ 0.1785 (LHS crosses 0.5 between
        // q_0 = 0.17 and q_0 = 0.18). The paper itself chose q_0 = 0.20
        // by reading off Figure 4(a); that's slightly above the strict
        // bound (LHS(0.20) ≈ 0.546 > 0.5) -- a deliberate permissive
        // choice the paper notes "exceeds the true change frequencies"
        // in § 6.4 sensitivity analysis. The picker returns the strict
        // upper bound, so it lands below 0.20 by design.
        let q = q0_upper_bound(0.1, 0.5, 4);
        assert!(
            (0.17..0.19).contains(&q),
            "q0_upper_bound(0.1, 0.5, 4) = {q}, expected ≈ 0.1785"
        );
        // Verify it's at the constraint surface: LHS at the returned
        // value should be ≈ λ_a from below (bisection's `lo` invariant).
        let lhs = lhs_eq14(0.1, q, 4);
        assert!(
            lhs < 0.5,
            "returned q_0={q} violates eq. 14: LHS={lhs} ≥ λ_a=0.5"
        );
        assert!(
            (0.5 - lhs).abs() < 1e-6,
            "returned q_0={q} not at constraint surface: LHS={lhs}, expected ≈ 0.5"
        );
    }

    #[test]
    fn upper_bound_grid_consistency() {
        // For a grid of (p_0, λ_a, Δt) pick the bound and verify two
        // properties: (a) LHS at q_0^* is ≤ λ_a (feasibility); (b) LHS
        // converges to within ε of λ_a (bisection landed on the surface).
        let p0_grid = [0.01, 0.05, 0.1, 0.25];
        let lambda_grid = [0.3, 0.5, 0.7, 0.9];
        let delta_t_grid = [2usize, 4, 8, 16];
        for &p0 in &p0_grid {
            for &la in &lambda_grid {
                for &dt in &delta_t_grid {
                    let q = q0_upper_bound(p0, la, dt);
                    let lhs = lhs_eq14(p0, q, dt);
                    assert!(
                        lhs < la + 1e-9,
                        "infeasible at (p_0={p0}, λ_a={la}, Δt={dt}): \
                         q_0={q}, LHS={lhs}, λ_a={la}"
                    );
                    // Bisection landed near the surface within tolerance.
                    // Tighter than the constructor's f64 step but loose
                    // enough to absorb log/exp roundoff at extreme grid
                    // corners.
                    assert!(
                        (la - lhs).abs() < 1e-6 || q == 1.0 - f64::EPSILON,
                        "bisection drifted at (p_0={p0}, λ_a={la}, Δt={dt}): \
                         q_0={q}, LHS={lhs} vs λ_a={la}"
                    );
                }
            }
        }
    }

    #[test]
    fn rejects_infeasible_inputs() {
        // λ_a ≤ 0: no q_0 satisfies LHS < λ_a (LHS ≥ 0). Picker returns 0.
        assert_eq!(q0_upper_bound(0.1, 0.0, 4), 0.0);
        assert_eq!(q0_upper_bound(0.1, -0.1, 4), 0.0);
        // λ_a ≥ 1: trivially satisfied. Picker returns 1 - ε.
        let q = q0_upper_bound(0.1, 1.0, 4);
        assert!(q > 0.99, "q0_upper_bound at λ_a=1 should be near 1, got {q}");
        // Non-finite λ_a: degenerate, return 0.
        assert_eq!(q0_upper_bound(0.1, f64::NAN, 4), 0.0);
        // `p_0 ∈ [0, 1)` is the constructor's accepted range; out-of-range
        // p_0 returns the conservative 0. p_0 = 0 itself is allowed.
        assert_eq!(q0_upper_bound(1.0, 0.5, 4), 0.0);
        let q_p0_zero = q0_upper_bound(0.0, 0.5, 4);
        assert!(
            q_p0_zero.is_finite() && (0.0..1.0).contains(&q_p0_zero),
            "p_0=0 should yield a finite q_0 ∈ [0,1), got {q_p0_zero}"
        );
        // Degenerate corner (p_0=0.5, λ_a=0.01, Δt=1):
        // LHS=q at Δt=1, so the bound is at q ≈ 0.01. No panic, finite result.
        let q_corner = q0_upper_bound(0.5, 0.01, 1);
        assert!(
            q_corner.is_finite() && (0.0..0.02).contains(&q_corner),
            "corner (0.5, 0.01, 1) should yield q ≈ 0.01, got {q_corner}"
        );
    }
}
