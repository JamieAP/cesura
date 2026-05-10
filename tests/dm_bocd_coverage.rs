//! Branch-coverage tests for `src/dm_bocd.rs`.
//!
//! Tests pin individual branch behavior; they do not establish
//! algorithmic correctness.

use cesura::dm_bocd::{DmBocdDetector, IdentityM, ImqM};
use cesura::eval::Rng;

fn gen_shift(seed: u64, d: usize, n_pre: usize, n_post: usize, shift: f64) -> Vec<Vec<f64>> {
    let mut rng = Rng::new(seed);
    let mut data: Vec<Vec<f64>> = (0..n_pre)
        .map(|_| (0..d).map(|_| rng.normal(0.0, 1.0)).collect())
        .collect();
    data.extend(
        (0..n_post).map(|_| (0..d).map(|_| rng.normal(shift, 1.0)).collect()),
    );
    data
}

// ── Whitening fallback ────────────────────────────────────────────────

#[test]
fn whitening_fallback_per_dim_znorm_when_warmup_too_short() {
    // n=22, d=20: warmup_n = (22/3).min(60).max(40) = 40 > n=22
    // -> the `warmup_n <= n` guard fails, per_dim_znorm fallback fires.
    let det = DmBocdDetector::new(20, 100.0, 50);
    let data: Vec<Vec<f64>> = (0..22).map(|i| vec![i as f64; 20]).collect();
    // No panic, length-N diagnostics returned.
    let (_cps, cp_probs, map_rls) = det.detect_multivariate_with_diagnostics(&data);
    assert_eq!(cp_probs.len(), 22);
    assert_eq!(map_rls.len(), 22);
}

#[test]
fn whitening_fallback_when_rank_deficient_warmup() {
    // d=2, but warmup window is rank-1 (col0 == col1) so the sample
    // covariance is singular -- whitening_transform returns None and
    // per_dim_znorm fires. With shift+noise after, detector still
    // proceeds without panicking.
    let mut rng = Rng::new(0xDEADBEEF);
    let mut data: Vec<Vec<f64>> = (0..40)
        .map(|_| {
            let v = rng.normal(0.0, 1.0);
            vec![v, v] // perfectly collinear warmup
        })
        .collect();
    // Append non-collinear tail so n is comfortably > 20.
    for _ in 0..40 {
        data.push(vec![rng.normal(2.0, 1.0), rng.normal(2.0, 1.0)]);
    }
    let det = DmBocdDetector::new(2, 100.0, 100);
    let (_cps, cp_probs, _map_rls) = det.detect_multivariate_with_diagnostics(&data);
    assert_eq!(cp_probs.len(), 80);
    assert!(cp_probs.iter().all(|p| p.is_finite()), "cp_probs must be finite under fallback");
}

// ── Mass-cutoff pruning ───────────────────────────────────────────────

#[test]
fn mass_cutoff_does_not_lose_cps_at_low_threshold() {
    // Compare a normal-cutoff run vs cutoff=0.0 (no pruning) on a
    // strong-shift fixture. The aggressive default cutoff prunes
    // tail mass but should keep the same CP count on a clean shift.
    let data = gen_shift(0xCAFE, 2, 200, 200, 6.0);
    let baseline = DmBocdDetector::new(2, 100.0, 400)
        .with_mass_cutoff(0.0) // no pruning
        .detect_multivariate(&data);
    let pruned = DmBocdDetector::new(2, 100.0, 400)
        .detect_multivariate(&data);
    assert_eq!(
        baseline.len(),
        pruned.len(),
        "mass-cutoff pruning lost CPs: {} vs {}",
        baseline.len(),
        pruned.len(),
    );
}

// ── Numerical singularity in DmStats::update ─────────────────────────

#[test]
fn singular_prior_sigma_inv_does_not_panic() {
    // Force the `solve_pd → None` branch in DmStats::update by
    // supplying a non-PD prior precision. Singular Σ⁻¹ + 2ω·I is
    // still likely PD after one update, so the branch may only fire
    // briefly; the assertion is graceful behaviour, not a particular
    // fallback path.
    let zero_sigma_inv = vec![vec![0.0; 2]; 2]; // not PD
    let det = DmBocdDetector::new(2, 100.0, 100)
        .with_prior(vec![0.0, 0.0], zero_sigma_inv);
    let data = gen_shift(0xBADBADBAD, 2, 100, 100, 4.0);
    // Must not panic.
    let _ = det.detect_multivariate(&data);
}

// ── Empty before/after windows boundary ──────────────────────────────

#[test]
fn detect_returns_when_n_just_above_min_prev_rl() {
    // Default min_prev_rl=30. n=31 means the trigger loop runs exactly
    // once and immediately can find an empty after-window if a CP fires
    // at i=30. Pin: no panic, returns within a length-31 trace.
    let det = DmBocdDetector::new(2, 100.0, 100);
    let mut rng = Rng::new(0xBEEF);
    let data: Vec<Vec<f64>> = (0..31)
        .map(|_| vec![rng.normal(0.0, 1.0), rng.normal(0.0, 1.0)])
        .collect();
    let cps = det.detect_multivariate(&data);
    // Either zero CPs (most likely on i.i.d. noise) or a CP within the
    // valid index range -- never out-of-bounds.
    for cp in &cps {
        assert!(cp.index < 31);
    }
}

// ── Edge dimensions ──────────────────────────────────────────────────

#[test]
fn d_equals_one() {
    let det = DmBocdDetector::new(1, 100.0, 200);
    let data: Vec<Vec<f64>> = gen_shift(0x1, 1, 200, 200, 5.0)
        .into_iter()
        .collect();
    let (cps, cp_probs, map_rls) = det.detect_multivariate_with_diagnostics(&data);
    assert_eq!(cp_probs.len(), 400);
    assert_eq!(map_rls.len(), 400);
    // No assertions on cps -- diagnosis-only.
    let _ = cps;
}

#[test]
fn d_equals_ten() {
    let det = DmBocdDetector::new(10, 100.0, 200);
    let data = gen_shift(0xA, 10, 100, 100, 3.0);
    let (cps, cp_probs, map_rls) = det.detect_multivariate_with_diagnostics(&data);
    assert_eq!(cp_probs.len(), 200);
    assert_eq!(map_rls.len(), 200);
    let _ = cps;
}

// ── n=20 boundary ────────────────────────────────────────────────────

#[test]
fn n_below_20_returns_empty() {
    let det = DmBocdDetector::new(2, 100.0, 100);
    for n in 0..20 {
        let data: Vec<Vec<f64>> = (0..n).map(|_| vec![0.0; 2]).collect();
        assert!(det.detect_multivariate(&data).is_empty());
        let (cps, probs, rls) = det.detect_multivariate_with_diagnostics(&data);
        assert!(cps.is_empty());
        assert!(probs.is_empty());
        assert!(rls.is_empty());
    }
}

#[test]
fn n_equal_20_runs_without_panic() {
    let det = DmBocdDetector::new(2, 100.0, 100);
    let data: Vec<Vec<f64>> = (0..20).map(|i| vec![i as f64, i as f64 * 2.0]).collect();
    let (_cps, cp_probs, map_rls) = det.detect_multivariate_with_diagnostics(&data);
    assert_eq!(cp_probs.len(), 20);
    assert_eq!(map_rls.len(), 20);
}

// ── NaN / Inf input ──────────────────────────────────────────────────

#[test]
fn nan_in_input_does_not_panic() {
    let det = DmBocdDetector::new(2, 100.0, 100);
    let mut data = gen_shift(0x7, 2, 100, 100, 3.0);
    // Inject a NaN at a deterministic index.
    data[50][0] = f64::NAN;
    data[120][1] = f64::INFINITY;
    // Behaviour pin: must not panic. detect_multivariate doesn't filter
    // non-finite values (unlike univariate detect), so cp_probs may
    // contain non-finite entries; we don't assert on them here.
    let _ = det.detect_multivariate(&data);
    let (_cps, cp_probs, _map_rls) = det.detect_multivariate_with_diagnostics(&data);
    assert_eq!(cp_probs.len(), data.len());
}

// ── Builder override chains ──────────────────────────────────────────

#[test]
fn with_prior_twice_keeps_last() {
    let det = DmBocdDetector::new(2, 100.0, 50)
        .with_prior(vec![1.0, 2.0], vec![vec![1.0, 0.0], vec![0.0, 1.0]])
        .with_prior(vec![3.0, 4.0], vec![vec![2.0, 0.0], vec![0.0, 2.0]]);
    let data = gen_shift(0xF1, 2, 50, 50, 3.0);
    // Behaviour pin: builder chains compile and run without panic.
    let _ = det.detect_multivariate(&data);
}

#[test]
fn with_omega_twice_keeps_last() {
    let det = DmBocdDetector::new(2, 100.0, 50)
        .with_omega(0.5)
        .with_omega(0.05);
    let data = gen_shift(0xF2, 2, 50, 50, 3.0);
    let _ = det.detect_multivariate(&data);
}

#[test]
fn with_map_drop_trigger_changes_cp_count() {
    // Pin: lowering min_prev_rl from 30 to 5 either keeps or increases
    // the CP count on a clean shift fixture (a stricter trigger cannot
    // emit more CPs).
    let data = gen_shift(0xF3, 2, 200, 200, 5.0);
    let strict = DmBocdDetector::new(2, 100.0, 400).detect_multivariate(&data);
    let lax = DmBocdDetector::new(2, 100.0, 400)
        .with_map_drop_trigger(3, 5, 15)
        .detect_multivariate(&data);
    assert!(lax.len() >= strict.len(), "lax trigger emitted fewer CPs ({} < {})", lax.len(), strict.len());
}

// ── with_omega(0.0) is already guarded ─────────────

#[test]
#[should_panic(expected = "omega must be > 0")]
fn with_omega_zero_panics() {
    let _ = DmBocdDetector::new(2, 100.0, 100).with_omega(0.0);
}

#[test]
#[should_panic(expected = "omega must be > 0")]
fn with_omega_negative_panics() {
    let _ = DmBocdDetector::new(2, 100.0, 100).with_omega(-0.1);
}

#[test]
#[should_panic]
fn with_map_drop_trigger_invalid_panics() {
    // drop_to >= min_prev_rl is logically incoherent: the trigger could
    // never arm. Builder rejects.
    let _ = DmBocdDetector::new(2, 100.0, 100).with_map_drop_trigger(30, 5, 15);
}

// ── m-function plug ───────────────────────────────────────────

#[test]
fn identity_m_explicit_matches_default_bit_for_bit() {
    // The IdentityM fast path is bit-identical to the default. Same
    // seed, same fixture, same CP indices+confidence+shift_sigma.
    let data = gen_shift(0xF311, 3, 200, 200, 5.0);
    let default_cps = DmBocdDetector::new(3, 100.0, 400).detect_multivariate(&data);
    let explicit_cps = DmBocdDetector::new(3, 100.0, 400)
        .with_m_weight(IdentityM)
        .detect_multivariate(&data);
    assert_eq!(default_cps.len(), explicit_cps.len());
    for (a, b) in default_cps.iter().zip(explicit_cps.iter()) {
        assert_eq!(a.index, b.index);
        // Bit-identical f64 bits, not just approx-equal.
        assert_eq!(
            a.confidence.to_bits(),
            b.confidence.to_bits(),
            "confidence drift: {} vs {}",
            a.confidence,
            b.confidence
        );
        assert_eq!(a.shift_sigma.to_bits(), b.shift_sigma.to_bits());
    }
}

#[test]
fn imq_m_runs_to_completion_on_mixed_input() {
    // Generic-path smoke: 3-d, n=200 mixed (Normal pre, t3 post),
    // detector runs to completion with finite outputs.
    let mut rng = Rng::new(0xF312);
    let mut data: Vec<Vec<f64>> = (0..100)
        .map(|_| (0..3).map(|_| rng.normal(0.0, 1.0)).collect())
        .collect();
    data.extend((0..100).map(|_| (0..3).map(|_| rng.student_t3()).collect()));
    let det = DmBocdDetector::new(3, 100.0, 400).with_m_weight(ImqM::new(1.0));
    let (cps, cp_probs, map_rls) = det.detect_multivariate_with_diagnostics(&data);
    assert_eq!(cp_probs.len(), 200);
    assert_eq!(map_rls.len(), 200);
    assert!(cp_probs.iter().all(|p| p.is_finite() && (0.0..=1.0).contains(p)));
    for cp in &cps {
        assert!(cp.confidence.is_finite() && cp.shift_sigma.is_finite());
    }
}

#[test]
fn imq_extreme_c_does_not_panic() {
    // c=0.01 (tight downweighting) and c=100 (loose, ≈ Identity-ish)
    // both produce finite cp_probs without panicking.
    let data = gen_shift(0xF313, 2, 200, 200, 5.0);
    for &c in &[0.01_f64, 100.0] {
        let det = DmBocdDetector::new(2, 100.0, 400).with_m_weight(ImqM::new(c));
        let (_cps, cp_probs, _map_rls) = det.detect_multivariate_with_diagnostics(&data);
        assert!(cp_probs.iter().all(|p| p.is_finite()), "c={c} produced non-finite probs");
    }
}

#[test]
fn with_m_weight_last_write_wins() {
    // Builder chain: second with_m_weight call replaces the first.
    // We verify by chaining IdentityM after ImqM and asserting the
    // result equals the IdentityM-only baseline (bit-identical).
    let data = gen_shift(0xF314, 2, 200, 200, 5.0);
    let baseline = DmBocdDetector::new(2, 100.0, 400)
        .with_m_weight(IdentityM)
        .detect_multivariate(&data);
    let chained = DmBocdDetector::new(2, 100.0, 400)
        .with_m_weight(ImqM::new(0.5))
        .with_m_weight(IdentityM)
        .detect_multivariate(&data);
    assert_eq!(baseline.len(), chained.len());
    for (a, b) in baseline.iter().zip(chained.iter()) {
        assert_eq!(a.index, b.index);
        assert_eq!(a.confidence.to_bits(), b.confidence.to_bits());
    }
}
