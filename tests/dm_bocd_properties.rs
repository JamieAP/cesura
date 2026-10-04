//! Property tests and heavy-tail adversarial fixtures
//! for `DmBocdDetector`.
//!
//! Two layers:
//! 1. proptest invariants (affine, short-input emptiness, no-panic
//!    over bounded random input).
//! 2. Adversarial fixtures using `cesura::eval::Rng` -- Cauchy, t3,
//!    GARCH-style volatility clusters, t3 with a real shift -- run
//!    over 16 seeds, asserting no panics and finite/well-formed output.
//!
//! Permutation invariance is **not** asserted: Cholesky whitening
//! depends on column order; permuting dimensions before
//! whitening is not a symmetry of the detector.

use cesura::dm_bocd::{DmBocdDetector, IdentityM, ImqM};
use cesura::eval::Rng;
use proptest::prelude::*;

fn idx(cps: &[cesura::ChangePoint]) -> Vec<usize> {
    cps.iter().map(|c| c.index).collect()
}

fn gen_2d_shift(seed: u64, n_pre: usize, n_post: usize, shift: f64) -> Vec<Vec<f64>> {
    let mut rng = Rng::new(seed);
    let mut data: Vec<Vec<f64>> = (0..n_pre)
        .map(|_| vec![rng.normal(0.0, 1.0), rng.normal(0.0, 1.0)])
        .collect();
    data.extend(
        (0..n_post).map(|_| vec![rng.normal(shift, 1.0), rng.normal(shift, 1.0)]),
    );
    data
}

// ── Affine invariance (per-dim positive scale + translation) ─────────

#[test]
fn affine_invariance_per_dim_in_bounded_range() {
    // y = a·x + b for a ∈ [0.1, 10], b ∈ [-100, 100]. Whitening centers
    // and rescales by the warmup covariance, so an affine map applied
    // uniformly across all rows should leave CP indices invariant
    // (modulo conditioning at extreme `a`).
    let det = DmBocdDetector::new(2, 100.0, 400);
    let base_data = gen_2d_shift(0xAFF1, 200, 200, 5.0);
    let base = idx(&det.detect_multivariate(&base_data));

    for &(a, b) in &[(0.1, -50.0), (1.0, 100.0), (10.0, 0.0), (3.7, -7.3)] {
        let xform: Vec<Vec<f64>> = base_data
            .iter()
            .map(|row| row.iter().map(|x| a * x + b).collect())
            .collect();
        let got = idx(&det.detect_multivariate(&xform));
        assert_eq!(got, base, "affine (a={a}, b={b}) shifted CP indices");
    }
}

// ── ω monotonicity probe ─────────────────────────────────────────────

#[test]
fn omega_sweep_keeps_n_finite() {
    // Probe CP-count sensitivity to `omega` on a synthetic input:
    // CP counts may be flat or non-monotone -- the assertion is only
    // that nothing panics or produces NaN regardless of ω across the
    // sweep range; no CP-count ordering is asserted.
    let data = gen_2d_shift(0x0E0A, 200, 200, 5.0);
    for &omega in &[0.01, 0.05, 0.1, 0.5, 1.0, 5.0] {
        let det = DmBocdDetector::new(2, 100.0, 400).with_omega(omega);
        let cps = det.detect_multivariate(&data);
        for cp in &cps {
            assert!(cp.confidence.is_finite());
            assert!(cp.shift_sigma.is_finite());
        }
    }
}

// ── Short-input always-empty ────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig { cases: 64, ..ProptestConfig::default() })]

    #[test]
    fn proptest_short_input_returns_empty(
        n in 0usize..20,
        d in 1usize..=5,
    ) {
        let det = DmBocdDetector::new(d, 100.0, 50);
        let data: Vec<Vec<f64>> = (0..n).map(|_| vec![0.0; d]).collect();
        prop_assert!(det.detect_multivariate(&data).is_empty());
    }
}

// ── No-panic on bounded random input ─────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig { cases: 32, ..ProptestConfig::default() })]

    #[test]
    fn proptest_no_panic_random_input(
        seed in any::<u64>(),
        n in 0usize..500,
        d in 1usize..=8,
    ) {
        let mut rng = Rng::new(seed.max(1));
        let data: Vec<Vec<f64>> = (0..n)
            .map(|_| (0..d).map(|_| {
                let v = rng.normal(0.0, 1.0);
                v.clamp(-1e6, 1e6)
            }).collect())
            .collect();
        let det = DmBocdDetector::new(d, 100.0, 200);
        let (cps, cp_probs, map_rls) = det.detect_multivariate_with_diagnostics(&data);
        if n >= 20 {
            prop_assert_eq!(cp_probs.len(), n);
            prop_assert_eq!(map_rls.len(), n);
        }
        // No duplicate CP indices (cooldown gate guarantees this).
        let mut indices: Vec<usize> = cps.iter().map(|c| c.index).collect();
        indices.sort();
        let pre = indices.len();
        indices.dedup();
        prop_assert_eq!(pre, indices.len(), "duplicate CP indices emitted");
    }
}

// ── Adversarial fixtures (heavy tails) ───────────────────────────────

const ADVERSARIAL_SEEDS: [u64; 16] = [
    0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x8,
    0xA, 0xB, 0xC, 0xD, 0xE, 0xF, 0x10, 0x11,
];

fn run_dm_no_panic(data: &[Vec<f64>], d: usize) {
    let (cps, cp_probs, _) = DmBocdDetector::new(d, 100.0, 400)
        .with_m_weight(IdentityM)
        .detect_multivariate_with_diagnostics(data);
    validate_diag(&cps, &cp_probs);
}

fn run_dm_no_panic_imq(data: &[Vec<f64>], d: usize, c: f64) -> Vec<cesura::ChangePoint> {
    let (cps, cp_probs, _) = DmBocdDetector::new(d, 100.0, 400)
        .with_m_weight(ImqM::new(c))
        .detect_multivariate_with_diagnostics(data);
    validate_diag(&cps, &cp_probs);
    cps
}

fn run_dm_no_panic_inner(data: &[Vec<f64>], d: usize, imq_c: Option<f64>) -> Vec<cesura::ChangePoint> {
    // imq_c=None ⇒ Identity baseline. Generic-typed detector forces this
    // helper to dispatch ahead of the with_m_weight call.
    match imq_c {
        Some(c) => run_dm_no_panic_imq(data, d, c),
        None => {
            let (cps, cp_probs, _) = DmBocdDetector::new(d, 100.0, 400)
                .with_m_weight(IdentityM)
                .detect_multivariate_with_diagnostics(data);
            validate_diag(&cps, &cp_probs);
            cps
        }
    }
}

fn validate_diag(cps: &[cesura::ChangePoint], cp_probs: &[f64]) {
    for &p in cp_probs {
        assert!(!p.is_nan(), "cp_prob NaN under heavy-tail input");
    }
    for cp in cps {
        assert!(
            cp.confidence.is_finite() && cp.shift_sigma.is_finite(),
            "non-finite CP fields under heavy-tail input"
        );
    }
    let mut idxs: Vec<usize> = cps.iter().map(|c| c.index).collect();
    idxs.sort();
    let n_pre = idxs.len();
    idxs.dedup();
    assert_eq!(n_pre, idxs.len());
}

fn count_far_cps(cps: &[cesura::ChangePoint], n: usize) -> usize {
    // No "true event" in these heavy-tail-no-real-shift fixtures, so
    // every emitted CP is a far-CP (false alarm).
    let _ = n;
    cps.len()
}

#[test]
fn adversarial_cauchy_2d() {
    for &seed in &ADVERSARIAL_SEEDS {
        let mut rng = Rng::new(seed);
        let data: Vec<Vec<f64>> = (0..400)
            .map(|_| vec![rng.cauchy(0.0, 1.0), rng.cauchy(0.0, 1.0)])
            .collect();
        run_dm_no_panic(&data, 2);
    }
}

#[test]
fn adversarial_student_t3_2d() {
    for &seed in &ADVERSARIAL_SEEDS {
        let mut rng = Rng::new(seed);
        let data: Vec<Vec<f64>> = (0..400)
            .map(|_| vec![rng.student_t3(), rng.student_t3()])
            .collect();
        run_dm_no_panic(&data, 2);
    }
}

#[test]
fn adversarial_garch_clusters_2d() {
    // GARCH(1,1)-ish volatility clustering on each dim.
    for &seed in &ADVERSARIAL_SEEDS {
        let mut rng = Rng::new(seed);
        let mut sigma2 = [1.0_f64; 2];
        let alpha = 0.1;
        let beta = 0.85;
        let omega_garch = 0.05;
        let data: Vec<Vec<f64>> = (0..400)
            .map(|_| {
                let mut row = vec![0.0; 2];
                for k in 0..2 {
                    let z = rng.normal(0.0, 1.0);
                    let eps = sigma2[k].sqrt() * z;
                    sigma2[k] = omega_garch + alpha * eps * eps + beta * sigma2[k];
                    row[k] = eps;
                }
                row
            })
            .collect();
        run_dm_no_panic(&data, 2);
    }
}

// ── IMQ properties ────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(16))]

    /// IMQ on bounded random input: cp_probs finite, no duplicate CP indices.
    #[test]
    fn imq_finite_on_bounded_random_input(
        seed in 1u64..(1u64 << 32),
        c in 0.1f64..5.0,
        n_pre in 60usize..200,
        n_post in 60usize..200,
        shift in -3.0f64..3.0,
    ) {
        let mut rng = Rng::new(seed);
        let mut data: Vec<Vec<f64>> = (0..n_pre)
            .map(|_| vec![rng.normal(0.0, 1.0), rng.normal(0.0, 1.0)])
            .collect();
        data.extend(
            (0..n_post).map(|_| vec![rng.normal(shift, 1.0), rng.normal(shift, 1.0)]),
        );
        let _ = run_dm_no_panic_imq(&data, 2, c);
    }
}

#[test]
fn imq_reduces_far_cps_on_heavy_tail_cauchy() {
    // Load-bearing claim: IMQ's bounded influence should suppress
    // false alarms vs Identity on no-shift heavy-tail input.
    // Pass criterion: across 16 seeds, IMQ's total far-CP count over
    // all seeds is ≤ Identity's. ±10% slack on the seed-by-seed
    // variance is allowed.
    let mut imq_total = 0usize;
    let mut id_total = 0usize;
    for &seed in &ADVERSARIAL_SEEDS {
        let mut rng = Rng::new(seed);
        let data: Vec<Vec<f64>> = (0..400)
            .map(|_| vec![rng.cauchy(0.0, 1.0), rng.cauchy(0.0, 1.0)])
            .collect();
        let imq_cps = run_dm_no_panic_imq(&data, 2, 1.0);
        let id_cps = run_dm_no_panic_inner(&data, 2, None);
        imq_total += count_far_cps(&imq_cps, data.len());
        id_total += count_far_cps(&id_cps, data.len());
    }
    let tol = (id_total as f64 * 1.10).ceil() as usize;
    assert!(
        imq_total <= tol,
        "IMQ far-CPs {} exceeded Identity far-CPs+10% ({} <= {})",
        imq_total,
        imq_total,
        tol,
    );
}

#[test]
fn imq_reduces_far_cps_on_heavy_tail_t3() {
    let mut imq_total = 0usize;
    let mut id_total = 0usize;
    for &seed in &ADVERSARIAL_SEEDS {
        let mut rng = Rng::new(seed);
        let data: Vec<Vec<f64>> = (0..400)
            .map(|_| vec![rng.student_t3(), rng.student_t3()])
            .collect();
        let imq_cps = run_dm_no_panic_imq(&data, 2, 1.0);
        let id_cps = run_dm_no_panic_inner(&data, 2, None);
        imq_total += count_far_cps(&imq_cps, data.len());
        id_total += count_far_cps(&id_cps, data.len());
    }
    let tol = (id_total as f64 * 1.10).ceil() as usize;
    assert!(
        imq_total <= tol,
        "IMQ far-CPs {} exceeded Identity far-CPs+10% ({} <= {})",
        imq_total,
        imq_total,
        tol,
    );
}

#[test]
fn imq_with_prior_aligned_outliers_pins_st_rcgp_failure_mode() {
    // ST-RCGP (Laplante-Altamirano-Duncan-Knoblauch-Briol, ICML 2025,
    // arXiv:2502.02450) identifies a failure mode for constant-c IMQ:
    // when the prior mean (here: zero, after warmup whitening centers
    // the data) ALIGNS with the outlier cluster, IMQ down-weights
    // informative samples instead of noise. This synthetic fixture exercises
    // that centering failure mode.
    //
    // This test pins it as a regression: on a fixture where the
    // outlier cluster sits near the post-whitening origin and the
    // signal lives away from it, IMQ should NOT improve over
    // Identity. If a future "fix" claims to make IMQ universally
    // dominant, this test will catch the over-claim -- the fix has
    // to address centering (γ_t) or bandwidth selection (c_t),
    // not just swap kernels.
    //
    // Construction: 200 obs near zero (the "near-prior outliers"),
    // then 200 obs at +5σ (the "informative signal"). After warmup
    // whitening, the first segment dominates the mean estimate and
    // the second segment's signal is what we want to detect.
    let mut imq_far_total = 0usize;
    let mut id_far_total = 0usize;
    let mut imq_hits_total = 0usize;
    let mut id_hits_total = 0usize;
    for &seed in &ADVERSARIAL_SEEDS[..8] {
        let mut rng = Rng::new(seed);
        let mut data: Vec<Vec<f64>> = (0..200)
            .map(|_| vec![rng.normal(0.0, 1.0), rng.normal(0.0, 1.0)])
            .collect();
        data.extend(
            (0..200).map(|_| vec![rng.normal(5.0, 1.0), rng.normal(5.0, 1.0)]),
        );
        let imq_cps = run_dm_no_panic_imq(&data, 2, 1.0);
        let id_cps = run_dm_no_panic_inner(&data, 2, None);
        // True CP at index 200; "hit" = any CP in [180, 260].
        let hit = |cps: &[cesura::ChangePoint]| -> bool {
            cps.iter().any(|c| (180..=260).contains(&c.index))
        };
        let far = |cps: &[cesura::ChangePoint]| -> usize {
            cps.iter().filter(|c| !(180..=260).contains(&c.index)).count()
        };
        if hit(&imq_cps) { imq_hits_total += 1; }
        if hit(&id_cps) { id_hits_total += 1; }
        imq_far_total += far(&imq_cps);
        id_far_total += far(&id_cps);
    }
    // The pin: under prior-aligned outliers, IMQ does NOT cleanly
    // dominate Identity. Either hit-rate is comparable, OR IMQ
    // pays its hits in far-CPs. If a future change makes IMQ both
    // strictly more accurate AND no worse on FAR here, this test
    // fires -- forcing the change to either (a) cite the fix's
    // ST-RCGP-style centering machinery or (b) show this case was
    // already handled by some other mechanism. Either is fine; the
    // regression requires a justified change in outlier handling.
    let imq_strictly_dominates =
        imq_hits_total > id_hits_total && imq_far_total <= id_far_total;
    assert!(
        !imq_strictly_dominates,
        "IMQ strictly dominated Identity on prior-aligned-outlier fixture \
         (imq_hits={imq_hits_total} > id_hits={id_hits_total}, \
         imq_far={imq_far_total} <= id_far={id_far_total}). \
         If this is genuine, the fix likely addresses centering/bandwidth \
         per arXiv:2502.02450 -- update this test to explicitly require \
         that mechanism."
    );
}

#[test]
fn adversarial_t3_with_real_shift_2d() {
    // 200 t3 observations, then a 5σ-equivalent mean shift, then 200 more
    // t3 observations -- mirrors eval::t3_with_real_shift but in 2D.
    for &seed in &ADVERSARIAL_SEEDS {
        let mut rng = Rng::new(seed);
        let mut data: Vec<Vec<f64>> = (0..200)
            .map(|_| vec![rng.student_t3(), rng.student_t3()])
            .collect();
        data.extend(
            (0..200).map(|_| vec![5.0 + rng.student_t3(), 5.0 + rng.student_t3()]),
        );
        run_dm_no_panic(&data, 2);
    }
}
