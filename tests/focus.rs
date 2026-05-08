//! Standalone FOCuS detector tests.
//!
//! Mirrors the BOCPD correctness suite shape: clean shift, false-alarm
//! rate, agreement against the reference detector on a multi-regime
//! fixture.
//!
//! Run: `cargo test --features test-utils --test focus`.

use cesura::eval::Rng;
use cesura::focus::FocusDetector;
use cesura::BocpdDetector;

#[test]
fn focus_matches_bocpd_within_25_steps() {
    // 3-regime fixture: same data fed to BOCPD and FOCuS, the two
    // detectors must agree on each major change point within ±25 steps.
    let mut rng = Rng::new(2026);
    let mut data: Vec<f64> = (0..100).map(|_| rng.normal(0.0, 1.0)).collect();
    data.extend((0..100).map(|_| rng.normal(4.0, 1.0)));
    data.extend((0..100).map(|_| rng.normal(-2.0, 1.0)));

    let bocpd_idx: Vec<usize> = BocpdDetector::new(200.0, 350)
        .detect(&data)
        .into_iter()
        .map(|c| c.index)
        .collect();
    let mut focus_det = FocusDetector::new(8.0);
    let focus_idx: Vec<usize> = focus_det
        .detect(&data)
        .into_iter()
        .map(|c| c.index)
        .collect();

    eprintln!("BOCPD: {bocpd_idx:?}  FOCuS: {focus_idx:?}");
    assert!(!bocpd_idx.is_empty(), "BOCPD must detect on this fixture");
    assert!(!focus_idx.is_empty(), "FOCuS must detect on this fixture");

    // For each ground-truth CP (100 and 200), at least one of each
    // detector's CPs is within ±25.
    for truth in [100usize, 200usize] {
        let near_bocpd = bocpd_idx
            .iter()
            .any(|&i| (i as i64 - truth as i64).abs() <= 25);
        let near_focus = focus_idx
            .iter()
            .any(|&i| (i as i64 - truth as i64).abs() <= 25);
        assert!(near_bocpd, "BOCPD missed truth {truth}: {bocpd_idx:?}");
        assert!(near_focus, "FOCuS missed truth {truth}: {focus_idx:?}");
    }
}

#[test]
fn focus_change_points_strictly_increasing() {
    // FOCuS CP indices in a single detect call must be strictly increasing.
    let mut rng = Rng::new(1234);
    let mut data = Vec::new();
    for k in 0..5 {
        let mu = if k % 2 == 0 { 0.0 } else { 4.0 };
        data.extend((0..100).map(|_| rng.normal(mu, 1.0)));
    }
    let mut det = FocusDetector::new(8.0);
    let cps = det.detect(&data);
    for w in cps.windows(2) {
        assert!(w[0].index < w[1].index, "FOCuS CPs not strictly increasing");
    }
}

#[test]
fn focus_confidence_in_unit_interval() {
    let mut rng = Rng::new(99);
    let mut data: Vec<f64> = (0..150).map(|_| rng.normal(0.0, 1.0)).collect();
    data.extend((0..150).map(|_| rng.normal(3.0, 1.0)));
    let mut det = FocusDetector::new(8.0);
    for cp in det.detect(&data) {
        assert!(
            (0.0..=1.0).contains(&cp.confidence),
            "confidence {} ∉ [0,1]",
            cp.confidence
        );
        assert!(cp.shift_sigma >= 0.0 && cp.shift_sigma.is_finite());
    }
}

#[test]
#[ignore]
fn focus_arl0_full_sweep_at_threshold_8() {
    // Heavy version: 30 trials × 1000 samples N(0,1) at threshold 8.
    // ARL₀ ≥ 1000 means most trials never trigger. Run with `--ignored`.
    let mut total = 0.0;
    let trials = 30;
    let length = 1000;
    for t in 0..trials {
        let mut rng = Rng::new(8000 + t);
        let mut det = FocusDetector::new(8.0);
        let mut fired = None;
        for i in 0..length {
            if det.step(rng.normal(0.0, 1.0)).is_some() {
                fired = Some(i);
                break;
            }
        }
        total += fired.map(|i| i as f64).unwrap_or(length as f64);
    }
    let arl0 = total / trials as f64;
    eprintln!("FOCuS ARL₀ at threshold=8 over {trials} × {length}: {arl0}");
    assert!(arl0 >= 500.0, "ARL₀ at threshold=8 too low: {arl0}");
}

// ── Functional-pruning parity ───────────────────────────
//
// Romano et al. 2023 functional pruning maintains a two-deque structure
// of candidate split points. Detection semantics MUST be bit-for-bit
// identical to the naive O(t) inner loop on every fixture below; if
// they ever diverge, the dominance proof or its implementation is
// wrong.

fn cps_idx(cps: Vec<cesura::ChangePoint>) -> Vec<usize> {
    cps.into_iter().map(|c| c.index).collect()
}

#[test]
fn focus_pruning_matches_naive_clean_shift() {
    let mut rng = Rng::new(11);
    let mut data: Vec<f64> = (0..150).map(|_| rng.normal(0.0, 1.0)).collect();
    data.extend((0..150).map(|_| rng.normal(5.0, 1.0)));
    let mut naive = FocusDetector::new(8.0);
    let mut pruned = FocusDetector::new(8.0).with_pruning();
    assert_eq!(
        cps_idx(naive.detect(&data)),
        cps_idx(pruned.detect(&data)),
        "pruned and naive must agree on clean 5σ shift"
    );
}

#[test]
fn focus_pruning_matches_naive_multi_regime() {
    // Same fixture as `focus_matches_bocpd_within_25_steps`.
    let mut rng = Rng::new(2026);
    let mut data: Vec<f64> = (0..100).map(|_| rng.normal(0.0, 1.0)).collect();
    data.extend((0..100).map(|_| rng.normal(4.0, 1.0)));
    data.extend((0..100).map(|_| rng.normal(-2.0, 1.0)));
    let mut naive = FocusDetector::new(8.0);
    let mut pruned = FocusDetector::new(8.0).with_pruning();
    assert_eq!(
        cps_idx(naive.detect(&data)),
        cps_idx(pruned.detect(&data)),
        "pruned and naive must agree on 3-regime fixture"
    );
}

#[test]
fn focus_pruning_matches_naive_arl0_calibrate() {
    // `arl0_calibrate` is itself a function of FocusDetector internals,
    // so we test its load-bearing primitive: identical first-fire indices
    // on the same N(0,1) trial streams used inside `arl0_calibrate`.
    // If these match across 30 trials × 600 samples for a representative
    // threshold, the calibrator returns the same threshold from either
    // mode (within 0%, not 5%).
    for trial in 0..30u64 {
        let mut rng = Rng::new(7000 + trial);
        let mut naive = FocusDetector::new(8.0);
        let mut pruned = FocusDetector::new(8.0).with_pruning();
        let length = 600;
        for _ in 0..length {
            let x = rng.normal(0.0, 1.0);
            let n_cp = naive.step(x);
            let p_cp = pruned.step(x);
            assert_eq!(
                n_cp, p_cp,
                "diverged at trial {trial} step {} (naive={n_cp:?}, pruned={p_cp:?})",
                naive.total_steps()
            );
        }
    }
}

#[test]
fn focus_pruning_matches_naive_eval_aggregate() {
    // Strongest parity contract: every scenario in the eval suite must
    // produce identical CP indices under naive and pruned modes.
    use cesura::eval;
    let scenarios = eval::all_scenarios();
    for s in &scenarios {
        let mut naive = FocusDetector::new(8.0);
        let mut pruned = FocusDetector::new(8.0).with_pruning();
        let n_idx = cps_idx(naive.detect(&s.data));
        let p_idx = cps_idx(pruned.detect(&s.data));
        assert_eq!(
            n_idx, p_idx,
            "scenario `{}`: naive={n_idx:?}, pruned={p_idx:?}",
            s.name
        );
    }
}

#[test]
#[ignore]
fn focus_pruning_fuzz_parity_long_streams() {
    // Heavy parity fuzzer. 100 seeds × random multi-regime streams up
    // to 10k samples. If naive and pruned ever disagree on CP indices
    // for any seed, the dominance proof or its implementation is wrong.
    // Run with `cargo test --features test-utils --test focus -- --ignored`.
    for seed in 0..100u64 {
        let mut rng = Rng::new(13_000 + seed);
        // 3-7 regimes of length 200..=1500 each, mean drawn uniform[-4, 4].
        let n_regimes = 3 + (seed as usize % 5);
        let mut data: Vec<f64> = Vec::with_capacity(8_000);
        for r in 0..n_regimes {
            let len = 200 + (seed.wrapping_mul(31).wrapping_add(r as u64) as usize % 1300);
            let mu = (rng.normal(0.0, 1.0) * 4.0).clamp(-4.0, 4.0);
            for _ in 0..len {
                data.push(rng.normal(mu, 1.0));
            }
        }
        let mut naive = FocusDetector::new(8.0);
        let mut pruned = FocusDetector::new(8.0).with_pruning();
        let n_idx = cps_idx(naive.detect(&data));
        let p_idx = cps_idx(pruned.detect(&data));
        assert_eq!(
            n_idx, p_idx,
            "fuzz parity break at seed={seed} (data.len={}): naive={n_idx:?}, pruned={p_idx:?}",
            data.len()
        );
    }
}

#[test]
#[ignore]
fn focus_pruning_fuzz_parity_step_lockstep() {
    // Per-step lockstep fuzzer: feed identical streams to naive and
    // pruned via `step()`; every emission must align exactly. Runs at
    // a long stationary length (50k) so pruned is well past where the
    // naive scan dominates -- if the deque drifts out of sync with
    // seg_sums, this catches it.
    for seed in 0..30u64 {
        let mut rng = Rng::new(20_000 + seed);
        let mut naive = FocusDetector::new(8.0);
        let mut pruned = FocusDetector::new(8.0).with_pruning();
        for _ in 0..50_000 {
            let x = rng.normal(0.0, 1.0);
            let n_cp = naive.step(x);
            let p_cp = pruned.step(x);
            assert_eq!(
                n_cp, p_cp,
                "step parity break at seed={seed} step={} (naive={n_cp:?}, pruned={p_cp:?})",
                naive.total_steps()
            );
        }
    }
}

#[test]
fn focus_pruning_step_matches_detect() {
    // Within the pruned mode, `step()` and `detect()` must emit the
    // same CP indices when fed the same data.
    let mut rng = Rng::new(31337);
    let mut data: Vec<f64> = (0..200).map(|_| rng.normal(0.0, 1.0)).collect();
    data.extend((0..200).map(|_| rng.normal(3.5, 1.0)));
    data.extend((0..200).map(|_| rng.normal(-1.5, 1.0)));

    let detect_idx: Vec<usize> = cps_idx(FocusDetector::new(8.0).with_pruning().detect(&data));

    let mut step_det = FocusDetector::new(8.0).with_pruning();
    let mut step_idx = Vec::new();
    for &x in &data {
        if let Some(t) = step_det.step(x) {
            step_idx.push(t);
        }
    }

    assert_eq!(
        detect_idx, step_idx,
        "pruned step()={step_idx:?} vs detect()={detect_idx:?}"
    );
}
