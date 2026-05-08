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
