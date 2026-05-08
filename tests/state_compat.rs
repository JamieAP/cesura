//! Streaming state size + serialization compatibility tests.
//!
//! Two production concerns:
//! - State size grows linearly with `total_steps` (map_rls + raw_index_map).
//!   A daemon running for a week must not need GBs to checkpoint.
//! - Serialized JSON state must round-trip and remain backward-compatible
//!   so a running deployment can deserialize state written by an older build.
//!
//! Run with: `cargo test --features test-utils --test state_compat`.

use cesura::eval::Rng;
use cesura::streaming::{DetectorState, StreamingDetector};

#[cfg(feature = "joint-detection")]
use cesura::streaming_chen_wu::{ChenWuDetectorState, StreamingChenWuDetector};

#[test]
fn state_size_is_bounded_after_long_run() {
    // Run 10K observations through a streaming detector, serialize, check size.
    // At 8 bytes per usize × 2 vectors × 10K = ~160KB raw; JSON overhead ~3-5x.
    // We assert ≤ 2MB as a generous, regression-catching upper bound.
    let mut det = StreamingDetector::new(200.0, 400);
    let mut rng = Rng::new(123);
    let chunk: Vec<f64> = (0..10_000).map(|_| rng.normal(0.0, 1.0)).collect();
    det.step(&chunk);

    let json = serde_json::to_string(&det.save_state()).unwrap();
    let bytes = json.len();
    eprintln!(
        "state size after 10K obs: {} bytes ({:.1} KB)",
        bytes,
        bytes as f64 / 1024.0
    );
    assert!(
        bytes < 2_000_000,
        "state {} bytes after 10K obs -- investigate growth",
        bytes
    );
    // Lower-bound sanity: should not be implausibly small (would mean state lost).
    assert!(bytes > 50_000, "state too small ({}b) -- looks broken", bytes);
}

#[test]
fn state_json_round_trip_preserves_behavior() {
    // Save → deserialize → restore → continue. The restored detector must produce
    // identical detections to the original on subsequent input -- that's the
    // contract production daemons rely on.
    //
    // We don't assert *byte-equal* JSON across two `to_string` calls: serde_json's
    // float formatter has minor representation drift for some values
    // (`-7.2…945` vs `-7.2…95`, `e308` vs `e+308`). Drift is in the encoding,
    // not the decoded f64 value, and the semantic behavior is what matters.
    let mut det = StreamingDetector::new(200.0, 250);
    let mut rng = Rng::new(7);
    let warmup: Vec<f64> = (0..100).map(|_| rng.normal(0.0, 1.0)).collect();
    det.step(&warmup);

    let json = serde_json::to_string(&det.save_state()).unwrap();
    let restored: DetectorState = serde_json::from_str(&json).unwrap();
    let mut det_b = StreamingDetector::restore(restored).unwrap();

    // Same total_steps, same NIG state, same Welford counters -- verify by
    // running identical input through both and comparing detections.
    let post: Vec<f64> = (0..150).map(|_| rng.normal(5.0, 1.0)).collect();
    let cps_a = det.step(&post);
    let cps_b = det_b.step(&post);
    assert_eq!(
        cps_a.len(),
        cps_b.len(),
        "restored detector returned different number of CPs"
    );
    for (a, b) in cps_a.iter().zip(cps_b.iter()) {
        assert_eq!(a.index, b.index, "restored detector diverged on index");
        // Confidences should match to at least 6 decimal places.
        assert!(
            (a.confidence - b.confidence).abs() < 1e-6,
            "confidence drift: original={} restored={}",
            a.confidence,
            b.confidence
        );
    }
    assert_eq!(det.total_steps(), det_b.total_steps());
}

#[test]
fn state_json_forward_compat_minimal_v1() {
    // A *minimal* JSON state representing a freshly-warmed detector.
    // If a future change breaks deserialization of this exact blob, you've
    // shipped a wire-format break to existing deployments. Bump `max_rl` to 4
    // to keep the blob tiny but exercise the vector-length validation.
    //
    // To regenerate: bump the schema, write a new pinned blob here, and
    // document the migration in the changelog.
    let blob = r#"{
        "rl_log": [0.0, -1.7976931348623157e308, -1.7976931348623157e308, -1.7976931348623157e308, -1.7976931348623157e308],
        "stats": [
            {"mu": 0.0, "kappa": 1.0, "alpha": 1.0, "beta": 1.0},
            {"mu": 0.0, "kappa": 1.0, "alpha": 1.0, "beta": 1.0},
            {"mu": 0.0, "kappa": 1.0, "alpha": 1.0, "beta": 1.0},
            {"mu": 0.0, "kappa": 1.0, "alpha": 1.0, "beta": 1.0},
            {"mu": 0.0, "kappa": 1.0, "alpha": 1.0, "beta": 1.0}
        ],
        "map_rls": [],
        "total_steps": 0,
        "last_detection": 0,
        "welford": {"count": 0, "mean": 0.0, "m2": 0.0},
        "hazard_log": -5.298317366548036,
        "growth_log": -0.005012541823544286,
        "max_rl": 4,
        "raw_steps": 0,
        "raw_index_map": []
    }"#;

    let state: DetectorState = serde_json::from_str(blob).expect("v1 blob must deserialize");
    let det = StreamingDetector::restore(state).expect("v1 blob must restore");
    assert_eq!(det.total_steps(), 0);

    // And it must keep working: feed it data, expect no panic.
    let mut det = det;
    let _ = det.step(&[0.0; 50]);
}

#[test]
fn v05_state_with_beta_round_trip() {
    // Save with β = 0.15, restore, then continue. Subsequent step() output
    // must match a fresh detector also configured with β = 0.15 on identical
    // input. Without β persistence the restored detector silently drops to the
    // standard path -- the regression Piece 1 closes.
    let beta = 0.15;
    let mut a = StreamingDetector::new(200.0, 250).with_beta(beta);
    let mut rng = Rng::new(42);
    let warmup: Vec<f64> = (0..120).map(|_| rng.normal(0.0, 1.0)).collect();
    a.step(&warmup);

    let json = serde_json::to_string(&a.save_state()).unwrap();
    let restored: DetectorState = serde_json::from_str(&json).unwrap();
    assert_eq!(restored.beta, beta, "β missing from serialised state");
    let mut a_restored = StreamingDetector::restore(restored).unwrap();

    // Fresh detector on the same warmup, also β = 0.15.
    let mut b = StreamingDetector::new(200.0, 250).with_beta(beta);
    b.step(&warmup);

    // Now feed both an identical post-warmup stream with a real shift.
    let mut rng2 = Rng::new(99);
    let post: Vec<f64> = (0..200)
        .map(|i| if i < 100 { rng2.normal(0.0, 1.0) } else { rng2.normal(4.0, 1.0) })
        .collect();
    let cps_a = a_restored.step(&post);
    let cps_b = b.step(&post);
    assert_eq!(
        cps_a.len(),
        cps_b.len(),
        "restored β-detector diverged in CP count: {cps_a:?} vs {cps_b:?}"
    );
    for (x, y) in cps_a.iter().zip(cps_b.iter()) {
        assert_eq!(x.index, y.index, "CP index drift after β restore");
        assert!((x.confidence - y.confidence).abs() < 1e-9);
    }
}

#[cfg(feature = "joint-detection")]
#[test]
fn chen_wu_state_round_trip() {
    // Save with hyperparameters + warmup, restore, run further input,
    // assert match against a fresh detector with the same hyperparameters.
    let mut a = StreamingChenWuDetector::new(0.1, 0.2, 4, 0.5, 0.5)
        .with_search_windows(128, 16)
        .with_localisation_tolerance(0)
        .with_min_post_change(5)
        .with_prior(0.0, 0.01, 0.5, 0.125);
    let mut rng = Rng::new(42);
    let warmup: Vec<f64> = (0..120).map(|_| rng.normal(0.0, 1.0)).collect();
    a.step(&warmup);

    let json = serde_json::to_string(&a.save_state()).unwrap();
    let restored: ChenWuDetectorState = serde_json::from_str(&json).unwrap();
    assert_eq!(restored.u_c, 128, "search window survived round-trip");
    assert_eq!(restored.delta_t, 4);
    let mut a_restored = StreamingChenWuDetector::restore(restored).unwrap();
    assert_eq!(a_restored.total_steps(), 120);

    let mut b = StreamingChenWuDetector::new(0.1, 0.2, 4, 0.5, 0.5)
        .with_search_windows(128, 16)
        .with_localisation_tolerance(0)
        .with_min_post_change(5)
        .with_prior(0.0, 0.01, 0.5, 0.125);
    b.step(&warmup);

    let mut rng2 = Rng::new(99);
    let post: Vec<f64> = (0..200)
        .map(|i| if i < 100 { rng2.normal(0.0, 1.0) } else { rng2.normal(4.0, 1.0) })
        .collect();
    let dets_a = a_restored.step(&post);
    let dets_b = b.step(&post);
    use cesura::chen_wu::Detection;
    assert_eq!(
        dets_a.len(),
        dets_b.len(),
        "restored streaming chen_wu diverged in detection count: {dets_a:?} vs {dets_b:?}"
    );
    for (a, b) in dets_a.iter().zip(dets_b.iter()) {
        match (a, b) {
            (Detection::ChangePoint(x), Detection::ChangePoint(y)) => {
                assert_eq!(x.index, y.index, "CP index drift after restore");
                assert!(
                    (x.confidence - y.confidence).abs() < 1e-9,
                    "CP confidence drift after restore: {} vs {}",
                    x.confidence,
                    y.confidence
                );
                assert!(
                    (x.shift_sigma - y.shift_sigma).abs() < 1e-9,
                    "CP shift_sigma drift after restore: {} vs {}",
                    x.shift_sigma,
                    y.shift_sigma
                );
            }
            (
                Detection::CollectiveAnomaly { start: s1, end: e1, confidence: c1 },
                Detection::CollectiveAnomaly { start: s2, end: e2, confidence: c2 },
            ) => {
                assert_eq!((s1, e1), (s2, e2), "anomaly window drift after restore");
                assert!(
                    (c1 - c2).abs() < 1e-9,
                    "anomaly confidence drift after restore: {c1} vs {c2}"
                );
            }
            _ => panic!("variant mismatch after restore: {a:?} vs {b:?}"),
        }
    }
}

#[cfg(feature = "joint-detection")]
#[test]
fn chen_wu_state_size_is_bounded() {
    // After 10K obs the serialised state should be < 5 MB (raw_history
    // + h_c_history + log_h vectors all bounded).
    let mut det = StreamingChenWuDetector::new(0.1, 0.2, 4, 0.5, 0.5);
    let mut rng = Rng::new(123);
    let chunk: Vec<f64> = (0..10_000).map(|_| rng.normal(0.0, 1.0)).collect();
    det.step(&chunk);

    let json = serde_json::to_string(&det.save_state()).unwrap();
    let bytes = json.len();
    eprintln!(
        "chen_wu state size after 10K obs: {} bytes ({:.1} KB)",
        bytes,
        bytes as f64 / 1024.0
    );
    assert!(
        bytes < 5_000_000,
        "chen_wu state {} bytes after 10K obs -- exceeds 5 MB budget",
        bytes
    );
    // Sanity floor: not implausibly small.
    assert!(bytes > 1_000, "state too small ({}b) -- looks broken", bytes);
}
