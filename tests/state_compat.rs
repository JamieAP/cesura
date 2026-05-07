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

#[test]
fn state_size_is_bounded_after_long_run() {
    // Run 10K observations through a streaming detector, serialize, check size.
    // At 8 bytes per usize × 2 vectors × 10K = ~160KB raw; JSON overhead ~3-5x.
    // We assert ≤ 2MB as a generous, regression-catching upper bound.
    let mut det = StreamingDetector::new(200.0, 400);
    let mut rng = Rng::new(123);
    let chunk: Vec<f64> = (0..10_000).map(|_| rng.normal(0.0, 1.0)).collect();
    det.step(&chunk, 0.5);

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
    det.step(&warmup, 0.3);

    let json = serde_json::to_string(&det.save_state()).unwrap();
    let restored: DetectorState = serde_json::from_str(&json).unwrap();
    let mut det_b = StreamingDetector::restore(restored).unwrap();

    // Same total_steps, same NIG state, same Welford counters -- verify by
    // running identical input through both and comparing detections.
    let post: Vec<f64> = (0..150).map(|_| rng.normal(5.0, 1.0)).collect();
    let cps_a = det.step(&post, 0.3);
    let cps_b = det_b.step(&post, 0.3);
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
    let _ = det.step(&[0.0; 50], 0.5);
}
