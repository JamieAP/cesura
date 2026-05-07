//! Resilience tests: adversarial inputs, edge config, malformed state.
//!
//! Production loops feed garbage to detectors. The contract: never panic,
//! never NaN-leak, never hang.
//!
//! Run: `cargo test --features test-utils --test resilience`.

use bocpd::eval::Rng;
use bocpd::streaming::{DetectorState, NigState, StreamingDetector, WelfordState};
use bocpd::BocpdDetector;

// ── Adversarial numerical inputs ─────────────────────────────────────

#[test]
fn alternating_extreme_values_does_not_overflow() {
    // ±1e300 alternating: sum-of-squares overflows to +inf → std=+inf →
    // normalized data = NaN → must not propagate to outputs.
    let det = BocpdDetector::new(200.0, 250);
    let mut data = Vec::with_capacity(200);
    for i in 0..200 {
        data.push(if i % 2 == 0 { 1e300 } else { -1e300 });
    }
    let cps = det.detect(&data, 0.3);
    for cp in &cps {
        assert!(cp.confidence.is_finite(), "non-finite confidence leaked");
        assert!(cp.shift_sigma.is_finite(), "non-finite shift_sigma leaked");
    }
}

#[test]
fn subnormal_floats_do_not_panic() {
    // Inputs near f64::MIN_POSITIVE → log/sqrt of tiny numbers.
    let det = BocpdDetector::new(200.0, 250);
    let data: Vec<f64> = (0..200)
        .map(|i| if i < 100 { 1e-300 } else { 1e-310 })
        .collect();
    let _ = det.detect(&data, 0.3);
}

#[test]
fn all_zeros_returns_no_detection() {
    // std = 0 → fallback to std=1 (per impl) → predictive on a zero stream
    // gives stable run-length growth → no detection.
    // Contract: max_run_length must be ≥ data.len() (per BocpdDetector::new
    // doc); we honor that here.
    let data = vec![0.0; 300];
    let det = BocpdDetector::new(200.0, data.len() + 50);
    let cps = det.detect(&data, 0.3);
    assert!(cps.is_empty(), "all-zeros should be silent, got {cps:?}");
}

#[test]
fn max_rl_truncation_does_not_emit_phantom_cp() {
    // Regression test for a numerical artifact: when max_rl < data.len(),
    // probability mass leaks at the top of the run-length distribution and
    // would produce a "change point" at t ≈ max_rl on a constant signal --
    // high confidence, zero observed shift. detect() now suppresses any
    // CP with shift_sigma < 1e-9 because no real regime change can have
    // identical before/after means.
    let data = vec![0.0; 300];
    let det = BocpdDetector::new(200.0, 250); // intentionally < data.len()
    let cps = det.detect(&data, 0.3);
    assert!(
        cps.is_empty(),
        "phantom CP from max_rl truncation leaked: {cps:?}"
    );
}

#[test]
fn alternating_high_low_no_phantom_changes() {
    // Pure 2-period alternation is not a regime change, just high-frequency
    // structure. The detector may fire on the implied variance, but should
    // not produce NaN or hang.
    let det = BocpdDetector::new(200.0, 250);
    let data: Vec<f64> = (0..300)
        .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
        .collect();
    let cps = det.detect(&data, 0.3);
    for cp in &cps {
        assert!(cp.confidence.is_finite());
        assert!(cp.confidence >= 0.0 && cp.confidence <= 1.0);
    }
}

// ── Boundary configuration ───────────────────────────────────────────

#[test]
fn max_rl_zero_does_not_panic() {
    // Pathological but legal config: max_rl = 0. Detector should produce no
    // detections (no run-length to grow into) without panicking.
    let det = BocpdDetector::new(200.0, 0);
    let data: Vec<f64> = (0..100).map(|i| if i < 50 { 0.0 } else { 5.0 }).collect();
    let cps = det.detect(&data, 0.3);
    assert!(cps.is_empty(), "max_rl=0 should produce no CPs");
}

#[test]
fn data_length_at_minimum_threshold() {
    // Contract: detect() returns empty for data.len() < 20.
    let det = BocpdDetector::new(200.0, 50);

    // len=19: rejected.
    assert!(det.detect(&[1.0; 19], 0.3).is_empty());

    // len=20: accepted, but the heuristic needs ≥ min_prev_rl=30 to fire.
    // So at len=20 there can be no detection regardless of input.
    assert!(det.detect(&[1.0; 20], 0.3).is_empty());
    let mut shifted = vec![0.0; 10];
    shifted.extend(vec![10.0; 10]);
    assert!(
        det.detect(&shifted, 0.3).is_empty(),
        "len=20 cannot fire -- under min_prev_rl warmup"
    );
}

#[test]
fn extreme_lambda_values() {
    // λ just above 1 (very sensitive) and λ enormous (very insensitive).
    // Both must produce finite output without overflow in hazard_log /
    // growth_log.
    for &lam in &[1.0001_f64, 1e9_f64] {
        let det = BocpdDetector::new(lam, 250);
        let data: Vec<f64> = (0..150)
            .map(|i| if i < 75 { 0.0 } else { 5.0 })
            .collect();
        let cps = det.detect(&data, 0.3);
        for cp in &cps {
            assert!(cp.confidence.is_finite(), "λ={lam}: non-finite confidence");
        }
    }
}

#[test]
fn streaming_empty_step_is_noop() {
    let mut det = StreamingDetector::new(200.0, 100);
    let cps = det.step(&[], 0.3);
    assert!(cps.is_empty());
    assert_eq!(det.total_steps(), 0);
}

#[test]
fn streaming_idempotent_save_restore_no_steps() {
    // Detector with no input survives save/restore.
    let det = StreamingDetector::new(200.0, 100);
    let s = det.save_state();
    let json = serde_json::to_string(&s).unwrap();
    let restored: DetectorState = serde_json::from_str(&json).unwrap();
    let _ = StreamingDetector::restore(restored).unwrap();
}

// ── Malformed / hostile state ────────────────────────────────────────

#[test]
fn malformed_json_state_returns_err_not_panic() {
    let bogus = r#"{"this": "is not", "a": "DetectorState"}"#;
    let result: Result<DetectorState, _> = serde_json::from_str(bogus);
    assert!(result.is_err(), "garbage JSON must fail to parse");

    let truncated = r#"{"rl_log": [0.0, "#;
    let result: Result<DetectorState, _> = serde_json::from_str(truncated);
    assert!(result.is_err(), "truncated JSON must fail");
}

#[test]
fn state_with_mismatched_lengths_is_rejected() {
    // restore() validates length consistency. Stress it from the wrong direction.
    let det = StreamingDetector::new(200.0, 50);
    let mut state = det.save_state();
    state.stats.truncate(10); // shorter than max_rl + 1
    let err = StreamingDetector::restore(state).err();
    assert!(
        err.is_some_and(|e| e.contains("stats")),
        "mismatched stats length must be rejected with a descriptive error"
    );
}

#[test]
fn state_with_nan_in_nig_does_not_propagate_silently() {
    // A corrupted state file with NaN in NIG fields. We don't require restore
    // to reject it (current impl doesn't), but we DO require subsequent
    // detection to either error gracefully or not produce NaN-tainted CPs.
    let mut det = StreamingDetector::new(200.0, 50);
    let mut rng = Rng::new(5);
    det.step(
        &(0..30).map(|_| rng.normal(0.0, 1.0)).collect::<Vec<_>>(),
        0.3,
    );
    let mut state = det.save_state();
    // Corrupt EVERY NIG slot -- guarantees the active MAP slot is poisoned
    // regardless of where the run-length distribution sits.
    for s in &mut state.stats {
        *s = NigState {
            mu: f64::NAN,
            kappa: f64::NAN,
            alpha: f64::NAN,
            beta: f64::NAN,
        };
    }
    let mut det2 = StreamingDetector::restore(state).expect("restore accepts NaN today");

    // Continued operation must not panic and must not emit NaN confidence.
    let cps = det2.step(&[1.0; 100], 0.3);
    for cp in &cps {
        assert!(
            cp.confidence.is_finite(),
            "NaN-corrupted state leaked NaN into a confidence value"
        );
    }
}

#[test]
fn state_with_negative_alpha_does_not_panic() {
    // α ≤ 0 makes the predictive Student-t df ≤ 0 → predictive returns -inf.
    // Detector must fall through that gracefully.
    let det = StreamingDetector::new(200.0, 30);
    let mut state = det.save_state();
    for s in &mut state.stats {
        s.alpha = -1.0;
        s.beta = -1.0;
    }
    let mut det2 = StreamingDetector::restore(state).unwrap();
    let _ = det2.step(&[0.0; 50], 0.3);
}

#[test]
fn welford_with_zero_count_does_not_divide_by_zero() {
    // Welford normalize with count=0 must not panic.
    let w = WelfordState::default();
    let _ = w.normalize(2.5); // no-op or fallback, must not panic
}
