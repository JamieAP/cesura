//! Resilience tests: adversarial inputs, edge config, malformed state.
//!
//! Production loops feed garbage to detectors. The contract: never panic,
//! never NaN-leak, never hang.
//!
//! Run: `cargo test --features test-utils --test resilience`.

use cesura::detrend::Detrender;
use cesura::eval::Rng;
use cesura::streaming::{DetectorState, NigState, StreamingDetector, WelfordState};
use cesura::BocpdDetector;
use std::f64::consts::PI;

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
    let cps = det.detect(&data);
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
    let _ = det.detect(&data);
}

#[test]
fn all_zeros_returns_no_detection() {
    // std = 0 → fallback to std=1 (per impl) → predictive on a zero stream
    // gives stable run-length growth → no detection.
    // Contract: max_run_length must be ≥ data.len() (per BocpdDetector::new
    // doc); we honor that here.
    let data = vec![0.0; 300];
    let det = BocpdDetector::new(200.0, data.len() + 50);
    let cps = det.detect(&data);
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
    let cps = det.detect(&data);
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
    let cps = det.detect(&data);
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
    let cps = det.detect(&data);
    assert!(cps.is_empty(), "max_rl=0 should produce no CPs");
}

#[test]
fn data_length_at_minimum_threshold() {
    // Contract: detect() returns empty for data.len() < 20.
    let det = BocpdDetector::new(200.0, 50);

    // len=19: rejected.
    assert!(det.detect(&[1.0; 19]).is_empty());

    // len=20: accepted, but the heuristic needs ≥ min_prev_rl=30 to fire.
    // So at len=20 there can be no detection regardless of input.
    assert!(det.detect(&[1.0; 20]).is_empty());
    let mut shifted = vec![0.0; 10];
    shifted.extend(vec![10.0; 10]);
    assert!(
        det.detect(&shifted).is_empty(),
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
        let cps = det.detect(&data);
        for cp in &cps {
            assert!(cp.confidence.is_finite(), "λ={lam}: non-finite confidence");
        }
    }
}

#[test]
fn streaming_empty_step_is_noop() {
    let mut det = StreamingDetector::new(200.0, 100);
    let cps = det.step(&[]);
    assert!(cps.is_empty());
    assert_eq!(det.total_steps(), 0);
}

#[test]
fn pre_v06_state_loads_with_beta_zero() {
    // Pinned JSON fixture written before `beta` was added to DetectorState.
    // Forward-compat contract: the field defaults to 0 (standard path).
    // A daemon checkpointing under 0.5 must continue to deserialise cleanly.
    let blob = r#"{
        "rl_log": [0.0, -1.7976931348623157e308, -1.7976931348623157e308],
        "stats": [
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
        "max_rl": 2,
        "raw_steps": 0,
        "raw_index_map": []
    }"#;
    let state: DetectorState = serde_json::from_str(blob).expect("pre-0.6 blob must deserialise");
    assert_eq!(state.beta, 0.0, "missing beta must default to 0");
    let mut det = StreamingDetector::restore(state).expect("pre-0.6 blob must restore");
    let _ = det.step(&[0.0; 20]);
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
    det.step(&(0..30).map(|_| rng.normal(0.0, 1.0)).collect::<Vec<_>>());
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
    let cps = det2.step(&[1.0; 100]);
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
    let _ = det2.step(&[0.0; 50]);
}

#[test]
fn welford_with_zero_count_does_not_divide_by_zero() {
    // Welford normalize with count=0 must not panic.
    let w = WelfordState::default();
    let _ = w.normalize(2.5); // no-op or fallback, must not panic
}

#[test]
fn long_stable_run_past_max_rl_does_not_phantom() {
    // 3000 stationary samples then a 5σ shift. Without mass-pruning the
    // run-length distribution piles up at max_rl=200 and the (now-removed)
    // band-aid suppresses one or more phantom MAP-drops. With mass-pruning
    // the tail trims itself well before 200, so the eventual real CP at
    // t=3000 is detected cleanly without spurious detections in the
    // stationary segment.
    let mut rng = Rng::new(0xCAFE_BABE);
    let mut data: Vec<f64> = (0..3000).map(|_| rng.normal(0.0, 1.0)).collect();
    data.extend((0..200).map(|_| rng.normal(5.0, 1.0)));

    let det = BocpdDetector::new(500.0, 200);
    let cps = det.detect(&data);

    let in_stable = cps.iter().filter(|c| c.index < 2990).count();
    let near_real = cps.iter().any(|c| (c.index as i64 - 3000).abs() <= 25);
    assert_eq!(
        in_stable, 0,
        "spurious detections in stationary segment: {:?}",
        cps.iter().filter(|c| c.index < 2990).map(|c| c.index).collect::<Vec<_>>()
    );
    assert!(near_real, "real CP at 3000 missed; got {:?}",
        cps.iter().map(|c| c.index).collect::<Vec<_>>()
    );
}

// ── Detrender long-stream stability ──────────────────────────────────

/// Pins the documented `update()` behaviour: the seasonal component is
/// EMA-blended on every update, but the linear trend coefficient and
/// intercept are frozen at fit time.
///
/// The interesting case is when underlying drift emerges *after* the fit
/// window -- Theil-Sen on the fit window returns ~0, and the post-fit drift
/// has nowhere to go but the residuals (or, via the EMA, into a slowly-
/// accumulating distortion of the seasonal component).
///
/// Test: 50,000-step pure-periodic signal for the fit window, then a
/// constant 0.001/step drift kicks in. Fit on first 200, then `update()`
/// for the remainder in 100-sample chunks. Compare residual *variance* on
/// the final window against the post-fit window. With the EMA absorbing
/// the per-period mean offset, the variance ratio is bounded even as the
/// absolute residual offset grows linearly with time.
#[test]
fn detrender_update_does_not_drift_over_long_stream() {
    let period = 60_usize;
    let amp = 3.0;
    let drift_after = 0.001_f64;
    let n_total = 50_000_usize;
    let fit_n = 200_usize;
    let chunk = 100_usize;

    let mut rng = Rng::new(0xD317_F71D);
    let signal: Vec<f64> = (0..n_total)
        .map(|i| {
            let drift_term = if i < fit_n {
                0.0
            } else {
                drift_after * (i - fit_n) as f64
            };
            amp * (i as f64 * 2.0 * PI / period as f64).sin()
                + drift_term
                + rng.normal(0.0, 0.3)
        })
        .collect();

    let mut det = Detrender::fit(&signal[..fit_n], period);

    let res_fit = det.detrend(&signal[..fit_n], 0);
    let mean_fit = res_fit.iter().sum::<f64>() / res_fit.len() as f64;
    let var_fit = res_fit.iter().map(|r| (r - mean_fit).powi(2)).sum::<f64>()
        / res_fit.len() as f64;

    let mut pos = fit_n;
    while pos < n_total {
        let end = (pos + chunk).min(n_total);
        det.update(&signal[pos..end], pos);
        pos = end;
    }

    let tail_start = n_total - 200;
    let res_tail = det.detrend(&signal[tail_start..], tail_start);
    let mean_tail = res_tail.iter().sum::<f64>() / res_tail.len() as f64;
    let var_tail = res_tail.iter().map(|r| (r - mean_tail).powi(2)).sum::<f64>()
        / res_tail.len() as f64;

    let ratio = var_tail / var_fit.max(1e-12);
    eprintln!(
        "detrender drift: post-fit var={var_fit:.4}  tail var={var_tail:.4}  ratio={ratio:.2}\n\
         tail residual mean (offset): {mean_tail:.2}  (linearly grows with t -- documented limitation)"
    );

    // Regression invariant: seasonal updates should bound residual variance.
    assert!(
        ratio < 2.5,
        "tail residual variance {var_tail:.4} > 5x post-fit {var_fit:.4} (ratio={ratio:.2}) \
         -- update()'s seasonal EMA stopped absorbing drift"
    );
}

#[test]
fn mass_cutoff_zero_disables_pruning() {
    let mut rng = Rng::new(0xBEEF_F00D);
    let data: Vec<f64> = (0..400).map(|_| rng.normal(0.0, 1.0)).collect();
    let pruned = BocpdDetector::new(200.0, 400).detect(&data);
    let unpruned = BocpdDetector::new(200.0, 400).with_mass_cutoff(0.0).detect(&data);
    // Both should agree on stationary input -- pruning a low-mass tail
    // shouldn't change MAP-drop behaviour.
    assert_eq!(pruned.len(), unpruned.len(),
        "pruned={} unpruned={} -- pruning changed detection on stationary input",
        pruned.len(), unpruned.len());
}

// ── Chen & Wu joint detector: hostile-state restore ──────────────────
//
// Mirrors the StreamingDetector hostile-state pins above for the joint
// detector's `ChenWuDetectorState`. A persisted/corrupted state reaches
// `restore()` either from disk or from a prior process crash; the
// recursion (eqs. 7-8) and emission loop in `streaming_chen_wu.rs`
// index `log_h_a` / `log_h_c` directly, so an empty / mismatched /
// oversized vector takes the daemon down on the next `step()` rather
// than failing closed at restore. These tests pin the validation that
// blocks each shape.

#[cfg(feature = "joint-detection")]
use cesura::streaming_chen_wu::{ChenWuDetectorState, StreamingChenWuDetector};

#[cfg(feature = "joint-detection")]
fn fresh_chen_wu_state() -> ChenWuDetectorState {
    StreamingChenWuDetector::new(0.1, 0.2, 4, 0.5, 0.5).save_state()
}

#[cfg(feature = "joint-detection")]
#[test]
fn chen_wu_state_with_mismatched_log_h_a_len_is_rejected() {
    let mut state = fresh_chen_wu_state();
    state.total_steps = 3;
    state.log_h_a = vec![f64::NEG_INFINITY, f64::NEG_INFINITY];
    state.log_h_c = vec![f64::NEG_INFINITY];
    let err = StreamingChenWuDetector::restore(state).err();
    assert!(
        err.as_deref().is_some_and(|e| e.contains("log_h_a") && e.contains("log_h_c")),
        "mismatched log_h_a/log_h_c lengths must be rejected, got {err:?}"
    );
}

#[cfg(feature = "joint-detection")]
#[test]
fn chen_wu_state_with_total_steps_but_empty_log_h_c_is_rejected() {
    let mut state = fresh_chen_wu_state();
    state.total_steps = 5;
    // log_h_a / log_h_c are already empty from `new()`.
    let err = StreamingChenWuDetector::restore(state).err();
    assert!(
        err.as_deref().is_some_and(|e| e.contains("total_steps") && e.contains("log_h_a")),
        "total_steps>0 with empty log_h_a/log_h_c must be rejected, got {err:?}"
    );
}

#[cfg(feature = "joint-detection")]
#[test]
fn chen_wu_state_with_oversized_log_h_a_is_rejected() {
    let mut state = fresh_chen_wu_state();
    // u_c default is 256, so log_h_a.len() must be <= 257.
    state.total_steps = 1;
    state.log_h_a = vec![f64::NEG_INFINITY; 1000];
    state.log_h_c = vec![f64::NEG_INFINITY; 1000];
    let err = StreamingChenWuDetector::restore(state).err();
    assert!(
        err.as_deref().is_some_and(|e| e.contains("log_h_a") && e.contains("u_c")),
        "log_h_a longer than u_c+1 must be rejected, got {err:?}"
    );
}

#[cfg(feature = "joint-detection")]
#[test]
fn chen_wu_state_with_oversized_h_c_history_is_rejected() {
    let mut state = fresh_chen_wu_state();
    // u_a default is 32, so h_c_history.len() must be <= 32.
    state.total_steps = 1;
    state.log_h_a = vec![f64::NEG_INFINITY];
    state.log_h_c = vec![f64::NEG_INFINITY];
    state.h_c_history = (0..100).map(|_| Vec::new()).collect();
    let err = StreamingChenWuDetector::restore(state).err();
    assert!(
        err.as_deref().is_some_and(|e| e.contains("h_c_history") && e.contains("u_a")),
        "oversized h_c_history must be rejected, got {err:?}"
    );
}

#[cfg(feature = "joint-detection")]
#[test]
fn chen_wu_state_with_oversized_h_c_history_entry_is_rejected() {
    let mut state = fresh_chen_wu_state();
    // delta_t is 4, so each h_c_history entry must be <= 4 long.
    state.total_steps = 1;
    state.log_h_a = vec![f64::NEG_INFINITY];
    state.log_h_c = vec![f64::NEG_INFINITY];
    state.h_c_history = vec![vec![f64::NEG_INFINITY; 100]];
    let err = StreamingChenWuDetector::restore(state).err();
    assert!(
        err.as_deref().is_some_and(|e| e.contains("h_c_history") && e.contains("delta_t")),
        "h_c_history entry longer than delta_t must be rejected, got {err:?}"
    );
}

#[cfg(feature = "joint-detection")]
#[test]
fn chen_wu_state_with_raw_history_overflow_is_rejected() {
    let mut state = fresh_chen_wu_state();
    state.total_steps = 1;
    state.log_h_a = vec![f64::NEG_INFINITY];
    state.log_h_c = vec![f64::NEG_INFINITY];
    state.raw_history = vec![0.0; 10_000];
    let err = StreamingChenWuDetector::restore(state).err();
    assert!(
        err.as_deref().is_some_and(|e| e.contains("raw_history")),
        "oversized raw_history must be rejected, got {err:?}"
    );
}

#[cfg(feature = "joint-detection")]
#[test]
fn chen_wu_state_with_lambda_out_of_range_is_rejected() {
    let mut state = fresh_chen_wu_state();
    state.lambda_a = 1.5;
    let err = StreamingChenWuDetector::restore(state).err();
    assert!(
        err.as_deref().is_some_and(|e| e.contains("lambda_a")),
        "lambda_a >= 1 must be rejected, got {err:?}"
    );

    let mut state = fresh_chen_wu_state();
    state.lambda_c = -0.1;
    let err = StreamingChenWuDetector::restore(state).err();
    assert!(
        err.as_deref().is_some_and(|e| e.contains("lambda_c")),
        "lambda_c < 0 must be rejected, got {err:?}"
    );
}

#[cfg(feature = "joint-detection")]
#[test]
fn chen_wu_state_with_non_finite_prior_is_rejected() {
    let mut state = fresh_chen_wu_state();
    state.prior.mu = f64::NAN;
    let err = StreamingChenWuDetector::restore(state).err();
    assert!(
        err.as_deref().is_some_and(|e| e.contains("prior.mu")),
        "NaN in prior.mu must be rejected, got {err:?}"
    );

    let mut state = fresh_chen_wu_state();
    state.prior.beta = f64::INFINITY;
    let err = StreamingChenWuDetector::restore(state).err();
    assert!(
        err.as_deref().is_some_and(|e| e.contains("prior.beta")),
        "+inf in prior.beta must be rejected, got {err:?}"
    );
}

#[cfg(feature = "joint-detection")]
#[test]
fn chen_wu_malformed_json_returns_err_not_panic() {
    let bogus = r#"{"this": "is not", "a": "ChenWuDetectorState"}"#;
    let result: Result<ChenWuDetectorState, _> = serde_json::from_str(bogus);
    assert!(result.is_err(), "garbage JSON must fail to parse");

    let truncated = r#"{"p0": 0.1, "q0":"#;
    let result: Result<ChenWuDetectorState, _> = serde_json::from_str(truncated);
    assert!(result.is_err(), "truncated JSON must fail");
}

#[cfg(feature = "joint-detection")]
#[test]
fn chen_wu_state_with_nan_in_raw_history_does_not_propagate_silently() {
    // raw_history is not range/finiteness-validated by restore (it's
    // user payload, not a recursion-shape invariant). NaN injected here
    // could poison rebuild_stats walks. Subsequent step() must not emit
    // NaN-tainted detections regardless of how the NaN got in.
    let mut det = StreamingChenWuDetector::new(0.1, 0.2, 4, 0.5, 0.5);
    let mut rng = Rng::new(7);
    let warmup: Vec<f64> = (0..30).map(|_| rng.normal(0.0, 1.0)).collect();
    let _ = det.step(&warmup);
    let mut state = det.save_state();
    if state.raw_history.len() >= 5 {
        state.raw_history[4] = f64::NAN;
    } else {
        state.raw_history.push(f64::NAN);
    }
    let mut det2 = StreamingChenWuDetector::restore(state)
        .expect("restore must accept NaN in raw_history (not in the validation list)");

    let mut post: Vec<f64> = (0..120).map(|_| rng.normal(0.0, 1.0)).collect();
    post.extend((0..120).map(|_| rng.normal(5.0, 1.0)));
    let dets = det2.step(&post);
    for d in &dets {
        match d {
            cesura::chen_wu::Detection::ChangePoint(cp) => {
                assert!(cp.confidence.is_finite(), "NaN leaked to CP confidence");
                assert!(cp.shift_sigma.is_finite(), "NaN leaked to shift_sigma");
            }
            cesura::chen_wu::Detection::CollectiveAnomaly { confidence, .. } => {
                assert!(confidence.is_finite(), "NaN leaked to anomaly confidence");
            }
        }
    }
}

#[cfg(feature = "joint-detection")]
#[test]
fn chen_wu_state_with_negative_alpha_does_not_panic() {
    // α ≤ 0 makes the predictive Student-t df ≤ 0 → predictive returns
    // NEG_INFINITY. Detector must fall through gracefully. Mirror of
    // `state_with_negative_alpha_does_not_panic` for StreamingDetector.
    let mut state = fresh_chen_wu_state();
    state.prior.alpha = -1.0;
    state.prior.beta = -1.0;
    let mut det = StreamingChenWuDetector::restore(state).unwrap();
    let _ = det.step(&[0.0; 50]);
}

#[cfg(feature = "joint-detection")]
#[test]
fn chen_wu_state_with_invalid_last_cp_is_rejected() {
    let mut state = fresh_chen_wu_state();
    state.total_steps = 1;
    state.log_h_a = vec![f64::NEG_INFINITY];
    state.log_h_c = vec![f64::NEG_INFINITY];
    state.last_cp = Some((100, 100));
    let err = StreamingChenWuDetector::restore(state).err();
    assert!(
        err.as_deref().is_some_and(|e| e.contains("last_cp")),
        "last_cp idx >= total_steps must be rejected, got {err:?}"
    );
}

#[cfg(feature = "joint-detection")]
#[test]
fn chen_wu_state_with_invalid_scalars_is_rejected() {
    // Walks scalar-validation branches that the
    // dedicated tests above don't individually cover.
    type Case = (&'static str, fn(&mut ChenWuDetectorState));
    let cases: Vec<Case> = vec![
        ("u_c", |s| s.u_a = s.u_c + 1),
        ("delta", |s| s.delta = s.u_c + 1),
        ("min_post_change_obs", |s| s.min_post_change_obs = 0),
        ("min_post_change_obs", |s| s.min_post_change_obs = s.u_c + 1),
        ("beta", |s| s.beta = 1.5),
        ("abs_origin", |s| s.abs_origin = 0),
    ];
    for (label, mutate) in cases {
        let mut state = fresh_chen_wu_state();
        mutate(&mut state);
        let err = StreamingChenWuDetector::restore(state).err();
        assert!(
            err.as_deref().is_some_and(|e| e.contains(label)),
            "case {label}: expected Err mentioning {label}, got {err:?}"
        );
    }
}

#[cfg(feature = "joint-detection")]
#[test]
fn chen_wu_state_with_zero_total_steps_but_nonempty_vectors_is_rejected() {
    let mut state = fresh_chen_wu_state();
    // total_steps stays 0; populate raw_history.
    state.raw_history = vec![1.0, 2.0, 3.0];
    let err = StreamingChenWuDetector::restore(state).err();
    assert!(
        err.as_deref().is_some_and(|e| e.contains("total_steps == 0")),
        "total_steps == 0 with non-empty raw_history must be rejected, got {err:?}"
    );
}

#[cfg(feature = "joint-detection")]
#[test]
fn chen_wu_state_with_excluded_range_violation_is_rejected() {
    let mut state = fresh_chen_wu_state();
    state.excluded_ranges = vec![(0, 5)];
    let err = StreamingChenWuDetector::restore(state).err();
    assert!(
        err.as_deref().is_some_and(|e| e.contains("excluded_ranges")),
        "excluded_ranges with start < 1 must be rejected, got {err:?}"
    );

    let mut state = fresh_chen_wu_state();
    state.excluded_ranges = vec![(10, 5)];
    let err = StreamingChenWuDetector::restore(state).err();
    assert!(
        err.as_deref().is_some_and(|e| e.contains("excluded_ranges")),
        "excluded_ranges with start > end must be rejected, got {err:?}"
    );
}
