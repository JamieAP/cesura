//! Chen & Wu joint detector integration tests.
//!
//! Run: `cargo test --features test-utils,joint-detection --test chen_wu`.
//! Heavy paper-fixture test gated `#[ignore]`; run with `--ignored`.

#![allow(clippy::needless_range_loop)]

use cesura::auto_q0::auto_q0;
use cesura::chen_wu::{ChenWuDetector, Detection};
use cesura::eval::Rng;

#[test]
fn confidence_in_unit_interval() {
    let mut rng = Rng::new(99);
    let mut data: Vec<f64> = (0..150).map(|_| rng.normal(0.0, 1.0)).collect();
    data.extend((0..150).map(|_| rng.normal(3.0, 1.0)));
    let det = ChenWuDetector::new(0.1, 0.2, 4, 0.5, 0.5);
    for d in det.detect(&data) {
        match d {
            Detection::ChangePoint(cp) => {
                assert!((0.0..=1.0).contains(&cp.confidence));
                assert!(cp.shift_sigma.is_finite());
                assert!(cp.shift_sigma >= 0.0);
            }
            Detection::CollectiveAnomaly { confidence, start, end, .. } => {
                assert!((0.0..=1.0).contains(&confidence));
                assert!(start <= end, "anomaly start {start} > end {end}");
            }
        }
    }
}

#[test]
fn shift_sigma_populated_for_change_points() {
    // Strong shift: 5σ. shift_sigma should land near 5 (within slop for the
    // 20-obs window pooling). Asserts the v0.7 fix that no longer hardcodes 0.0.
    let mut rng = Rng::new(123);
    let mut data: Vec<f64> = (0..150).map(|_| rng.normal(0.0, 1.0)).collect();
    data.extend((0..150).map(|_| rng.normal(5.0, 1.0)));
    let det = ChenWuDetector::new(0.1, 0.2, 4, 0.5, 0.5);
    let dets = det.detect(&data);
    let cps: Vec<_> = dets
        .iter()
        .filter_map(|d| match d {
            Detection::ChangePoint(cp) => Some(cp),
            _ => None,
        })
        .collect();
    assert!(!cps.is_empty(), "expected at least one CP");
    let max_shift = cps.iter().map(|c| c.shift_sigma).fold(0.0f64, f64::max);
    assert!(
        max_shift > 1.0,
        "expected shift_sigma > 1 on a 5σ shift, got max {max_shift}"
    );
}

#[test]
#[ignore = "paper § 6.1 fixture; run with --ignored"]
fn paper_section_6_1_fixture_qualitative() {
    // Paper § 6.1 simulation: length 1000, 6 CPs at {75,175,300,450,625,825},
    // means from {2,4,6,8} (adjacent differ), σ = 0.5, anomaly every 100 obs
    // with duration 1 or 4 and mean shift in {±2, ±4}. All anomalies are
    // collective except the spurious one at t = 300.
    //
    // We generate one realisation (not the full 1000-trial average from
    // table 3) and assert qualitative recovery: ≥ 4/6 known CPs detected,
    // and at least half the inserted anomalies emit something nearby.
    let mut rng = Rng::new(20260507);
    let n = 1000;
    let true_cps = [75usize, 175, 300, 450, 625, 825];
    let mean_choices = [2.0, 4.0, 6.0, 8.0];

    // Pick segment means with the adjacent-must-differ constraint.
    let mut segment_means = Vec::with_capacity(true_cps.len() + 1);
    segment_means.push(mean_choices[(rng.next_u64() % 4) as usize]);
    for _ in 0..true_cps.len() {
        loop {
            let candidate = mean_choices[(rng.next_u64() % 4) as usize];
            if candidate != *segment_means.last().unwrap() {
                segment_means.push(candidate);
                break;
            }
        }
    }

    // Generate the base series.
    let mut data = vec![0.0f64; n];
    let mut seg = 0;
    for t in 0..n {
        if seg < true_cps.len() && t == true_cps[seg] {
            seg += 1;
        }
        data[t] = rng.normal(segment_means[seg], 0.5);
    }

    // Insert anomalies every 100 obs (offset 50 to avoid exact CP overlap).
    let anom_centres: Vec<usize> = (50..n).step_by(100).collect();
    let mut inserted_anom: Vec<(usize, usize)> = Vec::new();
    for (i, &centre) in anom_centres.iter().enumerate() {
        let dur = if rng.uniform() < 0.5 { 1 } else { 4 };
        let signs = [-4.0, -2.0, 2.0, 4.0];
        let shift = signs[(rng.next_u64() % 4) as usize];
        let start = centre.saturating_sub(dur / 2);
        let end = (start + dur).min(n);
        for t in start..end {
            // Spurious anomaly at i ≈ time-300 region: keep the shift but
            // the paper marks it as not-a-collective-anomaly. Tracked
            // separately for reporting; the detector treats it the same.
            data[t] += shift;
        }
        // Record only collective anomalies (skip the spurious one).
        if i != 2 {
            // approximate: anomaly #3 (index 2) sits around t=250-300
            inserted_anom.push((start, end));
        }
    }

    let det = ChenWuDetector::new(0.1, 0.2, 4, 0.5, 0.5)
        .with_search_windows(299, 27)
        .with_localisation_tolerance(0)
        .with_prior(0.0, 0.01, 0.5, 0.125);

    let dets = det.detect(&data);
    let cp_indices: Vec<usize> = dets
        .iter()
        .filter_map(|d| match d {
            Detection::ChangePoint(cp) => Some(cp.index),
            _ => None,
        })
        .collect();
    let anom_windows: Vec<(usize, usize)> = dets
        .iter()
        .filter_map(|d| match d {
            Detection::CollectiveAnomaly { start, end, .. } => Some((*start, *end)),
            _ => None,
        })
        .collect();

    eprintln!(
        "\n=== Chen & Wu BOCD-AR on § 6.1 fixture ===\n  CPs detected: {} (truth: 6)\n  anomalies detected: {}",
        cp_indices.len(),
        anom_windows.len()
    );
    eprintln!("  CP indices: {cp_indices:?}");
    eprintln!("  CP truth:   {true_cps:?}");
    eprintln!("  anomaly windows: {anom_windows:?}");
    eprintln!("  inserted anomalies: {inserted_anom:?}");

    let cps_recovered = true_cps
        .iter()
        .filter(|&&truth| {
            cp_indices
                .iter()
                .any(|&det| (det as i64 - truth as i64).abs() <= 25)
        })
        .count();
    let anom_recovered = inserted_anom
        .iter()
        .filter(|(s, e)| {
            anom_windows
                .iter()
                .any(|(ds, de)| ranges_overlap(*ds, *de, *s, *e))
        })
        .count();
    eprintln!(
        "  CPs recovered: {cps_recovered}/6  anomalies recovered: {anom_recovered}/{}",
        inserted_anom.len()
    );

    assert!(
        cps_recovered >= 4,
        "expected ≥ 4/6 CPs recovered within ±25 obs, got {cps_recovered}"
    );
    assert!(
        anom_recovered * 2 >= inserted_anom.len(),
        "expected ≥ 50% of inserted anomalies recovered, got {anom_recovered}/{}",
        inserted_anom.len()
    );
    // Precision floor. Over-emission catches recursion bugs that produce
    // extra spurious CPs while still matching truth points (e.g. wrong r*
    // selection emitting CPs at anomaly endpoints). 6 truth + 4 slop.
    assert!(
        cp_indices.len() <= 10,
        "over-detection: {} CPs against 6 truth points -- recursion or \
         emission rule may be producing spurious CPs at anomaly windows",
        cp_indices.len()
    );

    // Regression snapshot. Pins the exact CP indices from this specific
    // seed + hyperparameter combination. A change here that doesn't also
    // come with a deliberate fixture update means a recursion-level
    // regression -- e.g. an off-by-one in the predictive, a renorm bug,
    // or wrong r* selection. Update intentionally; do not weaken to
    // `<=` set membership.
    // Snapshot updated when the inside_emitted_anomaly heuristic was
    // replaced by paper § 4.3 Υ_c^t persistence (v0.8). The previous
    // ±2-window CP suppression around emitted anomalies is gone; CPs
    // that fire one paper-step before an anomaly start now surface
    // (147, 747, 847). The over-detection cap (10) still gates this --
    // recall is unchanged at 6/6 truth points within ±25 obs.
    assert_eq!(
        cp_indices,
        vec![0, 75, 147, 175, 300, 452, 625, 747, 825, 847],
        "§ 6.1 CP indices changed -- recursion regression?"
    );
    assert_eq!(
        anom_windows,
        vec![
            (50, 51),
            (148, 152),
            (248, 252),
            (348, 352),
            (548, 552),
            (748, 752),
            (848, 852),
        ],
        "§ 6.1 anomaly windows changed -- emission regression?"
    );
}

#[test]
fn dense_anomalies_no_spurious_cps_at_boundaries() {
    // 10 short collective anomalies in 300 obs of N(0,1) noise. Paper
    // § 4.3 Υ_c^t persistence (v0.8) must suppress CPs that land *inside*
    // any emitted anomaly window. The previous code passed this via the
    // inside_emitted_anomaly heuristic; this test gates the heuristic's
    // removal -- it must keep passing once the search-range removal is
    // the sole mechanism.
    let mut rng = Rng::new(20260507);
    let n = 300;
    let mut data: Vec<f64> = (0..n).map(|_| rng.normal(0.0, 1.0)).collect();
    let centres: Vec<usize> = (15..n).step_by(28).take(10).collect();
    for &c in &centres {
        let dur = 3;
        let shift = if (c / 7) % 2 == 0 { 4.0 } else { -4.0 };
        for t in c..(c + dur).min(n) {
            data[t] += shift;
        }
    }

    let det = ChenWuDetector::new(0.1, 0.2, 4, 0.5, 0.5);
    let dets = det.detect(&data);
    let cps: Vec<usize> = dets
        .iter()
        .filter_map(|d| match d {
            Detection::ChangePoint(cp) => Some(cp.index),
            _ => None,
        })
        .collect();
    let anom_windows: Vec<(usize, usize)> = dets
        .iter()
        .filter_map(|d| match d {
            Detection::CollectiveAnomaly { start, end, .. } => Some((*start, *end)),
            _ => None,
        })
        .collect();

    eprintln!(
        "dense anomalies: {} CPs, {} anomalies\n  CPs: {cps:?}\n  windows: {anom_windows:?}",
        cps.len(),
        anom_windows.len()
    );

    for (s, e) in &anom_windows {
        for &cp in &cps {
            assert!(
                cp < *s || cp > *e,
                "CP {cp} lands *inside* emitted anomaly window ({s}, {e}) -- search-range removal regression"
            );
        }
    }
}


fn ranges_overlap(a_lo: usize, a_hi: usize, b_lo: usize, b_hi: usize) -> bool {
    a_lo <= b_hi && b_lo <= a_hi
}

#[test]
fn log_space_no_inf_under_pathological_input() {
    // Mixed-scale stream: alternating N(0, 1) and N(0, 1e6) blocks. The
    // pre-stage-5 detection rule materialised `h_a / h_c` via `.exp()`
    // before the argmax / ratio loops, so any per-r log mass that
    // accumulated near `f64::MAX_EXP` would surface as `f64::INFINITY`
    // in the linear copy and propagate to detection confidences. The
    // log-space rewrite computes ratios via `(log_num - log_den).exp()`
    // -- the subtraction lands the result in `[0, 1]` before the only
    // `.exp()` call. Verifies no `Detection::ChangePoint.confidence`,
    // `shift_sigma`, or `Detection::CollectiveAnomaly.confidence` is
    // non-finite.
    let mut rng = Rng::new(0xC1A0_5EFE);
    let mut data = Vec::with_capacity(5000);
    for k in 0..5000 {
        let scale = if (k / 250) % 2 == 0 { 1.0 } else { 1e6 };
        data.push(rng.normal(0.0, scale));
    }
    let det = ChenWuDetector::new(0.1, 0.2, 4, 0.5, 0.5);
    let dets = det.detect(&data);
    for d in &dets {
        match d {
            Detection::ChangePoint(cp) => {
                assert!(
                    cp.confidence.is_finite(),
                    "non-finite CP confidence: {cp:?}"
                );
                assert!(
                    cp.shift_sigma.is_finite(),
                    "non-finite shift_sigma: {cp:?}"
                );
                assert!(
                    (0.0..=1.0).contains(&cp.confidence),
                    "CP confidence out of [0,1]: {cp:?}"
                );
            }
            Detection::CollectiveAnomaly { confidence, start, end } => {
                assert!(
                    confidence.is_finite(),
                    "non-finite anomaly confidence at ({start},{end}): {confidence}"
                );
                assert!(
                    (0.0..=1.0).contains(confidence),
                    "anomaly confidence out of [0,1]: {confidence}"
                );
                assert!(start <= end, "anomaly start {start} > end {end}");
            }
        }
    }
}

#[test]
fn with_auto_q0_returns_eq_14_upper_bound() {
    // Smoke test the public `auto_q0` re-export and confirm the
    // builder applies the same value the helper computes.
    let p0 = 0.1_f64;
    let lambda_a = 0.5_f64;
    let delta_t = 4_usize;
    let helper_q = auto_q0(p0, lambda_a, delta_t);
    assert!(
        helper_q.is_finite() && (0.0..1.0).contains(&helper_q),
        "auto_q0 should return q_0 ∈ [0, 1), got {helper_q}"
    );
    // The picker must land strictly below the paper's permissive
    // q_0 = 0.2 (LHS at q_0 = 0.2 evaluates above λ_a = 0.5; see
    // `src/auto_q0.rs::tests::upper_bound_at_paper_section_6_1_settings`).
    assert!(
        helper_q < 0.2,
        "auto_q0(0.1, 0.5, 4) = {helper_q}; paper's q_0 = 0.2 is above the strict bound"
    );
}

#[test]
fn chen_wu_with_auto_q0_paper_section_6_1_fixture() {
    // § 6.1 fixture (subset: 300 obs, mid-stream shift) under
    // `with_auto_q0()`. With the strict upper bound replacing
    // q_0 = 0.2 the detector is *more conservative*; recall on the
    // strong shift must hold and detection count must not blow up.
    // Allow ±2 detection-count drift versus the hand-tuned
    // q_0 = 0.2 baseline.
    let mut rng_a = Rng::new(2026);
    let mut data: Vec<f64> = (0..150).map(|_| rng_a.normal(0.0, 1.0)).collect();
    data.extend((0..150).map(|_| rng_a.normal(4.0, 1.0)));

    let baseline = ChenWuDetector::new(0.1, 0.2, 4, 0.5, 0.5).detect(&data);
    let auto = ChenWuDetector::new(0.1, 0.2, 4, 0.5, 0.5)
        .with_auto_q0()
        .detect(&data);

    let baseline_cps: Vec<usize> = baseline
        .iter()
        .filter_map(|d| match d {
            Detection::ChangePoint(cp) => Some(cp.index),
            _ => None,
        })
        .collect();
    let auto_cps: Vec<usize> = auto
        .iter()
        .filter_map(|d| match d {
            Detection::ChangePoint(cp) => Some(cp.index),
            _ => None,
        })
        .collect();

    let truth = 150usize;
    let baseline_hit = baseline_cps
        .iter()
        .any(|&i| (i as i64 - truth as i64).abs() < 30);
    let auto_hit = auto_cps
        .iter()
        .any(|&i| (i as i64 - truth as i64).abs() < 30);
    assert!(baseline_hit, "baseline q_0 = 0.2 must catch the truth");
    assert!(
        auto_hit,
        "with_auto_q0 must keep recall on the truth: cps={auto_cps:?}"
    );
    let drift = (auto.len() as i64 - baseline.len() as i64).abs();
    assert!(
        drift <= 2,
        "with_auto_q0 detection count drifted by {drift} \
         (baseline={}, auto={})",
        baseline.len(),
        auto.len()
    );
}

#[test]
fn with_auto_q0_idempotent() {
    // The bound depends only on (p_0, λ_a, Δt); applying the builder
    // twice must return identical CP/anomaly streams to applying it
    // once.
    let mut rng = Rng::new(101);
    let mut data: Vec<f64> = (0..100).map(|_| rng.normal(0.0, 1.0)).collect();
    data.extend((0..100).map(|_| rng.normal(4.0, 1.0)));

    let once = ChenWuDetector::new(0.1, 0.2, 4, 0.5, 0.5)
        .with_auto_q0()
        .detect(&data);
    let twice = ChenWuDetector::new(0.1, 0.2, 4, 0.5, 0.5)
        .with_auto_q0()
        .with_auto_q0()
        .detect(&data);
    assert_eq!(
        once.len(),
        twice.len(),
        "double-apply changed detection count: {once:?} vs {twice:?}"
    );
}

#[test]
fn api_lock_every_builder_compiles() {
    // Locks the public API surface for v0.8: any rename, parameter
    // reorder, or visibility change of a builder method breaks this
    // compile-and-run check, which doubles as a worked example for
    // the README. `with_robust` lives behind `feature = "robust"` so
    // it's covered by the dedicated robust test below.
    let det = ChenWuDetector::new(0.1, 0.2, 4, 0.5, 0.5)
        .with_search_windows(128, 16)
        .with_localisation_tolerance(0)
        .with_min_post_change(5)
        .with_prior(0.0, 0.01, 0.5, 0.125);
    let mut rng = Rng::new(42);
    let data: Vec<f64> = (0..50).map(|_| rng.normal(0.0, 1.0)).collect();
    let dets = det.detect(&data);
    for d in &dets {
        match d {
            Detection::ChangePoint(cp) => {
                let _ = (cp.index, cp.confidence, cp.shift_sigma);
            }
            Detection::CollectiveAnomaly { start, end, confidence } => {
                let _ = (start, end, confidence);
            }
        }
    }

    // Streaming surface mirror.
    use cesura::streaming_chen_wu::{ChenWuDetectorState, StreamingChenWuDetector};
    let mut sdet = StreamingChenWuDetector::new(0.1, 0.2, 4, 0.5, 0.5)
        .with_search_windows(128, 16)
        .with_localisation_tolerance(0)
        .with_min_post_change(5)
        .with_prior(0.0, 0.01, 0.5, 0.125);
    let _: Vec<Detection> = sdet.step(&data);
    let state: ChenWuDetectorState = sdet.save_state();
    let _ = serde_json::to_string(&state).unwrap();
    let restored: ChenWuDetectorState = state;
    let _restored: StreamingChenWuDetector = StreamingChenWuDetector::restore(restored).unwrap();
}

#[test]
fn with_robust_zero_beta_matches_standard() {
    // β = 0 must short-circuit to bit-for-bit standard behaviour.
    let mut rng = Rng::new(2026);
    let mut data: Vec<f64> = (0..150).map(|_| rng.normal(0.0, 1.0)).collect();
    data.extend((0..150).map(|_| rng.normal(4.0, 1.0)));

    let standard = ChenWuDetector::new(0.1, 0.2, 4, 0.5, 0.5).detect(&data);
    let robust_zero = ChenWuDetector::new(0.1, 0.2, 4, 0.5, 0.5)
        .with_robust(0.0)
        .detect(&data);

    assert_eq!(
        standard.len(),
        robust_zero.len(),
        "β=0 must match standard count: standard={} robust={}",
        standard.len(),
        robust_zero.len()
    );
    for (a, b) in standard.iter().zip(robust_zero.iter()) {
        match (a, b) {
            (Detection::ChangePoint(x), Detection::ChangePoint(y)) => {
                assert_eq!(x.index, y.index, "CP index drift at β=0");
            }
            (
                Detection::CollectiveAnomaly { start: s1, end: e1, .. },
                Detection::CollectiveAnomaly { start: s2, end: e2, .. },
            ) => {
                assert_eq!((s1, e1), (s2, e2), "anomaly window drift at β=0");
            }
            _ => panic!("variant mismatch between standard and β=0 robust"),
        }
    }
}

#[test]
fn with_robust_reduces_spurious_cps_on_heavy_tail() {
    // The whole point of β-divergence is to suppress spurious detections
    // on heavy-tailed within-regime noise. Pin that contract: on pure t₃
    // noise (no real shift), β > 0 must emit fewer ChangePoints than
    // β = 0. Asserts the *qualitative* improvement, not a specific F1.
    let mut rng_a = Rng::new(424242);
    let mut rng_b = Rng::new(424242);
    let data_a: Vec<f64> = (0..500).map(|_| rng_a.student_t3()).collect();
    let data_b: Vec<f64> = (0..500).map(|_| rng_b.student_t3()).collect();
    assert_eq!(data_a, data_b, "RNG must be deterministic");

    let standard_cps = ChenWuDetector::new(0.1, 0.2, 4, 0.5, 0.5)
        .detect(&data_a)
        .iter()
        .filter(|d| matches!(d, Detection::ChangePoint(_)))
        .count();
    let robust_cps = ChenWuDetector::new(0.1, 0.2, 4, 0.5, 0.5)
        .with_robust(0.15)
        .detect(&data_b)
        .iter()
        .filter(|d| matches!(d, Detection::ChangePoint(_)))
        .count();

    eprintln!("t3 noise spurious CPs: standard={standard_cps} robust(β=0.15)={robust_cps}");
    assert!(
        robust_cps <= standard_cps,
        "β=0.15 should fire ≤ standard on pure t₃ noise, got robust={robust_cps} standard={standard_cps}"
    );
}

#[test]
fn with_robust_finite_under_heavy_tail() {
    // Heavy-tailed within-regime noise. β > 0 should produce finite,
    // sensible output; this is a smoke test, not an F1 comparison.
    let mut rng = Rng::new(7000);
    // t₃-distributed within-regime noise: kurtosis ≈ 6 over 300 obs.
    let data: Vec<f64> = (0..400).map(|_| rng.student_t3()).collect();
    let robust = ChenWuDetector::new(0.1, 0.2, 4, 0.5, 0.5)
        .with_robust(0.15)
        .detect(&data);
    for d in &robust {
        match d {
            Detection::ChangePoint(cp) => {
                assert!(cp.confidence.is_finite() && (0.0..=1.0).contains(&cp.confidence));
                assert!(cp.shift_sigma.is_finite() && cp.shift_sigma >= 0.0);
            }
            Detection::CollectiveAnomaly { confidence, start, end, .. } => {
                assert!(confidence.is_finite() && (0.0..=1.0).contains(confidence));
                assert!(start <= end);
            }
        }
    }
}
