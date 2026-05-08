//! Streaming Chen & Wu detector integration tests.
//!
//! Run: `cargo test --features test-utils,joint-detection --test streaming_chen_wu`.

#![allow(clippy::needless_range_loop)]

use cesura::chen_wu::{ChenWuDetector, Detection};
use cesura::eval::Rng;
use cesura::streaming_chen_wu::StreamingChenWuDetector;

fn detection_keys(dets: &[Detection]) -> Vec<(usize, usize, usize, u64, u64)> {
    // (kind, key1, key2, confidence_bits, shift_sigma_bits) sorted.
    // kind: 0 = ChangePoint, 1 = CollectiveAnomaly.
    // For CP, key1 = index, key2 = 0.
    // For anomaly, key1 = start, key2 = end.
    let mut keys: Vec<_> = dets
        .iter()
        .map(|d| match d {
            Detection::ChangePoint(cp) => (
                0usize,
                cp.index,
                0usize,
                cp.confidence.to_bits(),
                cp.shift_sigma.to_bits(),
            ),
            Detection::CollectiveAnomaly { start, end, confidence } => (
                1usize,
                *start,
                *end,
                confidence.to_bits(),
                0u64,
            ),
        })
        .collect();
    keys.sort();
    keys
}

#[test]
fn detector_runs_without_panic() {
    let mut rng = Rng::new(1);
    let data: Vec<f64> = (0..1000).map(|_| rng.normal(0.0, 1.0)).collect();
    let mut det = StreamingChenWuDetector::new(0.1, 0.2, 4, 0.5, 0.5);
    let _ = det.step(&data);
    assert_eq!(det.total_steps(), 1000);
}

#[test]
fn detect_clean_cp() {
    let mut rng = Rng::new(2);
    let mut data: Vec<f64> = (0..100).map(|_| rng.normal(0.0, 1.0)).collect();
    data.extend((0..100).map(|_| rng.normal(5.0, 1.0)));
    let mut det = StreamingChenWuDetector::new(0.1, 0.2, 4, 0.5, 0.5);
    let dets = det.step(&data);
    let any_cp_near_100 = dets.iter().any(|d| match d {
        Detection::ChangePoint(cp) => (cp.index as i64 - 100).abs() < 30,
        _ => false,
    });
    assert!(any_cp_near_100, "expected CP near 100, got {dets:?}");
}

#[test]
fn back_to_back_anomalies() {
    let mut rng = Rng::new(4);
    let mut data: Vec<f64> = (0..100).map(|_| rng.normal(0.0, 1.0)).collect();
    data.extend((0..3).map(|_| rng.normal(5.0, 1.0)));
    data.extend((0..1).map(|_| rng.normal(0.0, 1.0)));
    data.extend((0..3).map(|_| rng.normal(-5.0, 1.0)));
    data.extend((0..100).map(|_| rng.normal(0.0, 1.0)));
    let mut det = StreamingChenWuDetector::new(0.1, 0.2, 4, 0.5, 0.5);
    let dets = det.step(&data);
    let n_anom = dets
        .iter()
        .filter(|d| matches!(d, Detection::CollectiveAnomaly { .. }))
        .count();
    assert!(n_anom >= 2, "expected ≥ 2 anomalies, got {n_anom} in {dets:?}");
}

#[test]
fn matches_batch_detect_clean_shift() {
    // Recursion-equivalence check on the simplest fixture: streaming
    // step() over a clean shift produces the same multiset of
    // detections (with bit-equal confidences and shift_sigmas) as
    // batch detect(). This anchors the contract that batch and
    // streaming share the recursion + emission rules.
    let mut rng = Rng::new(2024);
    let mut data: Vec<f64> = (0..150).map(|_| rng.normal(0.0, 1.0)).collect();
    data.extend((0..150).map(|_| rng.normal(4.0, 1.0)));

    let batch = ChenWuDetector::new(0.1, 0.2, 4, 0.5, 0.5).detect(&data);
    let mut stream_det = StreamingChenWuDetector::new(0.1, 0.2, 4, 0.5, 0.5);
    let stream = stream_det.step(&data);

    assert_eq!(
        detection_keys(&batch),
        detection_keys(&stream),
        "streaming and batch produced different detections\n  batch  : {batch:?}\n  stream : {stream:?}"
    );
}

#[test]
fn matches_batch_detect_paper_section_6_1() {
    // The big test: § 6.1 fixture, 1000 obs, 6 truth CPs. Streaming
    // and batch must produce bit-equal detections including
    // shift_sigma values. Mirrors tests/chen_wu.rs::paper_section_6_1
    // but in matches-batch form.
    let mut rng = Rng::new(20260507);
    let n = 1000;
    let true_cps = [75usize, 175, 300, 450, 625, 825];
    let mean_choices = [2.0, 4.0, 6.0, 8.0];

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

    let mut data = vec![0.0f64; n];
    let mut seg = 0;
    for t in 0..n {
        if seg < true_cps.len() && t == true_cps[seg] {
            seg += 1;
        }
        data[t] = rng.normal(segment_means[seg], 0.5);
    }
    let anom_centres: Vec<usize> = (50..n).step_by(100).collect();
    for &centre in &anom_centres {
        let dur = if rng.uniform() < 0.5 { 1 } else { 4 };
        let signs = [-4.0, -2.0, 2.0, 4.0];
        let shift = signs[(rng.next_u64() % 4) as usize];
        let start = centre.saturating_sub(dur / 2);
        let end = (start + dur).min(n);
        for t in start..end {
            data[t] += shift;
        }
    }

    let batch = ChenWuDetector::new(0.1, 0.2, 4, 0.5, 0.5)
        .with_search_windows(299, 27)
        .with_localisation_tolerance(0)
        .with_prior(0.0, 0.01, 0.5, 0.125)
        .detect(&data);

    let mut stream_det = StreamingChenWuDetector::new(0.1, 0.2, 4, 0.5, 0.5)
        .with_search_windows(299, 27)
        .with_localisation_tolerance(0)
        .with_prior(0.0, 0.01, 0.5, 0.125);
    let stream = stream_det.step(&data);

    let bk = detection_keys(&batch);
    let sk = detection_keys(&stream);
    assert_eq!(
        bk, sk,
        "§ 6.1 streaming and batch diverged.\n  batch  : {batch:?}\n  stream : {stream:?}"
    );
}

// ── NaN-input contract ────────────────────────────────────────────────
//
// The streaming joint detector now mirrors `StreamingDetector`'s
// non-finite-input guard: NaN / ±inf samples are silently skipped --
// they do not advance `total_steps`, do not enter `raw_history`, and
// produce no detection. Before the guard, NaN poisoned
// `likelihood`, propagated through `log_pred`, and surfaced as
// NaN-tainted detections.

#[test]
fn nan_in_step_does_not_panic() {
    let mut rng = Rng::new(13);
    let mut data: Vec<f64> = (0..150).map(|_| rng.normal(0.0, 1.0)).collect();
    data.extend((0..150).map(|_| rng.normal(5.0, 1.0)));
    data[20] = f64::NAN;
    data[160] = f64::NAN;
    let mut det = StreamingChenWuDetector::new(0.1, 0.2, 4, 0.5, 0.5);
    let dets = det.step(&data);
    for d in &dets {
        match d {
            Detection::ChangePoint(cp) => {
                assert!(cp.confidence.is_finite(), "NaN leaked to CP confidence");
                assert!(cp.shift_sigma.is_finite(), "NaN leaked to shift_sigma");
            }
            Detection::CollectiveAnomaly { confidence, .. } => {
                assert!(confidence.is_finite(), "NaN leaked to anomaly confidence");
            }
        }
    }
}

#[test]
fn inf_in_step_does_not_panic() {
    let mut rng = Rng::new(14);
    let mut data: Vec<f64> = (0..150).map(|_| rng.normal(0.0, 1.0)).collect();
    data.extend((0..150).map(|_| rng.normal(5.0, 1.0)));
    data[40] = f64::INFINITY;
    data[180] = f64::NEG_INFINITY;
    let mut det = StreamingChenWuDetector::new(0.1, 0.2, 4, 0.5, 0.5);
    let dets = det.step(&data);
    for d in &dets {
        match d {
            Detection::ChangePoint(cp) => {
                assert!(cp.confidence.is_finite(), "inf leaked to CP confidence");
                assert!(cp.shift_sigma.is_finite(), "inf leaked to shift_sigma");
            }
            Detection::CollectiveAnomaly { confidence, .. } => {
                assert!(confidence.is_finite(), "inf leaked to anomaly confidence");
            }
        }
    }
}

#[test]
fn all_nan_step_returns_empty() {
    let mut det = StreamingChenWuDetector::new(0.1, 0.2, 4, 0.5, 0.5);
    let data = vec![f64::NAN; 200];
    let dets = det.step(&data);
    assert!(dets.is_empty(), "all-NaN chunk must produce no detections");
    assert_eq!(
        det.total_steps(),
        0,
        "all-NaN chunk must not advance total_steps"
    );
}

#[test]
fn nan_does_not_advance_total_steps() {
    let mut det = StreamingChenWuDetector::new(0.1, 0.2, 4, 0.5, 0.5);
    let data = [
        1.0,
        f64::NAN,
        2.0,
        f64::INFINITY,
        3.0,
        4.0,
        f64::NEG_INFINITY,
        5.0,
        6.0,
        7.0,
    ];
    let _ = det.step(&data);
    assert_eq!(
        det.total_steps(),
        7,
        "expected 7 valid samples (3 of 10 were non-finite), got {}",
        det.total_steps()
    );
}

#[test]
fn nan_survives_save_restore() {
    let mut rng = Rng::new(15);
    let warmup: Vec<f64> = (0..50).map(|_| rng.normal(0.0, 1.0)).collect();
    let mut det = StreamingChenWuDetector::new(0.1, 0.2, 4, 0.5, 0.5);
    let _ = det.step(&warmup);
    let _ = det.step(&[f64::NAN, f64::INFINITY, f64::NEG_INFINITY]);
    let pre_steps = det.total_steps();

    let json = serde_json::to_string(&det.save_state()).unwrap();
    let state: cesura::streaming_chen_wu::ChenWuDetectorState =
        serde_json::from_str(&json).unwrap();
    let mut det2 = StreamingChenWuDetector::restore(state).unwrap();
    assert_eq!(
        det2.total_steps(),
        pre_steps,
        "save/restore must preserve total_steps; non-finite inputs must not have advanced it"
    );

    let post: Vec<f64> = (0..200).map(|_| rng.normal(5.0, 1.0)).collect();
    let dets = det2.step(&post);
    for d in &dets {
        match d {
            Detection::ChangePoint(cp) => {
                assert!(cp.confidence.is_finite(), "NaN leaked across save/restore");
                assert!(cp.shift_sigma.is_finite());
            }
            Detection::CollectiveAnomaly { confidence, .. } => {
                assert!(confidence.is_finite());
            }
        }
    }
}

#[test]
fn matches_batch_detect_with_robust() {
    // β-divergence on a heavy-tail input. Both paths use β=0.15 and
    // must agree.
    let mut rng = Rng::new(7);
    let data: Vec<f64> = (0..400).map(|_| rng.student_t3()).collect();

    let batch = ChenWuDetector::new(0.1, 0.2, 4, 0.5, 0.5)
        .with_robust(0.15)
        .detect(&data);
    let mut stream_det =
        StreamingChenWuDetector::new(0.1, 0.2, 4, 0.5, 0.5).with_robust(0.15);
    let stream = stream_det.step(&data);

    assert_eq!(
        detection_keys(&batch),
        detection_keys(&stream),
        "robust streaming diverged from batch\n  batch  : {batch:?}\n  stream : {stream:?}"
    );
}
