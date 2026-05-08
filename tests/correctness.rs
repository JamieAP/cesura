//! Cross-API correctness tests.
//!
//! Verifies that independent code paths in the public API agree where they
//! claim to model the same thing. These are the tests that catch divergence
//! between `detect` and `detect_multivariate`, between `BocpdDetector` and
//! `StreamingDetector`, and between fresh and resumed streaming runs.
//!
//! Run: `cargo test --features test-utils --test correctness`.

use cesura::eval::Rng;
use cesura::streaming::StreamingDetector;
use cesura::BocpdDetector;

#[cfg(feature = "joint-detection")]
use cesura::chen_wu::Detection;
#[cfg(feature = "joint-detection")]
use cesura::streaming_chen_wu::StreamingChenWuDetector;

#[cfg(feature = "joint-detection")]
type DetectionKey = (usize, usize, usize, u64, u64);

#[cfg(feature = "joint-detection")]
fn detection_keys(dets: &[Detection]) -> Vec<DetectionKey> {
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
fn multivariate_d1_agrees_with_univariate_on_clean_shift() {
    // The d=1 multivariate path uses NIW(d=1, ν=3) -- different prior to the
    // univariate NIG(α=1) -- but on a strong, clean shift both must detect
    // and locate the change point within a small window of each other.
    let mut rng = Rng::new(987);
    let mut data: Vec<f64> = (0..150).map(|_| rng.normal(0.0, 1.0)).collect();
    data.extend((0..150).map(|_| rng.normal(5.0, 1.0)));
    let mv: Vec<Vec<f64>> = data.iter().map(|&x| vec![x]).collect();

    let det = BocpdDetector::new(200.0, 350);
    let univ = det.detect(&data);
    let multi = det.detect_multivariate(&mv);

    assert!(!univ.is_empty(), "univariate must detect 5σ shift");
    assert!(!multi.is_empty(), "d=1 multivariate must detect 5σ shift");

    let delta = (univ[0].index as i64 - multi[0].index as i64).abs();
    assert!(
        delta <= 10,
        "d=1 multivariate ({}) and univariate ({}) disagree by {} steps",
        multi[0].index,
        univ[0].index,
        delta
    );
}

#[test]
fn streaming_chunks_independent_of_chunk_size() {
    // Feeding the same data in different chunk sizes must produce the same
    // detections -- chunk boundaries must not influence the model.
    let mut rng = Rng::new(2024);
    let mut data: Vec<f64> = (0..150).map(|_| rng.normal(0.0, 1.0)).collect();
    data.extend((0..150).map(|_| rng.normal(4.0, 1.0)));

    let outputs: Vec<Vec<usize>> = [1, 7, 50, 300]
        .iter()
        .map(|&chunk_size| {
            let mut det = StreamingDetector::new(200.0, 350);
            let mut cps = Vec::new();
            for chunk in data.chunks(chunk_size) {
                cps.extend(det.step(chunk));
            }
            cps.into_iter().map(|c| c.index).collect()
        })
        .collect();

    let first = &outputs[0];
    for (i, out) in outputs.iter().enumerate().skip(1) {
        assert_eq!(
            out, first,
            "chunk size {} diverged from baseline (chunk=1): {:?} vs {:?}",
            [1, 7, 50, 300][i],
            out,
            first
        );
    }
}

#[test]
fn streaming_resume_is_observationally_equivalent() {
    // Run for N steps, save, restore, run M more = run N+M in one go.
    // Detection indices must match exactly.
    let mut rng = Rng::new(101);
    let pre: Vec<f64> = (0..120).map(|_| rng.normal(0.0, 1.0)).collect();
    let post: Vec<f64> = (0..180).map(|_| rng.normal(3.0, 1.0)).collect();

    // Path A: continuous run.
    let mut det_a = StreamingDetector::new(200.0, 350);
    let mut cps_a = det_a.step(&pre);
    cps_a.extend(det_a.step(&post));

    // Path B: save / restore between phases.
    let mut det_b = StreamingDetector::new(200.0, 350);
    let mut cps_b = det_b.step(&pre);
    let json = serde_json::to_string(&det_b.save_state()).unwrap();
    let restored: cesura::streaming::DetectorState = serde_json::from_str(&json).unwrap();
    let mut det_b2 = StreamingDetector::restore(restored).unwrap();
    cps_b.extend(det_b2.step(&post));

    let idx_a: Vec<usize> = cps_a.iter().map(|c| c.index).collect();
    let idx_b: Vec<usize> = cps_b.iter().map(|c| c.index).collect();
    assert_eq!(
        idx_a, idx_b,
        "save/restore changes detection: continuous={idx_a:?}, resumed={idx_b:?}"
    );
}

#[cfg(feature = "joint-detection")]
#[test]
fn streaming_chen_wu_chunks_independent_of_chunk_size() {
    // Feeding the same data in [1, 7, 50, 300]-sized chunks must
    // produce identical detections -- chunk boundaries must not
    // influence the model. Mirrors streaming_chunks_independent_of_chunk_size
    // for the joint detector.
    let mut rng = Rng::new(2024);
    let mut data: Vec<f64> = (0..150).map(|_| rng.normal(0.0, 1.0)).collect();
    data.extend((0..150).map(|_| rng.normal(4.0, 1.0)));

    let outputs: Vec<Vec<DetectionKey>> = [1, 7, 50, 300]
        .iter()
        .map(|&chunk_size| {
            let mut det = StreamingChenWuDetector::new(0.1, 0.2, 4, 0.5, 0.5);
            let mut all = Vec::new();
            for chunk in data.chunks(chunk_size) {
                all.extend(det.step(chunk));
            }
            detection_keys(&all)
        })
        .collect();

    let first = &outputs[0];
    for (i, out) in outputs.iter().enumerate().skip(1) {
        assert_eq!(
            out, first,
            "chunk size {} diverged from baseline (chunk=1)",
            [1, 7, 50, 300][i]
        );
    }
}

#[cfg(feature = "joint-detection")]
#[test]
fn streaming_chen_wu_resume_is_observationally_equivalent() {
    // Save mid-stream, restore, finish; assert identical to a
    // continuous run.
    use cesura::streaming_chen_wu::ChenWuDetectorState;

    let mut rng = Rng::new(101);
    let pre: Vec<f64> = (0..120).map(|_| rng.normal(0.0, 1.0)).collect();
    let post: Vec<f64> = (0..180).map(|_| rng.normal(3.0, 1.0)).collect();

    let mut det_a = StreamingChenWuDetector::new(0.1, 0.2, 4, 0.5, 0.5);
    let mut dets_a = det_a.step(&pre);
    dets_a.extend(det_a.step(&post));

    let mut det_b = StreamingChenWuDetector::new(0.1, 0.2, 4, 0.5, 0.5);
    let mut dets_b = det_b.step(&pre);
    let json = serde_json::to_string(&det_b.save_state()).unwrap();
    let restored: ChenWuDetectorState = serde_json::from_str(&json).unwrap();
    let mut det_b2 = StreamingChenWuDetector::restore(restored).unwrap();
    dets_b.extend(det_b2.step(&post));

    assert_eq!(
        detection_keys(&dets_a),
        detection_keys(&dets_b),
        "save/restore changes detection: continuous={dets_a:?}, resumed={dets_b:?}"
    );
}

#[test]
fn batch_detect_is_idempotent() {
    // Calling detect twice on the same data with the same detector must
    // produce identical results (no hidden mutation).
    let mut rng = Rng::new(55);
    let mut data: Vec<f64> = (0..100).map(|_| rng.normal(0.0, 1.0)).collect();
    data.extend((0..100).map(|_| rng.normal(4.0, 1.0)));

    let det = BocpdDetector::new(200.0, 250);
    let a: Vec<_> = det.detect(&data).into_iter().map(|c| (c.index, c.confidence.to_bits())).collect();
    let b: Vec<_> = det.detect(&data).into_iter().map(|c| (c.index, c.confidence.to_bits())).collect();
    assert_eq!(a, b, "detect() not idempotent -- hidden mutation?");
}

#[test]
fn confidence_in_unit_interval() {
    // Across a wide variety of inputs, every reported confidence ∈ [0, 1].
    let scenarios: Vec<Vec<f64>> = (0u64..20)
        .map(|seed| {
            let mut rng = Rng::new(seed.wrapping_mul(7919) + 1);
            let mut d: Vec<f64> = (0..150).map(|_| rng.normal(0.0, 1.0)).collect();
            d.extend((0..150).map(|_| rng.normal((seed % 7 + 1) as f64, 1.0)));
            d
        })
        .collect();

    let det = BocpdDetector::new(200.0, 350);
    for s in &scenarios {
        for cp in det.detect(s) {
            assert!(
                (0.0..=1.0).contains(&cp.confidence),
                "confidence {} ∉ [0, 1]",
                cp.confidence
            );
            assert!(cp.confidence.is_finite());
            assert!(cp.shift_sigma >= 0.0, "shift_sigma must be non-negative");
            assert!(cp.shift_sigma.is_finite());
        }
    }
}

#[test]
fn change_points_strictly_increasing() {
    // All reported CPs in a single detect call must be in strictly
    // increasing index order.
    let mut rng = Rng::new(8675309);
    let mut data = Vec::new();
    for k in 0..6 {
        let mu = if k % 2 == 0 { 0.0 } else { 4.0 };
        data.extend((0..80).map(|_| rng.normal(mu, 1.0)));
    }
    let det = BocpdDetector::new(200.0, data.len() + 50);
    let cps = det.detect(&data);
    for w in cps.windows(2) {
        assert!(
            w[0].index < w[1].index,
            "CPs not strictly increasing: {} >= {}",
            w[0].index,
            w[1].index
        );
    }
}

#[test]
fn multivariate_detects_correlated_shift_per_dim_invisible() {
    // Joint whitening should beat per-dim normalisation when the shift is
    // *anti-aligned* with the principal noise direction. Noise correlation
    // along (1,1); shift along (1,-1) -- per-dim each shift is ~ 0.6σ
    // (below typical detection threshold) but Mahalanobis ~ 1.6σ.
    //
    // Tightens the prior `>= max_univ` assertion: with correct whitening,
    // multivariate must fire strictly more than the best marginal.
    let mut rng = Rng::new(773);
    let mut data = Vec::with_capacity(300);
    for _ in 0..150 {
        let z = rng.normal(0.0, 1.0);
        data.push(vec![
            z * 0.7 + rng.normal(0.0, 0.3),
            z * 0.7 + rng.normal(0.0, 0.3),
        ]);
    }
    for _ in 0..150 {
        let z = rng.normal(0.0, 1.0);
        data.push(vec![
            0.6 + z * 0.7 + rng.normal(0.0, 0.3),
            -0.6 + z * 0.7 + rng.normal(0.0, 0.3),
        ]);
    }

    let det = BocpdDetector::new(200.0, 350);
    let dim1: Vec<f64> = data.iter().map(|x| x[0]).collect();
    let dim2: Vec<f64> = data.iter().map(|x| x[1]).collect();
    let univ_1 = det.detect(&dim1).len();
    let univ_2 = det.detect(&dim2).len();
    let multi = det.detect_multivariate(&data).len();

    eprintln!("anti-correlated shift: univ_1={univ_1} univ_2={univ_2} multi={multi}");
    let max_univ = univ_1.max(univ_2);
    assert!(
        multi > max_univ,
        "Mahalanobis multi must beat best marginal: multi={multi} max_univ={max_univ}"
    );
}

#[test]
fn multivariate_singular_covariance_falls_back() {
    // dim 0 is i.i.d. N(0,1); dim 1 is exactly 2× dim 0. Sample covariance
    // is rank-1 (singular), Cholesky must reject, and the path must fall
    // back to per-dim normalisation -- not panic.
    let mut rng = Rng::new(444);
    let mut data = Vec::with_capacity(200);
    for _ in 0..100 {
        let x = rng.normal(0.0, 1.0);
        data.push(vec![x, 2.0 * x]);
    }
    for _ in 0..100 {
        let x = rng.normal(3.0, 1.0);
        data.push(vec![x, 2.0 * x]);
    }
    let det = BocpdDetector::new(200.0, 250);
    // Just must not panic and must return finite output.
    let cps = det.detect_multivariate(&data);
    for cp in &cps {
        assert!(cp.confidence.is_finite());
        assert!(cp.shift_sigma.is_finite());
    }
}

#[test]
fn multivariate_warmup_too_small_falls_back() {
    // n = 25, d = 8 → warmup_n = 8, but warmup_n must be ≥ 2d = 16; the
    // path engages the per-dim fallback rather than attempting a
    // rank-deficient covariance estimate. Contract: no panic.
    let mut rng = Rng::new(901);
    let data: Vec<Vec<f64>> = (0..25)
        .map(|_| (0..8).map(|_| rng.normal(0.0, 1.0)).collect())
        .collect();
    let det = BocpdDetector::new(50.0, 50);
    let _ = det.detect_multivariate(&data);
}
