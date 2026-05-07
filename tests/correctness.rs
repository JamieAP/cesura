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
    let univ = det.detect(&data, 0.3);
    let multi = det.detect_multivariate(&mv, 0.3);

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
                cps.extend(det.step(chunk, 0.3));
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
    let mut cps_a = det_a.step(&pre, 0.3);
    cps_a.extend(det_a.step(&post, 0.3));

    // Path B: save / restore between phases.
    let mut det_b = StreamingDetector::new(200.0, 350);
    let mut cps_b = det_b.step(&pre, 0.3);
    let json = serde_json::to_string(&det_b.save_state()).unwrap();
    let restored: cesura::streaming::DetectorState = serde_json::from_str(&json).unwrap();
    let mut det_b2 = StreamingDetector::restore(restored).unwrap();
    cps_b.extend(det_b2.step(&post, 0.3));

    let idx_a: Vec<usize> = cps_a.iter().map(|c| c.index).collect();
    let idx_b: Vec<usize> = cps_b.iter().map(|c| c.index).collect();
    assert_eq!(
        idx_a, idx_b,
        "save/restore changes detection: continuous={idx_a:?}, resumed={idx_b:?}"
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
    let a: Vec<_> = det.detect(&data, 0.3).into_iter().map(|c| (c.index, c.confidence.to_bits())).collect();
    let b: Vec<_> = det.detect(&data, 0.3).into_iter().map(|c| (c.index, c.confidence.to_bits())).collect();
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
        for cp in det.detect(s, 0.0) {
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
    let cps = det.detect(&data, 0.3);
    for w in cps.windows(2) {
        assert!(
            w[0].index < w[1].index,
            "CPs not strictly increasing: {} >= {}",
            w[0].index,
            w[1].index
        );
    }
}
