use cesura::eval::Rng;
use cesura::streaming::StreamingDetector;
use cesura::{BocpdDetector, ChangePoint};

fn idx(cps: &[ChangePoint]) -> Vec<usize> {
    cps.iter().map(|c| c.index).collect()
}

fn fixture(seed: u64, n_pre: usize, n_post: usize, shift: f64) -> Vec<f64> {
    let mut rng = Rng::new(seed);
    let mut data: Vec<f64> = (0..n_pre).map(|_| rng.normal(0.0, 1.0)).collect();
    data.extend((0..n_post).map(|_| rng.normal(shift, 1.0)));
    data
}

#[test]
fn location_shift_invariance() {
    let det = BocpdDetector::new(200.0, 400);
    let data = fixture(0xC0FFEE, 150, 150, 4.0);

    let base = idx(&det.detect(&data));
    for c in [-1000.0, -1.0, 0.5, 17.3, 1e6] {
        let shifted: Vec<f64> = data.iter().map(|x| x + c).collect();
        let got = idx(&det.detect(&shifted));
        assert_eq!(got, base, "shift by {c} changed indices");
    }
}

#[test]
fn positive_scale_invariance() {
    let det = BocpdDetector::new(200.0, 400);
    let data = fixture(0xDEADBEEF, 150, 150, 3.0);

    let base = idx(&det.detect(&data));
    for k in [1e-6, 0.001, 0.5, 2.0, 1000.0, 1e6] {
        let scaled: Vec<f64> = data.iter().map(|x| x * k).collect();
        let got = idx(&det.detect(&scaled));
        assert_eq!(got, base, "scale by {k} changed indices");
    }
}

#[test]
fn sign_flip_symmetry() {
    let det = BocpdDetector::new(200.0, 400);
    let data = fixture(0x5EED, 150, 150, 4.0);

    let base = idx(&det.detect(&data));
    let flipped: Vec<f64> = data.iter().map(|x| -x).collect();
    let got = idx(&det.detect(&flipped));
    assert_eq!(got, base, "sign flip changed indices");
}

#[test]
fn time_reversal_mirror() {
    let det = BocpdDetector::new(200.0, 400);
    let data = fixture(0xFACE_BEEF, 150, 150, 4.0);
    let n = data.len();

    let fwd = idx(&det.detect(&data));
    let rev: Vec<f64> = data.iter().rev().copied().collect();
    let bwd = idx(&det.detect(&rev));

    eprintln!("fwd={fwd:?}  bwd={bwd:?}  n={n}");
    assert!(!fwd.is_empty() && !bwd.is_empty());

    // Slack of 25 absorbs prior-warmup asymmetry. Strict mirror would require
    // a flat prior, which BOCPD does not have.
    let mirror = (n - 1).saturating_sub(*bwd.last().unwrap()) as i64;
    let delta = (fwd[0] as i64 - mirror).abs();
    assert!(delta <= 25, "mirror delta {delta} > 25");
}

#[test]
fn streaming_beta_zero_matches_standard_streaming() {
    // Mirror of `beta_zero_matches_standard_bocpd` for the streaming path.
    // Pins that adding `with_beta(0.0)` to a StreamingDetector is a no-op.
    let data = fixture(0xCAFE_FACE, 150, 150, 4.0);

    let mut std = StreamingDetector::new(200.0, 400);
    let mut beta0 = StreamingDetector::new(200.0, 400).with_beta(0.0);

    let mut std_idx = Vec::new();
    let mut beta0_idx = Vec::new();
    for &x in &data {
        for cp in std.step(&[x]) {
            std_idx.push(cp.index);
        }
        for cp in beta0.step(&[x]) {
            beta0_idx.push(cp.index);
        }
    }
    assert_eq!(std_idx, beta0_idx);
}

#[test]
fn beta_zero_matches_standard_bocpd() {
    // The β-divergence path is short-circuited at β = 0. This pins that
    // contract: the regression snapshot is unchanged with `with_beta(0.0)`.
    let det_std = BocpdDetector::new(200.0, 350);
    let det_beta0 = BocpdDetector::new(200.0, 350).with_beta(0.0);

    // Same fixture as `regression_snapshot_three_regime` for cross-test
    // coherence.
    let mut rng = Rng::new(0xBEEF_CAFE);
    let mut data: Vec<f64> = (0..100).map(|_| rng.normal(0.0, 1.0)).collect();
    data.extend((0..100).map(|_| rng.normal(4.0, 1.0)));
    data.extend((0..100).map(|_| rng.normal(0.0, 1.0)));

    assert_eq!(idx(&det_std.detect(&data)), idx(&det_beta0.detect(&data)));

    // Also pin on the heavy-tail-with-shift scenario where the robust
    // path should produce different results at β > 0.
    let data2 = fixture(0xCAFE, 150, 150, 4.0);
    assert_eq!(idx(&det_std.detect(&data2)), idx(&det_beta0.detect(&data2)));
}

#[test]
fn streaming_vs_batch_on_truly_online_input() {
    let data = fixture(0xB0DEFA, 150, 150, 4.0);

    let batch = BocpdDetector::new(200.0, 400);
    let batch_idx = idx(&batch.detect(&data));

    let mut stream = StreamingDetector::new(200.0, 400);
    let mut stream_idx: Vec<usize> = Vec::new();
    for &x in &data {
        for cp in stream.step(&[x]) {
            stream_idx.push(cp.index);
        }
    }

    eprintln!("batch_idx={batch_idx:?}\nstream_idx={stream_idx:?}");
    assert!(!batch_idx.is_empty() && !stream_idx.is_empty());

    // Batch z-norms over the full slice; streaming uses online Welford.
    // They diverge for early indices and converge in the tail. 25-step
    // bound documents acceptable drift; a regression past it means
    // streaming has materially deviated from the batch reference.
    let batch_first = *batch_idx.iter().find(|&&i| i >= 145).expect("batch CP near 150");
    let stream_first = *stream_idx.iter().find(|&&i| i >= 145).expect("stream CP near 150");
    let drift = (batch_first as i64 - stream_first as i64).abs();
    eprintln!("drift = {drift}");
    assert!(drift <= 25, "batch/stream drift {drift} > 25");
}
