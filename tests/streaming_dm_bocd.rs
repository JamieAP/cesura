//! `StreamingDmBocd` batch-agreement test.
//!
//! Acceptance: feeding `data[..n]` to `StreamingDmBocd::step` one bar
//! at a time produces the same CP indices as
//! `DmBocdDetector::detect_multivariate(&data)`. Required `n >= 180`
//! so that the batch detector's `warmup_n = (n/3).min(60).max(d*2)`
//! resolves to 60 (matching `STREAMING_WARMUP_N`).

use cesura::dm_bocd::{DmBocdDetector, IdentityM, ImqM, StreamingDmBocd};
use cesura::eval::Rng;

fn gen_shift(seed: u64, d: usize, n_pre: usize, n_post: usize, shift: f64) -> Vec<Vec<f64>> {
    let mut rng = Rng::new(seed);
    let mut data: Vec<Vec<f64>> = (0..n_pre)
        .map(|_| (0..d).map(|_| rng.normal(0.0, 1.0)).collect())
        .collect();
    data.extend(
        (0..n_post).map(|_| (0..d).map(|_| rng.normal(shift, 1.0)).collect()),
    );
    data
}

#[test]
fn offline_agrees_with_batch_identity_m() {
    // n=300, d=2, shift=5σ at index 150. Both batch and streaming
    // resolve to warmup_n=60 -> whitening transforms agree.
    let data = gen_shift(0xA1, 2, 150, 150, 5.0);
    assert!(data.len() >= 180, "test needs n >= 180 for warmup parity");

    let batch_cps = DmBocdDetector::new(2, 100.0, 256).detect_multivariate(&data);

    let mut stream = StreamingDmBocd::new(2, 100.0, 256);
    let mut stream_cps = Vec::new();
    for x in &data {
        stream_cps.extend(stream.step(x));
    }

    let batch_idxs: Vec<usize> = batch_cps.iter().map(|c| c.index).collect();
    let stream_idxs: Vec<usize> = stream_cps.iter().map(|c| c.index).collect();
    assert_eq!(
        batch_idxs, stream_idxs,
        "CP indices diverge: batch={batch_idxs:?} stream={stream_idxs:?}",
    );
    // Confidence and shift_sigma should also agree to numerical noise
    // (both come from identical floating-point recursions).
    for (b, s) in batch_cps.iter().zip(stream_cps.iter()) {
        assert!(
            (b.confidence - s.confidence).abs() < 1e-9,
            "confidence diverges: batch={} stream={}",
            b.confidence,
            s.confidence,
        );
        assert!(
            (b.shift_sigma - s.shift_sigma).abs() < 1e-9,
            "shift_sigma diverges: batch={} stream={}",
            b.shift_sigma,
            s.shift_sigma,
        );
    }
    assert_eq!(stream.total_steps(), data.len());
}

#[test]
fn offline_agrees_with_batch_imq_m() {
    // Same fixture, IMQ m-weight. Verifies with_m_weight propagates.
    let data = gen_shift(0xA2, 2, 150, 150, 5.0);

    let batch_cps = DmBocdDetector::new(2, 100.0, 256)
        .with_m_weight(ImqM::new(1.0))
        .detect_multivariate(&data);

    let mut stream = StreamingDmBocd::new(2, 100.0, 256).with_m_weight(ImqM::new(1.0));
    let mut stream_cps = Vec::new();
    for x in &data {
        stream_cps.extend(stream.step(x));
    }

    let batch_idxs: Vec<usize> = batch_cps.iter().map(|c| c.index).collect();
    let stream_idxs: Vec<usize> = stream_cps.iter().map(|c| c.index).collect();
    assert_eq!(batch_idxs, stream_idxs);
}

#[test]
fn no_cps_until_warmup_plus_lookahead() {
    // Streaming emits nothing during the first 60+20 = 80 bars (warmup
    // window plus 20-bar trigger lookahead). After that it MAY emit.
    let data = gen_shift(0xA3, 2, 150, 150, 5.0);
    let mut stream = StreamingDmBocd::<IdentityM>::new(2, 100.0, 256);
    for x in &data[..80] {
        let cps = stream.step(x);
        assert!(cps.is_empty(), "streaming fired before warmup+lookahead");
    }
}

#[test]
fn invalid_observation_dim_returns_empty() {
    let mut stream = StreamingDmBocd::<IdentityM>::new(3, 100.0, 256);
    let cps = stream.step(&[0.0, 0.0]);
    assert!(cps.is_empty());
    assert_eq!(stream.total_steps(), 0, "rejected step shouldn't advance");
}

#[test]
fn long_stream_history_is_bounded() {
    // History buffers are pruned after each
    // scan_trigger call so a long-lived stream does not accumulate
    // observations indefinitely. Steady-state buffer size is
    // max(prev_max_lb=15, cooldown, lookahead=20) + lookahead=20 + small
    // margin.
    //
    // Feed a 5_000-bar stationary stream; the history-keep window must
    // stay below ~80 bars (80 = 20 + max(15,15,20) + 40 slack). Without
    // pruning, this would be 5000 entries.

    let mut rng = Rng::new(0xB1);
    let n = 5_000usize;
    let data: Vec<Vec<f64>> = (0..n)
        .map(|_| (0..2).map(|_| rng.normal(0.0, 1.0)).collect())
        .collect();

    let mut stream = StreamingDmBocd::<IdentityM>::new(2, 100.0, 256);
    for x in &data {
        let _ = stream.step(x);
    }
    assert_eq!(stream.total_steps(), n);
    let report = stream.history_buffer_report();
    let cap = 80;
    assert!(
        report.norm_history_len <= cap,
        "norm_history grew unboundedly: len={} expected <= {cap}",
        report.norm_history_len,
    );
    assert!(
        report.map_rls_len <= cap,
        "map_rls grew unboundedly: len={} expected <= {cap}",
        report.map_rls_len,
    );
    assert!(
        report.cp_probs_len <= cap,
        "cp_probs grew unboundedly: len={} expected <= {cap}",
        report.cp_probs_len,
    );
    // Sanity: history_offset advanced past most of the stream.
    assert!(
        report.history_offset > n - cap - 100,
        "history_offset stuck: offset={} total_steps={n}",
        report.history_offset,
    );
}

#[test]
fn with_map_drop_trigger_round_trip() {
    // Mirror the existing dm_bocd_coverage test for the trigger setter.
    let data = gen_shift(0xA4, 2, 150, 150, 5.0);
    let batch_cps = DmBocdDetector::new(2, 100.0, 256)
        .with_map_drop_trigger(3, 50, 15)
        .detect_multivariate(&data);
    let mut stream = StreamingDmBocd::new(2, 100.0, 256).with_map_drop_trigger(3, 50, 15);
    let mut stream_cps = Vec::new();
    for x in &data {
        stream_cps.extend(stream.step(x));
    }
    let batch_idxs: Vec<usize> = batch_cps.iter().map(|c| c.index).collect();
    let stream_idxs: Vec<usize> = stream_cps.iter().map(|c| c.index).collect();
    assert_eq!(batch_idxs, stream_idxs);
}
