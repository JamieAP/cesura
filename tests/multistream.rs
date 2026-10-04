//! Integration tests for the multistream aggregators against real
//! `StreamingDetector` instances.
//!
//! Scope: WIRING. Confirms the trait surface composes correctly --
//! `StreamingDetector` implements `ScoreStream`, aggregators consume
//! it, step()/save_state()/restore() round-trip. Per-aggregator
//! statistical behavior is tested against scripted streams in the
//! unit suites (`src/multistream/{hc,sum_cusum}.rs`).
//!
//! Real-data calibration (HC threshold tuning vs StreamingDetector's
//! BF/cp_probs distribution; persistence-filter to distinguish
//! transient noise from sustained shifts; per-application false-alarm
//! rate calibration) is not covered by this suite. The
//! integration tests here verify the COUPLING is correct, not that
//! defaults are well-tuned for any specific fixture.

#![cfg(feature = "test-utils")]

use cesura::eval::Rng;
use cesura::multistream::{
    HcAggregator, HcAggregatorState, ScoreKind, ScoreStream, SumCusumAggregator,
    SumCusumAggregatorState,
};
use cesura::streaming::StreamingDetector;

fn make_bf_streams(d: usize) -> Vec<StreamingDetector> {
    (0..d)
        .map(|_| StreamingDetector::new(200.0, 250).with_bayes_factor_rule(2.0, 3, 15))
        .collect()
}

fn make_map_streams(d: usize) -> Vec<StreamingDetector> {
    (0..d).map(|_| StreamingDetector::new(200.0, 250)).collect()
}

fn make_tape(n: usize, cp_step: usize, d: usize, affected: &[usize], shift_sigma: f64, seed: u64) -> Vec<Vec<f64>> {
    let mut rng = Rng::new(seed);
    let mut tape = Vec::with_capacity(n);
    for t in 0..n {
        let mut row = Vec::with_capacity(d);
        for d_i in 0..d {
            let mu = if t >= cp_step && affected.contains(&d_i) {
                shift_sigma
            } else {
                0.0
            };
            row.push(mu + rng.normal(0.0, 1.0));
        }
        tape.push(row);
    }
    tape
}

#[test]
fn streaming_detector_implements_scorestream_bf() {
    let mut det = StreamingDetector::new(200.0, 250).with_bayes_factor_rule(2.0, 3, 15);
    assert_eq!(det.score_kind(), ScoreKind::BayesFactor);
    let s = det.step_score(0.0);
    assert!(s.is_finite(), "BF score after one step must be finite, got {s}");
    assert_eq!(det.step_count(), 1);
}

#[test]
fn streaming_detector_implements_scorestream_map_drop() {
    let mut det = StreamingDetector::new(200.0, 250);
    assert_eq!(det.score_kind(), ScoreKind::CpProbability);
    let s = det.step_score(0.0);
    assert!((0.0..=1.0).contains(&s), "cp_prob ∈ [0,1], got {s}");
}

#[test]
fn hc_aggregator_consumes_streamingdetector_bf_streams_without_panic() {
    let mut hc = HcAggregator::new(make_bf_streams(3));
    let tape = make_tape(150, 10_000, 3, &[], 0.0, 7);
    let _ = hc.step(&tape);
    assert_eq!(hc.step_count(), 150);
}

#[test]
fn hc_aggregator_consumes_streamingdetector_cpprob_streams_without_panic() {
    // After rank-transform calibration shipped, HC accepts both
    // ScoreKind::BayesFactor and ScoreKind::CpProbability. Wiring
    // verification only -- behavioral calibration deferred.
    let mut hc = HcAggregator::new(make_map_streams(3));
    let tape = make_tape(150, 10_000, 3, &[], 0.0, 11);
    let _ = hc.step(&tape);
    assert_eq!(hc.step_count(), 150);
}

#[test]
fn sum_cusum_aggregator_consumes_streamingdetector_streams_without_panic() {
    let mut bf_agg = SumCusumAggregator::new(make_bf_streams(3));
    let cp_agg_streams = make_map_streams(3);
    let mut cp_agg = SumCusumAggregator::new(cp_agg_streams);

    let tape = make_tape(150, 10_000, 3, &[], 0.0, 13);
    let _ = bf_agg.step(&tape);
    let _ = cp_agg.step(&tape);
    assert_eq!(bf_agg.step_count(), 150);
    assert_eq!(cp_agg.step_count(), 150);
}

#[test]
fn hc_save_restore_round_trips_with_real_streams() {
    let mut hc = HcAggregator::new(make_bf_streams(3));
    let tape = make_tape(120, 10_000, 3, &[], 0.0, 17);
    let _ = hc.step(&tape);

    let json = serde_json::to_string(&hc.save_state()).unwrap();
    let restored: HcAggregatorState = serde_json::from_str(&json).unwrap();
    assert_eq!(restored.step_count, 120);
    assert_eq!(restored.score_kind, ScoreKind::BayesFactor);
    let _ = HcAggregator::restore(restored, make_bf_streams(3))
        .expect("restore must succeed with matching stream count");
}

#[test]
fn sum_cusum_save_restore_round_trips_with_real_streams() {
    let mut agg = SumCusumAggregator::new(make_bf_streams(3));
    let tape = make_tape(120, 10_000, 3, &[], 0.0, 19);
    let _ = agg.step(&tape);

    let json = serde_json::to_string(&agg.save_state()).unwrap();
    let restored: SumCusumAggregatorState = serde_json::from_str(&json).unwrap();
    assert_eq!(restored.step_count, 120);
    assert_eq!(restored.score_kind, ScoreKind::BayesFactor);
    assert_eq!(restored.per_stream_w.len(), 3);

    let _ = SumCusumAggregator::restore(restored, make_bf_streams(3))
        .expect("restore must succeed with matching stream count");
}

#[test]
#[ignore]
fn detection_delta_persistence_levels() {
    // Sparse detection rate as a function of persistence -- pair
    // with far_delta to see the FAR/detection tradeoff.
    let seeds: [u64; 5] = [7, 11, 17, 23, 31];
    let thresholds = [2.0, 3.0, 5.0];
    let persistences = [1usize, 2, 3];
    eprintln!("DEBUG detection_delta sparse 5σ 1-of-4:");
    eprintln!("  τ \\ persistence  1   2   3");
    for &tau in &thresholds {
        let mut row = format!("  τ={tau:.1}  ");
        for &pers in &persistences {
            let mut hits = 0usize;
            for &seed in &seeds {
                let streams = make_bf_streams(4);
                let mut hc = HcAggregator::new(streams)
                    .with_threshold(tau)
                    .with_persistence(pers);
                let tape = make_tape(300, 150, 4, &[0], 5.0, seed);
                let cps = hc.step(&tape);
                if cps
                    .iter()
                    .any(|c| (150..=230).contains(&c.index) && c.streams.contains(&0))
                {
                    hits += 1;
                }
            }
            row.push_str(&format!("  {hits}/5"));
        }
        eprintln!("{row}");
    }
}

#[test]
#[ignore]
fn far_delta_persistence_levels() {
    // Real-data FAR comparison: HC over BF streams on 5 stationary
    // seeds × 300 steps, sweep persistence ∈ {1, 2, 3} at τ ∈
    // {2.0, 3.0, 5.0}. Reports total fires per setting so we can
    // confirm persistence actually reduces FAR before shipping the
    // feature.
    let seeds: [u64; 5] = [7, 11, 17, 23, 31];
    let thresholds = [2.0, 3.0, 5.0];
    let persistences = [1usize, 2, 3];
    eprintln!("DEBUG far_delta_persistence_levels:");
    eprintln!("  τ \\ persistence    1     2     3");
    for &tau in &thresholds {
        let mut row = format!("  τ={tau:.1}  ");
        for &pers in &persistences {
            let mut total_fires = 0usize;
            for &seed in &seeds {
                let streams = make_bf_streams(4);
                let mut hc = HcAggregator::new(streams)
                    .with_threshold(tau)
                    .with_persistence(pers);
                let tape = make_tape(300, 10_000, 4, &[], 0.0, seed);
                let cps = hc.step(&tape);
                total_fires += cps.len();
            }
            row.push_str(&format!("  {total_fires:3}  "));
        }
        eprintln!("{row}");
    }
}

#[test]
#[should_panic(expected = "score_kind")]
fn hc_aggregator_refuses_mixed_score_kind_streams() {
    // Mixing BF and MAP-drop streams is rejected at construction --
    // even though rank-transform makes both p-value calibrations work
    // in isolation, mixing them in one aggregator confuses HC's
    // threshold.
    let mut streams = make_bf_streams(2);
    streams.push(StreamingDetector::new(200.0, 250)); // CpProbability
    let _ = HcAggregator::new(streams);
}
