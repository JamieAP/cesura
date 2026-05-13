//! Runner ≡ direct-detector equivalence. For each variant, feed the
//! same observations through `Runner::Uni(StreamingDetector)` (etc.)
//! and through the underlying detector directly; the emitted `Fire`
//! sequence must equal the mapped detector output bit-for-bit.

use cesura::dm_bocd::StreamingDmBocd;
use cesura::multistream::{
    FilterTickAggregator, HcAggregator, SumCusumAggregator,
};
use cesura::runtime::{DetectorKind, Fire, Obs, ObsShape, Runner};
use cesura::streaming::StreamingDetector;
#[cfg(feature = "joint-detection")]
use cesura::streaming_chen_wu::StreamingChenWuDetector;
#[cfg(feature = "joint-detection")]
use cesura::chen_wu::Detection;

fn synthetic_uni(n: usize, cp: usize) -> Vec<f64> {
    let mut v = Vec::with_capacity(n);
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    for t in 0..n {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u1 = ((state >> 33) as u32 as f64 + 1.0) / (u32::MAX as f64 + 2.0);
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u2 = ((state >> 33) as u32 as f64 + 1.0) / (u32::MAX as f64 + 2.0);
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        let mu = if t >= cp { 5.0 } else { 0.0 };
        v.push(mu + z);
    }
    v
}

fn synthetic_multi(n: usize, cp: usize, d: usize) -> Vec<Vec<f64>> {
    let mut out = Vec::with_capacity(n);
    let mut state: u64 = 0xDEAD_BEEF_C0FF_EE00;
    for t in 0..n {
        let mut row = Vec::with_capacity(d);
        for _ in 0..d {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u1 = ((state >> 33) as u32 as f64 + 1.0) / (u32::MAX as f64 + 2.0);
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u2 = ((state >> 33) as u32 as f64 + 1.0) / (u32::MAX as f64 + 2.0);
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            let mu = if t >= cp { 5.0 } else { 0.0 };
            row.push(mu + z);
        }
        out.push(row);
    }
    out
}

#[test]
fn runner_uni_matches_streaming_detector() {
    let data = synthetic_uni(400, 200);

    let mut direct = StreamingDetector::new(200.0, 250);
    let direct_cps = direct.step(&data);
    let direct_fires: Vec<Fire> = direct_cps
        .into_iter()
        .map(|c| Fire {
            index: c.index,
            confidence: c.confidence,
            shift_sigma: Some(c.shift_sigma),
            streams: Vec::new(),
            kind: std::borrow::Cow::Borrowed("cp"),
        })
        .collect();

    let mut runner = Runner::Uni(StreamingDetector::new(200.0, 250));
    let mut runner_fires = Vec::new();
    for &x in &data {
        runner_fires.extend(runner.feed(Obs::F64(x)));
    }

    assert_eq!(runner_fires, direct_fires);
    assert_eq!(runner.kind(), "streaming");
}

#[cfg(feature = "joint-detection")]
#[test]
fn runner_chen_wu_matches_streaming_chen_wu() {
    let data = synthetic_uni(300, 150);

    let mut direct = StreamingChenWuDetector::new(0.001, 0.05, 20, 0.999, 0.999);
    let direct_dets = direct.step(&data);
    let direct_fires: Vec<Fire> = direct_dets
        .into_iter()
        .map(|d| match d {
            Detection::ChangePoint(c) => Fire {
                index: c.index,
                confidence: c.confidence,
                shift_sigma: Some(c.shift_sigma),
                streams: Vec::new(),
                kind: std::borrow::Cow::Borrowed("cp"),
            },
            Detection::CollectiveAnomaly { end, confidence, .. } => Fire {
                index: end,
                confidence,
                shift_sigma: None,
                streams: Vec::new(),
                kind: std::borrow::Cow::Borrowed("anomaly"),
            },
        })
        .collect();

    let mut runner = Runner::UniChenWu(StreamingChenWuDetector::new(
        0.001, 0.05, 20, 0.999, 0.999,
    ));
    let mut runner_fires = Vec::new();
    for &x in &data {
        runner_fires.extend(runner.feed(Obs::F64(x)));
    }

    assert_eq!(runner_fires, direct_fires);
    assert_eq!(runner.kind(), "chen-wu");
}

fn streams(d: usize) -> Vec<StreamingDetector> {
    (0..d).map(|_| StreamingDetector::new(200.0, 250)).collect()
}

#[test]
fn runner_sum_cusum_matches_aggregator() {
    let data = synthetic_multi(400, 200, 3);

    let mut direct = SumCusumAggregator::new(streams(3)).with_threshold(0.1);
    let direct_cps = direct.step(&data);
    let direct_fires: Vec<Fire> = direct_cps
        .into_iter()
        .map(|c| Fire {
            index: c.index,
            confidence: c.confidence,
            shift_sigma: None,
            streams: c.streams,
            kind: std::borrow::Cow::Borrowed("cp"),
        })
        .collect();

    let mut runner = Runner::MultiSumCusum(
        SumCusumAggregator::new(streams(3)).with_threshold(0.1),
    );
    let mut runner_fires = Vec::new();
    for row in &data {
        runner_fires.extend(runner.feed(Obs::Vec(row.clone())));
    }

    assert_eq!(runner_fires, direct_fires);
    assert_eq!(runner.kind(), "sum-cusum");
}

#[test]
fn runner_hc_matches_aggregator() {
    let data = synthetic_multi(500, 250, 4);

    let mut direct = HcAggregator::new(streams(4));
    let direct_cps = direct.step(&data);
    let direct_fires: Vec<Fire> = direct_cps
        .into_iter()
        .map(|c| Fire {
            index: c.index,
            confidence: c.confidence,
            shift_sigma: None,
            streams: c.streams,
            kind: std::borrow::Cow::Borrowed("cp"),
        })
        .collect();

    let mut runner = Runner::MultiHc(HcAggregator::new(streams(4)));
    let mut runner_fires = Vec::new();
    for row in &data {
        runner_fires.extend(runner.feed(Obs::Vec(row.clone())));
    }

    assert_eq!(runner_fires, direct_fires);
    assert_eq!(runner.kind(), "hc");
}

#[test]
fn runner_filter_tick_matches_aggregator() {
    let data = synthetic_multi(400, 200, 4);

    let mut direct = FilterTickAggregator::new(4, 2, 0.5);
    let direct_cps = direct.step(&data);
    let direct_fires: Vec<Fire> = direct_cps
        .into_iter()
        .map(|c| Fire {
            index: c.index,
            confidence: c.confidence,
            shift_sigma: None,
            streams: c.streams,
            kind: std::borrow::Cow::Borrowed("cp"),
        })
        .collect();

    let mut runner = Runner::MultiFilterTick(FilterTickAggregator::new(4, 2, 0.5));
    let mut runner_fires = Vec::new();
    for row in &data {
        runner_fires.extend(runner.feed(Obs::Vec(row.clone())));
    }

    assert_eq!(runner_fires, direct_fires);
    assert_eq!(runner.kind(), "filter-tick");
}

#[test]
fn runner_dm_bocd_matches_streaming() {
    let data = synthetic_multi(300, 150, 2);

    let mut direct = StreamingDmBocd::new(2, 200.0, 250);
    let mut direct_fires: Vec<Fire> = Vec::new();
    for row in &data {
        for c in direct.step(row) {
            direct_fires.push(Fire {
                index: c.index,
                confidence: c.confidence,
                shift_sigma: Some(c.shift_sigma),
                streams: Vec::new(),
                kind: std::borrow::Cow::Borrowed("cp"),
            });
        }
    }

    let mut runner = Runner::DmBocd(StreamingDmBocd::new(2, 200.0, 250));
    let mut runner_fires = Vec::new();
    for row in &data {
        runner_fires.extend(runner.feed(Obs::Vec(row.clone())));
    }

    assert_eq!(runner_fires, direct_fires);
    assert_eq!(runner.kind(), "dm-bocd");
}

#[test]
fn runner_save_restore_uni_roundtrips() {
    let data = synthetic_uni(300, 100);
    let mut r1 = Runner::Uni(StreamingDetector::new(200.0, 250));
    for &x in data.iter().take(200) {
        r1.feed(Obs::F64(x));
    }
    let snap = r1.save().expect("save");
    let json = serde_json::to_string(&snap).expect("ser");
    let de: cesura::runtime::RunnerState = serde_json::from_str(&json).expect("de");

    let mut r2 = Runner::restore(de).expect("restore");
    let f1: Vec<Fire> = data[200..].iter().flat_map(|&x| r1.feed(Obs::F64(x))).collect();
    let f2: Vec<Fire> = data[200..].iter().flat_map(|&x| r2.feed(Obs::F64(x))).collect();
    assert_eq!(f1, f2);
}

#[test]
fn runner_save_dm_bocd_errors() {
    let runner = Runner::DmBocd(StreamingDmBocd::new(2, 200.0, 250));
    let err = runner.save().expect_err("dm-bocd has no save_state");
    assert!(err.contains("dm-bocd"), "expected dm-bocd in err, got: {err}");
}

#[test]
fn runner_version_mismatch_rejected() {
    let r = Runner::Uni(StreamingDetector::new(200.0, 250));
    let mut snap = r.save().unwrap();
    snap.version = 999;
    let result = Runner::restore(snap);
    assert!(matches!(result, Err(ref s) if s.contains("version mismatch")));
}

#[test]
fn runner_kind_payload_mismatch_rejected() {
    let r = Runner::Uni(StreamingDetector::new(200.0, 250));
    let snap = r.save().unwrap();
    // Swap kind tag but keep Uni payload.
    let bad = cesura::runtime::RunnerState {
        version: snap.version,
        kind: "hc".into(),
        payload: snap.payload,
    };
    assert!(matches!(Runner::restore(bad), Err(ref s) if s.contains("mismatch") || s.contains("unsupported")));
}

#[test]
fn runner_sum_cusum_save_restore_roundtrips() {
    let data = synthetic_multi(400, 200, 3);
    let mut r1 = Runner::MultiSumCusum(
        SumCusumAggregator::new(streams(3)).with_threshold(0.1),
    );
    for row in data.iter().take(250) {
        r1.feed(Obs::Vec(row.clone()));
    }
    let snap = r1.save().unwrap();
    let json = serde_json::to_string(&snap).unwrap();
    let de: cesura::runtime::RunnerState = serde_json::from_str(&json).unwrap();
    let mut r2 = Runner::restore(de).unwrap();

    let f1: Vec<Fire> = data[250..]
        .iter()
        .flat_map(|row| r1.feed(Obs::Vec(row.clone())))
        .collect();
    let f2: Vec<Fire> = data[250..]
        .iter()
        .flat_map(|row| r2.feed(Obs::Vec(row.clone())))
        .collect();
    assert_eq!(f1, f2);
}

#[test]
fn detector_kind_as_wire_matches_runner_kind() {
    // The wire form returned by DetectorKind::as_wire must equal what
    // Runner::kind() emits, for every variant. Single source of truth.
    let cases: &[(DetectorKind, Runner)] = &[
        (DetectorKind::Streaming, Runner::Uni(StreamingDetector::new(200.0, 250))),
        #[cfg(feature = "joint-detection")]
        (
            DetectorKind::ChenWu,
            Runner::UniChenWu(StreamingChenWuDetector::new(0.001, 0.05, 20, 0.999, 0.999)),
        ),
        (
            DetectorKind::SumCusum,
            Runner::MultiSumCusum(
                SumCusumAggregator::new(streams(3)).with_threshold(0.1),
            ),
        ),
        (DetectorKind::Hc, Runner::MultiHc(HcAggregator::new(streams(3)))),
        (
            DetectorKind::FilterTick,
            Runner::MultiFilterTick(FilterTickAggregator::new(3, 2, 0.5)),
        ),
        (DetectorKind::DmBocd, Runner::DmBocd(StreamingDmBocd::new(2, 200.0, 250))),
    ];
    for (kind, runner) in cases {
        assert_eq!(runner.kind(), kind.as_wire());
        assert_eq!(runner.kind_enum(), *kind);
    }
}

#[test]
fn detector_kind_from_str_roundtrips_with_display() {
    use std::str::FromStr;
    for kind in DetectorKind::all() {
        let s = kind.to_string();
        let back: DetectorKind = DetectorKind::from_str(&s).expect("from_str");
        assert_eq!(back, *kind, "roundtrip failed for {s}");
    }
}

#[test]
fn detector_kind_from_str_rejects_unknown() {
    use std::str::FromStr;
    let err = DetectorKind::from_str("not-a-real-kind").unwrap_err();
    assert!(err.contains("unknown"));
}

#[test]
fn detector_kind_obs_shape_partitions_univariate_vs_multistream() {
    assert_eq!(DetectorKind::Streaming.obs_shape(), ObsShape::Scalar);
    #[cfg(feature = "joint-detection")]
    assert_eq!(DetectorKind::ChenWu.obs_shape(), ObsShape::Scalar);
    assert_eq!(DetectorKind::SumCusum.obs_shape(), ObsShape::Vector);
    assert_eq!(DetectorKind::Hc.obs_shape(), ObsShape::Vector);
    assert_eq!(DetectorKind::FilterTick.obs_shape(), ObsShape::Vector);
    assert_eq!(DetectorKind::DmBocd.obs_shape(), ObsShape::Vector);
}

#[test]
fn detector_kind_serde_roundtrips_via_json() {
    for kind in DetectorKind::all() {
        let json = serde_json::to_string(kind).unwrap();
        let back: DetectorKind = serde_json::from_str(&json).unwrap();
        assert_eq!(back, *kind, "json roundtrip failed for {kind}");
    }
}

#[test]
fn detector_kind_floor_so_dropped_variants_fail_loudly() {
    // 5 = streaming, sum-cusum, hc, filter-tick, dm-bocd (the kinds
    // that exist regardless of feature flags). If this number drops,
    // a non-gated variant was silently removed.
    assert!(
        DetectorKind::all().len() >= 5,
        "DetectorKind::all() shrunk: {}",
        DetectorKind::all().len()
    );
}

#[cfg(feature = "cli")]
#[test]
fn clap_value_enum_wire_form_matches_as_wire() {
    // Pins clap's auto-kebab-case behaviour against `as_wire()`. If a
    // future clap release changes the variant→string convention, this
    // test fails loudly instead of `--kind chen-wu` silently breaking.
    use clap::ValueEnum;
    for kind in DetectorKind::value_variants() {
        let pv = kind.to_possible_value().expect("ValueEnum supplies a possible value");
        assert_eq!(
            pv.get_name(),
            kind.as_wire(),
            "clap ValueEnum drifted from as_wire() for {kind:?}"
        );
    }
}
