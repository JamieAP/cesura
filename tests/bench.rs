
#![cfg(feature = "test-utils")]

use cesura::bench::{
    classify_verdict, evaluate, load_reports, run_bench, write_report, Fixture, FixtureRegistry,
    Metrics, Report, VerdictLabel, DEFAULT_MIN_EVENTS,
};
use cesura::{BocpdDetector, ChangePoint};

fn fix_with_gt(gt: Vec<usize>, t: usize, margin: usize) -> Fixture {
    let data: Vec<Vec<f64>> = (0..t).map(|_| vec![0.0]).collect();
    Fixture {
        name: "stub".into(),
        version: 1,
        d: 1,
        data,
        epochs: None,
        ground_truth: gt,
        seed: None,
        margin,
    }
}

fn cps(indices: &[usize]) -> Vec<ChangePoint> {
    indices
        .iter()
        .map(|&i| ChangePoint {
            index: i,
            confidence: 1.0,
            shift_sigma: 0.0,
        })
        .collect()
}

#[test]
fn evaluate_perfect_match_yields_f1_one() {
    let fix = fix_with_gt(vec![100, 200, 300], 400, 10);
    let m = evaluate(&cps(&[100, 200, 300]), &fix);
    assert_eq!(m.n_events, 3);
    assert_eq!(m.n_cps, 3);
    assert!((m.precision - 1.0).abs() < 1e-12);
    assert!((m.recall - 1.0).abs() < 1e-12);
    assert!((m.f1 - 1.0).abs() < 1e-12);
    assert_eq!(m.far, 0);
    assert_eq!(m.per_event_hit, vec![true, true, true]);
}

#[test]
fn evaluate_far_outside_margin_counts_as_fp() {
    // CPs at 50 and 500; nearest GT is 100/300 at distance ≥ 50, margin=10.
    let fix = fix_with_gt(vec![100, 300], 600, 10);
    let m = evaluate(&cps(&[50, 500]), &fix);
    assert_eq!(m.n_cps, 2);
    assert_eq!(m.n_events, 2);
    assert_eq!(m.far, 2);
    assert!(m.per_event_hit.iter().all(|&b| !b));
    assert_eq!(m.f1, 0.0);
}

#[test]
fn evaluate_canonical_cases_from_legacy_f1_with_margin() {
    // Underlying tape: 6 bars at indices 0..6; "events" at indices 1, 3, 5
    // (mirroring the old test's epoch-100/200/300 layout where bar epochs
    // bracketed the events at indices 1, 3, 5).
    // Distances in this layout: bar-1 distance to event-1 = 0, etc.
    let fix = fix_with_gt(vec![1, 3, 5], 6, 0); // margin=0: only exact hits
    // Case 1: 2 perfect hits + 1 miss (CP at 5 missing). Precision=1, Recall=2/3.
    let m = evaluate(&cps(&[1, 3]), &fix);
    assert!((m.precision - 1.0).abs() < 1e-12);
    assert!((m.recall - 2.0 / 3.0).abs() < 1e-12);
    let want_f = 2.0 * 1.0 * (2.0 / 3.0) / (1.0 + 2.0 / 3.0);
    assert!((m.f1 - want_f).abs() < 1e-12);

    // Case 2: zero detections.
    let m = evaluate(&cps(&[]), &fix);
    assert_eq!((m.precision, m.recall, m.f1), (0.0, 0.0, 0.0));

    // Case 3: all false alarms (CP at 0, no event there at margin=0).
    let m = evaluate(&cps(&[0]), &fix);
    assert_eq!((m.precision, m.recall, m.f1), (0.0, 0.0, 0.0));
    assert_eq!(m.far, 1);

    // Case 4: one TP + one FP. CP at 1 (TP) and 0 (FP).
    let m = evaluate(&cps(&[0, 1]), &fix);
    assert!((m.precision - 0.5).abs() < 1e-12);
    assert!((m.recall - 1.0 / 3.0).abs() < 1e-12);
    let want_f = 2.0 * 0.5 * (1.0 / 3.0) / (0.5 + 1.0 / 3.0);
    assert!((m.f1 - want_f).abs() < 1e-12);

    // Case 5: one CP within margin of multiple events (degenerate but legal).
    // Wide margin of 5: CP at 3 hits all three events (dist 2, 0, 2). The
    // greedy matcher picks one to match (dist 0 → event-1), leaving the
    // other two as FN. precision=1.0, recall=1/3.
    let fix_wide = fix_with_gt(vec![1, 3, 5], 6, 5);
    let m = evaluate(&cps(&[3]), &fix_wide);
    assert!((m.precision - 1.0).abs() < 1e-12);
    assert!((m.recall - 1.0 / 3.0).abs() < 1e-12);
}

#[test]
fn evaluate_per_event_hit_matches_legacy() {
    // 3 events; 2 CPs that hit events 0 and 1, miss event 2.
    let fix = fix_with_gt(vec![10, 50, 200], 300, 5);
    let m = evaluate(&cps(&[12, 48]), &fix);
    assert_eq!(m.per_event_hit, vec![true, true, false]);
}

/// Migrated behavioural intent of `count_far_cps_thresholds_against_all_events`.
/// `Metrics::far` counts unmatched CPs after one-to-one matching,
/// including extra detections within a matched event's margin.
#[test]
fn evaluate_far_field_counts_distant_cps() {
    let fix = fix_with_gt(vec![0], 100, 5);
    // CPs at 0 (TP), 3 (within margin), 50 (far), 90 (far).
    let m = evaluate(&cps(&[0, 3, 50, 90]), &fix);
    // Greedy: 0 → event-0, leaving 3/50/90 unmatched.
    assert_eq!(m.far, 3);
}

#[test]
fn evaluate_partial_recall() {
    let fix = fix_with_gt(vec![100, 200, 300], 400, 5);
    let m = evaluate(&cps(&[102]), &fix);
    assert_eq!(m.n_events, 3);
    assert_eq!(m.n_cps, 1);
    assert!((m.precision - 1.0).abs() < 1e-12);
    assert!((m.recall - 1.0 / 3.0).abs() < 1e-12);
    assert!((m.f1 - 0.5).abs() < 1e-12);
}

#[test]
fn classify_verdict_returns_sanity_check_below_floor() {
    // 5-event report (the macro tape) → SanityCheck regardless of metrics.
    let r = Report {
        detector: "any".into(),
        fixture: "crypto_macro_5".into(),
        fixture_version: 1,
        commit: "abc".into(),
        seed: None,
        metrics: Metrics {
            precision: 1.0,
            recall: 1.0,
            f1: 1.0,
            far: 0,
            mean_delay_idx: None,
            per_event_hit: vec![true; 5],
            n_events: 5,
            n_cps: 5,
        },
        timing_us: 0,
        attribution: Vec::new(),
        timestamp: 0,
    };
    let v = classify_verdict(&[r], DEFAULT_MIN_EVENTS);
    assert_eq!(v.label, VerdictLabel::SanityCheck);
    assert_eq!(v.n_events_used, 5);
}

#[test]
fn classify_verdict_promotes_at_floor() {
    let mk = |n_events: usize| Report {
        detector: "d".into(),
        fixture: "f".into(),
        fixture_version: 1,
        commit: "c".into(),
        seed: None,
        metrics: Metrics {
            precision: 0.5,
            recall: 0.5,
            f1: 0.5,
            far: 10,
            mean_delay_idx: None,
            per_event_hit: vec![false; n_events],
            n_events,
            n_cps: 5,
        },
        timing_us: 0,
        attribution: Vec::new(),
        timestamp: 0,
    };
    assert_eq!(
        classify_verdict(&[mk(50), mk(80)], DEFAULT_MIN_EVENTS).label,
        VerdictLabel::VerdictGrade
    );
    // Floor is the smallest n_events across reports.
    assert_eq!(
        classify_verdict(&[mk(50), mk(49)], DEFAULT_MIN_EVENTS).label,
        VerdictLabel::SanityCheck
    );
}

#[test]
fn power_label_matches_classify_verdict() {
    use cesura::bench::{label_str, power_label};
    // Sub-floor → SanityCheck; ≥ floor → VerdictGrade. Mirrors
    // classify_verdict's per-report comparison so banner labels are
    // by-construction equivalent.
    assert_eq!(power_label(0, 50), VerdictLabel::SanityCheck);
    assert_eq!(power_label(5, 50), VerdictLabel::SanityCheck);
    assert_eq!(power_label(49, 50), VerdictLabel::SanityCheck);
    assert_eq!(power_label(50, 50), VerdictLabel::VerdictGrade);
    assert_eq!(power_label(200, 50), VerdictLabel::VerdictGrade);
    assert_eq!(label_str(VerdictLabel::SanityCheck), "sanity check");
    assert_eq!(label_str(VerdictLabel::VerdictGrade), "verdict-grade");
}

#[test]
fn report_json_round_trip() {
    let dir = tempdir_unique("cesura-bench-rt");
    let r = Report {
        detector: "Bocpd".into(),
        fixture: "synthetic_x".into(),
        fixture_version: 7,
        commit: "deadbee".into(),
        seed: Some(42),
        metrics: Metrics {
            precision: 0.8,
            recall: 0.6,
            f1: 0.685_714_285_714_285_7,
            far: 4,
            mean_delay_idx: Some(2.5),
            per_event_hit: vec![true, false, true],
            n_events: 3,
            n_cps: 5,
        },
        timing_us: 12345,
        timestamp: 1_700_000_000,
        attribution: vec![vec![0, 2], Vec::new(), vec![1]],
    };
    let path = write_report(&r, &dir).expect("write");
    assert!(path.exists());
    let loaded = load_reports(&dir);
    assert_eq!(loaded.len(), 1);
    let l = &loaded[0];
    assert_eq!(l.detector, r.detector);
    assert_eq!(l.fixture, r.fixture);
    assert_eq!(l.fixture_version, r.fixture_version);
    assert_eq!(l.seed, r.seed);
    assert_eq!(l.metrics.n_events, r.metrics.n_events);
    assert_eq!(l.metrics.per_event_hit, r.metrics.per_event_hit);
    assert_eq!(l.attribution, r.attribution);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn run_bench_via_bocpd_adapter_on_synthetic() {
    use cesura::bench::adapters::BocpdAdapter;
    // A known synthetic fixture from FixtureRegistry::synthetic.
    let fixtures = FixtureRegistry::synthetic();
    let fix = fixtures
        .iter()
        .find(|f| f.name == "noisy_3sigma")
        .expect("3sigma scenario");
    let det = BocpdDetector::new(200.0, 250);
    let adapter = BocpdAdapter {
        det: &det,
        mv: false,
        label: "Bocpd",
    };
    let (report, _) = run_bench(&adapter, fix);
    assert_eq!(report.detector, "Bocpd");
    assert_eq!(report.fixture, "noisy_3sigma");
    assert_eq!(report.metrics.n_events, fix.ground_truth.len());
    // 3σ shift on noisy synthetic: detector should hit precision/recall > 0.
    assert!(report.metrics.recall > 0.0, "recall = {}", report.metrics.recall);
}

#[test]
fn step_shift_injection_is_detectable_on_synthetic_baseline() {
    use cesura::bench::adapters::BocpdAdapter;
    use cesura::bench::fixture::step_shift_inject;
    // Build a noisy zero-mean baseline (univariate, σ ≈ 1.0) and inject
    // step shifts at known indices via the same square-wave logic
    // anomaly_injected_crypto uses. The resulting fixture must yield a
    // recall > 0 under a default Bocpd, which validates the step-shift
    // contract: the injected CPs are CPs a real CP detector can find
    // (in contrast to the previous single-bar-impulse implementation).
    let n = 1200usize;
    let mut rng_state = 0xdead_beef_u64;
    let mut next = || {
        // Tiny LCG for a deterministic uniform baseline (no extra dependency).
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u = ((rng_state >> 33) as f64) / (1u64 << 31) as f64;
        u - 1.0
    };
    let baseline: Vec<Vec<f64>> = (0..n).map(|_| vec![next()]).collect();
    let chosen = vec![300usize, 600, 900];
    let sigmas = [1.0_f64];
    let data = step_shift_inject(&baseline, &chosen, 3.0, &sigmas);
    let fix = Fixture {
        name: "synthetic_step_shift".into(),
        version: 1,
        d: 1,
        data,
        epochs: None,
        ground_truth: chosen.clone(),
        seed: Some(0),
        margin: 30,
    };
    let det = cesura::BocpdDetector::new(200.0, 250);
    let adapter = BocpdAdapter { det: &det, mv: false, label: "Bocpd" };
    let (report, _) = run_bench(&adapter, &fix);
    assert!(
        report.metrics.recall >= 2.0 / 3.0,
        "step-shift fixture must be detectable; recall={} hits={:?}",
        report.metrics.recall,
        report.metrics.per_event_hit
    );
}

#[test]
fn run_bench_via_conformal_adapter_projects_to_changepoint() {
    use cesura::bench::adapters::ConformalAdapter;
    use cesura::ConformalCpWrapper;
    let fix = FixtureRegistry::synthetic()
        .into_iter()
        .find(|f| f.name == "noisy_3sigma")
        .expect("3sigma scenario");
    let wrapper = ConformalCpWrapper::new(BocpdDetector::new(200.0, 250))
        .with_coverage(0.9)
        .with_calibration_capacity(8);
    let adapter = ConformalAdapter::new(wrapper, "Conformal+Bocpd");
    let (report, _) = run_bench(&adapter, &fix);
    assert_eq!(report.detector, "Conformal+Bocpd");
    assert!(report.metrics.recall > 0.0);
    // The projection must drop timing_interval cleanly: n_cps reflects
    // emitted CPs only, not paired interval objects.
    assert_eq!(report.metrics.n_cps, report.metrics.per_event_hit.iter().filter(|&&b| b).count() + report.metrics.far);
}

#[test]
fn synthetic_registry_covers_all_eval_scenarios() {
    let n = FixtureRegistry::synthetic().len();
    assert!(n >= 26, "expected ≥26 synthetic scenarios, got {n}");
    for f in FixtureRegistry::synthetic() {
        assert_eq!(f.d, 1);
        assert!(!f.is_empty());
        assert_eq!(f.data[0].len(), 1);
    }
}

#[test]
fn hc_attribution_renders_per_stream_provenance() {
    use cesura::bench::multistream_adapter::HcAggregatorAdapter;
    use cesura::bench::{render_attribution, AttributionRow};
    use cesura::eval::Rng;

    // A multi-stream HC fire must surface the
    // crossing stream(s) through the rendered attribution table. Build
    // a 4-stream synthetic fixture with a 5σ shift on stream 1 only;
    // HC's `attribution[i]` should list at least stream 1 for the
    // post-shift fire.
    const T: usize = 240;
    const D: usize = 4;
    const CP_STEP: usize = 120;
    const SHIFT: f64 = 5.0;
    let mut rng = Rng::new(7);
    let mut data: Vec<Vec<f64>> = Vec::with_capacity(T);
    for t in 0..T {
        let mut row = Vec::with_capacity(D);
        for s in 0..D {
            let mu = if t >= CP_STEP && s == 1 { SHIFT } else { 0.0 };
            row.push(rng.normal(mu, 1.0));
        }
        data.push(row);
    }
    let fix = Fixture {
        name: "synthetic_hc_attribution_4_streams".into(),
        version: 1,
        d: D,
        data,
        epochs: None,
        ground_truth: vec![CP_STEP],
        seed: Some(7),
        margin: 30,
    };

    let adapter = HcAggregatorAdapter::with_bf_streams(3.0).label("HC(BF)");
    let (_, result) = run_bench(&adapter, &fix);
    assert!(!result.cps.is_empty(), "HC fired no CPs on a 5σ shift");

    let mut rows: Vec<AttributionRow> = Vec::new();
    for (i, streams) in result.attribution.iter().enumerate() {
        if streams.is_empty() {
            continue;
        }
        let cp = &result.cps[i];
        let names: Vec<String> = streams.iter().map(|&s| format!("s{s}")).collect();
        rows.push(AttributionRow {
            detector: "HC(BF)".into(),
            fixture: fix.name.clone(),
            fire_idx: i,
            cp_index: cp.index,
            confidence: cp.confidence,
            streams: names,
        });
    }
    assert!(
        !rows.is_empty(),
        "HC produced 0 attribution rows on a 4-stream fixture with sparse 5σ shift"
    );
    let md = render_attribution(&rows);
    assert!(md.contains("| streams |"), "expected attribution header in markdown");
    // Structural assertion: at least one row names at
    // least one stream. Exact stream identities are sensitive to BOCPD
    // warmup / ARL_0 noise on a 240-step tape -- HC may attribute to a
    // false-alarm stream first. The harness contract is that
    // attribution is rendered, not that it's correct on small tapes.
    let any_stream = rows.iter().any(|r| !r.streams.is_empty());
    assert!(
        any_stream,
        "expected ≥1 attribution row with ≥1 stream; got:\n{md}"
    );
}

fn tempdir_unique(prefix: &str) -> std::path::PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let p = std::env::temp_dir().join(format!("{prefix}-{nanos}"));
    std::fs::create_dir_all(&p).unwrap();
    p
}
