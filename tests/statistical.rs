//! Production statistical test suite.
//!
//! Black-box tests that verify the detector's *operating characteristics* --
//! the properties a quant or ML practitioner must check before trusting it
//! in a production loop. Complements the unit tests in `src/lib.rs` which
//! verify the underlying math.
//!
//! Run with: `cargo test --features test-utils --test statistical`.

use cesura::eval::{self, Category, Rng};
use cesura::BocpdDetector;

// ── ARL₀: average run length to false alarm under H0 ─────────────────

/// Mean number of observations between false alarms on stationary N(0, 1).
///
/// Higher λ ⇒ higher ARL₀ ⇒ fewer false alarms. The classical CUSUM/EWMA
/// quality metric. We assert ordering and a loose lower bound; tight numerics
/// are unstable across seeds.
#[test]
fn arl0_increases_with_lambda() {
    let trials = 50;
    let trial_len = 1000;
    let threshold = 0.5;

    let mut alarm_rates = Vec::new();
    for &lam in &[100.0, 200.0, 500.0] {
        let det = BocpdDetector::new(lam, trial_len + 50);
        let mut total_alarms = 0usize;
        for s in 0..trials {
            let mut rng = Rng::new(0x9E37_u64.wrapping_mul(s as u64 + 1).wrapping_add(lam as u64));
            let data: Vec<f64> = (0..trial_len).map(|_| rng.normal(0.0, 1.0)).collect();
            total_alarms += det.detect(&data, threshold).len();
        }
        let rate = total_alarms as f64 / (trials * trial_len) as f64;
        let arl0 = if rate > 0.0 { 1.0 / rate } else { f64::INFINITY };
        eprintln!(
            "λ={lam:>4.0}  threshold={threshold}  alarms={total_alarms:>3}/{}  ARL₀≈{arl0:.0}",
            trials * trial_len
        );
        alarm_rates.push(rate);
    }

    // Ordering: λ=500 has fewer false alarms per sample than λ=100.
    assert!(
        alarm_rates[2] <= alarm_rates[0],
        "λ=500 alarm rate {} should be ≤ λ=100 alarm rate {}",
        alarm_rates[2],
        alarm_rates[0]
    );
    // Floor: at λ=500 the FAR should be < 1 alarm per 500 samples on stationary noise.
    assert!(
        alarm_rates[2] < 1.0 / 500.0,
        "λ=500 FAR {:.4} too high -- would saturate alerts in production",
        alarm_rates[2]
    );
}

// ── Operating characteristic: detection delay vs shift ───────────────

/// Mean detection delay should decrease monotonically as shift grows.
/// This is the curve a practitioner reads off the spec.
#[test]
fn detection_delay_decreases_with_shift_size() {
    let trials = 40;
    let pre = 150;
    let post = 150;
    let det = BocpdDetector::new(200.0, pre + post + 50);

    let shifts = [1.0_f64, 2.0, 3.0, 5.0];
    let mut report: Vec<(f64, f64, f64)> = Vec::new(); // (shift, hit_rate, mean_delay)
    for &shift in &shifts {
        let mut hits = 0usize;
        let mut delays = Vec::new();
        for s in 0..trials {
            let seed = (s as u64).wrapping_mul(7919) ^ ((shift * 1000.0) as u64);
            let mut rng = Rng::new(seed);
            let mut data: Vec<f64> = (0..pre).map(|_| rng.normal(0.0, 1.0)).collect();
            data.extend((0..post).map(|_| rng.normal(shift, 1.0)));
            let cps = det.detect(&data, 0.3);
            // First detection at-or-after the true change point.
            if let Some(cp) = cps.iter().find(|c| c.index >= pre) {
                hits += 1;
                delays.push((cp.index - pre) as f64);
            }
        }
        let hit_rate = hits as f64 / trials as f64;
        let mean_delay = if delays.is_empty() {
            f64::INFINITY
        } else {
            delays.iter().sum::<f64>() / delays.len() as f64
        };
        eprintln!("shift={shift:>3.1}σ  hit_rate={hit_rate:.2}  mean_delay={mean_delay:>5.1}");
        report.push((shift, hit_rate, mean_delay));
    }

    // Hit rate at 5σ ≥ at 1σ (monotone power).
    assert!(
        report[3].1 >= report[0].1,
        "5σ hit rate {} must dominate 1σ hit rate {}",
        report[3].1,
        report[0].1
    );
    // 5σ should hit on essentially every trial.
    assert!(
        report[3].1 >= 0.90,
        "5σ hit rate {} too low (expected ≥0.90)",
        report[3].1
    );
    // Mean delay at 5σ < at 2σ (sharper signal ⇒ faster detection).
    assert!(
        report[3].2 < report[1].2,
        "5σ delay {} must be < 2σ delay {}",
        report[3].2,
        report[1].2
    );
    // Production budget: 5σ shift detected within 25 steps on average.
    assert!(
        report[3].2 < 25.0,
        "5σ mean delay {} > 25 -- slower than acceptable",
        report[3].2
    );
}

// ── Regression snapshot: pinned indices for a fixed seed ─────────────

/// Lock the detector's behavior on a fixed-seed three-regime scenario.
/// Any change in the algorithm -- even a "harmless" tweak -- fires this.
///
/// Regenerate after deliberate algorithm changes: run the test once,
/// copy the printed `[…]` into `EXPECTED`.
#[test]
fn regression_snapshot_three_regime() {
    // Fixed scenario: N(0,1) for 100, then N(4,1) for 100, then N(0,1) for 100.
    let mut rng = Rng::new(0xBEEF_CAFE);
    let mut data: Vec<f64> = (0..100).map(|_| rng.normal(0.0, 1.0)).collect();
    data.extend((0..100).map(|_| rng.normal(4.0, 1.0)));
    data.extend((0..100).map(|_| rng.normal(0.0, 1.0)));

    let det = BocpdDetector::new(200.0, 350);
    let cps = det.detect(&data, 0.3);
    let indices: Vec<usize> = cps.iter().map(|c| c.index).collect();

    eprintln!("regression_snapshot_three_regime indices = {indices:?}");

    // Pinned. If this fails after an intentional change, copy the eprintln'd
    // vector here verbatim. If it fails for any other reason, investigate.
    const EXPECTED: &[usize] = &[101, 201];
    assert_eq!(
        indices.as_slice(),
        EXPECTED,
        "detector behavior drifted -- see eprintln"
    );

    // Even if the snapshot ever changes, these structural guarantees must hold:
    assert!(indices.len() >= 2, "should detect entry + exit");
    assert!(
        indices[0] >= 95 && indices[0] <= 130,
        "first detection {} outside [95, 130]",
        indices[0]
    );
}

// ── CUSUM baseline comparison ────────────────────────────────────────

/// Two-sided CUSUM with parameters (k, h) on standardized residuals.
/// Estimates running mean/std from a warmup window, then emits an alarm
/// when |S±| > h. Reference: Page (1954); Lai (1995) survey.
fn cusum_detect(data: &[f64], k: f64, h: f64, warmup: usize) -> Vec<usize> {
    if data.len() <= warmup {
        return vec![];
    }
    let win = &data[..warmup];
    let mu = win.iter().sum::<f64>() / win.len() as f64;
    let var = win.iter().map(|x| (x - mu).powi(2)).sum::<f64>() / win.len() as f64;
    let sigma = var.sqrt().max(1e-10);

    let mut s_pos = 0.0;
    let mut s_neg = 0.0;
    let cooldown = 15;
    let mut last = 0usize;
    let mut out = Vec::new();
    for (i, &x) in data.iter().enumerate().skip(warmup) {
        let z = (x - mu) / sigma;
        s_pos = (s_pos + z - k).max(0.0);
        s_neg = (s_neg - z - k).max(0.0);
        if (s_pos > h || s_neg > h) && i - last >= cooldown {
            out.push(i);
            last = i;
            s_pos = 0.0;
            s_neg = 0.0;
        }
    }
    out
}

/// BOCPD must not collapse against a strong classical baseline.
/// CUSUM is well-tuned for known mean shifts and routinely competitive;
/// a credible BOCPD implementation should at minimum match its aggregate F1
/// on a heterogeneous suite (mean shifts + variance + periodic patterns).
#[test]
fn bocpd_matches_or_beats_cusum_aggregate() {
    let scenarios = eval::all_scenarios();
    let det = BocpdDetector::new(200.0, 400);

    let mut bocpd_metrics = Vec::new();
    let mut cusum_metrics = Vec::new();

    for s in &scenarios {
        let bocpd_idx: Vec<usize> = det.detect(&s.data, 0.3).iter().map(|c| c.index).collect();
        let cusum_idx = cusum_detect(&s.data, 0.5, 4.0, 30);

        let mut bm = eval::match_detections(&bocpd_idx, &s.ground_truth, 20);
        bm.name = s.name.to_string();
        bm.category = s.category;
        bocpd_metrics.push(bm);

        let mut cm = eval::match_detections(&cusum_idx, &s.ground_truth, 20);
        cm.name = s.name.to_string();
        cm.category = s.category;
        cusum_metrics.push(cm);
    }

    let b_agg = eval::aggregate(&bocpd_metrics);
    let c_agg = eval::aggregate(&cusum_metrics);
    eprintln!("\n=== BOCPD vs CUSUM (k=0.5, h=4.0) ===");
    eprintln!(
        "BOCPD: F1={:.3}  P={:.3}  R={:.3}  TP={} FP={} FN={}",
        b_agg.f1, b_agg.precision, b_agg.recall, b_agg.tp, b_agg.fp, b_agg.r#fn
    );
    eprintln!(
        "CUSUM: F1={:.3}  P={:.3}  R={:.3}  TP={} FP={} FN={}",
        c_agg.f1, c_agg.precision, c_agg.recall, c_agg.tp, c_agg.fp, c_agg.r#fn
    );

    // Heterogeneous suite: BOCPD's structural advantage (variance changes,
    // periodic-aware via detrend) should land it within 5 F1 points of
    // CUSUM at worst, ideally above.
    assert!(
        b_agg.f1 >= c_agg.f1 - 0.05,
        "BOCPD F1 {:.3} more than 5pp below CUSUM F1 {:.3} -- investigate",
        b_agg.f1,
        c_agg.f1
    );

    // Per-scenario per-category accountability: BOCPD must not lose on the
    // MustReject category, where periodic/trending patterns trip CUSUM.
    let mr_bocpd_fp: usize = bocpd_metrics
        .iter()
        .filter(|m| m.category == Category::MustReject)
        .map(|m| m.fp)
        .sum();
    let mr_cusum_fp: usize = cusum_metrics
        .iter()
        .filter(|m| m.category == Category::MustReject)
        .map(|m| m.fp)
        .sum();
    eprintln!("MustReject FPs: BOCPD={mr_bocpd_fp}, CUSUM={mr_cusum_fp}");
    assert!(
        mr_bocpd_fp <= mr_cusum_fp,
        "BOCPD must reject as well as CUSUM on MustReject: {mr_bocpd_fp} > {mr_cusum_fp}"
    );
}
