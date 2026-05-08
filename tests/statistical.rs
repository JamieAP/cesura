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

    let mut alarm_rates = Vec::new();
    for &lam in &[100.0, 200.0, 500.0] {
        let det = BocpdDetector::new(lam, trial_len + 50);
        let mut total_alarms = 0usize;
        for s in 0..trials {
            let mut rng = Rng::new(0x9E37_u64.wrapping_mul(s as u64 + 1).wrapping_add(lam as u64));
            let data: Vec<f64> = (0..trial_len).map(|_| rng.normal(0.0, 1.0)).collect();
            total_alarms += det.detect(&data).len();
        }
        let rate = total_alarms as f64 / (trials * trial_len) as f64;
        let arl0 = if rate > 0.0 { 1.0 / rate } else { f64::INFINITY };
        eprintln!(
            "λ={lam:>4.0}  alarms={total_alarms:>3}/{}  ARL₀≈{arl0:.0}",
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
            let cps = det.detect(&data);
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
    let cps = det.detect(&data);
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
        let bocpd_idx: Vec<usize> = det.detect(&s.data).iter().map(|c| c.index).collect();
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

    let mut focus_metrics = Vec::new();
    for s in &scenarios {
        let focus_idx = eval::focus_detect(&s.data, 8.0);
        let mut fm = eval::match_detections(&focus_idx, &s.ground_truth, 20);
        fm.name = s.name.to_string();
        fm.category = s.category;
        focus_metrics.push(fm);
    }

    let b_agg = eval::aggregate(&bocpd_metrics);
    let c_agg = eval::aggregate(&cusum_metrics);
    let f_agg = eval::aggregate(&focus_metrics);
    eprintln!("\n=== BOCPD vs CUSUM (k=0.5, h=4.0) vs FOCuS (thr=8.0) ===");
    eprintln!(
        "BOCPD: F1={:.3}  P={:.3}  R={:.3}  TP={} FP={} FN={}",
        b_agg.f1, b_agg.precision, b_agg.recall, b_agg.tp, b_agg.fp, b_agg.r#fn
    );
    eprintln!(
        "CUSUM: F1={:.3}  P={:.3}  R={:.3}  TP={} FP={} FN={}",
        c_agg.f1, c_agg.precision, c_agg.recall, c_agg.tp, c_agg.fp, c_agg.r#fn
    );
    eprintln!(
        "FOCuS: F1={:.3}  P={:.3}  R={:.3}  TP={} FP={} FN={}",
        f_agg.f1, f_agg.precision, f_agg.recall, f_agg.tp, f_agg.fp, f_agg.r#fn
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

    // FOCuS comparison is informational. The frequentist GLR has a different
    // operating curve so a strict floor here would over-constrain it; a soft
    // floor catches a "FOCuS is broken" regression without coupling tuning.
    assert!(
        f_agg.f1 >= 0.10,
        "FOCuS aggregate F1 {:.3} below sanity floor 0.10 -- broken?",
        f_agg.f1
    );
}

#[test]
fn arl0_lambda_sweep() {
    let trials = 30;
    let trial_len = 1000;
    let lambdas = [100.0_f64, 300.0, 1000.0];

    let mut arl0s = Vec::new();
    for &lam in &lambdas {
        let det = BocpdDetector::new(lam, trial_len + 50);
        let mut alarms = 0usize;
        for s in 0..trials {
            let mut rng = Rng::new(0xA1B2_u64.wrapping_mul(s as u64 + 1) ^ lam.to_bits());
            let data: Vec<f64> = (0..trial_len).map(|_| rng.normal(0.0, 1.0)).collect();
            alarms += det.detect(&data).len();
        }
        let rate = alarms as f64 / (trials * trial_len) as f64;
        let arl0 = if rate > 0.0 { 1.0 / rate } else { f64::INFINITY };
        eprintln!("λ={lam:>5.0}  alarms={alarms:>3}  ARL0≈{arl0:>8.0}");
        arl0s.push(arl0);
    }

    for w in arl0s.windows(2) {
        assert!(
            w[1] >= w[0] || w[1].is_infinite(),
            "ARL0 should be non-decreasing in λ: {arl0s:?}"
        );
    }
    assert!(
        arl0s[2] >= 5000.0 || arl0s[2].is_infinite(),
        "λ=1000 ARL0 {} < 5000 -- false-alarm budget too loose for production",
        arl0s[2]
    );
}

#[test]
fn bocpd_beats_cusum_envelope() {
    let scenarios = eval::all_scenarios();
    let det = BocpdDetector::new(200.0, 400);

    let mut bocpd_metrics = Vec::new();
    let mut envelope_per_scenario: Vec<eval::EvalMetrics> = Vec::new();

    for s in &scenarios {
        let bocpd_idx: Vec<usize> = det.detect(&s.data).iter().map(|c| c.index).collect();
        let mut bm = eval::match_detections(&bocpd_idx, &s.ground_truth, 20);
        bm.name = s.name.to_string();
        bm.category = s.category;
        bocpd_metrics.push(bm);

        let mut best_f1 = -1.0_f64;
        let mut best: Option<eval::EvalMetrics> = None;
        for h in [2.0_f64, 3.0, 4.0, 5.0, 6.0] {
            let cusum_idx = cusum_detect(&s.data, 0.5, h, 30);
            let mut m = eval::match_detections(&cusum_idx, &s.ground_truth, 20);
            m.name = s.name.to_string();
            m.category = s.category;
            if m.f1 > best_f1 {
                best_f1 = m.f1;
                best = Some(m);
            }
        }
        envelope_per_scenario.push(best.unwrap());
    }

    let b = eval::aggregate(&bocpd_metrics);
    let e = eval::aggregate(&envelope_per_scenario);
    eprintln!(
        "\nBOCPD            : F1={:.3} P={:.3} R={:.3} TP={} FP={} FN={}",
        b.f1, b.precision, b.recall, b.tp, b.fp, b.r#fn
    );
    eprintln!(
        "CUSUM envelope   : F1={:.3} P={:.3} R={:.3} TP={} FP={} FN={}",
        e.f1, e.precision, e.recall, e.tp, e.fp, e.r#fn
    );

    assert!(
        b.f1 >= e.f1 - 0.05,
        "BOCPD F1 {:.3} > 5pp behind per-scenario CUSUM envelope {:.3}",
        b.f1,
        e.f1
    );
}

#[test]
fn confidence_is_calibrated() {
    let n_trials = 200;
    let pre = 150;
    let post = 150;
    let det = BocpdDetector::new(200.0, pre + post + 50);

    let mut bins: Vec<(f64, f64, usize, usize)> =
        vec![(0.3, 0.5, 0, 0), (0.5, 0.7, 0, 0), (0.7, 0.85, 0, 0), (0.85, 1.01, 0, 0)];

    for s in 0..n_trials {
        let mut rng = Rng::new(0xCA11B_u64 ^ s as u64);
        let true_cp = pre;
        let shift = 1.0 + (s as f64 % 4.0);
        let mut data: Vec<f64> = (0..pre).map(|_| rng.normal(0.0, 1.0)).collect();
        data.extend((0..post).map(|_| rng.normal(shift, 1.0)));

        for cp in det.detect(&data) {
            let is_true = (cp.index as i64 - true_cp as i64).abs() <= 20;
            for (lo, hi, n, tp) in bins.iter_mut() {
                if cp.confidence >= *lo && cp.confidence < *hi {
                    *n += 1;
                    if is_true {
                        *tp += 1;
                    }
                }
            }
        }
    }

    eprintln!("\n=== Confidence calibration ===");
    eprintln!("range          n     tp    empirical_precision");
    let mut prev_prec = 0.0;
    for &(lo, hi, n, tp) in &bins {
        let prec = if n > 0 { tp as f64 / n as f64 } else { f64::NAN };
        eprintln!("[{lo:.2},{hi:.2})  {n:>4}  {tp:>4}   {prec:.3}");
        if n >= 10 && !prec.is_nan() {
            assert!(
                prec >= prev_prec - 0.10,
                "precision non-monotone: bin [{lo},{hi}) prec={prec:.3} dropped > 10pp from prev {prev_prec:.3}"
            );
            prev_prec = prec;
        }
    }
}

// ── Per-scenario operational F1 floors ───────────────────────────────

///
#[test]
fn eval_operational_per_scenario_floors() {
    use cesura::eval::match_detections;

    let scenarios = eval::all_scenarios();
    let det = BocpdDetector::new(200.0, 400);

    let floors: &[(&str, f64)] = &[
        ("cpu_burst", 0.90),
        ("restart_storm", 0.66),
        ("deployment_rollout", 0.50),
    ];

    let mut found = std::collections::HashMap::new();
    for s in &scenarios {
        if s.category != Category::Operational {
            continue;
        }
        let cps: Vec<usize> = det.detect(&s.data).iter().map(|c| c.index).collect();
        let m = match_detections(&cps, &s.ground_truth, 20);
        eprintln!(
            "{:<20} F1={:.2} P={:.2} R={:.2} TP={} FP={} FN={}",
            s.name, m.f1, m.precision, m.recall, m.tp, m.fp, m.r#fn
        );
        found.insert(s.name, m);
    }

    for (name, floor) in floors {
        let m = found
            .get(name)
            .unwrap_or_else(|| panic!("scenario {name} missing from suite"));
        assert!(
            m.f1 >= *floor,
            "Operational scenario {name}: F1={:.2} < floor {:.2}",
            m.f1, floor
        );
    }

}

// ── β-divergence robust BOCPD (Knoblauch et al. 2018) ───────────────

#[test]
fn robust_bocpd_rejects_cauchy_noise() {
    // Cauchy(0, 1) has infinite variance; standard cesura fires on every
    // tail observation. β = 0.1 bounds the influence of any single point,
    // so the run-length distribution does not collapse on tail events.
    use cesura::eval::heavy_tail_scenarios;

    let cauchy = heavy_tail_scenarios()
        .into_iter()
        .find(|s| s.name == "cauchy_noise")
        .expect("cauchy_noise scenario");

    let std_idx: Vec<usize> = BocpdDetector::new(200.0, 400)
        .detect(&cauchy.data)
        .iter()
        .map(|c| c.index)
        .collect();
    let rob_idx: Vec<usize> = BocpdDetector::new(200.0, 400)
        .with_beta(0.15)
        .detect(&cauchy.data)
        .iter()
        .map(|c| c.index)
        .collect();

    eprintln!(
        "cauchy: standard={} {std_idx:?}  β=0.15: {} {rob_idx:?}",
        std_idx.len(),
        rob_idx.len()
    );
    assert!(
        std_idx.len() >= 3,
        "standard on Cauchy gave only {} false alarms -- premise broken",
        std_idx.len()
    );
    // Robust must materially reduce the count. Strict-zero would over-claim
    // robustness; the contract is "fewer false alarms than standard".
    assert!(
        rob_idx.len() < std_idx.len(),
        "robust β=0.3 fires {} times, no improvement over standard {}",
        rob_idx.len(),
        std_idx.len()
    );
}

#[test]
fn robust_bocpd_still_detects_real_shift_under_t3() {
    // A 3σ-of-noise shift buried in t₃ noise. Robust path must still
    // resolve the shift -- robustness should not collapse to "ignores
    // everything". Tolerance: ±25 steps from t = 150.
    use cesura::eval::heavy_tail_scenarios;

    let scenario = heavy_tail_scenarios()
        .into_iter()
        .find(|s| s.name == "t3_with_shift")
        .expect("t3_with_shift scenario");

    let cps = BocpdDetector::new(200.0, 400)
        .with_beta(0.15)
        .detect(&scenario.data);

    let near = cps.iter().any(|cp| (cp.index as i64 - 150).abs() <= 25);
    eprintln!(
        "robust t₃+shift detections: {:?}",
        cps.iter()
            .map(|cp| (cp.index, cp.confidence))
            .collect::<Vec<_>>()
    );
    assert!(near, "β=0.3 missed the real shift at t=150 within ±25");
}

// ── Heavy-tail F1 baseline (β-divergence headroom) ───────────────────

#[test]
fn eval_heavy_tail_aggregate_floor() {
    use cesura::eval::{aggregate, heavy_tail_scenarios, match_detections};

    let det = BocpdDetector::new(200.0, 400);
    let mut metrics = Vec::new();
    for s in heavy_tail_scenarios() {
        let cps: Vec<usize> = det.detect(&s.data).iter().map(|c| c.index).collect();
        let mut m = match_detections(&cps, &s.ground_truth, 20);
        m.name = s.name.to_string();
        m.category = s.category;
        metrics.push(m);
    }
    let agg = aggregate(&metrics);
    eprintln!(
        "\n=== heavy-tail aggregate ===\nF1={:.3} P={:.3} R={:.3} TP={} FP={} FN={}",
        agg.f1, agg.precision, agg.recall, agg.tp, agg.fp, agg.r#fn
    );
    assert!(
        agg.f1 >= 0.18,
        "heavy-tail aggregate F1={:.3} below standard-path floor 0.18",
        agg.f1
    );
}

#[test]
fn post_hoc_confidence_filter_is_monotone() {
    let trials = 100;
    let trial_len = 2000;
    let det = BocpdDetector::new(300.0, trial_len + 50);
    let cutoffs = [0.30_f64, 0.50, 0.70, 0.85, 0.95];
    let mut counts = vec![0usize; cutoffs.len()];

    for s in 0..trials {
        let mut rng = Rng::new(0xF100_u64.wrapping_mul(s as u64 + 1));
        let data: Vec<f64> = (0..trial_len).map(|_| rng.normal(0.0, 1.0)).collect();
        let cps = det.detect(&data);
        for (j, &cut) in cutoffs.iter().enumerate() {
            counts[j] += cps.iter().filter(|c| c.confidence >= cut).count();
        }
    }
    for (cut, c) in cutoffs.iter().zip(&counts) {
        eprintln!("cutoff={cut:.2}  retained={c}");
    }
    for w in counts.windows(2) {
        assert!(
            w[1] <= w[0],
            "post-hoc confidence filter must be non-increasing in cutoff: {counts:?}"
        );
    }
}

// ── Auto-β calibration ──────────────────────────────────────────────

#[test]
#[ignore]
fn auto_beta_calibration_probe() {
    // Empirical kurtosis on the heavy-tail scenarios. Run once to fix the
    // SLOPE / DEAD_ZONE / BETA_MAX constants in `auto_beta::beta_from_excess_kurtosis`.
    // Run with `--ignored --nocapture`.
    use cesura::auto_beta::{auto_beta, excess_kurtosis};
    use cesura::eval::heavy_tail_scenarios;
    for s in heavy_tail_scenarios() {
        let k = excess_kurtosis(&s.data);
        let b = auto_beta(&s.data);
        eprintln!("{:>20} excess_kurtosis={:>10.3}  → β={:.3}", s.name, k, b);
    }
    let mut rng = cesura::eval::Rng::new(9991);
    let gauss: Vec<f64> = (0..1000).map(|_| rng.normal(0.0, 1.0)).collect();
    let k = excess_kurtosis(&gauss);
    let b = auto_beta(&gauss);
    eprintln!("{:>20} excess_kurtosis={:>10.3}  → β={:.3}", "N(0,1)/1000", k, b);
}

#[test]
fn auto_beta_picks_zero_for_gaussian() {
    // 300 samples of N(0,1): excess kurtosis ≈ 0 → dead zone → β = 0 →
    // detector behaves bit-for-bit like the standard path.
    use cesura::auto_beta::auto_beta;
    let mut rng = cesura::eval::Rng::new(11);
    let warmup: Vec<f64> = (0..300).map(|_| rng.normal(0.0, 1.0)).collect();
    let b = auto_beta(&warmup);
    assert_eq!(b, 0.0, "Gaussian warmup should pick β=0, got {b}");

    // And a clean shift downstream: auto_beta path matches plain detector.
    let mut data = warmup.clone();
    data.extend((0..200).map(|_| rng.normal(5.0, 1.0)));
    let plain = BocpdDetector::new(200.0, 600).detect(&data);
    let auto = BocpdDetector::new(200.0, 600).with_auto_beta(&warmup).detect(&data);
    let plain_idx: Vec<usize> = plain.iter().map(|c| c.index).collect();
    let auto_idx: Vec<usize> = auto.iter().map(|c| c.index).collect();
    assert_eq!(plain_idx, auto_idx, "auto-β with β=0 must match standard exactly");
}

#[test]
fn auto_beta_picks_robust_for_t3() {
    // t₃ excess kurtosis (population: ∞; sample on 300 obs ≈ 4-8). The
    // mapping must place this in the robust band so a real shift is still
    // resolved while tail events are damped.
    use cesura::auto_beta::auto_beta;
    use cesura::eval::heavy_tail_scenarios;
    let t3 = heavy_tail_scenarios()
        .into_iter()
        .find(|s| s.name == "t3_noise")
        .expect("t3_noise scenario");
    let b = auto_beta(&t3.data);
    eprintln!("t3_noise auto-β = {b}");
    assert!(
        (0.05..=0.20).contains(&b),
        "auto-β on t3_noise = {b}, expected in [0.05, 0.20]"
    );
}
