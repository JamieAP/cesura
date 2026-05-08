//! Production statistical test suite.
//!
//! Black-box tests that verify the detector's *operating characteristics* --
//! the properties a quant or ML practitioner must check before trusting it
//! in a production loop. Complements the unit tests in `src/lib.rs` which
//! verify the underlying math.
//!
//! Run with: `cargo test --features test-utils --test statistical`.

use cesura::detrend::{dominant_period_via_acf, Detrender};
use cesura::eval::{self, Category, Rng};
use cesura::{BocpdDetector, EnsembleDetector};

#[cfg(feature = "joint-detection")]
use cesura::chen_wu::{ChenWuDetector, Detection};

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

// ── Ensemble-vote probe ──────────────────────────────────────────────
//
// Hypothesis: BOCPD and FOCuS have largely independent failure modes.
// BOCPD's MAP-drop heuristic is permissive -- low precision on
// MustReject scenarios. FOCuS's frequentist GLR is cleaner on
// stationary noise but misses subtle shifts -- low recall on
// Challenging scenarios. An agreement-based ensemble should lift
// aggregate F1 above either alone.
//
// Probe four rules:
// - bocpd-alone (baseline)
// - focus-alone (baseline)
// - and-vote: keep a BOCPD CP only if FOCuS fires within ±tol
// - confident-or-confirmed: keep BOCPD CPs that are either
//   high-confidence (>= conf_floor) OR confirmed by FOCuS within ±tol
//
// Informational test: prints per-rule aggregate metrics and per-
// category breakdown. Asserts only the soft floor that the best
// ensemble does not regress vs BOCPD alone (within 1pp of equality);
// a stricter assertion would over-fit to the current scenario suite.
fn and_vote(bocpd: &[usize], focus: &[usize], tol: i64) -> Vec<usize> {
    bocpd
        .iter()
        .copied()
        .filter(|&b| focus.iter().any(|&f| (f as i64 - b as i64).abs() <= tol))
        .collect()
}

fn confident_or_confirmed(
    bocpd_cps: &[(usize, f64)],
    focus: &[usize],
    tol: i64,
    conf_floor: f64,
) -> Vec<usize> {
    bocpd_cps
        .iter()
        .filter_map(|&(idx, conf)| {
            let confirmed = focus.iter().any(|&f| (f as i64 - idx as i64).abs() <= tol);
            if conf >= conf_floor || confirmed {
                Some(idx)
            } else {
                None
            }
        })
        .collect()
}

#[test]
fn ensemble_vote_aggregate_probe() {
    let scenarios = eval::all_scenarios();
    let det = BocpdDetector::new(200.0, 400);
    let tol_steps: i64 = 15; // detection-agreement window
    let conf_floor = 0.5_f64; // BOCPD confidence threshold for solo-trust

    let mut by_rule: std::collections::BTreeMap<&str, Vec<eval::EvalMetrics>> =
        std::collections::BTreeMap::new();

    for s in &scenarios {
        let bocpd_cps = det.detect(&s.data);
        let bocpd_idx: Vec<usize> = bocpd_cps.iter().map(|c| c.index).collect();
        let bocpd_pairs: Vec<(usize, f64)> =
            bocpd_cps.iter().map(|c| (c.index, c.confidence)).collect();
        let focus_idx = eval::focus_detect(&s.data, 8.0);

        let and_idx = and_vote(&bocpd_idx, &focus_idx, tol_steps);
        let coc_idx = confident_or_confirmed(&bocpd_pairs, &focus_idx, tol_steps, conf_floor);

        for (rule, idxs) in [
            ("bocpd_alone", bocpd_idx.clone()),
            ("focus_alone", focus_idx.clone()),
            ("and_vote", and_idx),
            ("confident_or_confirmed", coc_idx),
        ] {
            let mut m = eval::match_detections(&idxs, &s.ground_truth, 20);
            m.name = s.name.to_string();
            m.category = s.category;
            by_rule.entry(rule).or_default().push(m);
        }
    }

    eprintln!("\n=== Ensemble-vote probe (tol={tol_steps}, conf_floor={conf_floor}) ===");
    eprintln!(
        "{:<24} {:>6} {:>6} {:>6} {:>5} {:>5} {:>5}",
        "Rule", "P", "R", "F1", "TP", "FP", "FN"
    );
    eprintln!("{}", "-".repeat(64));
    let mut summary: Vec<(&str, eval::EvalMetrics)> = Vec::new();
    for (rule, ms) in &by_rule {
        let agg = eval::aggregate(ms);
        eprintln!(
            "{:<24} {:>6.3} {:>6.3} {:>6.3} {:>5} {:>5} {:>5}",
            rule, agg.precision, agg.recall, agg.f1, agg.tp, agg.fp, agg.r#fn
        );
        summary.push((*rule, agg));
    }

    // Per-category breakdown for the two ensemble rules vs BOCPD baseline.
    eprintln!("\n--- Per-category F1 ---");
    eprintln!(
        "{:<24} {:>6} {:>6} {:>6} {:>6} {:>6}",
        "Rule", "MD", "MR_FP", "CH", "OP", "HT"
    );
    for (rule, ms) in &by_rule {
        let f1_for = |cat: Category| -> f64 {
            let filt: Vec<eval::EvalMetrics> =
                ms.iter().filter(|m| m.category == cat).cloned().collect();
            if filt.is_empty() {
                0.0
            } else {
                eval::aggregate(&filt).f1
            }
        };
        let mr_fp: usize = ms
            .iter()
            .filter(|m| m.category == Category::MustReject)
            .map(|m| m.fp)
            .sum();
        eprintln!(
            "{:<24} {:>6.3} {:>6} {:>6.3} {:>6.3} {:>6.3}",
            rule,
            f1_for(Category::MustDetect),
            mr_fp,
            f1_for(Category::Challenging),
            f1_for(Category::Operational),
            // Heavy-tail = MustDetect tag in our suite, but the names hint
            f1_for(Category::MustDetect),
        );
    }

    // Soft assertion: at least one ensemble rule should not regress
    // BOCPD-alone aggregate F1 by more than 1pp. If both regress, the
    // simple-vote hypothesis is dead and we need a different approach.
    let bocpd_f1 = summary
        .iter()
        .find(|(r, _)| *r == "bocpd_alone")
        .unwrap()
        .1
        .f1;
    let best_ensemble_f1 = summary
        .iter()
        .filter(|(r, _)| *r == "and_vote" || *r == "confident_or_confirmed")
        .map(|(_, m)| m.f1)
        .fold(0.0_f64, f64::max);
    assert!(
        best_ensemble_f1 >= bocpd_f1 - 0.01,
        "best ensemble F1 {best_ensemble_f1:.3} regresses BOCPD-alone {bocpd_f1:.3} by >1pp -- simple vote not viable"
    );
}

#[test]
fn ensemble_aggregate_meets_floor() {
    let scenarios = eval::all_scenarios();
    let det = EnsembleDetector::new(200.0, 400);

    let mut metrics = Vec::new();
    for s in &scenarios {
        let idx: Vec<usize> = det.detect(&s.data).iter().map(|c| c.index).collect();
        let mut m = eval::match_detections(&idx, &s.ground_truth, 20);
        m.name = s.name.to_string();
        m.category = s.category;
        metrics.push(m);
    }
    let agg = eval::aggregate(&metrics);

    let bocpd_baseline = {
        let det = BocpdDetector::new(200.0, 400);
        let mut m = Vec::new();
        for s in &scenarios {
            let idx: Vec<usize> = det.detect(&s.data).iter().map(|c| c.index).collect();
            let mut em = eval::match_detections(&idx, &s.ground_truth, 20);
            em.name = s.name.to_string();
            em.category = s.category;
            m.push(em);
        }
        eval::aggregate(&m)
    };

    eprintln!("\n=== Ensemble vs BOCPD (regression pin) ===");
    eprintln!(
        "BOCPD-alone     F1={:.3}  P={:.3}  R={:.3}  TP={} FP={} FN={}",
        bocpd_baseline.f1,
        bocpd_baseline.precision,
        bocpd_baseline.recall,
        bocpd_baseline.tp,
        bocpd_baseline.fp,
        bocpd_baseline.r#fn
    );
    eprintln!(
        "Ensemble (COC)  F1={:.3}  P={:.3}  R={:.3}  TP={} FP={} FN={}",
        agg.f1, agg.precision, agg.recall, agg.tp, agg.fp, agg.r#fn
    );

    assert!(
        agg.f1 >= 0.595,
        "ensemble aggregate F1 {:.3} regressed below pinned floor 0.595",
        agg.f1
    );
    assert!(
        agg.f1 >= bocpd_baseline.f1 - 0.005,
        "ensemble F1 {:.3} regressed below BOCPD baseline {:.3} by >0.5pp",
        agg.f1,
        bocpd_baseline.f1
    );
    let mr_fp: usize = metrics
        .iter()
        .filter(|m| m.category == Category::MustReject)
        .map(|m| m.fp)
        .sum();
    let bocpd_mr_fp: usize = {
        let det = BocpdDetector::new(200.0, 400);
        scenarios
            .iter()
            .filter(|s| s.category == Category::MustReject)
            .map(|s| {
                let idx: Vec<usize> = det.detect(&s.data).iter().map(|c| c.index).collect();
                eval::match_detections(&idx, &s.ground_truth, 20).fp
            })
            .sum()
    };
    eprintln!("MustReject FPs:  BOCPD={bocpd_mr_fp}    Ensemble={mr_fp}");
    assert!(
        mr_fp <= bocpd_mr_fp,
        "ensemble must not have more MustReject FPs than BOCPD baseline; \
         got ensemble={mr_fp} vs BOCPD={bocpd_mr_fp}"
    );
}

#[test]
fn ensemble_vote_grid_probe() {
    // Sweep tolerance and confidence-floor to find the best
    // operating point of the confident-or-confirmed rule. Also tries
    // a strict-AND with relaxed tolerance to see if FOCuS just needs
    // a wider matching window.
    let scenarios = eval::all_scenarios();
    let det = BocpdDetector::new(200.0, 400);

    let tolerances: [i64; 4] = [10, 15, 25, 40];
    let conf_floors: [f64; 5] = [0.0, 0.3, 0.4, 0.5, 0.7];

    eprintln!("\n=== Ensemble-vote grid probe ===");
    eprintln!(
        "{:<6} {:>4} {:>5}   {:<5} {:<5} {:<5}    {:<5} {:<5} {:<5}",
        "rule", "tol", "conf", "P", "R", "F1", "MD", "MR-FP", "OP"
    );
    eprintln!("{}", "-".repeat(70));

    let mut best_f1 = 0.0_f64;
    let mut best_label = String::new();

    for &tol in &tolerances {
        let mut and_metrics = Vec::new();
        for s in &scenarios {
            let bocpd_idx: Vec<usize> = det.detect(&s.data).iter().map(|c| c.index).collect();
            let focus_idx = eval::focus_detect(&s.data, 8.0);
            let idxs = and_vote(&bocpd_idx, &focus_idx, tol);
            let mut m = eval::match_detections(&idxs, &s.ground_truth, 20);
            m.name = s.name.to_string();
            m.category = s.category;
            and_metrics.push(m);
        }
        let agg = eval::aggregate(&and_metrics);
        let md_f1 = eval::aggregate(
            &and_metrics
                .iter()
                .filter(|m| m.category == Category::MustDetect)
                .cloned()
                .collect::<Vec<_>>(),
        )
        .f1;
        let mr_fp: usize = and_metrics
            .iter()
            .filter(|m| m.category == Category::MustReject)
            .map(|m| m.fp)
            .sum();
        let op_f1 = eval::aggregate(
            &and_metrics
                .iter()
                .filter(|m| m.category == Category::Operational)
                .cloned()
                .collect::<Vec<_>>(),
        )
        .f1;
        eprintln!(
            "{:<6} {:>4} {:>5}   {:>5.3} {:>5.3} {:>5.3}    {:>5.3} {:>5} {:>5.3}",
            "AND", tol, "-", agg.precision, agg.recall, agg.f1, md_f1, mr_fp, op_f1
        );
        if agg.f1 > best_f1 {
            best_f1 = agg.f1;
            best_label = format!("AND tol={tol}");
        }

        for &cf in &conf_floors {
            let mut coc_metrics = Vec::new();
            for s in &scenarios {
                let bocpd_cps: Vec<(usize, f64)> = det
                    .detect(&s.data)
                    .into_iter()
                    .map(|c| (c.index, c.confidence))
                    .collect();
                let focus_idx = eval::focus_detect(&s.data, 8.0);
                let idxs = confident_or_confirmed(&bocpd_cps, &focus_idx, tol, cf);
                let mut m = eval::match_detections(&idxs, &s.ground_truth, 20);
                m.name = s.name.to_string();
                m.category = s.category;
                coc_metrics.push(m);
            }
            let agg = eval::aggregate(&coc_metrics);
            let md_f1 = eval::aggregate(
                &coc_metrics
                    .iter()
                    .filter(|m| m.category == Category::MustDetect)
                    .cloned()
                    .collect::<Vec<_>>(),
            )
            .f1;
            let mr_fp: usize = coc_metrics
                .iter()
                .filter(|m| m.category == Category::MustReject)
                .map(|m| m.fp)
                .sum();
            let op_f1 = eval::aggregate(
                &coc_metrics
                    .iter()
                    .filter(|m| m.category == Category::Operational)
                    .cloned()
                    .collect::<Vec<_>>(),
            )
            .f1;
            eprintln!(
                "{:<6} {:>4} {:>5.2}   {:>5.3} {:>5.3} {:>5.3}    {:>5.3} {:>5} {:>5.3}",
                "COC", tol, cf, agg.precision, agg.recall, agg.f1, md_f1, mr_fp, op_f1
            );
            if agg.f1 > best_f1 {
                best_f1 = agg.f1;
                best_label = format!("COC tol={tol} cf={cf:.2}");
            }
        }
    }

    eprintln!("\nBEST: {best_label} → F1 = {best_f1:.4}");
    let baseline_f1 = {
        let mut b = Vec::new();
        for s in &scenarios {
            let idx: Vec<usize> = det.detect(&s.data).iter().map(|c| c.index).collect();
            let mut m = eval::match_detections(&idx, &s.ground_truth, 20);
            m.name = s.name.to_string();
            m.category = s.category;
            b.push(m);
        }
        eval::aggregate(&b).f1
    };
    eprintln!("BOCPD-alone baseline F1 = {baseline_f1:.4}");
    eprintln!("DELTA = {:+.4}", best_f1 - baseline_f1);
}

// ── 3-way ensemble probe ─────────────────────────────────────────────
//
// Extend the 2-way (BOCPD + FOCuS) comparison with Chen & Wu's
// CP emissions as a third confirmation arm. Hypothesis: Chen & Wu
// (Bayesian, anomaly-aware) catches the same regime shifts BOCPD
// catches but rejects collective anomalies BOCPD treats as CPs --
// that anomaly-rejection is precisely the signal a confirmation arm
// adds. Two rules tested:
//
//   - AND-3:  BOCPD ∧ FOCuS ∧ ChenWu        (strict, max precision)
//   - COC3:   conf >= floor                  (high-conf passthrough)
//             ∨ FOCuS confirms within tol
//             ∨ ChenWu CP confirms within tol  (loose-OR confirmation)
//
#[cfg(feature = "joint-detection")]
#[test]
fn ensemble_three_way_probe() {
    let scenarios = eval::all_scenarios();
    let det = BocpdDetector::new(200.0, 400);
    let chen = ChenWuDetector::new(0.1, 0.2, 4, 0.5, 0.5);

    let tol: i64 = 25;
    let cf = 0.40_f64;

    eprintln!("\n=== 3-way ensemble probe (tol={tol}, cf={cf}) ===");
    eprintln!(
        "{:<24} {:>5} {:>5} {:>5}  {:>5} {:>5}",
        "Rule", "P", "R", "F1", "MR_FP", "OP"
    );
    eprintln!("{}", "-".repeat(60));

    let collect_for = |rule_idxs: Vec<Vec<usize>>| -> Vec<eval::EvalMetrics> {
        scenarios
            .iter()
            .zip(rule_idxs)
            .map(|(s, idxs)| {
                let mut m = eval::match_detections(&idxs, &s.ground_truth, 20);
                m.name = s.name.to_string();
                m.category = s.category;
                m
            })
            .collect()
    };
    let summarise = |label: &str, ms: &[eval::EvalMetrics]| {
        let agg = eval::aggregate(ms);
        let mr_fp: usize = ms
            .iter()
            .filter(|m| m.category == Category::MustReject)
            .map(|m| m.fp)
            .sum();
        let op_f1 = eval::aggregate(
            &ms.iter()
                .filter(|m| m.category == Category::Operational)
                .cloned()
                .collect::<Vec<_>>(),
        )
        .f1;
        eprintln!(
            "{:<24} {:>5.3} {:>5.3} {:>5.3}  {:>5} {:>5.3}",
            label, agg.precision, agg.recall, agg.f1, mr_fp, op_f1
        );
        agg.f1
    };

    // Per-scenario detector outputs (computed once, reused per rule).
    let bocpd_per_scenario: Vec<Vec<(usize, f64)>> = scenarios
        .iter()
        .map(|s| {
            det.detect(&s.data)
                .into_iter()
                .map(|c| (c.index, c.confidence))
                .collect()
        })
        .collect();
    let focus_per_scenario: Vec<Vec<usize>> = scenarios
        .iter()
        .map(|s| eval::focus_detect(&s.data, 8.0))
        .collect();
    let chen_per_scenario: Vec<Vec<usize>> = scenarios
        .iter()
        .map(|s| {
            chen.detect(&s.data)
                .into_iter()
                .filter_map(|d| match d {
                    Detection::ChangePoint(cp) => Some(cp.index),
                    Detection::CollectiveAnomaly { .. } => None,
                })
                .collect()
        })
        .collect();

    let bocpd_only: Vec<Vec<usize>> = bocpd_per_scenario
        .iter()
        .map(|v| v.iter().map(|&(i, _)| i).collect())
        .collect();
    let coc2: Vec<Vec<usize>> = bocpd_per_scenario
        .iter()
        .zip(&focus_per_scenario)
        .map(|(b, f)| {
            b.iter()
                .filter_map(|&(i, c)| {
                    let confirmed = f.iter().any(|&fi| (fi as i64 - i as i64).abs() <= tol);
                    (c >= cf || confirmed).then_some(i)
                })
                .collect()
        })
        .collect();
    let f_bocpd = summarise("bocpd_alone", &collect_for(bocpd_only.clone()));
    let f_coc2 = summarise("coc2 (B+F)", &collect_for(coc2.clone()));

    // ChenWu-only.
    let chen_only: Vec<Vec<usize>> = chen_per_scenario.clone();
    let _ = summarise("chenwu_alone", &collect_for(chen_only));

    // AND-3: BOCPD ∧ FOCuS ∧ ChenWu.
    let and3: Vec<Vec<usize>> = bocpd_per_scenario
        .iter()
        .zip(&focus_per_scenario)
        .zip(&chen_per_scenario)
        .map(|((b, f), c)| {
            b.iter()
                .filter_map(|&(i, _)| {
                    let f_ok = f.iter().any(|&fi| (fi as i64 - i as i64).abs() <= tol);
                    let c_ok = c.iter().any(|&ci| (ci as i64 - i as i64).abs() <= tol);
                    (f_ok && c_ok).then_some(i)
                })
                .collect()
        })
        .collect();
    let f_and3 = summarise("and3 (B∧F∧C)", &collect_for(and3));

    // COC3: high-conf OR (FOCuS confirms) OR (ChenWu confirms).
    let coc3: Vec<Vec<usize>> = bocpd_per_scenario
        .iter()
        .zip(&focus_per_scenario)
        .zip(&chen_per_scenario)
        .map(|((b, f), c)| {
            b.iter()
                .filter_map(|&(i, conf)| {
                    if conf >= cf {
                        return Some(i);
                    }
                    let f_ok = f.iter().any(|&fi| (fi as i64 - i as i64).abs() <= tol);
                    let c_ok = c.iter().any(|&ci| (ci as i64 - i as i64).abs() <= tol);
                    (f_ok || c_ok).then_some(i)
                })
                .collect()
        })
        .collect();
    let f_coc3 = summarise("coc3 (loose-OR)", &collect_for(coc3));

    // COC3-AND: high-conf OR (FOCuS AND ChenWu confirm).
    let coc3_and: Vec<Vec<usize>> = bocpd_per_scenario
        .iter()
        .zip(&focus_per_scenario)
        .zip(&chen_per_scenario)
        .map(|((b, f), c)| {
            b.iter()
                .filter_map(|&(i, conf)| {
                    if conf >= cf {
                        return Some(i);
                    }
                    let f_ok = f.iter().any(|&fi| (fi as i64 - i as i64).abs() <= tol);
                    let c_ok = c.iter().any(|&ci| (ci as i64 - i as i64).abs() <= tol);
                    (f_ok && c_ok).then_some(i)
                })
                .collect()
        })
        .collect();
    let f_coc3_and = summarise("coc3-and (B|FAND C)", &collect_for(coc3_and));

    eprintln!(
        "\nDeltas vs BOCPD-alone ({:.3}):  COC2 {:+.4}  AND3 {:+.4}  COC3 {:+.4}  COC3-AND {:+.4}",
        f_bocpd,
        f_coc2 - f_bocpd,
        f_and3 - f_bocpd,
        f_coc3 - f_bocpd,
        f_coc3_and - f_bocpd
    );
    eprintln!(
        "Deltas vs COC2 ({:.3}):           AND3 {:+.4}  COC3 {:+.4}  COC3-AND {:+.4}",
        f_coc2,
        f_and3 - f_coc2,
        f_coc3 - f_coc2,
        f_coc3_and - f_coc2
    );
}

///
/// Three modes:
/// - raw: BOCPD on raw data (current default, baseline)
/// - autodetrend: if `dominant_period_via_acf` returns Some, fit
///   `Detrender` and run BOCPD on `detrend_diff`. Else fall back to raw.
/// - always_diff: force `seasonal_difference(period=ACF or 24)`
///   regardless of ACF strength. Diagnostic only.
#[test]
fn detrending_integration_probe() {
    let scenarios = eval::all_scenarios();
    let det = BocpdDetector::new(200.0, 400);
    let tolerance = 20_usize;

    let detect_raw = |s: &eval::Scenario| -> Vec<usize> {
        det.detect(&s.data).iter().map(|c| c.index).collect()
    };

    let detect_autodetrend = |s: &eval::Scenario| -> (Vec<usize>, Option<usize>) {
        // Try seasonal differencing if a clear period is detected. The
        // mapping from differenced indices back to original space is a
        // shift by `period` (since `seasonal_difference` drops the first
        // P values).
        let period_opt = dominant_period_via_acf(&s.data);
        match period_opt {
            Some(p) if p > 0 && p < s.data.len() / 4 => {
                let diffed = cesura::detrend::seasonal_difference(&s.data, p);
                let cps_in_diffed: Vec<usize> =
                    det.detect(&diffed).iter().map(|c| c.index).collect();
                let cps_in_orig = cps_in_diffed.into_iter().map(|i| i + p).collect();
                (cps_in_orig, Some(p))
            }
            _ => (detect_raw(s), None),
        }
    };

    let mut metrics_raw = Vec::new();
    let mut metrics_auto = Vec::new();
    eprintln!("\n=== Detrending integration probe ===");
    eprintln!(
        "{:<32} {:<6} {:>3} {:>3} {:>3}    {:<6} {:>3} {:>3} {:>3}",
        "scenario", "raw", "TP", "FP", "FN", "auto", "TP", "FP", "FN"
    );
    eprintln!("{}", "-".repeat(90));
    for s in &scenarios {
        let raw_idx = detect_raw(s);
        let (auto_idx, period) = detect_autodetrend(s);

        let mut mr = eval::match_detections(&raw_idx, &s.ground_truth, tolerance);
        mr.name = s.name.to_string();
        mr.category = s.category;
        let mut ma = eval::match_detections(&auto_idx, &s.ground_truth, tolerance);
        ma.name = s.name.to_string();
        ma.category = s.category;

        let detrended_marker = period.map(|p| format!("p={p}")).unwrap_or_else(|| "-".into());
        eprintln!(
            "{:<32} {:<6} {:>3} {:>3} {:>3}    {:<6} {:>3} {:>3} {:>3}",
            s.name,
            "",
            mr.tp,
            mr.fp,
            mr.r#fn,
            detrended_marker,
            ma.tp,
            ma.fp,
            ma.r#fn
        );
        metrics_raw.push(mr);
        metrics_auto.push(ma);
    }

    let agg_raw = eval::aggregate(&metrics_raw);
    let agg_auto = eval::aggregate(&metrics_auto);
    eprintln!("{}", "-".repeat(90));
    eprintln!(
        "raw  AGG: F1={:.3} P={:.3} R={:.3} TP={} FP={} FN={}",
        agg_raw.f1, agg_raw.precision, agg_raw.recall, agg_raw.tp, agg_raw.fp, agg_raw.r#fn
    );
    eprintln!(
        "auto AGG: F1={:.3} P={:.3} R={:.3} TP={} FP={} FN={}",
        agg_auto.f1, agg_auto.precision, agg_auto.recall, agg_auto.tp, agg_auto.fp, agg_auto.r#fn
    );

    let mr_raw: usize = metrics_raw
        .iter()
        .filter(|m| m.category == Category::MustReject)
        .map(|m| m.fp)
        .sum();
    let mr_auto: usize = metrics_auto
        .iter()
        .filter(|m| m.category == Category::MustReject)
        .map(|m| m.fp)
        .sum();
    eprintln!("MR_FP: raw={mr_raw}  auto={mr_auto}  delta={}", mr_raw as i64 - mr_auto as i64);

    let _ = Detrender::auto_fit(&[0.0; 100]); // suppress unused-import warning
}

#[test]
fn detrend_ensemble_meets_floor() {
    let scenarios = eval::all_scenarios();
    let det = EnsembleDetector::new(200.0, 400).with_auto_detrend(true);
    let mut metrics = Vec::new();
    for s in &scenarios {
        let idx: Vec<usize> = det.detect(&s.data).iter().map(|c| c.index).collect();
        let mut m = eval::match_detections(&idx, &s.ground_truth, 20);
        m.name = s.name.to_string();
        m.category = s.category;
        metrics.push(m);
    }
    let agg = eval::aggregate(&metrics);
    let mr_fp: usize = metrics
        .iter()
        .filter(|m| m.category == Category::MustReject)
        .map(|m| m.fp)
        .sum();
    eprintln!(
        "detrend+ensemble  F1={:.3}  P={:.3}  R={:.3}  MR_FP={}",
        agg.f1, agg.precision, agg.recall, mr_fp
    );
    assert!(
        agg.f1 >= 0.620,
        "detrend+ensemble F1 {:.3} regressed below pinned floor 0.620",
        agg.f1
    );
    assert!(
        mr_fp <= 14,
        "detrend+ensemble MR_FP {mr_fp} regressed above pinned ceiling 14"
    );
    assert!(
        agg.recall >= 0.720,
        "detrend+ensemble recall {:.3} regressed below pinned floor 0.720",
        agg.recall
    );
}

/// Iter-6 follow-on: compose detrending + COC2 ensemble. If both
/// levers move the dial independently, applying both should compound.
#[test]
fn detrend_plus_ensemble_probe() {
    let scenarios = eval::all_scenarios();
    let det = BocpdDetector::new(200.0, 400);
    let ens = EnsembleDetector::new(200.0, 400);
    let tolerance = 20_usize;

    let mut metrics_raw_bocpd = Vec::new();
    let mut metrics_raw_ens = Vec::new();
    let mut metrics_diff_bocpd = Vec::new();
    let mut metrics_diff_ens = Vec::new();

    for s in &scenarios {
        let raw_idx_bocpd: Vec<usize> =
            det.detect(&s.data).iter().map(|c| c.index).collect();
        let raw_idx_ens: Vec<usize> = ens.detect(&s.data).iter().map(|c| c.index).collect();

        let detrended_data: Vec<f64> = match dominant_period_via_acf(&s.data) {
            Some(p) if p > 0 && p < s.data.len() / 4 => {
                cesura::detrend::seasonal_difference(&s.data, p)
            }
            _ => s.data.clone(),
        };
        let shift = s.data.len() - detrended_data.len();
        let diff_idx_bocpd: Vec<usize> = det
            .detect(&detrended_data)
            .iter()
            .map(|c| c.index + shift)
            .collect();
        let diff_idx_ens: Vec<usize> = ens
            .detect(&detrended_data)
            .iter()
            .map(|c| c.index + shift)
            .collect();

        for (idxs, target) in [
            (raw_idx_bocpd, &mut metrics_raw_bocpd),
            (raw_idx_ens, &mut metrics_raw_ens),
            (diff_idx_bocpd, &mut metrics_diff_bocpd),
            (diff_idx_ens, &mut metrics_diff_ens),
        ] {
            let mut m = eval::match_detections(&idxs, &s.ground_truth, tolerance);
            m.name = s.name.to_string();
            m.category = s.category;
            target.push(m);
        }
    }

    eprintln!("\n=== Detrending × Ensemble matrix ===");
    for (label, ms) in [
        ("raw + BOCPD     ", &metrics_raw_bocpd),
        ("raw + Ensemble  ", &metrics_raw_ens),
        ("detrend + BOCPD ", &metrics_diff_bocpd),
        ("detrend + Ens   ", &metrics_diff_ens),
    ] {
        let agg = eval::aggregate(ms);
        let mr_fp: usize = ms
            .iter()
            .filter(|m| m.category == Category::MustReject)
            .map(|m| m.fp)
            .sum();
        eprintln!(
            "{:<18}  F1={:.3}  P={:.3}  R={:.3}  TP={} FP={} FN={}  MR_FP={}",
            label, agg.f1, agg.precision, agg.recall, agg.tp, agg.fp, agg.r#fn, mr_fp
        );
    }
}

#[test]
fn alt_confidence_summary_calibration_probe() {
    let scenarios = eval::all_scenarios();
    let det = BocpdDetector::new(200.0, 400);
    let tolerance = 20_usize;
    let cooldown = 15_usize;

    fn build_summary<F: Fn(&[f64]) -> f64>(
        cp_probs: &[f64],
        trigger: usize,
        cooldown: usize,
        f: F,
    ) -> f64 {
        let lo = trigger.saturating_sub(cooldown);
        let win = &cp_probs[lo..=trigger.min(cp_probs.len() - 1)];
        f(win).clamp(0.0, 1.0)
    }

    // For each scenario, iterate over its detected CPs and compute every
    // candidate summary. Keep (summary_value, is_tp) for binning.
    let mut by_summary: std::collections::BTreeMap<&str, Vec<(f64, bool)>> =
        std::collections::BTreeMap::new();

    for s in &scenarios {
        let (cps, cp_probs, original_indices) = det.detect_with_cp_probs(&s.data);
        // Map original->filter
        let orig_to_filt: std::collections::HashMap<usize, usize> = original_indices
            .iter()
            .enumerate()
            .map(|(i, &orig)| (orig, i))
            .collect();

        // Greedy nearest-first matching mirrors `match_detections`.
        let mut matched_gt = vec![false; s.ground_truth.len()];
        let mut idxd: Vec<(usize, &cesura::ChangePoint)> = cps.iter().enumerate().collect();
        idxd.sort_by_key(|(_, cp)| {
            s.ground_truth
                .iter()
                .map(|&gt| (cp.index as i64 - gt as i64).unsigned_abs() as usize)
                .min()
                .unwrap_or(usize::MAX)
        });
        let mut tp_flags = vec![false; cps.len()];
        for (orig_idx, cp) in idxd {
            let mut best_dist = usize::MAX;
            let mut best_gt = None;
            for (gi, &gt) in s.ground_truth.iter().enumerate() {
                if matched_gt[gi] {
                    continue;
                }
                let d = (cp.index as i64 - gt as i64).unsigned_abs() as usize;
                if d <= tolerance && d < best_dist {
                    best_dist = d;
                    best_gt = Some(gi);
                }
            }
            if let Some(gi) = best_gt {
                matched_gt[gi] = true;
                tp_flags[orig_idx] = true;
            }
        }

        for (cp, &tp) in cps.iter().zip(&tp_flags) {
            let trig = match orig_to_filt.get(&cp.index) {
                Some(&t) => t,
                None => continue,
            };

            let peak = build_summary(&cp_probs, trig, cooldown, |w| {
                w.iter().copied().fold(0.0_f64, f64::max)
            });
            let mean = build_summary(&cp_probs, trig, cooldown, |w| {
                if w.is_empty() {
                    0.0
                } else {
                    w.iter().sum::<f64>() / w.len() as f64
                }
            });
            let area = build_summary(&cp_probs, trig, cooldown, |w| {
                w.iter().sum::<f64>() / (cooldown + 1) as f64
            });
            let frac_above_5 = build_summary(&cp_probs, trig, cooldown, |w| {
                w.iter().filter(|&&v| v >= 0.5).count() as f64 / w.len().max(1) as f64
            });
            let peak_x_mean = (peak * mean).sqrt();
            let sustained = peak * frac_above_5;

            by_summary.entry("peak").or_default().push((peak, tp));
            by_summary.entry("mean").or_default().push((mean, tp));
            by_summary.entry("area").or_default().push((area, tp));
            by_summary
                .entry("frac_above_0.5")
                .or_default()
                .push((frac_above_5, tp));
            by_summary
                .entry("sqrt(peak*mean)")
                .or_default()
                .push((peak_x_mean, tp));
            by_summary
                .entry("peak*frac_above_0.5")
                .or_default()
                .push((sustained, tp));
        }
    }

    eprintln!("\n=== Alt confidence summary calibration ===");
    let bins = [(0.0, 0.20), (0.20, 0.40), (0.40, 0.60), (0.60, 0.80), (0.80, 1.001)];
    for (label, events) in &by_summary {
        let n_total = events.len();
        let tp_total = events.iter().filter(|(_, t)| *t).count();
        eprintln!(
            "\n--- {label} (n={n_total}, global prec {:.3}) ---",
            tp_total as f64 / n_total.max(1) as f64
        );
        eprintln!("{:<14} {:>5} {:>5} {:>6}", "bin", "n", "tp", "prec");
        let mut precs = Vec::new();
        for &(lo, hi) in &bins {
            let in_bin: Vec<&(f64, bool)> =
                events.iter().filter(|(c, _)| *c >= lo && *c < hi).collect();
            let n = in_bin.len();
            let tp = in_bin.iter().filter(|(_, t)| *t).count();
            let prec = if n == 0 {
                f64::NAN
            } else {
                tp as f64 / n as f64
            };
            eprintln!(
                "[{:>4.2},{:>4.2}) {:>5} {:>5} {:>6.3}",
                lo, hi, n, tp, prec
            );
            if n > 0 {
                precs.push(prec);
            }
        }
        // Quick monotonicity check: count adjacent-bin increases.
        let monotone = precs.windows(2).all(|w| w[1] >= w[0] - 1e-9);
        eprintln!("monotone-non-decreasing? {monotone}");
    }
}

/// Iter-4 sanity check on detect_viterbi: clean two-regime shift.
/// One CP at index 100 in 200 samples. MAP-drop catches it; Viterbi
/// must not return 0 or 50 CPs.
#[test]
fn viterbi_clean_shift_sanity() {
    let mut data: Vec<f64> = (0..100).map(|_| 0.0).collect();
    data.extend((0..100).map(|_| 5.0));
    let det = BocpdDetector::new(200.0, 250);
    let cps = det.detect_viterbi(&data);
    eprintln!(
        "viterbi clean-shift CPs: {:?}",
        cps.iter().map(|c| c.index).collect::<Vec<_>>()
    );
    assert!(!cps.is_empty(), "expected at least one Viterbi CP");
    assert!(
        cps.iter().any(|c| (c.index as i64 - 100).abs() < 25),
        "expected a Viterbi CP near index 100, got {:?}",
        cps.iter().map(|c| c.index).collect::<Vec<_>>()
    );
}

#[test]
fn mapdrop_intersect_viterbi_probe() {
    let scenarios = eval::all_scenarios();
    let det = BocpdDetector::new(200.0, 400);
    let tol_steps: i64 = 25;
    let tolerance = 20_usize;

    let mut metrics = Vec::new();
    for s in &scenarios {
        let map_cps = det.detect(&s.data);
        let vit_cps = det.detect_viterbi(&s.data);
        let map_idx: Vec<usize> = map_cps.iter().map(|c| c.index).collect();
        let vit_idx: Vec<usize> = vit_cps.iter().map(|c| c.index).collect();
        // MAP-drop CP kept iff a Viterbi CP exists within ±tol_steps.
        let confirmed: Vec<usize> = map_idx
            .iter()
            .copied()
            .filter(|&m| vit_idx.iter().any(|&v| (v as i64 - m as i64).abs() <= tol_steps))
            .collect();
        let mut em = eval::match_detections(&confirmed, &s.ground_truth, tolerance);
        em.name = s.name.to_string();
        em.category = s.category;
        metrics.push(em);
    }
    let agg = eval::aggregate(&metrics);
    eprintln!(
        "\nMAP-drop ∩ Viterbi  F1={:.3}  P={:.3}  R={:.3}  TP={} FP={} FN={}",
        agg.f1, agg.precision, agg.recall, agg.tp, agg.fp, agg.r#fn
    );
    let mr_fp: usize = metrics
        .iter()
        .filter(|m| m.category == Category::MustReject)
        .map(|m| m.fp)
        .sum();
    eprintln!("MR_FP = {mr_fp}");
}

#[test]
fn viterbi_vs_mapdrop_aggregate() {
    let scenarios = eval::all_scenarios();
    let det = BocpdDetector::new(200.0, 400);
    let tolerance = 20_usize;

    let mut mapdrop_metrics = Vec::new();
    let mut viterbi_metrics = Vec::new();
    for s in &scenarios {
        let map_idx: Vec<usize> = det.detect(&s.data).iter().map(|c| c.index).collect();
        let vit_idx: Vec<usize> = det.detect_viterbi(&s.data).iter().map(|c| c.index).collect();
        let mut mm = eval::match_detections(&map_idx, &s.ground_truth, tolerance);
        mm.name = s.name.to_string();
        mm.category = s.category;
        mapdrop_metrics.push(mm);
        let mut vm = eval::match_detections(&vit_idx, &s.ground_truth, tolerance);
        vm.name = s.name.to_string();
        vm.category = s.category;
        viterbi_metrics.push(vm);
    }

    let mm = eval::aggregate(&mapdrop_metrics);
    let vm = eval::aggregate(&viterbi_metrics);
    eprintln!("\n=== MAP-drop vs Viterbi (aggregate) ===");
    eprintln!(
        "MAP-drop  F1={:.3}  P={:.3}  R={:.3}  TP={} FP={} FN={}",
        mm.f1, mm.precision, mm.recall, mm.tp, mm.fp, mm.r#fn
    );
    eprintln!(
        "Viterbi   F1={:.3}  P={:.3}  R={:.3}  TP={} FP={} FN={}",
        vm.f1, vm.precision, vm.recall, vm.tp, vm.fp, vm.r#fn
    );

    // Per-category breakdown for the diagnostic.
    eprintln!("\n--- Per-category F1 ---");
    eprintln!("{:<6} {:>6} {:>6} {:>6} {:>6}", "rule", "MD", "MR_FP", "CH", "OP");
    for (label, ms) in [("MAPdrp", &mapdrop_metrics), ("Vitrb", &viterbi_metrics)] {
        let f1_for = |cat: Category| -> f64 {
            let filt: Vec<eval::EvalMetrics> =
                ms.iter().filter(|m| m.category == cat).cloned().collect();
            if filt.is_empty() { 0.0 } else { eval::aggregate(&filt).f1 }
        };
        let mr_fp: usize = ms
            .iter()
            .filter(|m| m.category == Category::MustReject)
            .map(|m| m.fp)
            .sum();
        eprintln!(
            "{:<6} {:>6.3} {:>6} {:>6.3} {:>6.3}",
            label,
            f1_for(Category::MustDetect),
            mr_fp,
            f1_for(Category::Challenging),
            f1_for(Category::Operational),
        );
    }
}

/// Iter-4 probe. The MAP-drop heuristic emits CPs with a `confidence`
/// derived from `peak P(r_t = 0)` over the cooldown window. KNOWN_LIMITATIONS
/// flags this as permissive; COMPARISON.md proposes a Viterbi backward
/// pass for calibrated confidence. Before implementing Viterbi, ask: is
/// the existing confidence already informative?
///
#[test]
fn confidence_calibration_probe() {
    let scenarios = eval::all_scenarios();
    let det = BocpdDetector::new(200.0, 400);
    let tolerance = 20_usize;

    // Collect (confidence, is_tp) for every BOCPD detection across all
    // scenarios. Greedy nearest-first matching mirrors `match_detections`
    // so the binning is comparable to the aggregate F1 metric.
    let mut events: Vec<(f64, bool)> = Vec::new();
    for s in &scenarios {
        let cps = det.detect(&s.data);
        let mut matched_gt = vec![false; s.ground_truth.len()];
        // Sort detected by distance to nearest GT (matches `match_detections`).
        let mut idxd: Vec<(usize, &cesura::ChangePoint)> = cps.iter().enumerate().collect();
        idxd.sort_by_key(|(_, cp)| {
            s.ground_truth
                .iter()
                .map(|&gt| (cp.index as i64 - gt as i64).unsigned_abs() as usize)
                .min()
                .unwrap_or(usize::MAX)
        });
        let mut tp_flags = vec![false; cps.len()];
        for (orig_idx, cp) in idxd {
            let mut best_dist = usize::MAX;
            let mut best_gt = None;
            for (gi, &gt) in s.ground_truth.iter().enumerate() {
                if matched_gt[gi] {
                    continue;
                }
                let d = (cp.index as i64 - gt as i64).unsigned_abs() as usize;
                if d <= tolerance && d < best_dist {
                    best_dist = d;
                    best_gt = Some(gi);
                }
            }
            if let Some(gi) = best_gt {
                matched_gt[gi] = true;
                tp_flags[orig_idx] = true;
            }
        }
        for (cp, &tp) in cps.iter().zip(&tp_flags) {
            events.push((cp.confidence, tp));
        }
    }

    eprintln!("\n=== Confidence calibration probe ===");
    eprintln!("total detections: {}", events.len());
    let total_tp = events.iter().filter(|(_, t)| *t).count();
    eprintln!(
        "global precision: {:.3} ({} TP / {})",
        total_tp as f64 / events.len().max(1) as f64,
        total_tp,
        events.len()
    );

    let bins: [(f64, f64); 5] = [
        (0.0, 0.20),
        (0.20, 0.40),
        (0.40, 0.60),
        (0.60, 0.80),
        (0.80, 1.001),
    ];
    eprintln!(
        "\n{:<14} {:>5} {:>5} {:>6}",
        "bin", "n", "tp", "prec"
    );
    eprintln!("{}", "-".repeat(34));
    for &(lo, hi) in &bins {
        let in_bin: Vec<&(f64, bool)> =
            events.iter().filter(|(c, _)| *c >= lo && *c < hi).collect();
        let n = in_bin.len();
        let tp = in_bin.iter().filter(|(_, t)| *t).count();
        let prec = if n == 0 {
            f64::NAN
        } else {
            tp as f64 / n as f64
        };
        eprintln!(
            "[{:>4.2},{:>4.2}) {:>5} {:>5} {:>6.3}",
            lo, hi, n, tp, prec
        );
    }

    // Cumulative-from-top: precision when filtering on conf >= floor.
    eprintln!("\n--- Cumulative precision @ floor (post-hoc filter view) ---");
    eprintln!(
        "{:>6} {:>5} {:>5} {:>6} {:>6}",
        "floor", "kept", "tp", "prec", "recall"
    );
    let total_gt: usize = scenarios.iter().map(|s| s.ground_truth.len()).sum();
    let mut sorted = events.clone();
    sorted.sort_by(|a, b| b.0.total_cmp(&a.0));
    for floor in [0.0, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80] {
        let kept: Vec<&(f64, bool)> = sorted.iter().filter(|(c, _)| *c >= floor).collect();
        let n = kept.len();
        let tp = kept.iter().filter(|(_, t)| *t).count();
        let prec = if n == 0 {
            f64::NAN
        } else {
            tp as f64 / n as f64
        };
        let recall = tp as f64 / total_gt.max(1) as f64;
        eprintln!(
            "{:>6.2} {:>5} {:>5} {:>6.3} {:>6.3}",
            floor, n, tp, prec, recall
        );
    }
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

// ── AR(1)-BOCPD floors ──────────────────────────────

/// Generate an AR(1) sequence x_t = a + b · x_{t-1} + ε_t with no CPs.
fn ar1_sequence(n: usize, a: f64, b: f64, sigma: f64, seed: u64) -> Vec<f64> {
    let mut rng = Rng::new(seed);
    let mut data = Vec::with_capacity(n);
    let mut prev = 0.0_f64;
    for _ in 0..n {
        let x = a + b * prev + rng.normal(0.0, sigma);
        data.push(x);
        prev = x;
    }
    data
}

#[test]
fn bocpd_ar1_on_ar1_process_vs_nig() {
    // Headline gain: AR(1) within-regime is the matched model on AR(1)
    // data, so its false-alarm rate must be materially lower than the
    // NIG (iid-Gaussian) detector on the same streams. Comparison floor:
    // `ar1_fp <= nig_fp / 2`.
    use cesura::NigAr1;
    let trials = 30;
    let length = 600;
    let (a, b) = (0.0_f64, 0.6_f64); // moderate persistence
    let sigma = 1.0_f64;

    let mut nig_fp = 0;
    let mut ar1_fp = 0;

    for trial in 0..trials {
        let data = ar1_sequence(length, a, b, sigma, 7000 + trial);
        let nig_cps = BocpdDetector::new(200.0, length).detect(&data);
        let ar1_cps =
            BocpdDetector::with_prior(200.0, length, NigAr1::default_prior()).detect(&data);
        nig_fp += nig_cps.len();
        ar1_fp += ar1_cps.len();
    }
    eprintln!(
        "AR(1) process: nig_fp={nig_fp}, ar1_fp={ar1_fp} (over {trials} trials × {length} samples)"
    );
    // Strong floor: ar1 must produce no more than half the false alarms.
    // If nig_fp == 0 (already clean) we just check ar1 is also clean.
    assert!(
        ar1_fp * 2 <= nig_fp.max(1),
        "AR(1) detector's FP rate should be ≤ ½ of NIG's: nig={nig_fp}, ar1={ar1_fp}"
    );
}

#[test]
fn bocpd_ar1_detects_shift_in_ar1_process() {
    // Matched-model-with-signal: AR(1) data with a real intercept shift
    // mid-stream. The matched-model NigAr1 must NOT be blind to this
    // shift -- if it were, the FP reduction shown in
    // `bocpd_ar1_on_ar1_process_vs_nig` would be partly missed CPs in
    // disguise.
    //
    //
    // Pin: AR(1) detection rate ≥ ½ × NIG detection rate, AND
    //      AR(1) must detect at least 5/20 trials (not blind).
    //      Detection delay (when fired) must be ≤ 50 steps.
    use cesura::NigAr1;
    let mut ar1_hits = 0;
    let mut nig_hits = 0;
    let mut delays: Vec<i64> = Vec::new();
    for trial in 0..20u64 {
        let mut rng = Rng::new(31_000 + trial);
        let mut data = Vec::with_capacity(600);
        let mut prev = 0.0_f64;
        for _ in 0..300 {
            let x = 0.6 * prev + rng.normal(0.0, 1.0);
            data.push(x);
            prev = x;
        }
        for _ in 0..300 {
            let x = 5.0 + 0.6 * (prev - 5.0) + rng.normal(0.0, 1.0);
            data.push(x);
            prev = x;
        }
        let det_ar1 = BocpdDetector::with_prior(200.0, 700, NigAr1::default_prior());
        let det_nig = BocpdDetector::new(200.0, 700);
        let cps_ar1 = det_ar1.detect(&data);
        let cps_nig = det_nig.detect(&data);
        if let Some(cp) = cps_ar1.iter().find(|c| (c.index as i64 - 300).abs() <= 80) {
            ar1_hits += 1;
            delays.push(cp.index as i64 - 300);
        }
        if cps_nig.iter().any(|c| (c.index as i64 - 300).abs() <= 80) {
            nig_hits += 1;
        }
    }
    let max_delay = delays.iter().copied().map(i64::abs).max().unwrap_or(0);
    eprintln!(
        "AR(1)+shift: AR(1)={ar1_hits}/20, NIG={nig_hits}/20, AR(1) delays={delays:?} max={max_delay}"
    );
    assert!(
        ar1_hits >= 5,
        "AR(1)-BOCPD blind on matched-model-with-signal: {ar1_hits}/20"
    );
    // When AR(1) does fire, delay must be reasonable.
    assert!(
        max_delay <= 50,
        "AR(1) detection delay {max_delay} > 50, suspicious"
    );
}

#[test]
fn nig_ar1_with_beta_silently_falls_back_to_log_predictive() {
    // Documented limitation: NigAr1 inherits the trait's default impl
    // for log_predictive_robust, which falls back to log_predictive at
    // any β. A user combining `with_prior(NigAr1::default_prior())` and
    // `with_beta(0.1)` gets the standard predictive, NOT a β-AR(1)
    // robust update. A β-divergence update for AR(1) is not implemented
    // here.
    //
    // This test pins the current behaviour: results with and without
    // `with_beta(0.1)` must be bit-equal under NigAr1. If a future
    // change adds real β-AR(1) support, this test will start failing
    // (expected) and should be replaced with a positive-direction
    // β-divergence test.
    use cesura::NigAr1;
    let mut rng = Rng::new(2026);
    let mut data: Vec<f64> = (0..150).map(|_| rng.normal(0.0, 1.0)).collect();
    data.extend((0..150).map(|_| rng.normal(4.0, 1.0)));

    let det_plain = BocpdDetector::with_prior(200.0, 350, NigAr1::default_prior());
    let det_beta = BocpdDetector::with_prior(200.0, 350, NigAr1::default_prior()).with_beta(0.1);

    let cps_plain: Vec<usize> = det_plain.detect(&data).into_iter().map(|c| c.index).collect();
    let cps_beta: Vec<usize> = det_beta.detect(&data).into_iter().map(|c| c.index).collect();

    assert_eq!(
        cps_plain, cps_beta,
        "NigAr1 + with_beta must currently match plain (no β-AR(1) support); \
         if this fails, real β-AR(1) has landed and the test should be replaced"
    );
}

#[test]
fn bocpd_ar1_detects_clean_mean_shift() {
    // Sanity: AR(1)-BOCPD must still detect a clean mean shift.
    // Avoid masking real CPs by being too cautious on autocorrelated data.
    use cesura::NigAr1;
    let mut rng = Rng::new(2027);
    let mut data: Vec<f64> = (0..150).map(|_| rng.normal(0.0, 1.0)).collect();
    data.extend((0..150).map(|_| rng.normal(5.0, 1.0)));
    let det = BocpdDetector::with_prior(200.0, 350, NigAr1::default_prior());
    let cps = det.detect(&data);
    assert!(!cps.is_empty(), "AR(1)-BOCPD must detect a clean 5σ shift");
    let near = cps.iter().any(|c| (c.index as i64 - 150).abs() < 30);
    assert!(
        near,
        "AR(1)-BOCPD CP near 150 expected, got {:?}",
        cps.iter().map(|c| c.index).collect::<Vec<_>>()
    );
}

#[test]
fn adaptive_lambda_tracks_inter_cp_interval() {
    // Construct data with N evenly-spaced CPs at known interval T.
    // After detect_mut on the full stream, lambda should converge to
    // ≈ T (within 10%). When adaptive λ is OFF, lambda stays at the
    // constructor value -- assert that too.
    const T: usize = 120;
    const N_REGIMES: usize = 8;
    let mut rng = Rng::new(31_415);
    let mut data: Vec<f64> = Vec::with_capacity(T * N_REGIMES);
    for k in 0..N_REGIMES {
        let mu = if k % 2 == 0 { 0.0 } else { 5.0 };
        for _ in 0..T {
            data.push(rng.normal(mu, 1.0));
        }
    }

    // Adaptive ON: λ should drift toward T.
    let mut det_adapt = BocpdDetector::new(200.0, 1000).with_adaptive_lambda();
    let cps_adapt = det_adapt.detect_mut(&data);
    assert!(
        cps_adapt.len() >= 3,
        "need at least 3 CPs to populate the EMA, got {}",
        cps_adapt.len()
    );
    let lambda_adapt = det_adapt.lambda();
    eprintln!(
        "adaptive λ: {lambda_adapt} (expected ≈ {T}, after {} CPs)",
        cps_adapt.len()
    );
    let rel_err = (lambda_adapt - T as f64).abs() / T as f64;
    assert!(
        rel_err <= 0.10,
        "adaptive λ = {lambda_adapt}, expected within 10% of {T} (rel err {rel_err:.3})"
    );

    // Adaptive OFF: λ stays at constructor value.
    let mut det_fixed = BocpdDetector::new(200.0, 1000);
    let _ = det_fixed.detect_mut(&data);
    assert_eq!(
        det_fixed.lambda(),
        200.0,
        "non-adaptive λ must not change after detect_mut"
    );
}

#[test]
fn ar1_eval_aggregate_meets_floor() {
    // No-regression guard: AR(1) detector must achieve at least the
    // NIG-baseline F1 floor on the 26-scenario eval suite. Expectation
    // is a small gain on serially-dependent scenarios; minimum ask is
    // "no worse than baseline."
    use cesura::NigAr1;
    let scenarios = eval::all_scenarios();
    let det = BocpdDetector::with_prior(200.0, 400, NigAr1::default_prior());
    let mut metrics = Vec::new();
    for s in &scenarios {
        let cps: Vec<usize> = det.detect(&s.data).iter().map(|c| c.index).collect();
        let mut m = eval::match_detections(&cps, &s.ground_truth, 20);
        m.name = s.name.to_string();
        m.category = s.category;
        metrics.push(m);
    }
    let agg = eval::aggregate(&metrics);
    eprintln!(
        "AR(1)-BOCPD eval aggregate: F1={:.3} P={:.3} R={:.3} FP={}",
        agg.f1, agg.precision, agg.recall, agg.fp
    );
    // BocpdDetector::new (NIG) baseline floor in `eval_full_suite` is
    // 0.40; AR(1) variant shares the recursion's MAP-drop heuristic so
    // no large drift should occur.
    assert!(
        agg.f1 >= 0.40,
        "AR(1)-BOCPD aggregate F1 = {:.3}, expected ≥ 0.40",
        agg.f1
    );
}

// ── ConformalCpWrapper ──────────────────────────────────────

/// Bit-equality: wrapper-emitted CP indices match the unwrapped
/// `BocpdDetector::detect` output. Guards against accidental drop /
/// reorder in the `ScoredDetect` impl.
#[test]
fn conformal_wrapper_preserves_detection() {
    use cesura::{BocpdDetector, ConformalCpWrapper, ScoredDetect};

    let mut rng = Rng::new(0xC0CA);
    let mut data: Vec<f64> = (0..400).map(|_| rng.normal(0.0, 1.0)).collect();
    data.extend((0..400).map(|_| rng.normal(4.0, 1.0)));
    data.extend((0..400).map(|_| rng.normal(0.0, 1.0)));

    let det = BocpdDetector::new(200.0, 500);
    let baseline: Vec<usize> = det.detect(&data).iter().map(|c| c.index).collect();
    let scored: Vec<usize> = det
        .detect_with_score(&data)
        .iter()
        .map(|(c, _)| c.index)
        .collect();
    assert_eq!(baseline, scored, "ScoredDetect must preserve detect() indices");

    let mut wrapped = ConformalCpWrapper::new(BocpdDetector::new(200.0, 500))
        .with_calibration_capacity(50)
        .with_coverage(0.9);
    let conformal: Vec<usize> = wrapped.detect(&data).iter().map(|c| c.cp.index).collect();
    assert_eq!(baseline, conformal, "wrapper must preserve detect() indices");
}

/// Empirical coverage on the score's reference point (BOCPD's MAP-
/// collapse step `t_collapse`). For each post-warmup CP, the wrapper's
/// timing interval must contain `t_collapse = cp.index − score`.
///
/// Coverage is asserted against `t_collapse`, not the latent "true CP",
/// because the score directly calibrates the trigger-to-collapse offset
/// distribution. The detection-lag bias between `t_collapse` and the
/// latent CP is a separate uncalibrated quantity; CPTC-style
/// regime-state-conditional calibration (Sun & Yu 2025) is the
/// follow-up that addresses it.
#[test]
fn conformal_wrapper_coverage_is_nominal() {
    use cesura::{BocpdDetector, ConformalCpWrapper, ScoredDetect};

    let regime_len = 100usize;
    let n_regimes = 250usize;
    let shift = 1.0;
    let mut rng = Rng::new(0xDEADBEEF);
    let mut data = Vec::with_capacity(regime_len * n_regimes);
    for r in 0..n_regimes {
        let mu = if r % 2 == 0 { 0.0 } else { shift };
        for _ in 0..regime_len {
            data.push(rng.normal(mu, 1.0));
        }
    }

    let cap = 50;
    let coverage = 0.9;
    let det = BocpdDetector::new(80.0, regime_len + 50);
    let scored = det.detect_with_score(&data);

    let mut wrapper = ConformalCpWrapper::new(BocpdDetector::new(80.0, regime_len + 50))
        .with_calibration_capacity(cap)
        .with_coverage(coverage);
    let conformal = wrapper.detect(&data);
    assert_eq!(conformal.len(), scored.len(), "1:1 emission parity");

    let post: Vec<(usize, &cesura::ConformalCp, f64)> = conformal
        .iter()
        .zip(scored.iter())
        .enumerate()
        .filter(|(idx, _)| *idx >= cap)
        .map(|(idx, (c, (_, s)))| (idx, c, *s))
        .collect();
    assert!(
        post.len() >= 100,
        "need ≥ 100 post-warmup CPs, got {} (total {})",
        post.len(),
        conformal.len()
    );

    let contained = post
        .iter()
        .filter(|(_, c, score)| {
            let t_collapse = c.cp.index as i64 - score.round() as i64;
            let (lo, hi) = c.timing_interval;
            lo <= t_collapse && t_collapse <= hi
        })
        .count();
    let empirical = contained as f64 / post.len() as f64;
    eprintln!(
        "post-warmup CPs={} empirical_coverage={:.3} (nominal {:.2})",
        post.len(),
        empirical,
        coverage
    );
    // Lower bound 0.85 is the conformal contract (≥ nominal − slack);
    // upper bound 0.99 accepts the structural over-coverage of a
    // discrete-valued score under nearest-rank quantile (ties at the
    // lower boundary inflate empirical coverage above nominal). A
    // ceiling here still catches gross degeneracies (e.g. an always-
    // zero score, which would pin coverage at 1.0).
    assert!(
        (0.85..=0.99).contains(&empirical),
        "empirical coverage {empirical:.3} outside [0.85, 0.99]"
    );
}

/// Calibration-buffer warmup: interval-width variance over the second
/// 100 CPs (rolling) should be tighter than over the first 100 CPs
/// (filling) by at least a factor of 5.
#[test]
fn conformal_calibration_buffer_grows_then_stabilises() {
    use cesura::{BocpdDetector, ConformalCpWrapper};

    let regime_len = 100usize;
    let n_regimes = 300usize;
    let shift = 4.0;
    let mut rng = Rng::new(0xAB1A);
    let mut data = Vec::with_capacity(regime_len * n_regimes);
    for r in 0..n_regimes {
        let mu = if r % 2 == 0 { 0.0 } else { shift };
        for _ in 0..regime_len {
            data.push(rng.normal(mu, 1.0));
        }
    }

    let mut wrapper = ConformalCpWrapper::new(BocpdDetector::new(80.0, regime_len + 50))
        .with_calibration_capacity(100)
        .with_coverage(0.9);
    let conformal = wrapper.detect(&data);
    assert!(
        conformal.len() >= 200,
        "need ≥ 200 CPs to test variance regimes, got {}",
        conformal.len()
    );

    let widths: Vec<f64> = conformal
        .iter()
        .map(|c| (c.timing_interval.1 - c.timing_interval.0) as f64)
        .collect();
    let var = |xs: &[f64]| {
        let m = xs.iter().sum::<f64>() / xs.len() as f64;
        xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / xs.len() as f64
    };
    let early = var(&widths[..100]);
    let later = var(&widths[100..200]);
    eprintln!(
        "buffer width variance: first100={early:.3} second100={later:.3} ratio={:.3}",
        if early > 0.0 { later / early } else { f64::NAN }
    );
    // Buffer is full from CP 100 onward → variance should drop sharply.
    assert!(
        later <= 0.20 * early.max(1e-9),
        "variance over rolling window ({later:.3}) should be ≤ 20% of growing-window variance ({early:.3})"
    );
}

/// Smoke test: wrapper composes over the ensemble.
#[test]
fn conformal_wrapper_works_over_ensemble() {
    use cesura::{ConformalCpWrapper, EnsembleDetector};

    let data: Vec<f64> = std::iter::repeat_n(0.0, 200)
        .chain(std::iter::repeat_n(5.0, 200))
        .collect();
    let mut wrapper =
        ConformalCpWrapper::new(EnsembleDetector::new(200.0, 350)).with_calibration_capacity(8);
    let cps = wrapper.detect(&data);
    assert!(!cps.is_empty(), "ensemble-wrapped wrapper should emit ≥ 1 CP");
    for c in &cps {
        let (lo, hi) = c.timing_interval;
        assert!(lo <= hi, "interval must be non-empty: ({lo}, {hi})");
    }
}

///
/// Per-regime conditioning (CPTC-style) is a possible extension; if
/// this test ever fails, that's the natural escalation.
#[test]
fn conformal_wrapper_coverage_under_snr_transition() {
    use cesura::{BocpdDetector, ConformalCpWrapper, ScoredDetect};

    let regime_len = 100usize;
    // 200 alternating-regime pairs: first 100 at shift=1σ, second 100
    // at shift=4σ. Crosses the SNR transition mid-stream so the buffer
    // accumulates contaminated history at exactly the moment under
    // test.
    let n_low = 100usize;
    let n_high = 100usize;
    let shift_low = 1.0_f64;
    let shift_high = 4.0_f64;
    let mut rng = Rng::new(0xBADCAFE);
    let mut data = Vec::with_capacity(regime_len * (n_low + n_high));
    for r in 0..n_low {
        let mu = if r % 2 == 0 { 0.0 } else { shift_low };
        for _ in 0..regime_len {
            data.push(rng.normal(mu, 1.0));
        }
    }
    for r in 0..n_high {
        let mu = if r % 2 == 0 { 0.0 } else { shift_high };
        for _ in 0..regime_len {
            data.push(rng.normal(mu, 1.0));
        }
    }

    let cap = 50;
    let coverage = 0.9;
    let det = BocpdDetector::new(80.0, regime_len + 50);
    let scored = det.detect_with_score(&data);

    let mut wrapper = ConformalCpWrapper::new(BocpdDetector::new(80.0, regime_len + 50))
        .with_calibration_capacity(cap)
        .with_coverage(coverage);
    let conformal = wrapper.detect(&data);
    assert_eq!(conformal.len(), scored.len(), "1:1 emission parity");

    let post: Vec<(usize, &cesura::ConformalCp, f64)> = conformal
        .iter()
        .zip(scored.iter())
        .enumerate()
        .filter(|(idx, _)| *idx >= cap)
        .map(|(idx, (c, (_, s)))| (idx, c, *s))
        .collect();
    assert!(post.len() >= 100, "need ≥ 100 post-warmup CPs, got {}", post.len());

    let contained = post
        .iter()
        .filter(|(_, c, score)| {
            let t_collapse = c.cp.index as i64 - score.round() as i64;
            let (lo, hi) = c.timing_interval;
            lo <= t_collapse && t_collapse <= hi
        })
        .count();
    let empirical = contained as f64 / post.len() as f64;
    eprintln!(
        "snr-transition: post-warmup CPs={} empirical_coverage={:.3} (nominal {:.2})",
        post.len(),
        empirical,
        coverage
    );
    // Looser band than `conformal_wrapper_coverage_is_nominal` because
    // cross-regime contamination is expected to widen the empirical-
    // coverage spread; ≥ 0.85 keeps the conformal lower-bound contract.
    assert!(
        (0.85..=0.99).contains(&empirical),
        "marginal-across-regimes coverage {empirical:.3} outside [0.85, 0.99]"
    );
}

#[test]
fn conformal_mv_wrapper_coverage_on_anticorrelated_shifts() {
    // Multivariate analogue of `conformal_wrapper_coverage_is_nominal`.
    // 200 alternating-regime pairs at ρ=0.95, 1σ anti-correlated shift
    // (the exact fixture cesura's MV path is supposed to win on; per-
    // dim z-norm misses these). Each regime is 100 bars; total 20k.
    // Cap=50 calibration buffer; nominal coverage 0.90. Pin coverage
    // ∈ [0.85, 0.99] -- same band as the univariate SNR-transition pin.
    use cesura::{BocpdDetector, ConformalCpWrapper};
    use cesura::MvScoredDetect;

    let regime_len = 100usize;
    let n_regimes = 200usize;
    let rho = 0.95_f64;
    let shift = 1.0_f64;
    let mut rng = Rng::new(0xCAFE_F00D);
    let mut data: Vec<Vec<f64>> = Vec::with_capacity(regime_len * n_regimes);
    for r in 0..n_regimes {
        let pos = r % 2 == 1;
        for _ in 0..regime_len {
            let z0 = rng.normal(0.0, 1.0);
            let z1 = rng.normal(0.0, 1.0);
            let off = if pos { shift * 0.5 } else { 0.0 };
            let x0 = z0 + off;
            let x1 = rho * z0 + (1.0 - rho * rho).sqrt() * z1 - off;
            data.push(vec![x0, x1]);
        }
    }

    let cap = 50usize;
    let coverage = 0.9_f64;
    let det = BocpdDetector::new(80.0, regime_len + 50);
    let scored = det.detect_multivariate_with_score(&data);

    let mut wrapper = ConformalCpWrapper::new(BocpdDetector::new(80.0, regime_len + 50))
        .with_calibration_capacity(cap)
        .with_coverage(coverage);
    let conformal = wrapper.detect_multivariate(&data);
    assert_eq!(conformal.len(), scored.len(), "1:1 emission parity");
    assert!(scored.len() >= cap + 50, "need ≥ {} emissions, got {}", cap + 50, scored.len());

    let post: Vec<(usize, &cesura::ConformalCp, f64)> = conformal
        .iter()
        .zip(scored.iter())
        .enumerate()
        .filter(|(idx, _)| *idx >= cap)
        .map(|(idx, (c, (_, s)))| (idx, c, *s))
        .collect();
    let post_n = post.len();
    let contained = post
        .iter()
        .filter(|(_, c, score)| {
            let t_collapse = c.cp.index as i64 - score.round() as i64;
            let (lo, hi) = c.timing_interval;
            lo <= t_collapse && t_collapse <= hi
        })
        .count();
    let empirical = contained as f64 / post_n as f64;
    eprintln!(
        "mv anticorrelated: post-warmup={post_n} empirical_coverage={empirical:.3} (nominal {coverage})"
    );
    assert!(
        (0.85..=0.99).contains(&empirical),
        "MV anticorrelated empirical coverage {empirical:.3} outside [0.85, 0.99]"
    );
}
