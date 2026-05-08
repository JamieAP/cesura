//! Evaluation harness for change point detection.
//!
//! Gated behind the `test-utils` feature. Provides deterministic PRNG,
//! scenario generators with ground-truth change points, and precision/recall/F1
//! metrics with greedy nearest-first matching.

use std::f64::consts::PI;

// ── Deterministic PRNG (xoshiro256**) ────────────────────────

/// Deterministic PRNG for reproducible test scenarios.
pub struct Rng(pub [u64; 4]);

impl Rng {
    pub fn new(seed: u64) -> Self {
        let mut s = seed;
        let mut state = [0u64; 4];
        for slot in &mut state {
            s = s.wrapping_add(0x9e3779b97f4a7c15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            *slot = z ^ (z >> 31);
        }
        Self(state)
    }

    pub fn next_u64(&mut self) -> u64 {
        let result = (self.0[1].wrapping_mul(5)).rotate_left(7).wrapping_mul(9);
        let t = self.0[1] << 17;
        self.0[2] ^= self.0[0];
        self.0[3] ^= self.0[1];
        self.0[1] ^= self.0[2];
        self.0[0] ^= self.0[3];
        self.0[2] ^= t;
        self.0[3] = self.0[3].rotate_left(45);
        result
    }

    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    pub fn normal(&mut self, mean: f64, std: f64) -> f64 {
        let u1 = self.uniform().max(1e-300);
        let u2 = self.uniform();
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
        mean + std * z
    }

    /// Cauchy(loc, scale) via inverse-CDF.
    /// Heavy-tailed: undefined mean and variance. Used to stress-test
    /// detectors that assume Gaussian-like tails.
    pub fn cauchy(&mut self, loc: f64, scale: f64) -> f64 {
        let u = self.uniform().clamp(1e-12, 1.0 - 1e-12);
        loc + scale * (PI * (u - 0.5)).tan()
    }

    /// Student-t(df=3) via a standard normal divided by sqrt(chi²(3)/3).
    /// Its variance is 3; its fourth moment is undefined (infinite kurtosis).
    /// The tails are markedly heavier than Gaussian.
    pub fn student_t3(&mut self) -> f64 {
        // t_3 = Z / sqrt(Y/3) where Z ~ N(0,1) and Y ~ chi²(3).
        // chi²(3) = sum of 3 squared standard normals.
        let z = self.normal(0.0, 1.0);
        let n1 = self.normal(0.0, 1.0);
        let n2 = self.normal(0.0, 1.0);
        let n3 = self.normal(0.0, 1.0);
        let y = n1 * n1 + n2 * n2 + n3 * n3;
        z / (y / 3.0).sqrt().max(1e-12)
    }
}

// ── Scenario types ───────────────────────────────────────────

/// Category for expected behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Category {
    MustDetect,
    MustReject,
    Challenging,
    Operational,
}

/// A test scenario with data and ground truth.
pub struct Scenario {
    pub name: &'static str,
    pub category: Category,
    pub data: Vec<f64>,
    pub ground_truth: Vec<usize>,
    /// Known period for detrending (None = no known periodicity).
    pub period: Option<usize>,
}

// ── Metrics ──────────────────────────────────────────────────

/// Per-scenario evaluation metrics.
#[derive(Debug, Clone)]
pub struct EvalMetrics {
    pub name: String,
    pub category: Category,
    pub tp: usize,
    pub fp: usize,
    pub r#fn: usize,
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
    pub mean_delay: f64,
}

/// Greedy nearest-first matching with tolerance.
///
/// Matches detected change points to ground truth within ±tolerance.
/// Each ground truth can only be matched once.
pub fn match_detections(
    detected: &[usize],
    ground_truth: &[usize],
    tolerance: usize,
) -> EvalMetrics {
    let mut matched_gt = vec![false; ground_truth.len()];
    let mut tp = 0usize;
    let mut delays = Vec::new();

    // Sort detected by distance to nearest ground truth (greedy)
    let mut detected_sorted: Vec<usize> = detected.to_vec();
    detected_sorted.sort_by_key(|&d| {
        ground_truth
            .iter()
            .map(|&gt| (d as i64 - gt as i64).unsigned_abs() as usize)
            .min()
            .unwrap_or(usize::MAX)
    });

    for &det in &detected_sorted {
        let mut best_dist = usize::MAX;
        let mut best_idx = None;
        for (i, &gt) in ground_truth.iter().enumerate() {
            if matched_gt[i] {
                continue;
            }
            let dist = (det as i64 - gt as i64).unsigned_abs() as usize;
            if dist <= tolerance && dist < best_dist {
                best_dist = dist;
                best_idx = Some(i);
            }
        }
        if let Some(idx) = best_idx {
            matched_gt[idx] = true;
            tp += 1;
            delays.push(best_dist as f64);
        }
    }

    let fp = detected.len() - tp;
    let fn_count = ground_truth.len() - tp;
    let precision = if detected.is_empty() {
        if ground_truth.is_empty() {
            1.0
        } else {
            0.0
        }
    } else {
        tp as f64 / detected.len() as f64
    };
    let recall = if ground_truth.is_empty() {
        1.0
    } else {
        tp as f64 / ground_truth.len() as f64
    };
    let f1 = if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    };
    let mean_delay = if delays.is_empty() {
        0.0
    } else {
        delays.iter().sum::<f64>() / delays.len() as f64
    };

    EvalMetrics {
        name: String::new(),
        category: Category::MustDetect,
        tp,
        fp,
        r#fn: fn_count,
        precision,
        recall,
        f1,
        mean_delay,
    }
}

/// Aggregate metrics across scenarios.
pub fn aggregate(metrics: &[EvalMetrics]) -> EvalMetrics {
    let tp: usize = metrics.iter().map(|m| m.tp).sum();
    let fp: usize = metrics.iter().map(|m| m.fp).sum();
    let fn_count: usize = metrics.iter().map(|m| m.r#fn).sum();
    let precision = if tp + fp > 0 {
        tp as f64 / (tp + fp) as f64
    } else {
        1.0
    };
    let recall = if tp + fn_count > 0 {
        tp as f64 / (tp + fn_count) as f64
    } else {
        1.0
    };
    let f1 = if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    };
    let delays: Vec<f64> = metrics.iter().map(|m| m.mean_delay).collect();
    let mean_delay = if delays.is_empty() {
        0.0
    } else {
        delays.iter().sum::<f64>() / delays.len() as f64
    };
    EvalMetrics {
        name: "AGGREGATE".into(),
        category: Category::MustDetect,
        tp,
        fp,
        r#fn: fn_count,
        precision,
        recall,
        f1,
        mean_delay,
    }
}

/// Print a report table.
#[cfg(test)]
pub(crate) fn print_report(metrics: &[EvalMetrics]) {
    eprintln!(
        "{:<30} {:>5} {:>4} {:>4} {:>4} {:>6} {:>6} {:>6} {:>6}",
        "Scenario", "Cat", "TP", "FP", "FN", "Prec", "Rec", "F1", "Delay"
    );
    eprintln!("{}", "-".repeat(90));
    for m in metrics {
        let cat = match m.category {
            Category::MustDetect => "MD",
            Category::MustReject => "MR",
            Category::Challenging => "CH",
            Category::Operational => "OP",
        };
        eprintln!(
            "{:<30} {:>5} {:>4} {:>4} {:>4} {:>6.2} {:>6.2} {:>6.2} {:>6.1}",
            m.name, cat, m.tp, m.fp, m.r#fn, m.precision, m.recall, m.f1, m.mean_delay
        );
    }
    eprintln!("{}", "-".repeat(90));
    let agg = aggregate(metrics);
    eprintln!(
        "{:<30} {:>5} {:>4} {:>4} {:>4} {:>6.2} {:>6.2} {:>6.2} {:>6.1}",
        "AGGREGATE",
        "",
        agg.tp,
        agg.fp,
        agg.r#fn,
        agg.precision,
        agg.recall,
        agg.f1,
        agg.mean_delay
    );
}

// ── Scenario generators ──────────────────────────────────────

pub fn all_scenarios() -> Vec<Scenario> {
    vec![
        // MustDetect (8) -- all have realistic noise
        noisy_2sigma_shift(),
        noisy_3sigma_shift(),
        noisy_regime_change(),
        noisy_collective_anomaly(),
        noisy_variance_doubling(),
        noisy_step_down(),
        noisy_large_baseline_shift(),
        noisy_negative_shift(),
        // MustReject (6) -- patterns that look like changes but aren't
        white_noise(),
        ar1_colored_noise(),
        periodic_diurnal(),
        periodic_with_outliers(),
        trending_stationary(),
        bursty_but_stationary(),
        // Challenging (4) -- real detection difficulty
        one_sigma_shift(),
        gradual_ramp(),
        seasonal_with_shift(),
        shift_in_noisy_periodic(),
        // Operational (4) -- K8s metric patterns
        cpu_idle_to_burst(),
        memory_leak(),
        restart_storm(),
        deployment_rollout(),
        // Heavy-tail (4) -- stress tests for the Gaussian-tail assumption
        cauchy_noise(),
        t3_noise(),
        garch_volatility_clusters(),
        t3_with_real_shift(),
    ]
}

/// Subset selector: heavy-tail scenarios only. Used by tests that gauge
/// improvements (e.g. β-divergence) on the data class where standard
/// NIG-Gaussian under-models the tails.
pub fn heavy_tail_scenarios() -> Vec<Scenario> {
    vec![
        cauchy_noise(),
        t3_noise(),
        garch_volatility_clusters(),
        t3_with_real_shift(),
    ]
}

/// FOCuS GLR baseline detector for comparison.
/// Returns CP indices from `FocusDetector::detect` at the given threshold.
pub fn focus_detect(data: &[f64], threshold: f64) -> Vec<usize> {
    let mut det = crate::focus::FocusDetector::new(threshold);
    det.detect(data).into_iter().map(|c| c.index).collect()
}

/// Naive z-score baseline detector for comparison.
/// Returns indices where |z| > threshold using a rolling window.
#[cfg(test)]
pub(crate) fn naive_zscore_detect(data: &[f64], window: usize, z_thresh: f64) -> Vec<usize> {
    let mut detections = Vec::new();
    let cooldown = 15;
    let mut last_det = 0usize;
    for i in window..data.len() {
        if i - last_det < cooldown {
            continue;
        }
        let win = &data[i - window..i];
        let mean = win.iter().sum::<f64>() / win.len() as f64;
        let std = (win.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / win.len() as f64).sqrt();
        if std < 1e-10 {
            continue;
        }
        let z = (data[i] - mean).abs() / std;
        if z > z_thresh {
            detections.push(i);
            last_det = i;
        }
    }
    detections
}

// ── MustDetect scenarios (all noisy) ─────────────────────────

fn noisy_2sigma_shift() -> Scenario {
    let mut rng = Rng::new(1001);
    let mut data: Vec<f64> = (0..150).map(|_| rng.normal(0.0, 1.0)).collect();
    data.extend((0..150).map(|_| rng.normal(2.0, 1.0)));
    Scenario {
        name: "noisy_2sigma",
        category: Category::MustDetect,
        data,
        period: None,
        ground_truth: vec![150],
    }
}

fn noisy_3sigma_shift() -> Scenario {
    let mut rng = Rng::new(1002);
    let mut data: Vec<f64> = (0..150).map(|_| rng.normal(0.0, 1.0)).collect();
    data.extend((0..150).map(|_| rng.normal(3.0, 1.0)));
    Scenario {
        name: "noisy_3sigma",
        category: Category::MustDetect,
        data,
        period: None,
        ground_truth: vec![150],
    }
}

fn noisy_regime_change() -> Scenario {
    let mut rng = Rng::new(1003);
    let mut data: Vec<f64> = (0..100).map(|_| rng.normal(0.0, 1.0)).collect();
    data.extend((0..100).map(|_| rng.normal(4.0, 1.5)));
    data.extend((0..100).map(|_| rng.normal(-2.0, 0.8)));
    Scenario {
        name: "noisy_regime_change",
        category: Category::MustDetect,
        data,
        period: None,
        ground_truth: vec![100, 200],
    }
}

fn noisy_collective_anomaly() -> Scenario {
    let mut rng = Rng::new(1004);
    let mut data: Vec<f64> = (0..100).map(|_| rng.normal(0.0, 1.0)).collect();
    data.extend((0..50).map(|_| rng.normal(4.0, 1.0)));
    data.extend((0..100).map(|_| rng.normal(0.0, 1.0)));
    Scenario {
        name: "noisy_collective",
        category: Category::MustDetect,
        data,
        period: None,
        ground_truth: vec![100, 150],
    }
}

fn noisy_variance_doubling() -> Scenario {
    let mut rng = Rng::new(1005);
    let mut data: Vec<f64> = (0..150).map(|_| rng.normal(5.0, 0.5)).collect();
    data.extend((0..150).map(|_| rng.normal(5.0, 2.0)));
    Scenario {
        name: "noisy_variance_double",
        category: Category::MustDetect,
        data,
        period: None,
        ground_truth: vec![150],
    }
}

fn noisy_step_down() -> Scenario {
    let mut rng = Rng::new(1006);
    let mut data: Vec<f64> = (0..150).map(|_| rng.normal(10.0, 1.5)).collect();
    data.extend((0..150).map(|_| rng.normal(5.0, 1.5)));
    Scenario {
        name: "noisy_step_down",
        category: Category::MustDetect,
        data,
        period: None,
        ground_truth: vec![150],
    }
}

fn noisy_large_baseline_shift() -> Scenario {
    let mut rng = Rng::new(1007);
    let mut data: Vec<f64> = (0..150).map(|_| rng.normal(1000.0, 50.0)).collect();
    data.extend((0..150).map(|_| rng.normal(1200.0, 50.0)));
    Scenario {
        name: "noisy_large_baseline",
        category: Category::MustDetect,
        data,
        period: None,
        ground_truth: vec![150],
    }
}

fn noisy_negative_shift() -> Scenario {
    let mut rng = Rng::new(1008);
    let mut data: Vec<f64> = (0..150).map(|_| rng.normal(-5.0, 1.0)).collect();
    data.extend((0..150).map(|_| rng.normal(-2.0, 1.0)));
    Scenario {
        name: "noisy_negative",
        category: Category::MustDetect,
        data,
        period: None,
        ground_truth: vec![150],
    }
}

// ── MustReject scenarios ─────────────────────────────────────

fn white_noise() -> Scenario {
    let mut rng = Rng::new(2001);
    let data: Vec<f64> = (0..300).map(|_| rng.normal(0.0, 1.0)).collect();
    Scenario {
        name: "white_noise",
        category: Category::MustReject,
        data,
        period: None,
        ground_truth: vec![],
    }
}

fn ar1_colored_noise() -> Scenario {
    let mut rng = Rng::new(2002);
    let phi = 0.95;
    let mut data = Vec::with_capacity(300);
    let mut x = 0.0;
    for _ in 0..300 {
        x = phi * x + rng.normal(0.0, 0.3);
        data.push(x);
    }
    Scenario {
        name: "ar1_colored",
        category: Category::MustReject,
        data,
        period: None,
        ground_truth: vec![],
    }
}

fn periodic_diurnal() -> Scenario {
    // Simulates daily CPU pattern: low overnight, high during day
    let mut rng = Rng::new(2003);
    let data: Vec<f64> = (0..360)
        .map(|i| {
            let hour_frac = (i as f64 / 60.0) % 24.0;
            let diurnal = if (8.0..18.0).contains(&hour_frac) {
                0.6
            } else {
                0.1
            };
            diurnal + rng.normal(0.0, 0.05)
        })
        .collect();
    Scenario {
        name: "periodic_diurnal",
        category: Category::MustReject,
        data,
        period: Some(360),
        ground_truth: vec![],
    }
}

fn periodic_with_outliers() -> Scenario {
    // Periodic signal with random spikes -- not a change point
    let mut rng = Rng::new(2004);
    let data: Vec<f64> = (0..300)
        .map(|i| {
            let base = (i as f64 * 2.0 * PI / 50.0).sin() * 2.0;
            let spike = if rng.uniform() < 0.03 { 5.0 } else { 0.0 };
            base + spike + rng.normal(0.0, 0.3)
        })
        .collect();
    Scenario {
        name: "periodic_outliers",
        category: Category::MustReject,
        data,
        period: Some(50),
        ground_truth: vec![],
    }
}

fn trending_stationary() -> Scenario {
    // Gentle linear trend -- not a change point, just drift
    let mut rng = Rng::new(2005);
    let data: Vec<f64> = (0..300)
        .map(|i| i as f64 * 0.01 + rng.normal(0.0, 0.5))
        .collect();
    Scenario {
        name: "trending_stationary",
        category: Category::MustReject,
        data,
        period: None,
        ground_truth: vec![],
    }
}

fn bursty_but_stationary() -> Scenario {
    // Intermittent bursts that return to baseline -- no real change point
    let mut rng = Rng::new(2006);
    let data: Vec<f64> = (0..300)
        .map(|i| {
            let burst = if i % 50 < 5 { 3.0 } else { 0.0 };
            burst + rng.normal(0.0, 0.5)
        })
        .collect();
    Scenario {
        name: "bursty_stationary",
        category: Category::MustReject,
        data,
        period: Some(50),
        ground_truth: vec![],
    }
}

// ── Challenging scenarios ────────────────────────────────────

fn one_sigma_shift() -> Scenario {
    let mut rng = Rng::new(3001);
    let mut data: Vec<f64> = (0..150).map(|_| rng.normal(0.0, 1.0)).collect();
    data.extend((0..150).map(|_| rng.normal(1.0, 1.0)));
    Scenario {
        name: "one_sigma_shift",
        category: Category::Challenging,
        data,
        period: None,
        ground_truth: vec![150],
    }
}

fn gradual_ramp() -> Scenario {
    // Linear ramp from 0 to 5 over 100 steps, then stable at 5
    let mut rng = Rng::new(3002);
    let data: Vec<f64> = (0..300)
        .map(|i| {
            let level = if i < 100 {
                0.0
            } else if i < 200 {
                (i - 100) as f64 * 0.05
            } else {
                5.0
            };
            rng.normal(level, 0.5)
        })
        .collect();
    Scenario {
        name: "gradual_ramp",
        category: Category::Challenging,
        data,
        period: None,
        ground_truth: vec![100], // onset of the ramp
    }
}

fn seasonal_with_shift() -> Scenario {
    let mut rng = Rng::new(3003);
    let data: Vec<f64> = (0..360)
        .map(|i| {
            let seasonal = (i as f64 * 2.0 * PI / 60.0).sin() * 2.0;
            let level = if i < 180 { 0.0 } else { 3.0 };
            seasonal + level + rng.normal(0.0, 0.5)
        })
        .collect();
    Scenario {
        name: "seasonal_with_shift",
        category: Category::Challenging,
        data,
        period: Some(60),
        ground_truth: vec![180],
    }
}

fn shift_in_noisy_periodic() -> Scenario {
    // Diurnal pattern + mean shift mid-series
    let mut rng = Rng::new(3004);
    let data: Vec<f64> = (0..360)
        .map(|i| {
            let hour = (i as f64 / 60.0) % 24.0;
            let diurnal = (hour * PI / 12.0).sin() * 0.3;
            let level = if i < 180 { 1.0 } else { 2.5 };
            diurnal + level + rng.normal(0.0, 0.2)
        })
        .collect();
    Scenario {
        name: "shift_in_periodic",
        category: Category::Challenging,
        data,
        period: Some(360),
        ground_truth: vec![180],
    }
}

// ── Operational scenarios (K8s patterns) ─────────────────────

fn cpu_idle_to_burst() -> Scenario {
    // Idle namespace suddenly starts using CPU (deployment scaled up)
    let mut rng = Rng::new(4001);
    let mut data: Vec<f64> = (0..150).map(|_| rng.normal(0.05, 0.02).max(0.0)).collect();
    data.extend((0..50).map(|_| rng.normal(0.8, 0.15).clamp(0.0, 1.0)));
    data.extend((0..100).map(|_| rng.normal(0.05, 0.02).max(0.0)));
    Scenario {
        name: "cpu_burst",
        category: Category::Operational,
        data,
        period: None,
        ground_truth: vec![150, 200],
    }
}

fn memory_leak() -> Scenario {
    // Linear memory growth with noise -- should detect onset
    let mut rng = Rng::new(4002);
    let data: Vec<f64> = (0..300)
        .map(|i| {
            let base = if i < 100 {
                500.0
            } else {
                500.0 + (i - 100) as f64 * 3.0
            };
            base + rng.normal(0.0, 15.0)
        })
        .collect();
    Scenario {
        name: "memory_leak",
        category: Category::Operational,
        data,
        period: None,
        ground_truth: vec![100],
    }
}

fn restart_storm() -> Scenario {
    // Container restarts: near-zero → burst → back to zero
    let mut rng = Rng::new(4003);
    let mut data: Vec<f64> = (0..150).map(|_| rng.normal(0.1, 0.3).max(0.0)).collect();
    data.extend((0..30).map(|_| rng.normal(5.0, 2.0).max(0.0)));
    data.extend((0..120).map(|_| rng.normal(0.1, 0.3).max(0.0)));
    Scenario {
        name: "restart_storm",
        category: Category::Operational,
        data,
        period: None,
        ground_truth: vec![150, 180],
    }
}

// ── Heavy-tail scenarios ─────────────────────────────────────
//
// Real-world inputs are rarely Gaussian. Crypto returns, network latencies,
// and queue depths exhibit kurtosis 5-15. Standard BOCPD with NIG-Gaussian
// likelihood under-models the tails and fires on legitimate tail events as
// if they were change points. These scenarios pin that behaviour.

fn cauchy_noise() -> Scenario {
    // Cauchy(0, 1): undefined mean and variance. Standard cesura fires
    // on every tail event; β-divergence should suppress them.
    let mut rng = Rng::new(5001);
    let data: Vec<f64> = (0..300).map(|_| rng.cauchy(0.0, 1.0)).collect();
    Scenario {
        name: "cauchy_noise",
        category: Category::MustReject,
        data,
        period: None,
        ground_truth: vec![],
    }
}

fn t3_noise() -> Scenario {
    // Student-t with df=3: heavy tails but finite variance. Milder
    // than Cauchy; a sanity check that β-divergence's robustness
    // does not require pathological inputs to show.
    let mut rng = Rng::new(5002);
    let data: Vec<f64> = (0..300).map(|_| rng.student_t3()).collect();
    Scenario {
        name: "t3_noise",
        category: Category::MustReject,
        data,
        period: None,
        ground_truth: vec![],
    }
}

fn garch_volatility_clusters() -> Scenario {
    // GARCH(1,1): σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}. Volatility clusters
    // but the unconditional mean is zero -- no regime change. A naive
    // detector confuses high-vol periods for shifts.
    // Parameters chosen so the process is stationary (α + β < 1) but
    // exhibits visible clustering: ω=0.05, α=0.20, β=0.75.
    let mut rng = Rng::new(5003);
    let omega = 0.05_f64;
    let alpha = 0.20_f64;
    let beta = 0.75_f64;
    let mut sigma_sq = omega / (1.0 - alpha - beta); // unconditional var
    let mut prev_eps = 0.0_f64;
    let data: Vec<f64> = (0..300)
        .map(|_| {
            sigma_sq = omega + alpha * prev_eps * prev_eps + beta * sigma_sq;
            let eps = rng.normal(0.0, sigma_sq.sqrt());
            prev_eps = eps;
            eps
        })
        .collect();
    Scenario {
        name: "garch_clusters",
        category: Category::MustReject,
        data,
        period: None,
        ground_truth: vec![],
    }
}

fn t3_with_real_shift() -> Scenario {
    // t₃ noise (scale 1) with a 3σ-of-noise mean shift at t=150. The
    // detector must still resolve the shift even with heavy-tailed
    // observations dominating the predictive likelihood. t₃ has
    // variance 3, so std ≈ 1.732; a "3σ" shift in noise units is 3*1.732 ≈ 5.2.
    let mut rng = Rng::new(5004);
    let mut data: Vec<f64> = (0..150).map(|_| rng.student_t3()).collect();
    let shift = 3.0 * 3.0_f64.sqrt();
    data.extend((0..150).map(|_| shift + rng.student_t3()));
    Scenario {
        name: "t3_with_shift",
        category: Category::MustDetect,
        data,
        period: None,
        ground_truth: vec![150],
    }
}

fn deployment_rollout() -> Scenario {
    // CPU spike during rollout, settles to new (higher) baseline
    let mut rng = Rng::new(4004);
    let mut data: Vec<f64> = (0..100).map(|_| rng.normal(0.3, 0.08).max(0.0)).collect();
    data.extend((0..20).map(|_| rng.normal(0.9, 0.2).max(0.0)));
    data.extend((0..180).map(|_| rng.normal(0.45, 0.08).max(0.0)));
    Scenario {
        name: "deployment_rollout",
        category: Category::Operational,
        data,
        period: None,
        ground_truth: vec![100, 120],
    }
}
