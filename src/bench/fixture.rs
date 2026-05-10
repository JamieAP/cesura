//! Canonical fixture shape.
//!
//! `data` is always `[Vec<f64>]`; univariate fixtures set `d=1` with a
//! single inner vec. `ground_truth` indexes into `data`. `margin` is
//! tolerance in indices (epoch-window→index conversion happens at
//! fixture construction, not in `evaluate`).

#[derive(Debug, Clone)]
pub struct Fixture {
    pub name: String,
    /// Bump on data/annotation change. Reports record this for audit.
    pub version: u32,
    pub d: usize,
    /// Shape `[T][d]`: `data[t][i]` is channel `i` at time step `t`.
    /// Univariate fixtures are `T` rows each of length 1.
    /// Matches `*::detect_multivariate(&[Vec<f64>])` calling convention.
    pub data: Vec<Vec<f64>>,
    /// `Some` for real-world fixtures with timestamps; `None` for
    /// synthetic.
    pub epochs: Option<Vec<i64>>,
    /// CP indices into `data` (column index for multivariate, time
    /// index for univariate).
    pub ground_truth: Vec<usize>,
    pub seed: Option<u64>,
    /// Hit tolerance in indices. Greedy nearest-first matching.
    pub margin: usize,
}

impl Fixture {
    /// Length of the time axis (number of rows in `data`).
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Flatten a univariate (`d=1`) fixture's data to `Vec<f64>`.
    /// Adapters use this to feed the univariate `detect` paths.
    /// Empty vec if `d != 1`.
    pub fn as_univariate(&self) -> Vec<f64> {
        if self.d != 1 {
            return Vec::new();
        }
        self.data.iter().map(|r| r[0]).collect()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[derive(Debug, Clone)]
pub enum FixtureError {
    Io(String),
    Missing(String),
    Malformed(String),
}

impl std::fmt::Display for FixtureError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(s) => write!(f, "fixture io: {s}"),
            Self::Missing(s) => write!(f, "fixture missing: {s}"),
            Self::Malformed(s) => write!(f, "fixture malformed: {s}"),
        }
    }
}

impl std::error::Error for FixtureError {}

/// Test-helper: build a step-shift fixture from a base tape using the
/// same square-wave injection as `anomaly_injected_crypto`. Lets the
/// injection logic be unit-tested without the real parquet dependency.
#[doc(hidden)]
pub fn step_shift_inject(
    base: &[Vec<f64>],
    chosen: &[usize],
    sigma: f64,
    sigmas: &[f64],
) -> Vec<Vec<f64>> {
    let n = base.len();
    let d = sigmas.len();
    let mut data: Vec<Vec<f64>> = base.to_vec();
    let mut offset = vec![0f64; d];
    let mut k = 0usize;
    for (t, row) in data.iter_mut().enumerate().take(n) {
        if k < chosen.len() && t == chosen[k] {
            let sign = if k.is_multiple_of(2) { 1.0 } else { -1.0 };
            for i in 0..d {
                offset[i] = sign * sigma * sigmas[i];
            }
            k += 1;
        }
        for i in 0..d {
            row[i] += offset[i];
        }
    }
    data
}

// ── Registry ─────────────────────────────────────────────────

use crate::bench::events::INDICES_MACRO_EVENTS;
use crate::bench::loaders::{
    align_assets, load_indices_1m, load_real_csv, parse_iso_date, prepare_real_csvs, AssetSeries,
    ASSETS, INDICES_DEFAULT_SYMBOLS,
};
use crate::eval::{all_scenarios, heavy_tail_scenarios, Rng, Scenario};

pub const KNOWN_EVENTS: &[(&str, &str)] = &[
    ("SVB / USDC depeg", "2023-03-10"),
    ("Binance/CFTC suit", "2023-03-27"),
    ("BTC ETF approval", "2024-01-10"),
    ("Bitcoin halving", "2024-04-19"),
    ("US election", "2024-11-05"),
];

/// 48h hit window for the 5-macro tape, in seconds. Hourly tape;
/// macro shocks propagate within hours.
pub const CRYPTO_MACRO_WINDOW_SECS: i64 = 48 * 3600;

pub struct FixtureRegistry;

impl FixtureRegistry {
    /// Wraps `eval::all_scenarios()`. Each `Scenario` becomes a univariate
    /// `Fixture` with `d=1`. Margin defaults to 30 indices (synthetic
    /// scenarios use this in the existing eval suites).
    pub fn synthetic() -> Vec<Fixture> {
        all_scenarios().into_iter().map(scenario_to_fixture).collect()
    }

    pub fn synthetic_heavy_tail() -> Vec<Fixture> {
        heavy_tail_scenarios()
            .into_iter()
            .map(scenario_to_fixture)
            .collect()
    }

    pub fn crypto_macro_5() -> Result<Fixture, FixtureError> {
        prepare_real_csvs()?;
        let mut per_asset: Vec<AssetSeries> = Vec::new();
        for &asset in ASSETS {
            let (e, y) = load_real_csv(&format!("{asset}-logret"))?;
            per_asset.push((asset.into(), e, y));
        }
        let (epochs, tape) = align_assets(&per_asset, ASSETS.len());
        let n = tape.len();
        if n == 0 {
            return Err(FixtureError::Malformed("empty aligned tape".into()));
        }
        let parquet_max = *epochs.last().unwrap();
        let mut gt: Vec<usize> = Vec::new();
        for (_label, date) in KNOWN_EVENTS {
            let target = parse_iso_date(date);
            if target > parquet_max {
                continue;
            }
            // Nearest-bar lookup on a sorted epoch vector.
            let pos = epochs
                .binary_search(&target)
                .unwrap_or_else(|i| i.min(n - 1));
            gt.push(pos);
        }
        Ok(Fixture {
            name: "crypto_macro_5".into(),
            version: 1,
            d: ASSETS.len(),
            data: tape,
            epochs: Some(epochs),
            ground_truth: gt,
            seed: None,
            margin: 48, // hourly bars ↔ 48h window
        })
    }

    /// Inject `n_inj` synthetic step shifts of `sigma · σ_global` into
    /// the real BTC/ETH/SOL hourly tape at random epochs, dodging the
    /// 5 KNOWN_EVENTS by ±7 days and keeping injections ≥ 24h apart.
    /// Each chosen index marks a CP: from that bar onward, every
    /// channel's offset toggles by `±sigma · σ_i` until the next CP.
    /// The signs alternate, so the cumulative offset is a square-wave
    /// riding on the real returns -- piecewise-constant between CPs,
    /// which matches the regime-shift semantics every CP detector in
    /// this crate is designed for.
    ///
    /// Yields exactly `chosen.len()` CPs on real-data structure.
    /// Single-bar impulses (the previous implementation) are
    /// undetectable by step-shift detectors and would gut statistical
    /// power.
    pub fn anomaly_injected_crypto(
        seed: u64,
        n_inj: usize,
        sigma: f64,
    ) -> Result<Fixture, FixtureError> {
        let base = Self::crypto_macro_5()?;
        let n = base.data.len();
        let d = base.d;
        let epochs = base.epochs.as_ref().expect("crypto_macro_5 has epochs");
        let buffer = 7 * 86_400i64;

        let event_epochs: Vec<i64> = KNOWN_EVENTS
            .iter()
            .map(|(_, date)| parse_iso_date(date))
            .collect();

        // Per-channel global σ from the underlying tape.
        let mut sigmas = vec![0f64; d];
        for i in 0..d {
            let mean: f64 = base.data.iter().map(|r| r[i]).sum::<f64>() / n as f64;
            let var: f64 =
                base.data.iter().map(|r| (r[i] - mean).powi(2)).sum::<f64>() / n.max(1) as f64;
            sigmas[i] = var.sqrt();
        }

        let mut rng = Rng::new(seed);
        let min_sep = 24usize; // hourly: keep CPs ≥ 1 day apart
        let mut chosen: Vec<usize> = Vec::with_capacity(n_inj);
        let mut attempts = 0usize;
        let max_attempts = n_inj * 50;
        while chosen.len() < n_inj && attempts < max_attempts {
            attempts += 1;
            let r = rng.uniform();
            let idx = ((r * n as f64) as usize).min(n - 1);
            // Reserve the first / last min_sep bars so a step shift has
            // room on both sides for detectors to estimate baselines.
            if idx < min_sep || idx > n - min_sep {
                continue;
            }
            let e = epochs[idx];
            if event_epochs.iter().any(|ev| (e - ev).abs() <= buffer) {
                continue;
            }
            if chosen
                .iter()
                .any(|&c| (c as i64 - idx as i64).unsigned_abs() < min_sep as u64)
            {
                continue;
            }
            chosen.push(idx);
        }
        chosen.sort_unstable();

        let data = step_shift_inject(&base.data, &chosen, sigma, &sigmas);

        Ok(Fixture {
            name: format!("anomaly_injected_crypto_v1_seed{seed}"),
            version: 1,
            d,
            data,
            epochs: base.epochs,
            ground_truth: chosen,
            seed: Some(seed),
            margin: 48,
        })
    }

    /// US-equity / vol indices joint fixture. Hourly log-returns
    /// across `symbols` (default: `INDICES_DEFAULT_SYMBOLS` = SPX,
    /// NDX, DJI, VIX). Ground truth = `INDICES_MACRO_EVENTS` dates
    /// projected to bar indices via nearest-bar lookup; events
    /// outside the parquet's epoch window are dropped.
    ///
    /// Margin is 4 hours (US session resolution -- macro releases
    /// land within 1 candle and propagate by intraday close).
    /// Verdict-grade by construction: events_count ≥ 50 (compile-time
    /// asserted in `events.rs`); after window-clipping the test
    /// `indices_macro_v1_has_above_floor_events` re-asserts the
    /// runtime count.
    pub fn indices_macro_v1(symbols: &[&str]) -> Result<Fixture, FixtureError> {
        let symbols: Vec<&str> = if symbols.is_empty() {
            INDICES_DEFAULT_SYMBOLS.to_vec()
        } else {
            symbols.to_vec()
        };
        let per_asset = load_indices_1m(&symbols)?;
        let d = per_asset.len();
        let (epochs, tape) = align_assets(&per_asset, d);
        let n = tape.len();
        if n == 0 {
            return Err(FixtureError::Malformed(
                "empty aligned indices tape".into(),
            ));
        }
        let parquet_max = *epochs.last().unwrap();
        let parquet_min = *epochs.first().unwrap();
        let mut gt: Vec<usize> = Vec::new();
        for (_cat, _label, date) in INDICES_MACRO_EVENTS {
            let target = parse_iso_date(date);
            if target < parquet_min || target > parquet_max {
                continue;
            }
            let pos = epochs
                .binary_search(&target)
                .unwrap_or_else(|i| i.min(n - 1));
            gt.push(pos);
        }
        // Dedupe (collisions when two events share a date, e.g.
        // FOMC + BOJ on the same day in Oct 2025).
        gt.sort_unstable();
        gt.dedup();

        let symbol_slug = symbols
            .iter()
            .map(|s| s.trim_start_matches("I:"))
            .collect::<Vec<_>>()
            .join("-")
            .to_lowercase();
        Ok(Fixture {
            name: format!("indices_macro_v1_{symbol_slug}"),
            version: 1,
            d,
            data: tape,
            epochs: Some(epochs),
            ground_truth: gt,
            seed: None,
            margin: 4,
        })
    }

    pub fn anomaly_injected_indices(
        symbols: &[&str],
        seed: u64,
        n_inj: usize,
        sigma: f64,
    ) -> Result<Fixture, FixtureError> {
        let base = Self::indices_macro_v1(symbols)?;
        let n = base.data.len();
        let d = base.d;
        let epochs = base.epochs.as_ref().expect("indices_macro_v1 has epochs");
        let buffer = 86_400i64;

        let event_epochs: Vec<i64> = INDICES_MACRO_EVENTS
            .iter()
            .map(|(_, _, date)| parse_iso_date(date))
            .collect();

        let mut sigmas = vec![0f64; d];
        for i in 0..d {
            let mean: f64 = base.data.iter().map(|r| r[i]).sum::<f64>() / n as f64;
            let var: f64 =
                base.data.iter().map(|r| (r[i] - mean).powi(2)).sum::<f64>() / n.max(1) as f64;
            sigmas[i] = var.sqrt();
        }

        let mut rng = Rng::new(seed);
        let min_sep = 8usize;
        let mut chosen: Vec<usize> = Vec::with_capacity(n_inj);
        let mut attempts = 0usize;
        let max_attempts = n_inj * 50;
        while chosen.len() < n_inj && attempts < max_attempts {
            attempts += 1;
            let r = rng.uniform();
            let idx = ((r * n as f64) as usize).min(n - 1);
            if idx < min_sep || idx > n - min_sep {
                continue;
            }
            let e = epochs[idx];
            if event_epochs.iter().any(|ev| (e - ev).abs() <= buffer) {
                continue;
            }
            if chosen
                .iter()
                .any(|&c| (c as i64 - idx as i64).unsigned_abs() < min_sep as u64)
            {
                continue;
            }
            chosen.push(idx);
        }
        chosen.sort_unstable();

        let data = step_shift_inject(&base.data, &chosen, sigma, &sigmas);

        let symbol_slug = symbols
            .iter()
            .map(|s| s.trim_start_matches("I:"))
            .collect::<Vec<_>>()
            .join("-")
            .to_lowercase();
        let name = if symbols.is_empty() {
            format!(
                "anomaly_injected_indices_v1_{}_seed{seed}",
                INDICES_DEFAULT_SYMBOLS
                    .iter()
                    .map(|s| s.trim_start_matches("I:"))
                    .collect::<Vec<_>>()
                    .join("-")
                    .to_lowercase()
            )
        } else {
            format!("anomaly_injected_indices_v1_{symbol_slug}_seed{seed}")
        };
        Ok(Fixture {
            name,
            version: 1,
            d,
            data,
            epochs: base.epochs,
            ground_truth: chosen,
            seed: Some(seed),
            margin: 4,
        })
    }
}

fn scenario_to_fixture(s: Scenario) -> Fixture {
    let t = s.data.len();
    let data: Vec<Vec<f64>> = s.data.into_iter().map(|x| vec![x]).collect();
    let f = Fixture {
        name: s.name.to_string(),
        version: 1,
        d: 1,
        data,
        epochs: None,
        ground_truth: s.ground_truth,
        seed: None,
        margin: 30,
    };
    debug_assert_eq!(f.len(), t, "scenario→fixture length mismatch");
    f
}
