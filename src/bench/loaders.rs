//! Local Parquet/CSV and date helpers for optional evaluation fixtures.
//! This version uses fixed paths under `data/`; datasets are supplied separately.
//! Parquet ingestion uses native Polars and may create derived CSVs in /tmp.

use std::fs;
use std::path::Path;

use crate::bench::fixture::FixtureError;

pub const BTC_PARQUET: &str = "data/crypto_macro_1m.parquet";
pub const TMP_DIR: &str = "/tmp";
pub const ASSETS: &[&str] = &["btc", "eth", "sol"];

/// Directory of per-day parquets for US equity / vol indices. Schema:
/// `timestamp:int64 (ms), symbol:str, open/high/low/close:float64`.
/// Filtered + hour-bucketed by [`load_indices_1m`].
pub const INDICES_PARQUET_DIR: &str =
    "data/indices_1m";

/// Default symbols for [`load_indices_1m`] / [`crate::bench::FixtureRegistry::indices_macro_v1`].
/// 4 streams: SPX (large-cap), NDX (tech), DJI (industrials), VIX (vol).
pub const INDICES_DEFAULT_SYMBOLS: &[&str] = &["I:SPX", "I:NDX", "I:DJI", "I:VIX"];

/// Directory of per-day Hydromancer Reservoir parquets for Hyperliquid
/// 1s OHLCV. Layout: `<dir>/date=YYYY-MM-DD/candles.parquet`. Files are
/// downloaded ahead-of-time via:
///
/// ```bash
/// aws s3 cp \
///   s3://hydromancer-reservoir/by_dex/hyperliquid/candles/1s/date=YYYY-MM-DD/candles.parquet \
///   <dir>/date=YYYY-MM-DD/candles.parquet \
///   --request-payer requester --region ap-northeast-1
/// ```
///
/// Expected source schema:
///   `coin: string, dex: string, asset_class: string ∈ {perp, spot},
///    base_symbol, quote_symbol, timestamp: timestamp[ms,UTC],
///    open/high/low/close/volume/volume_quote: decimal128(20,10),
///    trade_count: uint32`.
pub const HYPERLIQUID_PARQUET_DIR: &str =
    "data/hyperliquid_1s";

/// Default symbols for [`load_hyperliquid_1s`] / [`crate::bench::FixtureRegistry::hyperliquid_1s_v1`].
pub const HYPERLIQUID_DEFAULT_SYMBOLS: &[&str] = &["BTC"];

/// Per-asset tape: `(name, epochs, log-returns)`.
pub type AssetSeries = (String, Vec<i64>, Vec<f64>);

/// Idempotent prepare: writes `cesura-<asset>-logret.csv` to `TMP_DIR`
/// if absent. Errors out if the parquet file is not present. Uses
/// native polars without a subprocess, matching [`load_indices_1m`].
pub fn prepare_real_csvs() -> Result<(), FixtureError> {
    if ASSETS
        .iter()
        .all(|n| Path::new(&format!("{TMP_DIR}/cesura-{n}-logret.csv")).exists())
    {
        return Ok(());
    }
    if !Path::new(BTC_PARQUET).exists() {
        return Err(FixtureError::Missing(format!(
            "parquet not found: {BTC_PARQUET}"
        )));
    }
    use polars::prelude::*;

    let pl_path = PlPath::new(BTC_PARQUET);
    let scan_args = ScanArgsParquet::default();
    let scan = LazyFrame::scan_parquet(pl_path, scan_args)
        .map_err(|e| FixtureError::Io(format!("scan_parquet {BTC_PARQUET}: {e}")))?;

    // Mirror the original Python: per asset, filter + bucket + log-diff + write.
    for asset in ASSETS {
        let csv_path = format!("{TMP_DIR}/cesura-{asset}-logret.csv");
        if Path::new(&csv_path).exists() {
            continue;
        }
        let col_name = format!("{asset}_vwap");
        let agg_lf = scan
            .clone()
            .select([col("timestamp_ms"), col(col_name.as_str())])
            .filter(
                col(col_name.as_str())
                    .is_not_null()
                    .and(col(col_name.as_str()).gt(lit(0.0))),
            )
            .with_column((col("timestamp_ms") / lit(3_600_000i64)).alias("hour"))
            .sort_by_exprs([col("timestamp_ms")], SortMultipleOptions::default())
            .group_by_stable([col("hour")])
            .agg([col(col_name.as_str()).last().alias("v")])
            .sort_by_exprs([col("hour")], SortMultipleOptions::default())
            .with_columns([
                (col("hour") * lit(3600i64)).alias("epoch_s"),
                col("v")
                    .log(std::f64::consts::E)
                    .diff(lit(1), Default::default())
                    .alias("y"),
            ])
            .filter(col("y").is_not_null().and(col("y").is_finite()))
            .select([col("epoch_s"), col("y")]);
        let mut df = agg_lf
            .collect()
            .map_err(|e| FixtureError::Io(format!("aggregate {asset}: {e}")))?;
        let mut writer = std::fs::File::create(&csv_path)
            .map_err(|e| FixtureError::Io(format!("create {csv_path}: {e}")))?;
        CsvWriter::new(&mut writer)
            .include_header(false)
            .finish(&mut df)
            .map_err(|e| FixtureError::Io(format!("write_csv {csv_path}: {e}")))?;
    }
    Ok(())
}

/// Read a per-asset CSV produced by [`prepare_real_csvs`].
pub fn load_real_csv(name: &str) -> Result<(Vec<i64>, Vec<f64>), FixtureError> {
    let raw = fs::read_to_string(format!("{TMP_DIR}/cesura-{name}.csv"))
        .map_err(|e| FixtureError::Io(format!("read {name}: {e}")))?;
    let mut e = Vec::new();
    let mut y = Vec::new();
    for line in raw.lines() {
        let mut it = line.split(',');
        e.push(
            it.next()
                .ok_or_else(|| FixtureError::Malformed("missing epoch".into()))?
                .parse()
                .map_err(|err| FixtureError::Malformed(format!("epoch: {err}")))?,
        );
        y.push(
            it.next()
                .ok_or_else(|| FixtureError::Malformed("missing y".into()))?
                .parse()
                .map_err(|err| FixtureError::Malformed(format!("y: {err}")))?,
        );
    }
    Ok((e, y))
}

/// `YYYY-MM-DD` to Unix-seconds (UTC midnight). Howard Hinnant's
/// civil-from-days formula. Verbatim from the example helpers.
pub fn parse_iso_date(s: &str) -> i64 {
    let mut it = s.split('-');
    let y: i64 = it.next().unwrap().parse().unwrap();
    let m: i64 = it.next().unwrap().parse().unwrap();
    let d: i64 = it.next().unwrap().parse().unwrap();
    let (m_adj, y_adj) = if m <= 2 { (m + 12, y - 1) } else { (m, y) };
    let era = y_adj.div_euclid(400);
    let yoe = y_adj.rem_euclid(400);
    let doy = (153 * (m_adj - 3) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days_since_epoch = era * 146097 + doe - 719468;
    days_since_epoch * 86400
}

/// Inner-join `d` per-asset tapes on epoch. Drops any timestamp that
/// is missing in any one asset. The output `tape[t]` has `d` entries
/// in input order. `_d` is informational; the join uses `per_asset.len()`.
pub fn align_assets(per_asset: &[AssetSeries], _d: usize) -> (Vec<i64>, Vec<Vec<f64>>) {
    let cap = per_asset.iter().map(|a| a.1.len()).min().unwrap_or(0);
    let mut epochs = Vec::with_capacity(cap);
    let mut tape: Vec<Vec<f64>> = Vec::with_capacity(cap);
    let mut idx = vec![0usize; per_asset.len()];
    while idx.iter().zip(per_asset.iter()).all(|(&i, a)| i < a.1.len()) {
        let max_e = idx
            .iter()
            .zip(per_asset.iter())
            .map(|(&i, a)| a.1[i])
            .max()
            .unwrap();
        let mut all_match = true;
        for (i, asset) in idx.iter_mut().zip(per_asset.iter()) {
            while *i < asset.1.len() && asset.1[*i] < max_e {
                *i += 1;
            }
            if *i >= asset.1.len() || asset.1[*i] != max_e {
                all_match = false;
                break;
            }
        }
        if !all_match {
            continue;
        }
        let row: Vec<f64> = per_asset
            .iter()
            .zip(idx.iter())
            .map(|(asset, &i)| asset.2[i])
            .collect();
        epochs.push(max_e);
        tape.push(row);
        for i in &mut idx {
            *i += 1;
        }
    }
    (epochs, tape)
}

/// Load hourly log-returns from the indices_1m parquet shard for each
/// requested symbol. Aggregates 1m bars to hourly by `last(close)` per
/// `timestamp // 3_600_000`, then `log().diff()`. Returns one
/// `AssetSeries` per symbol in input order.
///
pub fn load_indices_1m(symbols: &[&str]) -> Result<Vec<AssetSeries>, FixtureError> {
    use polars::prelude::*;
    if symbols.is_empty() {
        return Err(FixtureError::Malformed("no symbols requested".into()));
    }
    if !Path::new(INDICES_PARQUET_DIR).exists() {
        return Err(FixtureError::Missing(format!(
            "indices parquet dir not found: {INDICES_PARQUET_DIR}"
        )));
    }
    let mut paths: Vec<std::path::PathBuf> = fs::read_dir(INDICES_PARQUET_DIR)
        .map_err(|e| FixtureError::Io(format!("read_dir {INDICES_PARQUET_DIR}: {e}")))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "parquet"))
        .collect();
    paths.sort();
    if paths.is_empty() {
        return Err(FixtureError::Missing(format!(
            "no parquet files in {INDICES_PARQUET_DIR}"
        )));
    }

    // Build symbol-IN filter. polars 0.50 doesn't expose `is_in` for
    // string lits without the `is_in` feature flag, so OR-fold matches.
    let mut syms_iter = symbols.iter();
    let first = syms_iter.next().unwrap();
    let mut filter_expr = col("symbol").eq(lit(*first));
    for s in syms_iter {
        filter_expr = filter_expr.or(col("symbol").eq(lit(*s)));
    }

    // Scan each parquet lazily, concat into a single LazyFrame.
    let scan_args = ScanArgsParquet::default();
    let mut lazy_frames: Vec<LazyFrame> = Vec::with_capacity(paths.len());
    for path in &paths {
        let pl_path = PlPath::new(&path.to_string_lossy());
        let lf = LazyFrame::scan_parquet(pl_path, scan_args.clone())
            .map_err(|e| FixtureError::Io(format!("scan_parquet {path:?}: {e}")))?
            .filter(filter_expr.clone())
            .filter(col("close").is_not_null().and(col("close").gt(lit(0.0))))
            .select([col("timestamp"), col("symbol"), col("close")]);
        lazy_frames.push(lf);
    }
    let combined_lf = polars::prelude::concat(
        &lazy_frames,
        UnionArgs {
            rechunk: false,
            parallel: true,
            ..Default::default()
        },
    )
    .map_err(|e| FixtureError::Io(format!("concat: {e}")))?;

    // Hour-bucket → last(close) per (symbol, hour) → log-diff over symbol.
    let agg_lf = combined_lf
        .with_column((col("timestamp") / lit(3_600_000i64)).alias("hour"))
        .sort_by_exprs(
            [col("symbol"), col("timestamp")],
            SortMultipleOptions::default(),
        )
        .group_by_stable([col("symbol"), col("hour")])
        .agg([col("close").last().alias("v")])
        .sort_by_exprs(
            [col("symbol"), col("hour")],
            SortMultipleOptions::default(),
        )
        .with_columns([(col("hour") * lit(3600i64)).alias("epoch_s")])
        .with_columns([col("v").log(std::f64::consts::E).diff(lit(1), Default::default()).over([col("symbol")]).alias("y")])
        .filter(col("y").is_not_null().and(col("y").is_finite()))
        .select([col("symbol"), col("epoch_s"), col("y")]);
    let agg = agg_lf
        .collect()
        .map_err(|e| FixtureError::Io(format!("aggregate: {e}")))?;

    // Project to per-symbol AssetSeries.
    let mut out: Vec<AssetSeries> = Vec::with_capacity(symbols.len());
    for sym in symbols {
        let mask = agg
            .column("symbol")
            .map_err(|e| FixtureError::Io(format!("col(symbol): {e}")))?
            .as_materialized_series()
            .equal(*sym)
            .map_err(|e| FixtureError::Io(format!("equal({sym}): {e}")))?;
        let f = agg
            .filter(&mask)
            .map_err(|e| FixtureError::Io(format!("filter({sym}): {e}")))?;
        let n = f.height();
        if n == 0 {
            return Err(FixtureError::Malformed(format!(
                "no rows for symbol {sym}"
            )));
        }
        let epochs: Vec<i64> = f
            .column("epoch_s")
            .map_err(|e| FixtureError::Io(format!("col(epoch_s): {e}")))?
            .i64()
            .map_err(|e| FixtureError::Malformed(format!("epoch_s.i64: {e}")))?
            .into_no_null_iter()
            .collect();
        let ys: Vec<f64> = f
            .column("y")
            .map_err(|e| FixtureError::Io(format!("col(y): {e}")))?
            .f64()
            .map_err(|e| FixtureError::Malformed(format!("y.f64: {e}")))?
            .into_no_null_iter()
            .collect();
        out.push(((*sym).to_string(), epochs, ys));
    }
    Ok(out)
}

///
/// Reads every `<dir>/date=YYYY-MM-DD/candles.parquet` it finds,
/// concatenates, filters `coin == sym AND asset_class == "perp"`,
/// projects `[timestamp, close]`, sorts by timestamp, forward-fills the
/// 1s grid, then `log().diff()` to log-returns.
///
/// Returns one `AssetSeries` per symbol in input order, where the
/// `epochs` entries are in **seconds** (matches [`load_indices_1m`]
/// convention; 1s granularity for sub-hourly tapes is still
/// representable). Empty-day files are silently skipped.
pub fn load_hyperliquid_1s(symbols: &[&str]) -> Result<Vec<AssetSeries>, FixtureError> {
    use polars::prelude::*;
    if symbols.is_empty() {
        return Err(FixtureError::Malformed("no symbols requested".into()));
    }
    if !Path::new(HYPERLIQUID_PARQUET_DIR).exists() {
        return Err(FixtureError::Missing(format!(
            "hyperliquid parquet dir not found: {HYPERLIQUID_PARQUET_DIR}"
        )));
    }

    // Discover per-day parquet files. Each `date=YYYY-MM-DD/` dir holds
    // exactly one `candles.parquet`.
    let mut paths: Vec<std::path::PathBuf> = Vec::new();
    for entry in fs::read_dir(HYPERLIQUID_PARQUET_DIR)
        .map_err(|e| FixtureError::Io(format!("read_dir {HYPERLIQUID_PARQUET_DIR}: {e}")))?
    {
        let entry = entry.map_err(|e| FixtureError::Io(format!("read_dir entry: {e}")))?;
        let p = entry.path();
        if p.is_dir() {
            let f = p.join("candles.parquet");
            if f.exists() {
                paths.push(f);
            }
        }
    }
    paths.sort();
    if paths.is_empty() {
        return Err(FixtureError::Missing(format!(
            "no candles.parquet files in {HYPERLIQUID_PARQUET_DIR}"
        )));
    }

    // Build a coin-IN filter (OR-fold over symbols).
    let mut syms_iter = symbols.iter();
    let first = syms_iter.next().unwrap();
    let mut coin_filter = col("coin").eq(lit(*first));
    for s in syms_iter {
        coin_filter = coin_filter.or(col("coin").eq(lit(*s)));
    }

    let scan_args = ScanArgsParquet::default();
    let mut lazy_frames: Vec<LazyFrame> = Vec::with_capacity(paths.len());
    for path in &paths {
        let pl_path = PlPath::new(&path.to_string_lossy());
        let lf = LazyFrame::scan_parquet(pl_path, scan_args.clone())
            .map_err(|e| FixtureError::Io(format!("scan_parquet {path:?}: {e}")))?
            .filter(coin_filter.clone())
            .filter(col("asset_class").eq(lit("perp")))
            .select([
                col("coin"),
                col("timestamp").cast(DataType::Int64).alias("timestamp_ms"),
                col("close").cast(DataType::Float64).alias("close_f64"),
            ]);
        lazy_frames.push(lf);
    }
    let combined_lf = polars::prelude::concat(
        &lazy_frames,
        UnionArgs {
            rechunk: false,
            parallel: true,
            ..Default::default()
        },
    )
    .map_err(|e| FixtureError::Io(format!("concat: {e}")))?;

    let collected = combined_lf
        .sort_by_exprs(
            [col("coin"), col("timestamp_ms")],
            SortMultipleOptions::default(),
        )
        .collect()
        .map_err(|e| FixtureError::Io(format!("collect: {e}")))?;

    // Per-symbol: extract (ts_ms, close), forward-fill 1s grid,
    // compute log-returns, convert ms→s epochs.
    let mut out: Vec<AssetSeries> = Vec::with_capacity(symbols.len());
    for sym in symbols {
        let mask = collected
            .column("coin")
            .map_err(|e| FixtureError::Io(format!("col(coin): {e}")))?
            .as_materialized_series()
            .equal(*sym)
            .map_err(|e| FixtureError::Io(format!("equal({sym}): {e}")))?;
        let f = collected
            .filter(&mask)
            .map_err(|e| FixtureError::Io(format!("filter({sym}): {e}")))?;
        let n = f.height();
        if n == 0 {
            return Err(FixtureError::Malformed(format!(
                "no rows for coin {sym}"
            )));
        }
        let ts_ms: Vec<i64> = f
            .column("timestamp_ms")
            .map_err(|e| FixtureError::Io(format!("col(timestamp_ms): {e}")))?
            .i64()
            .map_err(|e| FixtureError::Malformed(format!("timestamp_ms.i64: {e}")))?
            .into_no_null_iter()
            .collect();
        let closes: Vec<f64> = f
            .column("close_f64")
            .map_err(|e| FixtureError::Io(format!("col(close_f64): {e}")))?
            .f64()
            .map_err(|e| FixtureError::Malformed(format!("close_f64.f64: {e}")))?
            .into_no_null_iter()
            .collect();

        let (dense_epochs_s, dense_closes) = densify_seconds(&ts_ms, &closes);
        let logrets = log_diff(&dense_closes);
        // log_diff returns N-1 values; align epochs to the diff window
        // by dropping the first epoch (matches `load_indices_1m`'s
        // `is_not_null().and(is_finite())` filter at the end).
        let aligned_epochs: Vec<i64> = dense_epochs_s.iter().skip(1).copied().collect();
        out.push(((*sym).to_string(), aligned_epochs, logrets));
    }
    Ok(out)
}

/// Forward-fill a sparse `(ts_ms, close)` sequence to a dense 1s grid.
/// `ts_ms` must be sorted ascending. Returns `(epoch_s, close)` pairs
/// at 1s spacing covering `[ts_ms[0]/1000, ts_ms.last()/1000]`.
pub fn densify_seconds(ts_ms: &[i64], closes: &[f64]) -> (Vec<i64>, Vec<f64>) {
    if ts_ms.is_empty() {
        return (Vec::new(), Vec::new());
    }
    let start_s = ts_ms[0] / 1000;
    let end_s = ts_ms[ts_ms.len() - 1] / 1000;
    let n = (end_s - start_s + 1).max(0) as usize;
    let mut epochs = Vec::with_capacity(n);
    let mut out = Vec::with_capacity(n);
    let mut idx = 0usize;
    let mut last_close = closes[0];
    for s in start_s..=end_s {
        // Advance source cursor through all samples landing on or before this second.
        while idx < ts_ms.len() && ts_ms[idx] / 1000 <= s {
            last_close = closes[idx];
            idx += 1;
        }
        epochs.push(s);
        out.push(last_close);
    }
    (epochs, out)
}

/// `ln(x[i]) - ln(x[i-1])` over a slice. Returns `N-1` values. Skips
/// non-finite results (e.g. log of zero or negative) by replacing with
/// `0.0`; callers are expected to drop the first epoch.
pub fn log_diff(closes: &[f64]) -> Vec<f64> {
    if closes.len() < 2 {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(closes.len() - 1);
    for w in closes.windows(2) {
        let a = w[0];
        let b = w[1];
        let r = if a > 0.0 && b > 0.0 {
            b.ln() - a.ln()
        } else {
            0.0
        };
        out.push(if r.is_finite() { r } else { 0.0 });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_iso_date_matches_unix_reference() {
        assert_eq!(parse_iso_date("1970-01-01"), 0);
        assert_eq!(parse_iso_date("2024-01-10"), 1_704_844_800); // BTC ETF approval
        assert_eq!(parse_iso_date("2023-03-10"), 1_678_406_400); // SVB depeg
        assert_eq!(parse_iso_date("2024-04-19"), 1_713_484_800); // Bitcoin halving
        // 2000-03-01 is the Jan/Feb branch boundary in the days_from_civil
        // formula; guard against off-by-one.
        assert_eq!(parse_iso_date("2000-03-01"), 951_868_800);
        assert_eq!(parse_iso_date("2000-02-29"), 951_782_400);
    }

    #[test]
    fn densify_seconds_forward_fills_gap() {
        // Sparse: trades at t=1000ms (close=100) and t=4000ms (close=102).
        // Expected dense: t=1, 2, 3, 4 (epoch_s); close = 100, 100, 100, 102.
        let ts = [1000i64, 4000];
        let cl = [100.0f64, 102.0];
        let (e, c) = densify_seconds(&ts, &cl);
        assert_eq!(e, vec![1, 2, 3, 4]);
        assert_eq!(c, vec![100.0, 100.0, 100.0, 102.0]);
    }

    #[test]
    fn densify_seconds_collapses_same_second() {
        // Two trades land within the same UTC second: last close wins.
        let ts = [1500i64, 1700, 3000];
        let cl = [100.0f64, 101.0, 105.0];
        let (e, c) = densify_seconds(&ts, &cl);
        assert_eq!(e, vec![1, 2, 3]);
        // Second 1: last trade ≤1000ms is index 1 (price 101).
        assert_eq!(c[0], 101.0);
        // Second 2: same forward-fill.
        assert_eq!(c[1], 101.0);
        // Second 3: catches index 2.
        assert_eq!(c[2], 105.0);
    }

    #[test]
    fn densify_seconds_empty_input() {
        let (e, c) = densify_seconds(&[], &[]);
        assert!(e.is_empty() && c.is_empty());
    }

    #[test]
    fn log_diff_basic() {
        let r = log_diff(&[1.0, std::f64::consts::E, 1.0]);
        assert_eq!(r.len(), 2);
        assert!((r[0] - 1.0).abs() < 1e-12);   // ln(e/1) = 1
        assert!((r[1] - -1.0).abs() < 1e-12);  // ln(1/e) = -1
    }

    #[test]
    fn log_diff_zero_returns_zero() {
        // Zero or negative closes → 0.0 (loader's defensive fallback).
        let r = log_diff(&[0.0, 1.0, -1.0, 2.0]);
        assert_eq!(r, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn log_diff_short_input() {
        assert!(log_diff(&[]).is_empty());
        assert!(log_diff(&[100.0]).is_empty());
    }

    #[test]
    fn align_assets_drops_unmatched_epochs() {
        let a: AssetSeries = ("a".into(), vec![1, 2, 3, 4], vec![1.0, 2.0, 3.0, 4.0]);
        let b: AssetSeries = ("b".into(), vec![2, 3, 5], vec![20.0, 30.0, 50.0]);
        let c: AssetSeries = ("c".into(), vec![1, 2, 3, 5], vec![100.0, 200.0, 300.0, 500.0]);
        let (epochs, tape) = align_assets(&[a, b, c], 3);
        assert_eq!(epochs, vec![2, 3]);
        assert_eq!(tape, vec![vec![2.0, 20.0, 200.0], vec![3.0, 30.0, 300.0]]);
    }
}
