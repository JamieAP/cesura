//! Parquet/CSV/date helpers shared by the example emitters.
//! This version uses a fixed parquet path; it has no environment-variable
//! path override.
//!
//! Parquet ingestion shells out to `uv run python` with a small polars
//! script (no Rust parquet dep). The script writes per-asset hourly
//! log-return CSVs to `/tmp/cesura-<asset>-logret.csv`; subsequent
//! runs reuse them via a `Once`-guarded skip.

use std::fs;
use std::path::Path;
use std::process::Command;
use std::sync::Once;

use crate::bench::fixture::FixtureError;

pub const BTC_PARQUET: &str = "data/crypto_macro_1m.parquet";
pub const TMP_DIR: &str = "/tmp";
pub const ASSETS: &[&str] = &["btc", "eth", "sol"];

/// Per-asset tape: `(name, epochs, log-returns)`.
pub type AssetSeries = (String, Vec<i64>, Vec<f64>);

static PREPARE_REAL: Once = Once::new();

/// Idempotent prepare: writes `cesura-<asset>-logret.csv` to `TMP_DIR`
/// if absent. Errors out if the parquet file is not present.
pub fn prepare_real_csvs() -> Result<(), FixtureError> {
    if ASSETS
        .iter()
        .all(|n| Path::new(&format!("{TMP_DIR}/cesura-{n}-logret.csv")).exists())
    {
        return Ok(());
    }
    if !Path::new(BTC_PARQUET).exists() {
        return Err(FixtureError::Missing(format!("parquet not found: {BTC_PARQUET}")));
    }
    let mut err: Option<String> = None;
    PREPARE_REAL.call_once(|| {
        let py = format!(
            r#"
import polars as pl
src = pl.scan_parquet("{BTC_PARQUET}").select(["timestamp_ms", "btc_vwap", "eth_vwap", "sol_vwap"])
def hourly_logret(col):
    return (
        src.filter(pl.col(col).is_not_null() & pl.col(col).is_finite() & (pl.col(col) > 0))
           .with_columns((pl.col("timestamp_ms") // 3600_000).alias("hour"))
           .group_by("hour", maintain_order=True)
           .agg(pl.col(col).last().alias("v"))
           .sort("hour")
           .with_columns([
               (pl.col("hour") * 3600).alias("epoch_s"),
               pl.col("v").log().diff().alias("y"),
           ])
           .filter(pl.col("y").is_not_null() & pl.col("y").is_finite())
           .select(["epoch_s", "y"])
           .collect()
    )
hourly_logret("btc_vwap").write_csv("{TMP_DIR}/cesura-btc-logret.csv", include_header=False)
hourly_logret("eth_vwap").write_csv("{TMP_DIR}/cesura-eth-logret.csv", include_header=False)
hourly_logret("sol_vwap").write_csv("{TMP_DIR}/cesura-sol-logret.csv", include_header=False)
"#
        );
        let res = Command::new("uv")
            .args(["run", "--with", "polars", "--no-project", "python3", "-c", &py])
            .env("POLARS_MAX_THREADS", "1")
            .env("RAYON_NUM_THREADS", "1")
            .output();
        match res {
            Err(e) => err = Some(format!("uv spawn: {e}")),
            Ok(out) if !out.status.success() => {
                err = Some(format!("polars: {}", String::from_utf8_lossy(&out.stderr)))
            }
            Ok(_) => {}
        }
    });
    if let Some(e) = err {
        return Err(FixtureError::Io(e));
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
    fn align_assets_drops_unmatched_epochs() {
        let a: AssetSeries = ("a".into(), vec![1, 2, 3, 4], vec![1.0, 2.0, 3.0, 4.0]);
        let b: AssetSeries = ("b".into(), vec![2, 3, 5], vec![20.0, 30.0, 50.0]);
        let c: AssetSeries = ("c".into(), vec![1, 2, 3, 5], vec![100.0, 200.0, 300.0, 500.0]);
        let (epochs, tape) = align_assets(&[a, b, c], 3);
        assert_eq!(epochs, vec![2, 3]);
        assert_eq!(tape, vec![vec![2.0, 20.0, 200.0], vec![3.0, 30.0, 300.0]]);
    }
}
