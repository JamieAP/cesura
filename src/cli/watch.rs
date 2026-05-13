//! `cesura watch` -- JSONL stdin → JSONL stdout, line per fire.
//!
//! Each emitted stdout line is one Claude Monitor notification.
//! Parse errors emit a `{"type":"parse_error",...}` line and continue
//! (Monitor channel must not die on a single bad line).

use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;

use anyhow::{anyhow, bail, Context, Result};
use clap::Args;
use serde::{Deserialize, Serialize};

use crate::canonical::recommended_streams;
use crate::dm_bocd::StreamingDmBocd;
use crate::multistream::{FilterTickAggregator, HcAggregator, SumCusumAggregator};
use crate::runtime::{DetectorKind, Fire, Obs, ObsShape, Runner, RunnerState};
use crate::streaming::StreamingDetector;
#[cfg(feature = "joint-detection")]
use crate::streaming_chen_wu::StreamingChenWuDetector;

#[derive(Args, Debug)]
pub struct WatchArgs {
    /// Detector kind.
    #[arg(long, value_enum)]
    pub kind: DetectorKind,

    /// Expected run length (BOCPD hazard = 1/lambda).
    #[arg(long, default_value_t = 200.0)]
    pub lambda: f64,

    /// Maximum tracked run length.
    #[arg(long, default_value_t = 250)]
    pub max_rl: usize,

    /// Observation dimension for multistream / multivariate detectors.
    #[arg(long, default_value_t = 4)]
    pub d: usize,

    /// k-of-d for filter-tick. Ignored for other kinds.
    #[arg(long, default_value_t = 2)]
    pub k: usize,

    /// Aggregator threshold (sum-cusum, hc, filter-tick).
    #[arg(long)]
    pub threshold: Option<f64>,

    /// HC persistence filter (consecutive above-threshold steps).
    #[arg(long)]
    pub persistence: Option<usize>,

    /// Chen-Wu prior CP probability per step.
    #[arg(long, default_value_t = 0.001)]
    pub p0: f64,

    /// Chen-Wu prior anomaly-end probability conditional on open anomaly.
    #[arg(long, default_value_t = 0.05)]
    pub q0: f64,

    /// Chen-Wu maximum collective-anomaly duration.
    #[arg(long, default_value_t = 20)]
    pub delta_t: usize,

    /// Chen-Wu anomaly alarm threshold.
    #[arg(long, default_value_t = 0.999)]
    pub lambda_a: f64,

    /// Chen-Wu CP alarm threshold.
    #[arg(long, default_value_t = 0.999)]
    pub lambda_c: f64,

    /// Drop fires below this confidence threshold. Default 0.0 = emit all.
    #[arg(long, default_value_t = 0.0)]
    pub min_confidence: f64,

    /// Load detector state from this JSON file (Runner snapshot) before reading stdin.
    #[arg(long)]
    pub state: Option<PathBuf>,

    /// On EOF, write the final Runner snapshot to this JSON file.
    #[arg(long)]
    pub snapshot_on_exit: Option<PathBuf>,
}

#[derive(Deserialize)]
struct ObsLine {
    x: ObsValue,
    #[serde(default)]
    t: Option<String>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum ObsValue {
    Scalar(f64),
    Vector(Vec<f64>),
}

#[derive(Serialize)]
struct FireLine<'a> {
    #[serde(rename = "type")]
    ty: &'a str,
    index: usize,
    confidence: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    shift_sigma: Option<f64>,
    streams: &'a [usize],
    kind: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    t: Option<&'a str>,
}

#[derive(Serialize)]
struct ParseErrorLine<'a> {
    #[serde(rename = "type")]
    ty: &'a str,
    line: usize,
    msg: String,
}

pub fn run(args: WatchArgs) -> Result<()> {
    let mut runner = build_runner(&args)?;
    let stdin = std::io::stdin();
    let reader = BufReader::new(stdin.lock());
    let stdout = std::io::stdout();
    let mut out = stdout.lock();

    for (line_no, line) in reader.lines().enumerate() {
        let line_no = line_no + 1;
        let line = line.with_context(|| format!("read stdin at line {line_no}"))?;
        if line.trim().is_empty() {
            continue;
        }
        let parsed: serde_json::Result<ObsLine> = serde_json::from_str(&line);
        let obs_line = match parsed {
            Ok(p) => p,
            Err(e) => {
                emit_parse_error(&mut out, line_no, e.to_string())?;
                continue;
            }
        };
        let obs = match (obs_line.x, runner.kind_enum().obs_shape()) {
            (ObsValue::Scalar(x), ObsShape::Scalar) => Obs::F64(x),
            (ObsValue::Vector(v), ObsShape::Vector) => {
                if v.len() != args.d {
                    emit_parse_error(
                        &mut out,
                        line_no,
                        format!("observation length {} does not match --d {}", v.len(), args.d),
                    )?;
                    continue;
                }
                Obs::Vec(v)
            }
            (got, _) => {
                emit_parse_error(
                    &mut out,
                    line_no,
                    format!(
                        "observation shape does not match kind={}: got {}",
                        runner.kind(),
                        match got {
                            ObsValue::Scalar(_) => "scalar",
                            ObsValue::Vector(_) => "vector",
                        }
                    ),
                )?;
                continue;
            }
        };
        let fires = runner.feed(obs);
        for f in fires {
            if f.confidence < args.min_confidence {
                continue;
            }
            emit_fire(&mut out, &f, obs_line.t.as_deref())?;
        }
    }

    if let Some(path) = args.snapshot_on_exit.as_ref() {
        let snap = runner
            .save()
            .map_err(|e| anyhow!("save snapshot: {e}"))?;
        let json = serde_json::to_string(&snap)?;
        std::fs::write(path, json).with_context(|| format!("write {}", path.display()))?;
    }

    Ok(())
}

fn emit_fire(out: &mut impl Write, f: &Fire, t: Option<&str>) -> Result<()> {
    let line = FireLine {
        ty: "fire",
        index: f.index,
        confidence: f.confidence,
        shift_sigma: f.shift_sigma,
        streams: &f.streams,
        kind: &f.kind,
        t,
    };
    writeln!(out, "{}", serde_json::to_string(&line)?)?;
    Ok(())
}

fn emit_parse_error(out: &mut impl Write, line: usize, msg: String) -> Result<()> {
    let line = ParseErrorLine {
        ty: "parse_error",
        line,
        msg,
    };
    writeln!(out, "{}", serde_json::to_string(&line)?)?;
    Ok(())
}

fn build_runner(args: &WatchArgs) -> Result<Runner> {
    if let Some(path) = args.state.as_ref() {
        let json = std::fs::read_to_string(path)
            .with_context(|| format!("read state file {}", path.display()))?;
        let state: RunnerState = serde_json::from_str(&json)
            .with_context(|| format!("parse state file {}", path.display()))?;
        let restored = Runner::restore(state).map_err(|e| anyhow!("restore: {e}"))?;
        if restored.kind_enum() != args.kind {
            bail!(
                "state file kind {} does not match --kind {}",
                restored.kind(),
                args.kind
            );
        }
        return Ok(restored);
    }
    match args.kind {
        DetectorKind::Streaming => {
            Ok(Runner::Uni(StreamingDetector::new(args.lambda, args.max_rl)))
        }
        #[cfg(feature = "joint-detection")]
        DetectorKind::ChenWu => Ok(Runner::UniChenWu(StreamingChenWuDetector::new(
            args.p0, args.q0, args.delta_t, args.lambda_a, args.lambda_c,
        ))),
        DetectorKind::SumCusum => {
            let agg = if args.lambda == 200.0 && args.max_rl == 250 && args.threshold.is_none() {
                recommended_streams(args.d)
            } else {
                let streams: Vec<StreamingDetector> = (0..args.d)
                    .map(|_| StreamingDetector::new(args.lambda, args.max_rl))
                    .collect();
                let agg = SumCusumAggregator::new(streams);
                match args.threshold {
                    Some(t) => agg.with_threshold(t),
                    None => agg,
                }
            };
            Ok(Runner::MultiSumCusum(agg))
        }
        DetectorKind::Hc => {
            let streams: Vec<StreamingDetector> = (0..args.d)
                .map(|_| StreamingDetector::new(args.lambda, args.max_rl))
                .collect();
            let mut agg = HcAggregator::new(streams);
            if let Some(t) = args.threshold {
                agg = agg.with_threshold(t);
            }
            if let Some(p) = args.persistence {
                agg = agg.with_persistence(p);
            }
            Ok(Runner::MultiHc(agg))
        }
        DetectorKind::FilterTick => {
            let threshold = args.threshold.unwrap_or(0.5);
            Ok(Runner::MultiFilterTick(FilterTickAggregator::new(
                args.d, args.k, threshold,
            )))
        }
        DetectorKind::DmBocd => Ok(Runner::DmBocd(StreamingDmBocd::new(
            args.d, args.lambda, args.max_rl,
        ))),
    }
}
