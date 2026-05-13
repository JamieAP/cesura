//! `cesura info` -- detector discovery surface for agents.
//!
//! Single JSON document, hand-written. Drifts at constructor speed
//! (detector taxonomy moves at research-arc speed, so this is fine).

use anyhow::Result;
use clap::Args;
use serde::Serialize;

use crate::runtime::DetectorKind;

#[derive(Args, Debug)]
pub struct InfoArgs {
    /// Filter to a single detector kind. Omit to dump every kind.
    #[arg(long, value_enum)]
    pub kind: Option<DetectorKind>,
}

#[derive(Serialize)]
struct InfoOutput<'a> {
    cesura_version: &'static str,
    detectors: Vec<DetectorInfo<'a>>,
}

#[derive(Serialize)]
struct DetectorInfo<'a> {
    kind: &'a str,
    obs_shape: &'a str,
    params: Vec<ParamInfo<'a>>,
    notes: &'a str,
}

#[derive(Serialize)]
struct ParamInfo<'a> {
    name: &'a str,
    #[serde(rename = "type")]
    ty: &'a str,
    default: serde_json::Value,
    doc: &'a str,
}

fn all_detectors() -> Vec<DetectorInfo<'static>> {
    vec![
        DetectorInfo {
            kind: "streaming",
            obs_shape: "scalar",
            params: vec![
                ParamInfo {
                    name: "lambda",
                    ty: "f64",
                    default: serde_json::json!(200.0),
                    doc: "Expected run length (hazard = 1/lambda).",
                },
                ParamInfo {
                    name: "max-rl",
                    ty: "usize",
                    default: serde_json::json!(250),
                    doc: "Maximum tracked run length.",
                },
            ],
            notes: "Univariate BOCPD with NIG conjugate prior + MAP-drop trigger.",
        },
        #[cfg(feature = "joint-detection")]
        DetectorInfo {
            kind: "chen-wu",
            obs_shape: "scalar",
            params: vec![
                ParamInfo {
                    name: "p0",
                    ty: "f64",
                    default: serde_json::json!(0.001),
                    doc: "Prior CP probability per step.",
                },
                ParamInfo {
                    name: "q0",
                    ty: "f64",
                    default: serde_json::json!(0.05),
                    doc: "Prior anomaly-end probability conditional on open anomaly.",
                },
                ParamInfo {
                    name: "delta-t",
                    ty: "usize",
                    default: serde_json::json!(20),
                    doc: "Maximum collective-anomaly duration.",
                },
                ParamInfo {
                    name: "lambda-a",
                    ty: "f64",
                    default: serde_json::json!(0.999),
                    doc: "Anomaly alarm threshold.",
                },
                ParamInfo {
                    name: "lambda-c",
                    ty: "f64",
                    default: serde_json::json!(0.999),
                    doc: "CP alarm threshold.",
                },
            ],
            notes: "Chen & Wu (2025) joint CP + collective-anomaly detector. \
                    CollectiveAnomaly fires emit as kind=\"anomaly\" at the end \
                    index of the anomalous segment.",
        },
        DetectorInfo {
            kind: "sum-cusum",
            obs_shape: "vector",
            params: vec![
                ParamInfo {
                    name: "d",
                    ty: "usize",
                    default: serde_json::json!(4),
                    doc: "Number of streams (observation dim).",
                },
                ParamInfo {
                    name: "lambda",
                    ty: "f64",
                    default: serde_json::json!(200.0),
                    doc: "Per-stream BOCPD lambda.",
                },
                ParamInfo {
                    name: "max-rl",
                    ty: "usize",
                    default: serde_json::json!(250),
                    doc: "Per-stream BOCPD max_run_length.",
                },
                ParamInfo {
                    name: "threshold",
                    ty: "f64",
                    default: serde_json::json!(0.1),
                    doc: "Sum-CUSUM fire threshold.",
                },
            ],
            notes: "Dense-change-optimal multi-stream aggregator (Mei 2010).",
        },
        DetectorInfo {
            kind: "hc",
            obs_shape: "vector",
            params: vec![
                ParamInfo {
                    name: "d",
                    ty: "usize",
                    default: serde_json::json!(4),
                    doc: "Number of streams.",
                },
                ParamInfo {
                    name: "lambda",
                    ty: "f64",
                    default: serde_json::json!(200.0),
                    doc: "Per-stream BOCPD lambda.",
                },
                ParamInfo {
                    name: "max-rl",
                    ty: "usize",
                    default: serde_json::json!(250),
                    doc: "Per-stream BOCPD max_run_length.",
                },
                ParamInfo {
                    name: "threshold",
                    ty: "f64",
                    default: serde_json::json!(3.0),
                    doc: "Higher-Criticism fire threshold.",
                },
                ParamInfo {
                    name: "persistence",
                    ty: "usize",
                    default: serde_json::json!(2),
                    doc: "Consecutive above-threshold steps required.",
                },
            ],
            notes: "Sparse-aware Higher-Criticism aggregator (Gong-Kipnis-Xie 2024).",
        },
        DetectorInfo {
            kind: "filter-tick",
            obs_shape: "vector",
            params: vec![
                ParamInfo {
                    name: "d",
                    ty: "usize",
                    default: serde_json::json!(4),
                    doc: "Number of streams.",
                },
                ParamInfo {
                    name: "k",
                    ty: "usize",
                    default: serde_json::json!(2),
                    doc: "k-of-d threshold count.",
                },
                ParamInfo {
                    name: "threshold",
                    ty: "f64",
                    default: serde_json::json!(0.5),
                    doc: "Per-stream score threshold.",
                },
            ],
            notes: "k-of-d filter over per-stream scores.",
        },
        DetectorInfo {
            kind: "dm-bocd",
            obs_shape: "vector",
            params: vec![
                ParamInfo {
                    name: "d",
                    ty: "usize",
                    default: serde_json::json!(4),
                    doc: "Observation dimension.",
                },
                ParamInfo {
                    name: "lambda",
                    ty: "f64",
                    default: serde_json::json!(200.0),
                    doc: "Expected run length.",
                },
                ParamInfo {
                    name: "max-rl",
                    ty: "usize",
                    default: serde_json::json!(250),
                    doc: "Max tracked run length.",
                },
            ],
            notes: "Multivariate Dm-BOCD (Altamirano-Briol-Knoblauch ICML 2023). \
                    Streaming variant; emits with 20-bar lookahead latency.",
        },
    ]
}

pub fn run(args: InfoArgs) -> Result<()> {
    let mut detectors = all_detectors();
    // Build the typed-kind allow-list so adding a new DetectorKind variant
    // forces an info.rs update via the assertion below.
    let known_kinds: Vec<&'static str> =
        DetectorKind::all().iter().map(|k| k.as_wire()).collect();
    detectors.retain(|d| known_kinds.contains(&d.kind));
    if let Some(k) = args.kind {
        let wire = k.as_wire();
        detectors.retain(|d| d.kind == wire);
        if detectors.is_empty() {
            anyhow::bail!("no info entry for kind: {wire}");
        }
    }
    let out = InfoOutput {
        cesura_version: env!("CARGO_PKG_VERSION"),
        detectors,
    };
    println!("{}", serde_json::to_string(&out)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn info_covers_every_runtime_detector_kind() {
        // Drift guard: every variant in DetectorKind::all() must have a
        // matching entry in info.rs hand-written table. Adding a kind
        // without updating info.rs fails this test loudly.
        let info_kinds: std::collections::HashSet<&str> =
            all_detectors().iter().map(|d| d.kind).collect();
        for kind in DetectorKind::all() {
            assert!(
                info_kinds.contains(kind.as_wire()),
                "DetectorKind::{:?} ({}) has no info entry",
                kind,
                kind.as_wire()
            );
        }
    }
}
