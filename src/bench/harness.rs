//! Run reports, JSON IO, verdict-floor classification, and a
//! markdown renderer used by the migrated examples.
//!
//! Verdict floor: a fixture with `n_events < min_events` (default 50)
//! never produces a `VerdictGrade` label. The 5-macro tape is a
//! sanity check by construction.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::bench::detector::CpDetector;
use crate::bench::fixture::Fixture;
use crate::bench::metrics::{evaluate, Metrics};

/// Default min ground-truth events for a `VerdictGrade` label.
/// Below this the fixture is treated as a sanity check only.
pub const DEFAULT_MIN_EVENTS: usize = 50;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Report {
    pub detector: String,
    pub fixture: String,
    pub fixture_version: u32,
    pub commit: String,
    pub seed: Option<u64>,
    pub metrics: Metrics,
    pub timing_us: u128,
    /// Unix seconds at run time.
    pub timestamp: i64,
    /// Per-CP stream attribution. `attribution[i]` is the streams
    /// (sorted ascending) that drove `cps[i]`. Empty per-CP for
    /// univariate / joint-aggregator detectors. Aligned with the CP
    /// list the metrics were computed over.
    #[serde(default)]
    pub attribution: Vec<Vec<usize>>,
}

pub fn run_bench<D: CpDetector + ?Sized>(d: &D, fix: &Fixture) -> Report {
    let t0 = Instant::now();
    let result = d.detect_full(fix);
    let timing_us = t0.elapsed().as_micros();
    let metrics = evaluate(&result.cps, fix);
    Report {
        detector: d.name(),
        fixture: fix.name.clone(),
        fixture_version: fix.version,
        commit: current_commit(),
        seed: fix.seed,
        metrics,
        timing_us,
        timestamp: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0),
        attribution: result.attribution,
    }
}

/// Writes one report to `<dir>/<fixture>/<detector>-<commit>.json`.
pub fn write_report(r: &Report, dir: &Path) -> io::Result<PathBuf> {
    let fixture_dir = dir.join(&r.fixture);
    fs::create_dir_all(&fixture_dir)?;
    let safe_det = sanitize(&r.detector);
    let safe_commit = sanitize(&r.commit);
    let path = fixture_dir.join(format!("{safe_det}-{safe_commit}.json"));
    let json = serde_json::to_string_pretty(r)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    fs::write(&path, json)?;
    Ok(path)
}

pub fn load_reports(dir: &Path) -> Vec<Report> {
    let mut out = Vec::new();
    let Ok(entries) = fs::read_dir(dir) else {
        return out;
    };
    for entry in entries.flatten() {
        let p = entry.path();
        if p.is_dir() {
            out.extend(load_reports(&p));
        } else if p.extension().map(|e| e == "json").unwrap_or(false) {
            if let Ok(s) = fs::read_to_string(&p) {
                if let Ok(r) = serde_json::from_str::<Report>(&s) {
                    out.push(r);
                }
            }
        }
    }
    out
}

pub fn render_markdown(reports: &[Report]) -> String {
    let mut s = String::new();
    s.push_str("| detector | fixture | n_events | n_cps | precision | recall | F1 | FAR | timing_µs |\n");
    s.push_str("|---|---|---:|---:|---:|---:|---:|---:|---:|\n");
    for r in reports {
        let m = &r.metrics;
        s.push_str(&format!(
            "| {} | {} (v{}) | {} | {} | {:.3} | {:.3} | {:.3} | {} | {} |\n",
            r.detector,
            r.fixture,
            r.fixture_version,
            m.n_events,
            m.n_cps,
            m.precision,
            m.recall,
            m.f1,
            m.far,
            r.timing_us,
        ));
    }
    s
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VerdictLabel {
    VerdictGrade,
    SanityCheck,
    NoiseRange,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Verdict {
    pub label: VerdictLabel,
    pub reason: String,
    pub n_events_used: usize,
}

pub fn power_label(n_events: usize, min_events: usize) -> VerdictLabel {
    if n_events < min_events {
        VerdictLabel::SanityCheck
    } else {
        VerdictLabel::VerdictGrade
    }
}

/// Human-facing string form of [`VerdictLabel`]. Stable wording used
/// across regenerated markdowns and banners.
pub fn label_str(label: VerdictLabel) -> &'static str {
    match label {
        VerdictLabel::VerdictGrade => "verdict-grade",
        VerdictLabel::SanityCheck => "sanity check",
        VerdictLabel::NoiseRange => "noise range",
    }
}

/// `SanityCheck` if the smallest `n_events` across reports is below
/// `min_events`. `VerdictGrade` only when every report has enough
/// ground-truth power. Caller may downgrade `VerdictGrade` to
/// `NoiseRange` based on metric deltas; this function never does.
pub fn classify_verdict(reports: &[Report], min_events: usize) -> Verdict {
    let n_min = reports
        .iter()
        .map(|r| r.metrics.n_events)
        .min()
        .unwrap_or(0);
    if n_min < min_events {
        Verdict {
            label: VerdictLabel::SanityCheck,
            reason: format!(
                "n_events={n_min} < min_events={min_events} \u{2192} sanity check only"
            ),
            n_events_used: n_min,
        }
    } else {
        Verdict {
            label: VerdictLabel::VerdictGrade,
            reason: format!("n_events={n_min} \u{2265} min_events={min_events}"),
            n_events_used: n_min,
        }
    }
}

fn sanitize(s: &str) -> String {
    s.chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' { c } else { '_' })
        .collect()
}

fn current_commit() -> String {
    Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .and_then(|o| {
            if o.status.success() {
                Some(String::from_utf8_lossy(&o.stdout).trim().to_string())
            } else {
                None
            }
        })
        .unwrap_or_else(|| "unknown".into())
}
