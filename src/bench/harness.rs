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

/// Run a single (detector, fixture) cell. Returns the aggregated
/// `Report` (suitable for JSON dump) AND the live `DetectionResult`
/// for in-process rendering of per-CP detail (confidence, attribution
/// names) the aggregate cannot carry.
///
/// `Report.attribution` mirrors `result.attribution` so the JSON
/// dump alone is sufficient for downstream replays; the second tuple
/// element is just-don't-redo-detection convenience for the caller.
pub fn run_bench<D: CpDetector + ?Sized>(
    d: &D,
    fix: &Fixture,
) -> (Report, crate::bench::DetectionResult) {
    let t0 = Instant::now();
    let result = d.detect_full(fix);
    let timing_us = t0.elapsed().as_micros();
    let metrics = evaluate(&result.cps, fix);
    let report = Report {
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
        attribution: result.attribution.clone(),
    };
    (report, result)
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

/// Per-fire attribution row for markdown rendering. Streams are
/// pre-resolved to display names by the caller (numeric stream
/// index → asset name happens at the example layer, since fixture
/// → asset-name mapping is fixture-specific).
#[derive(Debug, Clone)]
pub struct AttributionRow {
    pub detector: String,
    pub fixture: String,
    pub fire_idx: usize,
    pub cp_index: usize,
    pub confidence: f64,
    pub streams: Vec<String>,
}

/// Render the per-stream attribution table. Fires with empty streams
/// (univariate / sum-CUSUM) are skipped by the caller before passing
/// rows in. Empty `rows` slice produces an empty string.
pub fn render_attribution(rows: &[AttributionRow]) -> String {
    if rows.is_empty() {
        return String::new();
    }
    let mut s = String::new();
    s.push_str("| detector | fixture | fire_idx | cp_index | confidence | streams |\n");
    s.push_str("|---|---|---:|---:|---:|---|\n");
    for row in rows {
        s.push_str(&format!(
            "| {} | {} | {} | {} | {:.3} | {} |\n",
            row.detector,
            row.fixture,
            row.fire_idx,
            row.cp_index,
            row.confidence,
            row.streams.join(", "),
        ));
    }
    s
}

/// Asset-name resolver: `(fixture_name, stream_index) → display name`.
type AssetNameFn<'a> = Box<dyn Fn(&str, usize) -> String + 'a>;

///
/// The builder owns:
/// - `intro`: the section text emitted ahead of the audit trail block
/// - `results_dir`: where per-cell JSON dumps land
/// - `asset_name_for`: closure resolving `(fixture_name, stream_idx)` to
///   a display name (BTC/ETH/SOL for crypto_macro_5, SPX/NDX/DJI/VIX for
///   indices_macro_v1, fallback `s{n}`). Per-fixture knowledge stays
///   at the example layer; the builder threads it through.
///
/// `run` records the `Report` AND the live `DetectionResult` so
/// per-fire attribution (which `Report` aggregates away) is available
/// at `render` time without re-running detection.
pub struct BenchAuditTrail<'a> {
    intro: String,
    results_dir: &'a std::path::Path,
    asset_name_for: AssetNameFn<'a>,
    cells: Vec<(Report, crate::bench::DetectionResult)>,
    /// Cap on rows rendered in the markdown table. `None` = render all.
    /// Used by `comprehensive_report` (large detector × scenario panel)
    /// where 30 rows is enough sample to validate the harness output;
    /// the rich matrix above the audit trail is the load-bearing read.
    render_limit: Option<usize>,
}

impl<'a> BenchAuditTrail<'a> {
    pub fn new(results_dir: &'a std::path::Path, intro: impl Into<String>) -> Self {
        Self {
            intro: intro.into(),
            results_dir,
            // Default resolver: numeric stream indices `s{n}`.
            asset_name_for: Box::new(|_fix, idx| format!("s{idx}")),
            cells: Vec::new(),
            render_limit: None,
        }
    }

    /// Cap the rendered markdown table at `n` rows. JSON dumps and the
    /// classify_verdict banner still cover all cells; only the inline
    /// table is truncated. Used by examples with very large
    /// detector × fixture panels where 30 rows is enough to validate
    /// the harness pipeline and the rich matrix above carries the load.
    pub fn with_render_limit(mut self, n: usize) -> Self {
        self.render_limit = Some(n);
        self
    }

    /// Override the default `s{n}` stream-index resolver. The closure
    /// receives `(fixture_name, stream_index)` and returns a display
    /// name. Fixture-specific name maps stay in the example layer.
    pub fn with_asset_resolver(
        mut self,
        f: impl Fn(&str, usize) -> String + 'a,
    ) -> Self {
        self.asset_name_for = Box::new(f);
        self
    }

    /// Run a single (detector, fixture) cell. The aggregated `Report`
    /// is dumped to `results_dir/<fixture>/<detector>-<commit>.json`.
    /// The live `DetectionResult` is retained for attribution rendering.
    /// Returns the cell's `Metrics` so callers needing per-cell numbers
    /// (per-seed F1, custom verdict math) don't have to re-look-up the
    /// last report.
    pub fn run<D: CpDetector + ?Sized>(&mut self, det: &D, fix: &Fixture) -> Metrics {
        let (report, result) = run_bench(det, fix);
        let _ = write_report(&report, self.results_dir);
        let metrics = report.metrics.clone();
        self.cells.push((report, result));
        metrics
    }

    /// Read-only access to the aggregated reports (drops the live
    /// detection results). Useful when an example wants to compute
    /// custom per-detector aggregates on top of the canonical render.
    pub fn reports(&self) -> Vec<&Report> {
        self.cells.iter().map(|(r, _)| r).collect()
    }

    pub fn classify(&self) -> Verdict {
        let reports: Vec<Report> = self.cells.iter().map(|(r, _)| r.clone()).collect();
        classify_verdict(&reports, DEFAULT_MIN_EVENTS)
    }

    /// Emit the canonical audit-trail block:
    /// 1. `intro` text (caller-provided)
    /// 2. classify_verdict banner
    /// 3. `render_markdown(reports)` table
    /// 4. (when any HC fire has non-empty attribution)
    ///    `### HC per-stream attribution` + `render_attribution(rows)`
    pub fn render(&self, out: &mut String) {
        let reports: Vec<Report> = self.cells.iter().map(|(r, _)| r.clone()).collect();
        let verdict = classify_verdict(&reports, DEFAULT_MIN_EVENTS);
        out.push_str(&self.intro);
        if !self.intro.ends_with("\n\n") {
            out.push_str("\n\n");
        }
        out.push_str(&format!(
            "**classify_verdict** ({} reports, min_events={DEFAULT_MIN_EVENTS}): \
             label = **{}**; {}.\n\n",
            reports.len(),
            label_str(verdict.label),
            verdict.reason,
        ));
        if let Some(limit) = self.render_limit {
            out.push_str(&format!(
                "Canonical Report table (`render_markdown`, first {limit} rows):\n\n"
            ));
            let head: Vec<Report> = reports.iter().take(limit).cloned().collect();
            out.push_str(&render_markdown(&head));
        } else {
            out.push_str("Canonical Report table (`render_markdown`):\n\n");
            out.push_str(&render_markdown(&reports));
        }
        out.push('\n');

        // Attribution sub-table: collect non-empty rows from any HC-named
        // detector. Sum-CUSUM and univariate cells produce zero rows.
        let mut attribution_rows: Vec<AttributionRow> = Vec::new();
        for (report, result) in &self.cells {
            if !report.detector.starts_with("HC") {
                continue;
            }
            for (i, streams) in result.attribution.iter().enumerate() {
                if streams.is_empty() {
                    continue;
                }
                let cp = &result.cps[i];
                let names: Vec<String> = streams
                    .iter()
                    .map(|&s| (self.asset_name_for)(&report.fixture, s))
                    .collect();
                attribution_rows.push(AttributionRow {
                    detector: report.detector.clone(),
                    fixture: report.fixture.clone(),
                    fire_idx: i,
                    cp_index: cp.index,
                    confidence: cp.confidence,
                    streams: names,
                });
            }
        }
        if !attribution_rows.is_empty() {
            out.push_str(
                "\n### HC per-stream attribution\n\n\
                 Per-fire attribution from the live `DetectionResult`. For \
                 HC, `attribution[i]` lists the streams whose p-value crossed \
                 the threshold for fire `i` (sum-CUSUM and univariate \
                 detectors are empty by contract; filtered to non-empty rows).\n\n",
            );
            out.push_str(&render_attribution(&attribution_rows));
            out.push('\n');
        }
    }
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
