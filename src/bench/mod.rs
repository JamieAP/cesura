//! Unified eval harness for change-point detectors.
//!
//!
//! The five sub-modules carry distinct responsibilities:
//! - [`detector`] -- the `CpDetector` lowest-common-denominator trait.
//! - [`fixture`] -- canonical `Fixture` shape + registry.
//! - [`metrics`] -- single source of truth for precision/recall/F1.
//! - `harness` -- run reports, JSON IO, verdict floor.
//! - `loaders` -- parquet/CSV/date helpers.
//! - `adapters` -- bridges existing detectors to `CpDetector`.

pub mod adapters;
pub mod detector;
pub mod fixture;
pub mod harness;
pub mod loaders;
pub mod metrics;
pub mod multistream_adapter;

pub use detector::{CpDetector, DetectionResult};
pub use fixture::{Fixture, FixtureError, FixtureRegistry, KNOWN_EVENTS};
pub use harness::{
    classify_verdict, label_str, load_reports, power_label, render_markdown, run_bench,
    write_report, Report, Verdict, VerdictLabel, DEFAULT_MIN_EVENTS,
};
pub use metrics::{
    count_event_hits, count_far_cps, evaluate, f1_with_margin, per_event_hits, Metrics,
};
