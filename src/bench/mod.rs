//! Evaluation helpers for change-point detectors, behind `test-utils`.
//!
//! Provides fixtures, detector adapters, precision/recall metrics, and report
//! rendering. Optional real-data fixtures require separately supplied datasets;
//! synthetic fixtures do not require private data. These APIs are experimental.

pub mod adapters;
pub mod detector;
pub mod events;
pub mod fixture;
pub mod harness;
pub mod hyperliquid_events;
pub mod loaders;
pub mod metrics;
pub mod multistream_adapter;

pub use detector::{CpDetector, DetectionResult};
pub use events::{INDICES_MACRO_EVENTS, INDICES_MACRO_EVENTS_VERSION, INDICES_MACRO_EVENT_COUNT};
pub use fixture::{Fixture, FixtureError, FixtureRegistry, KNOWN_EVENTS};
pub use hyperliquid_events::{
    HYPERLIQUID_EVENTS_V1, HYPERLIQUID_EVENTS_V1_COUNT, HYPERLIQUID_EVENTS_V1_VERSION,
};
pub use harness::{
    classify_verdict, label_str, load_reports, power_label, render_attribution, render_markdown,
    run_bench, write_report, AttributionRow, BenchAuditTrail, Report, Verdict, VerdictLabel,
    DEFAULT_MIN_EVENTS,
};
pub use metrics::{
    count_event_hits, count_far_cps, evaluate, f1_with_margin, per_event_hits, Metrics,
};
