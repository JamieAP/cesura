//! Unified eval harness for change-point detectors.
//!
//!
//! The seven sub-modules carry distinct responsibilities:
//! - [`detector`] -- the `CpDetector` lowest-common-denominator trait.
//! - [`fixture`] -- canonical `Fixture` shape + registry.
//! - [`metrics`] -- single source of truth for precision/recall/F1.
//! - `harness` -- run reports, JSON IO, verdict floor.
//! - `loaders` -- parquet/date helpers via native polars.
//! - `adapters` -- bridges existing detectors to `CpDetector`.
//! - [`events`] -- INDICES_MACRO_EVENTS dated event list.
//!
//!
//! Two `pub` patterns surface as duplication on a quick read but are
//! intentional:
//!
//! - **Per-fixture asset-name resolvers (`asset_name_for(fixture, idx)`)**.
//!   Each multi-stream example owns its own `(fixture, stream_index) →
//!   display-name` mapping (BTC/ETH/SOL for `crypto_macro_5`, SPX/NDX/DJI/VIX
//!   for `indices_macro_v1`). The mapping is fixture-specific knowledge
//!   that does not belong in `src/bench/`; promoting it would couple the
//!   harness to the example layer's display choices. The cost is that
//!   each consumer of attribution-rendering writes the same shape of
//!   match block.
//!

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
