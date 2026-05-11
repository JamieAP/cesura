//! Hand-curated dated events for the `hyperliquid_1s_v1` fixture.
//!
//! Mirrors the shape of `INDICES_MACRO_EVENTS` (string-tagged tuples).
//! Category tags follow the indices set verbatim for the MACRO subset
//! (FOMC/CPI/NFP/ECB/BOJ/BOE/PCE + non-calendar TARIFF/AI/GEO). New
//! crypto-specific tags planned for v2:
//!
//!
//! Window covered: 2025-01 → 2026-02 (matches `INDICES_MACRO_EVENTS`).
//! The Hydromancer Reservoir parquet archive starts 2025-07-31; events
//! outside the available data window are clipped at fixture-construction
//! time (same pattern as `indices_macro_v1`).
//!

pub const HYPERLIQUID_EVENTS_V1_VERSION: u32 = 1;

/// `(category, label, "YYYY-MM-DD")`. Sorted ascending by date.
/// Macro subset is verbatim copy of `INDICES_MACRO_EVENTS` (any drift
/// across the two modules is a bug -- they describe the same calendar).
#[allow(clippy::type_complexity)]
pub const HYPERLIQUID_EVENTS_V1: &[(&str, &str, &str)] = &[
    // ── 2025 ─────────────────────────────────────────────────
    ("CPI",  "Dec 2024 CPI release",        "2025-01-15"),
    ("BOJ",  "Jan 2025 rate decision",      "2025-01-24"),
    ("FOMC", "Jan 2025 rate decision",      "2025-01-29"),
    ("ECB",  "Jan 2025 rate decision",      "2025-01-30"),
    ("NFP",  "Jan 2025 jobs report",        "2025-02-07"),
    ("CPI",  "Jan 2025 CPI release",        "2025-02-12"),
    ("NFP",  "Feb 2025 jobs report",        "2025-03-07"),
    ("ECB",  "Mar 2025 rate decision",      "2025-03-06"),
    ("CPI",  "Feb 2025 CPI release",        "2025-03-12"),
    ("BOJ",  "Mar 2025 rate decision",      "2025-03-19"),
    ("FOMC", "Mar 2025 rate decision",      "2025-03-19"),
    ("NFP",  "Mar 2025 jobs report",        "2025-04-04"),
    ("CPI",  "Mar 2025 CPI release",        "2025-04-10"),
    ("ECB",  "Apr 2025 rate decision",      "2025-04-17"),
    ("BOJ",  "May 2025 rate decision",      "2025-05-01"),
    ("NFP",  "Apr 2025 jobs report",        "2025-05-02"),
    ("FOMC", "May 2025 rate decision",      "2025-05-07"),
    ("CPI",  "Apr 2025 CPI release",        "2025-05-13"),
    ("ECB",  "Jun 2025 rate decision",      "2025-06-05"),
    ("NFP",  "May 2025 jobs report",        "2025-06-06"),
    ("CPI",  "May 2025 CPI release",        "2025-06-11"),
    ("BOJ",  "Jun 2025 rate decision",      "2025-06-17"),
    ("FOMC", "Jun 2025 rate decision",      "2025-06-18"),
    ("NFP",  "Jun 2025 jobs report",        "2025-07-03"),
    ("CPI",  "Jun 2025 CPI release",        "2025-07-15"),
    ("ECB",  "Jul 2025 rate decision",      "2025-07-24"),
    ("FOMC", "Jul 2025 rate decision",      "2025-07-30"),
    ("BOJ",  "Jul 2025 rate decision",      "2025-07-31"),
    ("NFP",  "Jul 2025 jobs report",        "2025-08-01"),
    ("CPI",  "Jul 2025 CPI release",        "2025-08-12"),
    ("NFP",  "Aug 2025 jobs report",        "2025-09-05"),
    ("ECB",  "Sep 2025 rate decision",      "2025-09-11"),
    ("CPI",  "Aug 2025 CPI release",        "2025-09-11"),
    ("FOMC", "Sep 2025 rate decision",      "2025-09-17"),
    ("BOJ",  "Sep 2025 rate decision",      "2025-09-19"),
    ("NFP",  "Sep 2025 jobs report",        "2025-10-03"),
    ("CPI",  "Sep 2025 CPI release",        "2025-10-15"),
    ("ECB",  "Oct 2025 rate decision",      "2025-10-30"),
    ("FOMC", "Oct 2025 rate decision",      "2025-10-29"),
    ("BOJ",  "Oct 2025 rate decision",      "2025-10-30"),
    ("NFP",  "Oct 2025 jobs report",        "2025-11-07"),
    ("CPI",  "Oct 2025 CPI release",        "2025-11-13"),
    ("NFP",  "Nov 2025 jobs report",        "2025-12-05"),
    ("ECB",  "Dec 2025 rate decision",      "2025-12-11"),
    ("CPI",  "Nov 2025 CPI release",        "2025-12-10"),
    ("FOMC", "Dec 2025 rate decision",      "2025-12-10"),
    ("BOJ",  "Dec 2025 rate decision",      "2025-12-19"),
    // ── 2026 ─────────────────────────────────────────────────
    ("NFP",  "Dec 2025 jobs report",        "2026-01-09"),
    ("CPI",  "Dec 2025 CPI release",        "2026-01-14"),
    ("ECB",  "Jan 2026 rate decision",      "2026-01-29"),
    ("FOMC", "Jan 2026 rate decision",      "2026-01-28"),
    ("BOJ",  "Jan 2026 rate decision",      "2026-01-23"),
    ("NFP",  "Jan 2026 jobs report",        "2026-02-06"),
    ("CPI",  "Jan 2026 CPI release",        "2026-02-11"),
    // BOE rate decisions.
    ("BOE",  "Feb 2025 rate decision",      "2025-02-06"),
    ("BOE",  "Mar 2025 rate decision",      "2025-03-20"),
    ("BOE",  "May 2025 rate decision",      "2025-05-08"),
    ("BOE",  "Jun 2025 rate decision",      "2025-06-19"),
    ("BOE",  "Aug 2025 rate decision",      "2025-08-07"),
    ("BOE",  "Sep 2025 rate decision",      "2025-09-18"),
    ("BOE",  "Nov 2025 rate decision",      "2025-11-06"),
    ("BOE",  "Dec 2025 rate decision",      "2025-12-18"),
    ("BOE",  "Feb 2026 rate decision",      "2026-02-05"),
    // PCE inflation.
    ("PCE",  "Dec 2024 PCE release",        "2025-01-31"),
    ("PCE",  "Jan 2025 PCE release",        "2025-02-28"),
    ("PCE",  "Feb 2025 PCE release",        "2025-03-28"),
    ("PCE",  "Mar 2025 PCE release",        "2025-04-30"),
    ("PCE",  "Apr 2025 PCE release",        "2025-05-30"),
    ("PCE",  "May 2025 PCE release",        "2025-06-27"),
    ("PCE",  "Jun 2025 PCE release",        "2025-07-31"),
    ("PCE",  "Jul 2025 PCE release",        "2025-08-29"),
    ("PCE",  "Aug 2025 PCE release",        "2025-09-26"),
    ("PCE",  "Sep 2025 PCE release",        "2025-10-31"),
    ("PCE",  "Oct 2025 PCE release",        "2025-11-26"),
    ("PCE",  "Nov 2025 PCE release",        "2025-12-19"),
    ("PCE",  "Dec 2025 PCE release",        "2026-01-30"),
    // Non-calendar events mirror INDICES_MACRO_EVENTS.
    ("TARIFF", "Tariffs announced on MX/CA/CN",       "2025-02-01"),
    ("TARIFF", "Liberation Day reciprocal tariffs",   "2025-04-02"),
    ("TARIFF", "90-day tariff pause + 145% China",    "2025-04-09"),
    ("TARIFF", "US-China Geneva de-escalation",       "2025-05-12"),
    ("TARIFF", "Reciprocal tariffs effective (post-pause)", "2025-08-07"),
    ("AI", "DeepSeek R1 fallout / NVDA -17%",   "2025-01-27"),
    ("AI", "NVDA Q4 FY25 earnings",             "2025-02-26"),
    ("AI", "NVDA Q1 FY26 earnings",             "2025-05-28"),
    ("AI", "NVDA Q2 FY26 earnings",             "2025-08-27"),
    ("AI", "NVDA Q3 FY26 earnings",             "2025-11-19"),
    ("GEO", "Israel strikes Iran (Rising Lion)", "2025-06-13"),
    ("GEO", "US strikes Iranian nuclear sites",  "2025-06-22"),
    ("GEO", "Iran-Israel ceasefire",             "2025-06-24"),
];

pub const HYPERLIQUID_EVENTS_V1_COUNT: usize = HYPERLIQUID_EVENTS_V1.len();

const _: () = assert!(
    HYPERLIQUID_EVENTS_V1_COUNT >= 50,
    "HYPERLIQUID_EVENTS_V1 must clear the verdict-grade floor of 50"
);
