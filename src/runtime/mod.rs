//! Agent-facing runtime: uniform `feed(obs) -> Vec<Fire>` over every
//! shipped detector. Backs the `cesura watch` CLI and the `cesura-mcp`
//! stdio MCP server.
//!
//! Sum type, not trait -- observation shapes differ across detectors
//! (`f64`, `Vec<f64>`), and a trait forces an enum-of-shapes anyway.
//!
//! ## Runtime representation
//!
//! - `Fire.shift_sigma`: `Some(v)` for univariate variants (carries the
//!   per-detector `ChangePoint::shift_sigma`); `None` for multistream
//!   aggregators (no analogue).
//! - `Fire.streams`: empty `vec![]` for univariate; populated by the
//!   aggregator for multistream.
//! - `Fire.kind`: `"cp"` for every plain change-point. ChenWu
//!   `Detection::CollectiveAnomaly{start,end}` emits one `Fire` at
//!   `index = end` with `kind = "anomaly"` (no extra fire at `start`
//!   -- the closing edge is the actionable signal for a Monitor).
//! - The DmBocd variant wraps `StreamingDmBocd` for incremental input.
//!   It has no `save_state`/`restore`, so `Runner::save` returns `Err`
//!   for that variant; restore is unsupported. Regression tests compare
//!   its change-point indices with batch detection for `n ≥ 180`.
//! - Multistream aggregator save persists both the aggregator state
//!   and the inner per-stream `DetectorState`s (HC / SumCusum
//!   `restore` take `Vec<S>`). `FilterTickAggregator` has no inner
//!   streams.
//! - Generic parameters: `SumCusumAggregator<StreamingDetector>` and
//!   `HcAggregator<StreamingDetector>` (matches `canonical::recommended_streams`).
//!   `StreamingDmBocd<IdentityM>` (matches `DmBocdDetector::new` default).

use serde::{Deserialize, Serialize};

use std::fmt;
use std::str::FromStr;

use crate::dm_bocd::{IdentityM, StreamingDmBocd};
use crate::multistream::{
    FilterTickAggregator, FilterTickAggregatorState, HcAggregator, HcAggregatorState,
    SumCusumAggregator, SumCusumAggregatorState,
};
use crate::streaming::{DetectorState, StreamingDetector};
#[cfg(feature = "joint-detection")]
use crate::streaming_chen_wu::{ChenWuDetectorState, StreamingChenWuDetector};
#[cfg(feature = "joint-detection")]
use crate::chen_wu::Detection;

/// Single source of truth for the six shipped detector kinds. Used by
/// `--kind` parsing in the CLI, by tool dispatch in the planned MCP
/// server, and by `Runner::kind_enum()`. The wire string form is
/// auto-kebab-case (`ChenWu` ↔ `"chen-wu"`, `SumCusum` ↔ `"sum-cusum"`, etc.)
/// and matches what `Runner::kind()` returns.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "cli", derive(clap::ValueEnum))]
pub enum DetectorKind {
    Streaming,
    #[cfg(feature = "joint-detection")]
    ChenWu,
    SumCusum,
    Hc,
    FilterTick,
    DmBocd,
}

impl DetectorKind {
    /// Stable wire string, matching `Runner::kind()` output.
    pub fn as_wire(&self) -> &'static str {
        match self {
            Self::Streaming => "streaming",
            #[cfg(feature = "joint-detection")]
            Self::ChenWu => "chen-wu",
            Self::SumCusum => "sum-cusum",
            Self::Hc => "hc",
            Self::FilterTick => "filter-tick",
            Self::DmBocd => "dm-bocd",
        }
    }

    /// Observation shape this detector expects. Scalar = `f64`,
    /// Vector = `Vec<f64>`. CLI + MCP use this to validate inputs
    /// before dispatching to `Runner::feed`.
    pub fn obs_shape(&self) -> ObsShape {
        match self {
            Self::Streaming => ObsShape::Scalar,
            #[cfg(feature = "joint-detection")]
            Self::ChenWu => ObsShape::Scalar,
            Self::SumCusum | Self::Hc | Self::FilterTick | Self::DmBocd => ObsShape::Vector,
        }
    }

    /// Every variant in declaration order. Drives discovery surfaces
    /// (CLI `info`, MCP `ServerInfo` instructions) so a future
    /// detector variant only needs to be added here once.
    pub fn all() -> &'static [DetectorKind] {
        &[
            Self::Streaming,
            #[cfg(feature = "joint-detection")]
            Self::ChenWu,
            Self::SumCusum,
            Self::Hc,
            Self::FilterTick,
            Self::DmBocd,
        ]
    }
}

impl fmt::Display for DetectorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_wire())
    }
}

impl FromStr for DetectorKind {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        for k in Self::all() {
            if k.as_wire() == s {
                return Ok(*k);
            }
        }
        Err(format!("unknown detector kind: {s:?}"))
    }
}

/// Observation shape expected by a `DetectorKind`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObsShape {
    Scalar,
    Vector,
}

/// Uniform fire envelope. Collapses `ChangePoint` (univariate) and
/// `MultiStreamChangePoint` (multistream) into one wire shape so the
/// CLI and MCP layers don't dispatch per variant.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Fire {
    /// Observation index where the fire occurred (model step count,
    /// NaN-skipped where applicable).
    pub index: usize,
    /// Confidence in `[0, 1]` -- higher = stronger evidence. Source
    /// depends on detector kind (BOCPD posterior, HC statistic,
    /// sum-CUSUM normalised level, etc.).
    pub confidence: f64,
    /// Shift magnitude in units of global σ, for univariate detectors
    /// that compute it. `None` for multistream aggregators.
    pub shift_sigma: Option<f64>,
    /// Contributing stream indices (HC: above-τ set; SumCusum: empty;
    /// FilterTick: k-of-d set). Empty for univariate detectors.
    pub streams: Vec<usize>,
    /// Kind tag: `"cp"` for change points, `"anomaly"` for
    /// `Detection::CollectiveAnomaly`. `Cow<'static, str>` so library
    /// construction stays alloc-free while `serde_json::from_str` on an
    /// owned input can deserialize without a `'static` borrow.
    pub kind: std::borrow::Cow<'static, str>,
}

/// One observation -- scalar for univariate detectors, vector for
/// multistream / multivariate.
#[derive(Debug, Clone)]
pub enum Obs {
    /// Univariate sample.
    F64(f64),
    /// `d`-dimensional sample. Length must equal the detector's `d`.
    Vec(Vec<f64>),
}

/// Detector dispatched at runtime. One variant per shipped detector
/// kind. Construction goes through `Runner::*_new` factory functions
/// keyed to the `--kind` flag.
pub enum Runner {
    Uni(StreamingDetector),
    #[cfg(feature = "joint-detection")]
    UniChenWu(StreamingChenWuDetector),
    MultiSumCusum(SumCusumAggregator<StreamingDetector>),
    MultiHc(HcAggregator<StreamingDetector>),
    MultiFilterTick(FilterTickAggregator),
    DmBocd(StreamingDmBocd<IdentityM>),
}

/// Wire-format snapshot of a `Runner`. Versioned -- `restore` rejects
/// mismatched `version`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunnerState {
    /// Snapshot schema version. Bump on any breaking layout change.
    pub version: u32,
    /// Detector kind tag (`Runner::kind()`). Used to disambiguate the
    /// payload variant on restore.
    pub kind: String,
    /// Inner payload -- kind-specific.
    pub payload: RunnerStatePayload,
}

pub const RUNNER_STATE_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RunnerStatePayload {
    Uni(DetectorState),
    #[cfg(feature = "joint-detection")]
    UniChenWu(ChenWuDetectorState),
    MultiSumCusum {
        agg: SumCusumAggregatorState,
        streams: Vec<DetectorState>,
    },
    MultiHc {
        agg: HcAggregatorState,
        streams: Vec<DetectorState>,
    },
    MultiFilterTick(FilterTickAggregatorState),
}

impl Runner {
    /// Feed a single observation. Returns every fire produced by this
    /// step (zero or more). `Obs` shape must match the variant; a
    /// mismatch returns an empty `Vec` and increments no state.
    pub fn feed(&mut self, obs: Obs) -> Vec<Fire> {
        match (self, obs) {
            (Runner::Uni(d), Obs::F64(x)) => d
                .step(&[x])
                .into_iter()
                .map(|c| Fire {
                    index: c.index,
                    confidence: c.confidence,
                    shift_sigma: Some(c.shift_sigma),
                    streams: Vec::new(),
                    kind: std::borrow::Cow::Borrowed("cp"),
                })
                .collect(),
            #[cfg(feature = "joint-detection")]
            (Runner::UniChenWu(d), Obs::F64(x)) => d
                .step(&[x])
                .into_iter()
                .map(|det| match det {
                    Detection::ChangePoint(c) => Fire {
                        index: c.index,
                        confidence: c.confidence,
                        shift_sigma: Some(c.shift_sigma),
                        streams: Vec::new(),
                        kind: std::borrow::Cow::Borrowed("cp"),
                    },
                    Detection::CollectiveAnomaly { end, confidence, .. } => Fire {
                        index: end,
                        confidence,
                        shift_sigma: None,
                        streams: Vec::new(),
                        kind: std::borrow::Cow::Borrowed("anomaly"),
                    },
                })
                .collect(),
            (Runner::MultiSumCusum(a), Obs::Vec(v)) => a
                .step(std::slice::from_ref(&v))
                .into_iter()
                .map(multi_to_fire)
                .collect(),
            (Runner::MultiHc(a), Obs::Vec(v)) => a
                .step(std::slice::from_ref(&v))
                .into_iter()
                .map(multi_to_fire)
                .collect(),
            (Runner::MultiFilterTick(a), Obs::Vec(v)) => a
                .step(std::slice::from_ref(&v))
                .into_iter()
                .map(multi_to_fire)
                .collect(),
            (Runner::DmBocd(d), Obs::Vec(v)) => d
                .step(&v)
                .into_iter()
                .map(|c| Fire {
                    index: c.index,
                    confidence: c.confidence,
                    shift_sigma: Some(c.shift_sigma),
                    streams: Vec::new(),
                    kind: std::borrow::Cow::Borrowed("cp"),
                })
                .collect(),
            _ => Vec::new(),
        }
    }

    /// Wire kind tag. Matches the `--kind` CLI value.
    pub fn kind(&self) -> &'static str {
        self.kind_enum().as_wire()
    }

    /// Typed kind. Prefer this over `kind()` in dispatch code so the
    /// exhaustiveness checker catches missing variants when a new
    /// detector is added.
    pub fn kind_enum(&self) -> DetectorKind {
        match self {
            Runner::Uni(_) => DetectorKind::Streaming,
            #[cfg(feature = "joint-detection")]
            Runner::UniChenWu(_) => DetectorKind::ChenWu,
            Runner::MultiSumCusum(_) => DetectorKind::SumCusum,
            Runner::MultiHc(_) => DetectorKind::Hc,
            Runner::MultiFilterTick(_) => DetectorKind::FilterTick,
            Runner::DmBocd(_) => DetectorKind::DmBocd,
        }
    }

    /// Capture a serialisable snapshot. Returns `Err` for variants
    /// whose underlying detector has no `save_state` -- currently only
    /// `Runner::DmBocd` (StreamingDmBocd save/restore deferred).
    pub fn save(&self) -> Result<RunnerState, String> {
        let payload = match self {
            Runner::Uni(d) => RunnerStatePayload::Uni(d.save_state()),
            #[cfg(feature = "joint-detection")]
            Runner::UniChenWu(d) => RunnerStatePayload::UniChenWu(d.save_state()),
            Runner::MultiSumCusum(a) => RunnerStatePayload::MultiSumCusum {
                agg: a.save_state(),
                streams: a.streams().iter().map(|s| s.save_state()).collect(),
            },
            Runner::MultiHc(a) => RunnerStatePayload::MultiHc {
                agg: a.save_state(),
                streams: a.streams().iter().map(|s| s.save_state()).collect(),
            },
            Runner::MultiFilterTick(a) => RunnerStatePayload::MultiFilterTick(a.save_state()),
            Runner::DmBocd(_) => {
                return Err(
                    "dm-bocd: StreamingDmBocd does not yet implement save_state \
                     (deferred per src/dm_bocd.rs:869)"
                        .into(),
                )
            }
        };
        Ok(RunnerState {
            version: RUNNER_STATE_VERSION,
            kind: self.kind().to_string(),
            payload,
        })
    }

    /// Restore a runner from a saved snapshot. Rejects mismatched
    /// `version` and rejects payloads whose tag disagrees with `kind`.
    pub fn restore(state: RunnerState) -> Result<Self, String> {
        if state.version != RUNNER_STATE_VERSION {
            return Err(format!(
                "RunnerState version mismatch: got {}, expected {}",
                state.version, RUNNER_STATE_VERSION
            ));
        }
        match (state.kind.as_str(), state.payload) {
            ("streaming", RunnerStatePayload::Uni(d)) => {
                Ok(Runner::Uni(StreamingDetector::restore(d)?))
            }
            #[cfg(feature = "joint-detection")]
            ("chen-wu", RunnerStatePayload::UniChenWu(d)) => {
                Ok(Runner::UniChenWu(StreamingChenWuDetector::restore(d)?))
            }
            ("sum-cusum", RunnerStatePayload::MultiSumCusum { agg, streams }) => {
                let det_streams: Vec<StreamingDetector> = streams
                    .into_iter()
                    .map(StreamingDetector::restore)
                    .collect::<Result<_, _>>()?;
                Ok(Runner::MultiSumCusum(SumCusumAggregator::restore(
                    agg,
                    det_streams,
                )?))
            }
            ("hc", RunnerStatePayload::MultiHc { agg, streams }) => {
                let det_streams: Vec<StreamingDetector> = streams
                    .into_iter()
                    .map(StreamingDetector::restore)
                    .collect::<Result<_, _>>()?;
                Ok(Runner::MultiHc(HcAggregator::restore(agg, det_streams)?))
            }
            ("filter-tick", RunnerStatePayload::MultiFilterTick(s)) => {
                Ok(Runner::MultiFilterTick(FilterTickAggregator::restore(s)))
            }
            (kind, _) => Err(format!(
                "RunnerState kind/payload mismatch or unsupported kind: {kind}"
            )),
        }
    }
}

fn multi_to_fire(c: crate::multistream::MultiStreamChangePoint) -> Fire {
    Fire {
        index: c.index,
        confidence: c.confidence,
        shift_sigma: None,
        streams: c.streams,
        kind: std::borrow::Cow::Borrowed("cp"),
    }
}
