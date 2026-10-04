//! Bridges existing detectors to [`CpDetector`] without touching their
//! public APIs. Pure dispatch wrappers.
//!
//! Adapter caveats:
//! - `ChenWuAdapter` flattens `Vec<Detection>` to `Vec<ChangePoint>` by
//!   keeping `Detection::ChangePoint(_)` and dropping `CollectiveAnomaly`.
//!   A separate `ChenWuFullAdapter` can be added if downstream cares.
//! - `FocusAdapter` is multivariate-only at the trait surface: even `d=1`
//!   fixtures are routed through `detect_multivariate(&[vec])` to avoid
//!   the `&mut self` univariate path (FocusDetector is not Clone).
//! - `ConformalAdapter` / `ConformalMvAdapter` project `ConformalCp →
//!   ChangePoint` (drop `timing_interval` + `coverage`). The score
//!   channel is *not* exposed through `CpDetector`; downstream callers
//!   that need timing intervals must call `ConformalCpWrapper::detect`
//!   directly. The wrapper requires `&mut self`, so the adapter holds
//!   an owned `RefCell<ConformalCpWrapper<D>>` so detection can borrow
//!   the wrapper mutably through the adapter's shared reference.

use std::cell::RefCell;

use crate::bench::detector::CpDetector;
use crate::bench::fixture::Fixture;
use crate::conformal::{ConformalCpWrapper, MvScoredDetect, ScoredDetect};
use crate::dm_bocd::{DmBocdDetector, MWeight};
use crate::focus::FocusDetector;
use crate::nig::Nig;
use crate::predictive::Predictive;
use crate::{BocpdDetector, ChangePoint, EnsembleDetector};

#[cfg(feature = "joint-detection")]
use crate::chen_wu::{ChenWuDetector, Detection};

/// Bench adapter for `BocpdDetector<P>`. Generic over the predictive
/// `P` so non-default plugs (e.g., `NigAr1`) can be evaluated without
/// custom adapter glue. Default `P = Nig` keeps existing call sites
/// (`BocpdAdapter { det, mv, label }`) compiling unchanged.
pub struct BocpdAdapter<'a, P: Predictive = Nig> {
    pub det: &'a BocpdDetector<P>,
    pub mv: bool,
    pub label: &'a str,
}

impl<P: Predictive> CpDetector for BocpdAdapter<'_, P> {
    fn name(&self) -> String {
        self.label.to_string()
    }
    fn detect(&self, fix: &Fixture) -> Vec<ChangePoint> {
        if self.mv {
            self.det.detect_multivariate(&fix.data)
        } else {
            self.det.detect(&fix.as_univariate())
        }
    }
}

/// Joint multivariate `BocpdDetector` bench adapter with the Cholesky
/// whitening preamble disabled. Wraps
/// [`BocpdDetector::detect_multivariate_no_whiten`].
///
/// Evaluates joint NIW on per-dimension normalized data, allowing comparison
/// with the whitened variant.
///
/// Same `P`-agnostic caveat as [`BocpdAdapter`] with `mv: true`: the
/// underlying recursion is hard-coded NIW regardless of `P`.
pub struct BocpdJointZnormAdapter<'a, P: Predictive = Nig> {
    pub det: &'a BocpdDetector<P>,
    pub label: &'a str,
}

impl<P: Predictive> CpDetector for BocpdJointZnormAdapter<'_, P> {
    fn name(&self) -> String {
        self.label.to_string()
    }
    fn detect(&self, fix: &Fixture) -> Vec<ChangePoint> {
        self.det.detect_multivariate_no_whiten(&fix.data)
    }
}

/// Bench adapter for [`crate::pro_bocd::PrOBocpdDetector`].
///
/// Wraps `detect_multivariate_seeded(data, seed)` so seeded sweeps
/// share infrastructure with the other detector adapters. Callers can
/// provide the same seed schedule for comparisons across detectors.
pub struct PrOBocpdAdapter<'a> {
    pub det: &'a crate::pro_bocd::PrOBocpdDetector,
    pub seed: u64,
    pub label: &'a str,
}

impl CpDetector for PrOBocpdAdapter<'_> {
    fn name(&self) -> String {
        self.label.to_string()
    }
    fn detect(&self, fix: &Fixture) -> Vec<ChangePoint> {
        self.det.detect_multivariate_seeded(&fix.data, self.seed)
    }
}

/// Per-channel univariate `BocpdDetector::<P>::detect` + OR-union
/// across channels with margin-aware dedup. Mirrors
/// [`FocusDetector::detect_multivariate`]'s per-dim pattern. Generic
/// over the predictive `P`, defaulting to `Nig`. Use this when:
///
/// - The fixture is multivariate but events are per-asset specific
///   (a single channel reacts; joint posterior dilutes the signal).
/// - You want to exercise a univariate predictive (`NigAr1`,
///   custom `Predictive` plug) on multivariate input. The joint
///   `BocpdDetector::detect_multivariate` ignores the `P` type
///   parameter and always uses NIW (per its doc-comment), so this
///   adapter is the only correctness-preserving way to test a
///   non-NIW likelihood multivariately today.
///
/// `prior_factory` builds a fresh `P` per channel (each channel has
/// its own warmup / sufficient stats). For default `Nig` use
/// `Default::default`. For `NigAr1` use `NigAr1::default_prior` etc.
///
/// `dedup_window` collapses union output entries within `window` bars
/// of the previous emitted CP. Default `0` = no dedup. Setting to the
/// fixture's `margin` keeps the union compact without losing distinct
/// nearby events (since `match_detections` does bipartite GT-event
/// matching, multi-fires don't stack into TPs anyway).
pub struct BocpdPerChannelAdapter<'a, P: Predictive = Nig> {
    pub label: &'a str,
    pub lambda: f64,
    pub max_rl: usize,
    pub prior_factory: Box<dyn Fn() -> P + 'a>,
    pub dedup_window: usize,
}

impl<'a> BocpdPerChannelAdapter<'a, Nig> {
    /// Constructor for the default `Nig` predictive. Each channel
    /// gets a fresh `BocpdDetector::new(lambda, max_rl)`-equivalent
    /// (NIG prior `(μ=0, κ=1, α=1, β=1)`).
    pub fn new(label: &'a str, lambda: f64, max_rl: usize) -> Self {
        Self {
            label,
            lambda,
            max_rl,
            prior_factory: Box::new(|| Nig::new(0.0, 1.0, 1.0, 1.0)),
            dedup_window: 0,
        }
    }
}

impl<'a, P: Predictive + 'a> BocpdPerChannelAdapter<'a, P> {
    /// Constructor for a non-default predictive `P`. `prior_factory`
    /// is called once per channel to get a fresh `P`.
    pub fn new_with_prior(
        label: &'a str,
        lambda: f64,
        max_rl: usize,
        prior_factory: impl Fn() -> P + 'a,
    ) -> Self {
        Self {
            label,
            lambda,
            max_rl,
            prior_factory: Box::new(prior_factory),
            dedup_window: 0,
        }
    }

    /// Builder: install a `prior_factory` closure that produces a
    /// fresh `P` per channel. Use this for non-default predictives.
    pub fn with_prior_factory(mut self, f: impl Fn() -> P + 'a) -> Self {
        self.prior_factory = Box::new(f);
        self
    }

    /// Builder: set the union dedup window. `0` keeps every fire.
    pub fn with_dedup_window(mut self, window: usize) -> Self {
        self.dedup_window = window;
        self
    }
}

impl<P: Predictive> CpDetector for BocpdPerChannelAdapter<'_, P> {
    fn name(&self) -> String {
        self.label.to_string()
    }
    fn detect(&self, fix: &Fixture) -> Vec<ChangePoint> {
        let n = fix.data.len();
        let d = fix.d;
        if n == 0 || d == 0 {
            return Vec::new();
        }
        let mut all: Vec<ChangePoint> = Vec::new();
        for k in 0..d {
            let series: Vec<f64> = fix.data.iter().map(|row| row[k]).collect();
            let det =
                BocpdDetector::with_prior(self.lambda, self.max_rl, (self.prior_factory)());
            for cp in det.detect(&series) {
                all.push(cp);
            }
        }
        all.sort_by_key(|c| c.index);
        if self.dedup_window == 0 {
            return all;
        }
        let mut out: Vec<ChangePoint> = Vec::with_capacity(all.len());
        for cp in all {
            if let Some(last) = out.last() {
                if cp.index <= last.index + self.dedup_window {
                    continue;
                }
            }
            out.push(cp);
        }
        out
    }
}

pub struct DmBocdAdapter<'a, M: MWeight> {
    pub det: &'a DmBocdDetector<M>,
    pub label: &'a str,
}

impl<M: MWeight> CpDetector for DmBocdAdapter<'_, M> {
    fn name(&self) -> String {
        self.label.to_string()
    }
    fn detect(&self, fix: &Fixture) -> Vec<ChangePoint> {
        self.det.detect_multivariate(&fix.data)
    }
}

pub struct EnsembleAdapter<'a> {
    pub det: &'a EnsembleDetector,
    pub mv: bool,
    pub label: &'a str,
}

impl CpDetector for EnsembleAdapter<'_> {
    fn name(&self) -> String {
        self.label.to_string()
    }
    fn detect(&self, fix: &Fixture) -> Vec<ChangePoint> {
        if self.mv {
            self.det.detect_multivariate(&fix.data)
        } else {
            self.det.detect(&fix.as_univariate())
        }
    }
}

/// Multivariate-only at the trait surface; `d=1` is wrapped into
/// `&[vec]` and routed through `detect_multivariate`. Avoids the
/// univariate `&mut self` path since `FocusDetector` is not `Clone`.
pub struct FocusAdapter<'a> {
    pub det: &'a FocusDetector,
    pub label: &'a str,
}

impl CpDetector for FocusAdapter<'_> {
    fn name(&self) -> String {
        self.label.to_string()
    }
    fn detect(&self, fix: &Fixture) -> Vec<ChangePoint> {
        self.det.detect_multivariate(&fix.data)
    }
}

#[cfg(feature = "joint-detection")]
pub struct ChenWuAdapter<'a> {
    pub det: &'a ChenWuDetector,
    pub label: &'a str,
}

#[cfg(feature = "joint-detection")]
impl CpDetector for ChenWuAdapter<'_> {
    fn name(&self) -> String {
        self.label.to_string()
    }
    fn detect(&self, fix: &Fixture) -> Vec<ChangePoint> {
        self.det
            .detect(&fix.as_univariate())
            .into_iter()
            .filter_map(|d| match d {
                Detection::ChangePoint(cp) => Some(cp),
                Detection::CollectiveAnomaly { .. } => None,
            })
            .collect()
    }
}

/// Univariate conformal adapter. Projects `ConformalCp → ChangePoint`
/// (timing_interval / coverage are dropped at the `CpDetector`
/// boundary by design). Holds the wrapper owned in a `RefCell`
/// because `ConformalCpWrapper::detect` requires `&mut self`.
pub struct ConformalAdapter<D: ScoredDetect> {
    pub det: RefCell<ConformalCpWrapper<D>>,
    pub label: String,
}

impl<D: ScoredDetect> ConformalAdapter<D> {
    pub fn new(wrapper: ConformalCpWrapper<D>, label: impl Into<String>) -> Self {
        Self {
            det: RefCell::new(wrapper),
            label: label.into(),
        }
    }
}

impl<D: ScoredDetect> CpDetector for ConformalAdapter<D> {
    fn name(&self) -> String {
        self.label.clone()
    }
    fn detect(&self, fix: &Fixture) -> Vec<ChangePoint> {
        self.det
            .borrow_mut()
            .detect(&fix.as_univariate())
            .into_iter()
            .map(|c| c.cp)
            .collect()
    }
}

/// Multivariate counterpart. Same projection (`ConformalCp.cp`) and
/// same `RefCell` ownership rationale.
pub struct ConformalMvAdapter<D: MvScoredDetect> {
    pub det: RefCell<ConformalCpWrapper<D>>,
    pub label: String,
}

impl<D: MvScoredDetect> ConformalMvAdapter<D> {
    pub fn new(wrapper: ConformalCpWrapper<D>, label: impl Into<String>) -> Self {
        Self {
            det: RefCell::new(wrapper),
            label: label.into(),
        }
    }
}

impl<D: MvScoredDetect> CpDetector for ConformalMvAdapter<D> {
    fn name(&self) -> String {
        self.label.clone()
    }
    fn detect(&self, fix: &Fixture) -> Vec<ChangePoint> {
        self.det
            .borrow_mut()
            .detect_multivariate(&fix.data)
            .into_iter()
            .map(|c| c.cp)
            .collect()
    }
}
