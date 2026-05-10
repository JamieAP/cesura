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
use crate::{BocpdDetector, ChangePoint, EnsembleDetector};

#[cfg(feature = "joint-detection")]
use crate::chen_wu::{ChenWuDetector, Detection};

pub struct BocpdAdapter<'a> {
    pub det: &'a BocpdDetector,
    pub mv: bool,
    pub label: &'a str,
}

impl CpDetector for BocpdAdapter<'_> {
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
