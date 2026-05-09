#![forbid(unsafe_code)]
//! Bayesian Online Change Point Detection (Adams & MacKay 2007)
//! with Normal-Inverse-Gamma conjugate prior.
//!
//! Collective anomalies -- a temporary regime that returns to baseline --
//! surface as a pair of detections (start + end) without any special-case
//! handling. See `tests::detects_collective_anomaly`.
//!
//! # Example
//!
//! ```
//! use cesura::{BocpdDetector, ChangePoint};
//!
//! let data: Vec<f64> = std::iter::repeat(0.0).take(100)
//!     .chain(std::iter::repeat(5.0).take(100))
//!     .collect();
//!
//! let detector = BocpdDetector::new(200.0, 250);
//! let change_points = detector.detect(&data);
//!
//! assert!(!change_points.is_empty());
//! assert!((change_points[0].index as i64 - 100).abs() < 15);
//! ```

pub mod auto_beta;
#[cfg(feature = "joint-detection")]
pub mod auto_q0;
#[cfg(feature = "joint-detection")]
pub mod chen_wu;
pub mod conformal;
pub mod detrend;
pub mod dm_bocd;
pub mod ensemble;
#[cfg(any(test, feature = "test-utils"))]
pub mod eval;
pub mod focus;
pub mod nig_ar1;
pub mod niw;
pub mod predictive;
pub mod streaming;
#[cfg(feature = "joint-detection")]
pub mod streaming_chen_wu;
pub mod multistream;

mod bocpd;
mod math;
mod nig;

pub use dm_bocd::DmBocdDetector;
pub use bocpd::BocpdDetector;
pub use conformal::{ConformalCp, ConformalCpWrapper, MvScoredDetect, ScoredDetect};
pub use ensemble::EnsembleDetector;
pub use multistream::{
    HcAggregator, HcAggregatorState, MultiStreamChangePoint, ScoreKind, ScoreStream,
    SumCusumAggregator, SumCusumAggregatorState,
};
pub use nig::Nig;
pub use nig_ar1::NigAr1;
pub use predictive::Predictive;
pub(crate) use math::{lgamma, log_add_exp};

/// A detected change point with its index and confidence score.
#[derive(Debug, Clone)]
pub struct ChangePoint {
    /// Index in the input data where the change was detected.
    pub index: usize,
    /// Confidence score in `[0, 1]`. Higher means more confident.
    pub confidence: f64,
    /// Absolute shift magnitude in units of global σ.
    /// Computed from normalized data: `|mean_after - mean_before|`.
    pub shift_sigma: f64,
}

/// Default mass-pruning cutoff: tail-end run lengths whose posterior mass
/// falls below 1e-4 are dropped. Matches the order of magnitude of
/// `changepoint::BocpdTruncated`. See `BocpdDetector::with_mass_cutoff`.
pub const DEFAULT_MASS_CUTOFF: f64 = 1e-4;
