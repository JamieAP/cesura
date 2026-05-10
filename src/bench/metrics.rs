//! Single source of truth for precision / recall / F1 / FAR / delay.
//!
//! Greedy nearest-first matching with index tolerance `fix.margin`.
//! For real-world fixtures with epochs, the caller converts the
//! epoch-window margin to an index margin at fixture build time;
//! `evaluate` itself is epoch-agnostic.

use crate::ChangePoint;
use crate::bench::fixture::Fixture;
use crate::eval::match_detections;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Metrics {
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
    /// CPs farther than `margin` from every ground-truth event.
    /// (Same as FP under the F1-with-margin protocol; surfaced
    /// separately for legacy reports.)
    pub far: usize,
    pub mean_delay_idx: Option<f64>,
    /// `[i]` is true iff `ground_truth[i]` got a CP within margin.
    pub per_event_hit: Vec<bool>,
    pub n_events: usize,
    pub n_cps: usize,
}

/// F1-with-margin per van den Burg & Williams (2022), §3.
/// - TP = CPs whose nearest unmatched ground-truth is within margin.
/// - FP = CPs with no unmatched ground-truth in margin.
/// - FN = ground-truth without any CP in margin.
///
/// Greedy nearest-first matching is delegated to
/// [`crate::eval::match_detections`]. `per_event_hit` is computed
/// separately because `match_detections` doesn't surface it.
pub fn evaluate(cps: &[ChangePoint], fix: &Fixture) -> Metrics {
    let gt = &fix.ground_truth;
    let margin = fix.margin;
    let n_cps = cps.len();
    let detected: Vec<usize> = cps.iter().map(|c| c.index).collect();
    let m = match_detections(&detected, gt, margin);

    let mean_delay_idx = if m.tp == 0 { None } else { Some(m.mean_delay) };
    let per_event_hit: Vec<bool> = gt
        .iter()
        .map(|&g| {
            detected
                .iter()
                .any(|&i| (i as i64 - g as i64).unsigned_abs() as usize <= margin)
        })
        .collect();

    Metrics {
        precision: m.precision,
        recall: m.recall,
        f1: m.f1,
        far: m.fp,
        mean_delay_idx,
        per_event_hit,
        n_events: gt.len(),
        n_cps,
    }
}

// ── Epoch-window event matching ────────────────────────────────────
//
// Used by examples that match CPs against dated KNOWN_EVENTS rather
// than index-level ground truth: `dm_bocd_eval`, `comprehensive_report`.
// `events[i] = (label, epoch_seconds)`. `epochs[i]` is the unix-second
// timestamp of bar `i` in the tape. A CP at index `i` matches event
// `e` iff `|epochs[i] - e.epoch| ≤ window_secs`.

/// Count events with at least one CP within `±window_secs`.
pub fn count_event_hits(
    epochs: &[i64],
    cp_indices: &[usize],
    events: &[(&str, i64)],
    window_secs: i64,
) -> usize {
    events
        .iter()
        .filter(|(_, target)| {
            cp_indices
                .iter()
                .any(|&i| (epochs[i] - target).abs() <= window_secs)
        })
        .count()
}

/// Per-event hit booleans aligned with `events`.
pub fn per_event_hits(
    epochs: &[i64],
    cp_indices: &[usize],
    events: &[(&str, i64)],
    window_secs: i64,
) -> Vec<bool> {
    events
        .iter()
        .map(|(_, target)| {
            cp_indices
                .iter()
                .any(|&i| (epochs[i] - target).abs() <= window_secs)
        })
        .collect()
}

/// F1 with margin τ on epoch-window event matching (van den Burg &
/// Williams 2022, arXiv:2003.06222 §3). Returns
/// `(precision, recall, f1)`.
///
/// - **TP** = CPs with at least one event within `±window_secs`.
/// - **FP** = CPs with no event within margin.
/// - **FN** = events with no CP within margin.
pub fn f1_with_margin(
    epochs: &[i64],
    cp_indices: &[usize],
    events: &[(&str, i64)],
    window_secs: i64,
) -> (f64, f64, f64) {
    let tp = cp_indices
        .iter()
        .filter(|&&i| {
            events
                .iter()
                .any(|(_, target)| (epochs[i] - target).abs() <= window_secs)
        })
        .count();
    let fp = cp_indices.len().saturating_sub(tp);
    let fn_ = events
        .iter()
        .filter(|(_, target)| {
            cp_indices
                .iter()
                .all(|&i| (epochs[i] - target).abs() > window_secs)
        })
        .count();
    let precision = if tp + fp == 0 { 0.0 } else { tp as f64 / (tp + fp) as f64 };
    let recall = if tp + fn_ == 0 { 0.0 } else { tp as f64 / (tp + fn_) as f64 };
    let f1 = if precision + recall == 0.0 {
        0.0
    } else {
        2.0 * precision * recall / (precision + recall)
    };
    (precision, recall, f1)
}

/// Count CPs farther than `far_threshold_secs` from every known event
/// (false-alarm proxy when ground-truth CP labels are unavailable).
pub fn count_far_cps(
    epochs: &[i64],
    cp_indices: &[usize],
    events: &[(&str, i64)],
    far_threshold_secs: i64,
) -> usize {
    cp_indices
        .iter()
        .filter(|&&i| {
            events
                .iter()
                .all(|(_, target)| (epochs[i] - target).abs() > far_threshold_secs)
        })
        .count()
}

#[cfg(test)]
mod epoch_window_tests {
    use super::*;

    #[test]
    fn count_event_hits_first_fire_only() {
        let epochs = vec![0, 3600, 7200, 10800, 14400];
        let events = vec![("E", 7200)];
        assert_eq!(count_event_hits(&epochs, &[1, 2], &events, 3600), 1);
        assert_eq!(count_event_hits(&epochs, &[0, 4], &events, 3600), 0);
        assert_eq!(count_event_hits(&epochs, &[3], &events, 3600), 1);
        assert_eq!(count_event_hits(&epochs, &[3], &events, 3500), 0);
    }

    #[test]
    fn per_event_hits_returns_per_event_booleans() {
        let epochs = vec![0, 3600, 7200, 10800, 14400, 18000];
        let events = vec![("A", 3600), ("B", 14400), ("C", 100_000)];
        let cps = [1, 4];
        let h = per_event_hits(&epochs, &cps, &events, 1800);
        assert_eq!(h, vec![true, true, false]);
    }

    #[test]
    fn count_far_cps_thresholds_against_all_events() {
        let epochs = vec![0, 86_400, 86_400 * 5, 86_400 * 30];
        let events = vec![("E", 0)];
        let far_secs: i64 = 7 * 86_400;
        assert_eq!(count_far_cps(&epochs, &[0, 1, 2, 3], &events, far_secs), 1);
    }

    #[test]
    fn f1_with_margin_perfect_match() {
        let epochs = vec![0, 3600, 7200, 10800];
        let events = vec![("E", 3600)];
        let (p, r, f1) = f1_with_margin(&epochs, &[1], &events, 1800);
        assert_eq!((p, r, f1), (1.0, 1.0, 1.0));
    }

    #[test]
    fn f1_with_margin_zero_when_no_match() {
        let epochs = vec![0, 3600, 7200, 10800];
        let events = vec![("E", 100_000)];
        let (p, r, f1) = f1_with_margin(&epochs, &[1, 2], &events, 1800);
        assert_eq!((p, r, f1), (0.0, 0.0, 0.0));
    }
}
