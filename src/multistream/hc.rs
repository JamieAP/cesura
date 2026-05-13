//! Higher-Criticism aggregator over per-stream scores.
//!
//! Implements the change-detection variant of Donoho & Jin (2004)
//! Higher Criticism, calibrated for sparse multi-stream change as in
//! Gong, Kipnis & Xie (2024-25, arXiv:2409.15597).
//!
//! Per step:
//!  1. Each stream produces a per-step change-evidence score.
//!  2. Scores → p-values via the [`ScoreKind`]-specific mapping.
//!  3. p-values sorted ascending → `p_(1) ≤ ... ≤ p_(d)`.
//!  4. HC statistic
//!     `HC* = max_{i ∈ 1..=⌈α₀·d⌉}  √d · (i/d - p_(i)) / √(p_(i)·(1-p_(i)))`
//!     where `α₀ = 0.5` (textbook default).
//!  5. Fire if `HC* > τ` and posterior is mature and cooldown elapsed.
//!
//! Posterior-maturity arming mirrors `detect_bayes_factor`: HC at
//! initialisation is undefined / NaN-ish for the all-mass-at-r=0
//! BOCPD start, so arming only flips true after HC first drops below
//! 1.0 (i.e. the per-stream posteriors have matured enough that the
//! d-vector of p-values isn't degenerate).
//!
//! ## p-value calibration: rank-transform
//!
//! Rather than analytic score → p mappings (which differ per
//! [`super::ScoreKind`] and are sensitive to the model's H_0
//! distribution), HC uses a per-stream **rank-transform**: maintain
//! a sliding window of recent scores; p_t = empirical proportion of
//! window values ≥ current score. Approximate uniform-rank behavior
//! requires comparable, exchangeable scores; dependence, drift and ties
//! can alter calibration for BF, cp_probs or other scores. Calibration is the
//! aggregator's responsibility; streams just emit raw scores.
//!
//! Window capacity defaults to 100
//! and is configurable via [`HcAggregator::with_rank_window`]. The
//! warmup gate skips aggregator firing until each stream's window
//! has at least `warmup` samples -- empirical CDF on too-few samples
//! gives high-variance p-values.
//!
//! ## Aggregator ownership
//!
//! Aggregators consume their streams (`Vec<S>`) and call `step()` /
//! `step_score()` on them per observation. **Do not call `step()`
//! on a stream owned by an aggregator out-of-band** -- the aggregator
//! relies on each stream advancing exactly once per aggregator step.
//! Use [`HcAggregator::streams`] for read-only access (e.g. inspecting
//! per-stream state). Mutable access via [`HcAggregator::streams_mut`]
//! exists but should be reserved for batch state save/restore, not
//! per-step interaction.

use std::collections::VecDeque;

use serde::{Deserialize, Serialize};

use super::{MultiStreamChangePoint, ScoreKind, ScoreStream};

/// Per-stream sliding window for empirical-CDF rank-based p-values.
/// Drops to a midrank-tie-breaker for finite-sample stability.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct RankWindow {
    capacity: usize,
    ring: VecDeque<f64>,
}

impl RankWindow {
    fn new(capacity: usize) -> Self {
        Self {
            capacity,
            ring: VecDeque::with_capacity(capacity),
        }
    }

    /// One-sided upper-tail rank p-value: proportion of window values
    /// `≥ score`, mid-rank tie-breaker, ε-floor against zero.
    fn pvalue(&self, score: f64) -> f64 {
        if self.ring.is_empty() {
            return 0.5;
        }
        let count_above = self.ring.iter().filter(|&&v| v >= score).count();
        let n = self.ring.len() as f64;
        ((count_above as f64 + 0.5) / (n + 1.0)).clamp(1e-12, 1.0)
    }

    fn update(&mut self, score: f64) {
        if self.ring.len() >= self.capacity {
            self.ring.pop_front();
        }
        self.ring.push_back(score);
    }

    #[allow(dead_code)]
    fn len(&self) -> usize {
        self.ring.len()
    }
}

/// Higher-Criticism multi-stream aggregator. Generic over any
/// [`ScoreStream`] -- typically `StreamingDetector` per asset.
pub struct HcAggregator<S: ScoreStream> {
    streams: Vec<S>,
    score_kind: ScoreKind,
    threshold: f64,
    cooldown: usize,
    /// Steps to skip before HC computation begins. Guards against the
    /// per-stream BOCPD posterior-maturity startup AND ensures the
    /// rank windows have enough samples for stable empirical CDF.
    /// Default 100.
    warmup: usize,
    persistence: usize,
    armed: bool,
    last_emit: Option<usize>,
    step_count: usize,
    /// Consecutive steps where HC has been > threshold (for the
    /// persistence filter). Reset to 0 on any step where HC ≤ τ.
    consecutive_above: usize,
    /// Streams that were in the contributing set on the most recent
    /// step where HC > τ. Captured at trigger time so persistence-
    /// confirmed fires can attribute to the original signal even if
    /// later above-τ steps shifted the contributing set slightly.
    pending_contributors: Vec<usize>,
    /// Per-stream rank-transform windows. Calibrate scores → p-values
    /// empirically so HC's threshold is meaningful regardless of the
    /// stream's score distribution under H_0.
    rank_windows: Vec<RankWindow>,
}

/// Aggregator-level state for save/restore. Streams are saved/restored
/// independently via their own `save_state` / `restore`; the user
/// passes the restored streams to [`HcAggregator::restore`] alongside
/// this state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HcAggregatorState {
    pub score_kind: ScoreKind,
    pub threshold: f64,
    pub cooldown: usize,
    #[serde(default = "default_hc_warmup")]
    pub warmup: usize,
    #[serde(default = "default_hc_persistence")]
    pub persistence: usize,
    pub armed: bool,
    pub last_emit: Option<usize>,
    pub step_count: usize,
    #[serde(default)]
    pub consecutive_above: usize,
    #[serde(default)]
    pub pending_contributors: Vec<usize>,
    #[serde(default)]
    pub(crate) rank_windows: Vec<RankWindow>,
}

fn default_hc_persistence() -> usize {
    2
}

fn default_hc_warmup() -> usize {
    100
}

impl<S: ScoreStream> HcAggregator<S> {
    /// Construct an HC aggregator over `streams`. All streams must
    /// emit the same [`ScoreKind`]; the constructor panics otherwise
    /// (HC's threshold is not invariant under p-value calibration).
    ///
    /// Defaults: `threshold = 3.0`,
    /// `cooldown = 15`. Override via [`Self::with_threshold`] /
    /// [`Self::with_cooldown`].
    ///
    /// # Panics
    /// - `streams` is empty.
    /// - Streams disagree on `score_kind()`.
    pub fn new(streams: Vec<S>) -> Self {
        assert!(!streams.is_empty(), "HcAggregator requires ≥ 1 stream");
        let kind = streams[0].score_kind();
        for (i, s) in streams.iter().enumerate().skip(1) {
            assert_eq!(
                s.score_kind(),
                kind,
                "stream {i} score_kind {:?} ≠ stream 0 score_kind {kind:?} -- HC requires same-kind streams",
                s.score_kind()
            );
        }
        let d = streams.len();
        Self {
            streams,
            score_kind: kind,
            // Default τ=3.0 calibrated against rank-transform
            // p-values: integration tests against StreamingDetector
            // (BF mode and CpProbability mode) fire on 1-of-4 5σ
            // sparse shifts and do not fire on stationary 300-step
            // tapes. HC asymptotic 95th percentile under H_0 for
            // d ∈ [3..5] is ≈ √(2 log log d) ≈ 1.5; τ=3.0 leaves
            // headroom for finite-sample p-value variance.
            threshold: 3.0,
            cooldown: 15,
            // warmup = rank window capacity so the empirical CDF is
            // populated from a full window before HC fires.
            warmup: 100,
            persistence: 2,
            armed: false,
            last_emit: None,
            step_count: 0,
            consecutive_above: 0,
            pending_contributors: Vec::new(),
            rank_windows: (0..d).map(|_| RankWindow::new(100)).collect(),
        }
    }

    pub fn with_threshold(mut self, tau: f64) -> Self {
        assert!(tau > 0.0, "threshold must be > 0, got {tau}");
        self.threshold = tau;
        self
    }

    pub fn with_cooldown(mut self, c: usize) -> Self {
        self.cooldown = c;
        self
    }

    /// Steps to skip before HC computation begins. Default 100. Must
    /// be ≥ rank-window capacity for stable empirical-CDF p-values.
    pub fn with_warmup(mut self, w: usize) -> Self {
        self.warmup = w;
        self
    }

    /// Rank-window capacity per stream. Larger = more stable
    /// empirical-CDF p-values but slower adaptation. Default 100.
    pub fn with_rank_window(mut self, capacity: usize) -> Self {
        assert!(capacity > 0, "rank window capacity must be > 0");
        let d = self.streams.len();
        self.rank_windows = (0..d).map(|_| RankWindow::new(capacity)).collect();
        self
    }

    /// Persistence filter: require HC > threshold for `n` consecutive
    /// steps before firing. Default 2.
    ///
    ///
    ///
    ///
    ///
    /// # Panics
    /// Panics if `n == 0`.
    pub fn with_persistence(mut self, n: usize) -> Self {
        assert!(n > 0, "persistence must be ≥ 1, got {n}");
        self.persistence = n;
        self
    }

    /// Process one d-vector observation per step. `observations[t][d]`
    /// is the value at step `t`, stream `d`. Returns any new
    /// detections (each carries the streams that crossed below the
    /// threshold p-value).
    pub fn step(&mut self, observations: &[Vec<f64>]) -> Vec<MultiStreamChangePoint> {
        let d = self.streams.len();
        let mut out = Vec::new();
        for obs in observations {
            assert_eq!(
                obs.len(),
                d,
                "observation dim {} ≠ aggregator dim {d}",
                obs.len()
            );

            let mut scores = Vec::with_capacity(d);
            for (s, &x) in self.streams.iter_mut().zip(obs.iter()) {
                scores.push(s.step_score(x));
            }
            self.step_count += 1;
            let i = self.step_count - 1;

            // Rank-transform p-values: query CDF rank BEFORE updating
            // the windows so the current score isn't included in its
            // own reference distribution.
            let pvals: Vec<f64> = scores
                .iter()
                .zip(self.rank_windows.iter())
                .map(|(&s, w)| w.pvalue(s))
                .collect();
            for (s, w) in scores.iter().zip(self.rank_windows.iter_mut()) {
                w.update(*s);
            }

            // Posterior-maturity warmup: skip until each stream's
            // rank window has filled enough for stable p-values AND
            // per-stream BOCPD posteriors have matured past the
            // all-mass-at-r=0 startup transient.
            if i < self.warmup {
                continue;
            }

            let in_cd = matches!(
                self.last_emit,
                Some(le) if i.saturating_sub(le) <= self.cooldown
            );
            if in_cd {
                continue;
            }

            let (hc_stat, contributing) = hc_statistic(&pvals);

            if !self.armed {
                if hc_stat < 1.0 {
                    self.armed = true;
                }
                self.consecutive_above = 0;
                self.pending_contributors.clear();
                continue;
            }

            if hc_stat > self.threshold {
                if self.consecutive_above == 0 {
                    // First step of a potential persistent excursion --
                    // capture the contributing set at the leading edge.
                    self.pending_contributors = contributing;
                }
                self.consecutive_above += 1;
                if self.consecutive_above >= self.persistence {
                    let confidence = (hc_stat / self.threshold / 2.0).clamp(0.0, 1.0);
                    out.push(MultiStreamChangePoint {
                        index: i,
                        confidence,
                        streams: std::mem::take(&mut self.pending_contributors),
                        per_stream_weights: pvals.clone(),
                    });
                    self.last_emit = Some(i);
                    self.armed = false;
                    self.consecutive_above = 0;
                }
            } else {
                // HC dipped below τ -- reset the persistence counter.
                self.consecutive_above = 0;
                self.pending_contributors.clear();
            }
        }
        out
    }

    pub fn save_state(&self) -> HcAggregatorState {
        HcAggregatorState {
            score_kind: self.score_kind,
            threshold: self.threshold,
            cooldown: self.cooldown,
            warmup: self.warmup,
            persistence: self.persistence,
            armed: self.armed,
            last_emit: self.last_emit,
            step_count: self.step_count,
            consecutive_above: self.consecutive_above,
            pending_contributors: self.pending_contributors.clone(),
            rank_windows: self.rank_windows.clone(),
        }
    }

    /// Reconstruct an `HcAggregator` from persisted state + restored
    /// streams. Caller is responsible for restoring each stream from
    /// its own `save_state` / `restore` round-trip.
    ///
    /// # Errors
    /// - Stream count mismatch with the saved state's implicit dim.
    /// - Streams disagree on `score_kind()` or on the saved
    ///   `score_kind`.
    pub fn restore(state: HcAggregatorState, streams: Vec<S>) -> Result<Self, String> {
        if streams.is_empty() {
            return Err("restore requires ≥ 1 stream".into());
        }
        for (i, s) in streams.iter().enumerate() {
            if s.score_kind() != state.score_kind {
                return Err(format!(
                    "stream {i} score_kind {:?} ≠ saved state score_kind {:?}",
                    s.score_kind(),
                    state.score_kind
                ));
            }
        }
        let d = streams.len();
        // Rank windows: forward-compat. If the saved state was empty
        // (pre-rank-transform snapshot), seed fresh windows -- the
        // aggregator will need a fresh warmup before firing again.
        let rank_windows = if state.rank_windows.is_empty() {
            (0..d).map(|_| RankWindow::new(200)).collect()
        } else if state.rank_windows.len() == d {
            state.rank_windows
        } else {
            return Err(format!(
                "saved rank_windows count {} ≠ stream count {d}",
                state.rank_windows.len()
            ));
        };
        Ok(Self {
            streams,
            score_kind: state.score_kind,
            threshold: state.threshold,
            cooldown: state.cooldown,
            warmup: state.warmup,
            persistence: state.persistence.max(1),
            armed: state.armed,
            last_emit: state.last_emit,
            step_count: state.step_count,
            consecutive_above: state.consecutive_above,
            pending_contributors: state.pending_contributors,
            rank_windows,
        })
    }

    pub fn step_count(&self) -> usize {
        self.step_count
    }

    pub fn streams(&self) -> &[S] {
        &self.streams
    }

    pub fn streams_mut(&mut self) -> &mut [S] {
        &mut self.streams
    }
}

/// Donoho-Jin Higher-Criticism statistic on p-values. Returns
/// `(HC*, contributing_streams)` where `contributing_streams` is the
/// set of original-stream indices whose p-values are at or below the
/// `i*`-th order statistic that maximised HC*.
fn hc_statistic(pvals: &[f64]) -> (f64, Vec<usize>) {
    let d = pvals.len();
    if d == 0 {
        return (0.0, Vec::new());
    }
    let mut idx: Vec<usize> = (0..d).collect();
    idx.sort_by(|&a, &b| pvals[a].total_cmp(&pvals[b]));
    let sorted: Vec<f64> = idx.iter().map(|&i| pvals[i]).collect();

    // α₀ = 0.5 textbook default; ceil to avoid empty range for small d.
    let n0 = ((d as f64) * 0.5).ceil() as usize;
    let n0 = n0.clamp(1, d);

    let sqrt_d = (d as f64).sqrt();
    let mut hc_max = f64::NEG_INFINITY;
    let mut i_star = 1usize;
    for i in 1..=n0 {
        let p_i = sorted[i - 1];
        let denom = (p_i * (1.0 - p_i)).max(1e-24).sqrt();
        let stat = sqrt_d * (i as f64 / d as f64 - p_i) / denom;
        if stat > hc_max {
            hc_max = stat;
            i_star = i;
        }
    }

    let mut contributing: Vec<usize> = idx.into_iter().take(i_star).collect();
    contributing.sort_unstable();
    (hc_max, contributing)
}

#[cfg(test)]
mod tests {
    use super::super::ScoreKind;
    use super::*;

    /// Toy ScoreStream for unit tests -- emits a pre-determined sequence.
    struct ScriptedStream {
        scripted: Vec<f64>,
        i: usize,
        kind: ScoreKind,
    }

    impl ScriptedStream {
        fn new(scripted: Vec<f64>, kind: ScoreKind) -> Self {
            Self {
                scripted,
                i: 0,
                kind,
            }
        }
    }

    impl ScoreStream for ScriptedStream {
        fn step_score(&mut self, _x: f64) -> f64 {
            let v = self.scripted[self.i];
            self.i += 1;
            v
        }
        fn score_kind(&self) -> ScoreKind {
            self.kind
        }
        fn step_count(&self) -> usize {
            self.i
        }
    }

    #[test]
    #[should_panic(expected = "score_kind")]
    fn refuses_mixed_score_kind() {
        let s1 = ScriptedStream::new(vec![1.0; 5], ScoreKind::BayesFactor);
        let s2 = ScriptedStream::new(vec![1.0; 5], ScoreKind::CpProbability);
        let _ = HcAggregator::new(vec![s1, s2]);
    }

    #[test]
    fn accepts_cpprobability_streams_with_rank_transform() {
        // After rank-transform shipped, HC works with any monotone
        // score stream. CpProbability is a first-class citizen.
        let s1 = ScriptedStream::new(vec![0.05; 50], ScoreKind::CpProbability);
        let s2 = ScriptedStream::new(vec![0.05; 50], ScoreKind::CpProbability);
        let _ = HcAggregator::new(vec![s1, s2]);
    }

    #[test]
    fn rank_window_pvalue_uniform_under_iid_noise() {
        // p-values from rank-transform on iid noise should land near
        // the median 0.5 -- empirical CDF rank is uniform under H_0.
        let mut w = RankWindow::new(100);
        // Seed window with iid samples
        let samples: Vec<f64> = (0..100).map(|i| (i as f64).sin()).collect();
        for &s in &samples {
            w.update(s);
        }
        // Query a "median" score (around 0)
        let p = w.pvalue(0.0);
        assert!(
            (0.3..=0.7).contains(&p),
            "median rank should give p ≈ 0.5, got {p}"
        );
    }

    #[test]
    fn rank_window_pvalue_extreme_for_outlier() {
        let mut w = RankWindow::new(100);
        for i in 0..100 {
            w.update(i as f64);
        }
        let p = w.pvalue(1000.0);
        assert!(p < 0.02, "extreme outlier should give very small p, got {p}");
    }

    #[test]
    fn fires_on_sparse_change_one_stream() {
        // d=4, 60 steps. Stream 0 spikes (BF=20) at step 50; others
        // stay at BF=1.0 throughout. Rank window = 30, warmup = 30 →
        // by step 50 the windows are filled and the spike registers
        // as an extreme outlier on stream 0 only.
        let mk = |spike_step: Option<usize>, spike: f64| {
            let mut v = vec![1.0; 60];
            if let Some(idx) = spike_step {
                v[idx] = spike;
            }
            ScriptedStream::new(v, ScoreKind::BayesFactor)
        };
        let s0 = mk(Some(50), 20.0);
        let s1 = mk(None, 0.0);
        let s2 = mk(None, 0.0);
        let s3 = mk(None, 0.0);
        let mut agg = HcAggregator::new(vec![s0, s1, s2, s3])
            .with_threshold(1.5)
            .with_rank_window(30)
            .with_warmup(30)
            .with_persistence(1); // single-step fixture; persistence=1 behavior
        let dummy_obs: Vec<Vec<f64>> = (0..60).map(|_| vec![0.0, 0.0, 0.0, 0.0]).collect();
        let cps = agg.step(&dummy_obs);
        assert!(!cps.is_empty(), "HC should fire on a 1-of-4 spike");
        let cp = &cps[0];
        assert_eq!(
            cp.streams,
            vec![0],
            "attribution should identify stream 0 only, got {:?}",
            cp.streams
        );
        assert!(cp.index >= 50, "fires at or after the spike step");
    }

    #[test]
    fn no_fire_on_uniform_constant_scores() {
        // d=3. All streams emit a constant score; under rank-
        // transform every value gets p ≈ 0.5 (no rank, no extreme).
        // HC stat ≈ 0; never crosses default threshold 2.0.
        let mk = || ScriptedStream::new(vec![0.8; 100], ScoreKind::BayesFactor);
        let mut agg = HcAggregator::new(vec![mk(), mk(), mk()])
            .with_rank_window(30)
            .with_warmup(30);
        let dummy: Vec<Vec<f64>> = (0..100).map(|_| vec![0.0; 3]).collect();
        let cps = agg.step(&dummy);
        assert!(
            cps.is_empty(),
            "uniform constant evidence should not fire, got {} CPs",
            cps.len()
        );
    }

    #[test]
    fn save_restore_round_trips() {
        let mk = || ScriptedStream::new(vec![0.7; 20], ScoreKind::BayesFactor);
        let mut agg = HcAggregator::new(vec![mk(), mk(), mk()])
            .with_threshold(2.5)
            .with_rank_window(10)
            .with_warmup(10);
        let dummy: Vec<Vec<f64>> = (0..15).map(|_| vec![0.0; 3]).collect();
        agg.step(&dummy);

        let state = agg.save_state();
        let json = serde_json::to_string(&state).unwrap();
        let restored: HcAggregatorState = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.threshold, 2.5);
        assert_eq!(restored.cooldown, 15);
        assert_eq!(restored.score_kind, ScoreKind::BayesFactor);
        assert_eq!(restored.step_count, 15);
        assert_eq!(restored.rank_windows.len(), 3);

        let _ = HcAggregator::restore(restored, vec![mk(), mk(), mk()]).unwrap();
    }

    #[test]
    fn hc_stat_zero_for_uniform_pvalues() {
        let p = vec![0.5; 10];
        let (hc, _) = hc_statistic(&p);
        // For uniform p-values: i/d - p_(i) = i/d - 0.5; for i=1..5:
        // 0.1-0.5, 0.2-0.5, ..., 0.5-0.5. Max is at i=5: 0. So HC=0.
        assert!(hc.abs() < 1e-9, "uniform p=0.5 should give HC≈0, got {hc}");
    }

    #[test]
    fn hc_stat_positive_for_extreme_minimum() {
        let p = vec![1e-6, 0.5, 0.5, 0.5];
        let (hc, contrib) = hc_statistic(&p);
        assert!(hc > 1.0, "extreme minimum should drive HC > 1, got {hc}");
        assert_eq!(contrib, vec![0]);
    }

    #[test]
    fn persistence_blocks_transient_single_step_spike() {
        // Stream 0 has a SINGLE-step spike at step 50; sub-neutral
        // before and after. With persistence=3, HC sees one above-τ
        // step then drops back -- the persistence counter resets,
        // never fires.
        let mk = |spike_step: Option<usize>, spike: f64| {
            let mut v = vec![1.0; 100];
            if let Some(idx) = spike_step {
                v[idx] = spike;
            }
            ScriptedStream::new(v, ScoreKind::BayesFactor)
        };
        let s0 = mk(Some(50), 50.0); // huge isolated spike
        let s1 = mk(None, 0.0);
        let s2 = mk(None, 0.0);
        let s3 = mk(None, 0.0);
        let mut agg = HcAggregator::new(vec![s0, s1, s2, s3])
            .with_threshold(1.5)
            .with_rank_window(30)
            .with_warmup(30)
            .with_persistence(3);
        let dummy: Vec<Vec<f64>> = (0..100).map(|_| vec![0.0; 4]).collect();
        let cps = agg.step(&dummy);
        assert!(
            cps.is_empty(),
            "persistence=3 should block a 1-step transient spike, got {:?}",
            cps.iter().map(|c| c.index).collect::<Vec<_>>()
        );
    }

    #[test]
    fn persistence_allows_sustained_excursion() {
        // Stream 0 has a SUSTAINED elevated score from step 50 onward
        // (10 consecutive steps of 50.0). With persistence=2 HC fires
        // after the second consecutive above-τ step.
        //
        // Note: rank-transform adapts -- as the sustained value enters
        // the window, its rank rises (less "extreme") and HC drops.
        // For windows of capacity 30, sustained values stay rank-
        // extreme for ~2-3 steps before the window saturates with the
        // shifted value. persistence=2 fits within that budget;
        // persistence=3 wouldn't on this fixture. Real-world detectors
        // pair persistence with a larger window OR sliding-baseline
        // window that excludes recent samples.
        let mk = |sustain_from: Option<usize>, value: f64| {
            let mut v = vec![1.0; 100];
            if let Some(start) = sustain_from {
                for s in &mut v[start..(start + 10).min(100)] {
                    *s = value;
                }
            }
            ScriptedStream::new(v, ScoreKind::BayesFactor)
        };
        let s0 = mk(Some(50), 50.0);
        let s1 = mk(None, 0.0);
        let s2 = mk(None, 0.0);
        let s3 = mk(None, 0.0);
        let mut agg = HcAggregator::new(vec![s0, s1, s2, s3])
            .with_threshold(1.5)
            .with_rank_window(30)
            .with_warmup(30)
            .with_persistence(2);
        let dummy: Vec<Vec<f64>> = (0..100).map(|_| vec![0.0; 4]).collect();
        let cps = agg.step(&dummy);
        assert!(!cps.is_empty(), "persistence=2 should fire on sustained excursion");
        assert_eq!(
            cps[0].streams,
            vec![0],
            "attribution captured at leading edge, got {:?}",
            cps[0].streams
        );
        assert!(
            cps[0].index >= 50 && cps[0].index <= 60,
            "fires within sustained window, got {}",
            cps[0].index
        );
    }

    #[test]
    fn persistence_default_is_two() {
        // Default persistence=2 blocks 1-step transients
        // and fires on 2-step sustained excursions. Sustained spike
        // of 50.0 over 5 steps from step 50; rank-extreme for the
        // first ~2-3 steps before window saturates → fires.
        let mk = |sustain_from: Option<usize>, value: f64| {
            let mut v = vec![1.0; 100];
            if let Some(start) = sustain_from {
                for s in &mut v[start..(start + 5).min(100)] {
                    *s = value;
                }
            }
            ScriptedStream::new(v, ScoreKind::BayesFactor)
        };
        let s0 = mk(Some(50), 50.0);
        let s1 = mk(None, 0.0);
        let s2 = mk(None, 0.0);
        let s3 = mk(None, 0.0);
        let mut agg = HcAggregator::new(vec![s0, s1, s2, s3])
            .with_threshold(1.5)
            .with_rank_window(30)
            .with_warmup(30);
        let dummy: Vec<Vec<f64>> = (0..100).map(|_| vec![0.0; 4]).collect();
        let cps = agg.step(&dummy);
        assert!(
            !cps.is_empty(),
            "persistence=2 (default) should fire on sustained excursion"
        );
    }

    #[test]
    fn persistence_state_round_trips() {
        let mk = || ScriptedStream::new(vec![0.7; 60], ScoreKind::BayesFactor);
        let mut agg = HcAggregator::new(vec![mk(), mk(), mk()])
            .with_rank_window(20)
            .with_warmup(20)
            .with_persistence(3);
        let dummy: Vec<Vec<f64>> = (0..40).map(|_| vec![0.0; 3]).collect();
        agg.step(&dummy);

        let json = serde_json::to_string(&agg.save_state()).unwrap();
        let restored: HcAggregatorState = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.persistence, 3, "persistence must survive serde");
        let _ = HcAggregator::restore(restored, vec![mk(), mk(), mk()]).unwrap();
    }

    #[test]
    fn persistence_forward_compat_zero_clamped_to_one() {
        // Snapshots without `persistence` use the default of 2. But
        // a hand-crafted snapshot with 0 should clamp to 1 on restore
        // (avoid the divide-by-zero / fire-immediately edge case).
        let mk = || ScriptedStream::new(vec![0.7; 60], ScoreKind::BayesFactor);
        let mut agg = HcAggregator::new(vec![mk(), mk(), mk()]);
        let _ = agg.step(&(0..30).map(|_| vec![0.0; 3]).collect::<Vec<_>>());
        let mut state = agg.save_state();
        state.persistence = 0;
        let restored = HcAggregator::restore(state, vec![mk(), mk(), mk()]).unwrap();
        assert!(
            restored.persistence >= 1,
            "persistence must be clamped to ≥ 1"
        );
    }

    #[test]
    #[should_panic(expected = "persistence must be ≥ 1")]
    fn with_persistence_zero_panics() {
        let mk = || ScriptedStream::new(vec![0.7; 10], ScoreKind::BayesFactor);
        let _ = HcAggregator::new(vec![mk(), mk()]).with_persistence(0);
    }
}
