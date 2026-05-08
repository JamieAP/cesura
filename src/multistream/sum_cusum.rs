//! Sum-CUSUM aggregator over per-stream scores. Joint detector --
//! optimal in the dense-change regime where most streams shift
//! together (Mei 2010, *Biometrika*; Halme-Koivunen 2026 §III).
//!
//! Per step:
//!   1. Each stream produces a per-step change-evidence score.
//!   2. Per-stream CUSUM accumulates `(score - reference)`, reset to
//!      0 when the cumulative sum goes negative.
//!   3. Aggregate: sum across streams.
//!   4. Fire if `sum > τ` and cooldown elapsed.
//!
//! Unlike HC, no per-stream attribution: the sum is dimension-
//! agnostic and the joint detection doesn't separate streams.

use serde::{Deserialize, Serialize};

use super::{MultiStreamChangePoint, ScoreKind, ScoreStream};

/// Sum-CUSUM multi-stream aggregator. Generic over any [`ScoreStream`].
pub struct SumCusumAggregator<S: ScoreStream> {
    streams: Vec<S>,
    score_kind: ScoreKind,
    threshold: f64,
    /// Per-step subtraction in the per-stream CUSUM update -- the
    /// neutral score level. Defaults to `score_kind.neutral_reference()`.
    reference: f64,
    cooldown: usize,
    /// Steps before per-stream CUSUM accumulation begins. Constructor default 50.
    warmup: usize,
    last_emit: Option<usize>,
    step_count: usize,
    per_stream_w: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SumCusumAggregatorState {
    pub score_kind: ScoreKind,
    pub threshold: f64,
    pub reference: f64,
    pub cooldown: usize,
    #[serde(default = "default_sum_warmup")]
    pub warmup: usize,
    pub last_emit: Option<usize>,
    pub step_count: usize,
    pub per_stream_w: Vec<f64>,
}

fn default_sum_warmup() -> usize {
    30
}

impl<S: ScoreStream> SumCusumAggregator<S> {
    /// Construct a sum-CUSUM aggregator. All streams must agree on
    /// `score_kind()`. Defaults: threshold =
    /// `kind.default_sum_cusum_threshold()`, reference =
    /// `kind.neutral_reference()`, cooldown = 15, warmup = 50.
    ///
    /// # Panics
    /// - `streams` is empty.
    /// - Streams disagree on `score_kind()`.
    pub fn new(streams: Vec<S>) -> Self {
        assert!(!streams.is_empty(), "SumCusumAggregator requires ≥ 1 stream");
        let kind = streams[0].score_kind();
        for (i, s) in streams.iter().enumerate().skip(1) {
            assert_eq!(
                s.score_kind(),
                kind,
                "stream {i} score_kind {:?} ≠ stream 0 score_kind {kind:?}",
                s.score_kind()
            );
        }
        let d = streams.len();
        Self {
            streams,
            score_kind: kind,
            threshold: kind.default_sum_cusum_threshold(),
            reference: kind.neutral_reference(),
            cooldown: 15,
            warmup: 50,
            last_emit: None,
            step_count: 0,
            per_stream_w: vec![0.0; d],
        }
    }

    pub fn with_threshold(mut self, tau: f64) -> Self {
        assert!(tau > 0.0, "threshold must be > 0, got {tau}");
        self.threshold = tau;
        self
    }

    pub fn with_reference(mut self, r: f64) -> Self {
        self.reference = r;
        self
    }

    pub fn with_cooldown(mut self, c: usize) -> Self {
        self.cooldown = c;
        self
    }

    /// Steps before per-stream CUSUM accumulation begins (the all-mass-
    /// at-r=0 startup transient produces extreme scores that would
    /// bias the CUSUM). Constructor default 50.
    pub fn with_warmup(mut self, w: usize) -> Self {
        self.warmup = w;
        self
    }

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

            // During warmup we still consume the per-stream score
            // (so the underlying detectors advance their state) but
            // skip the CUSUM accumulation. Otherwise the all-mass-at-
            // r=0 startup permanently biases W upward.
            let i_pending = self.step_count;
            if i_pending >= self.warmup {
                for (idx, (s, &x)) in self.streams.iter_mut().zip(obs.iter()).enumerate() {
                    let score = s.step_score(x);
                    let w = self.per_stream_w[idx];
                    self.per_stream_w[idx] = (w + (score - self.reference)).max(0.0);
                }
            } else {
                for (s, &x) in self.streams.iter_mut().zip(obs.iter()) {
                    let _ = s.step_score(x);
                }
            }
            self.step_count += 1;
            let i = self.step_count - 1;

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

            let sum: f64 = self.per_stream_w.iter().sum();
            if sum > self.threshold {
                let confidence = (sum / self.threshold / 2.0).clamp(0.0, 1.0);
                out.push(MultiStreamChangePoint {
                    index: i,
                    confidence,
                    streams: Vec::new(),
                });
                self.last_emit = Some(i);
                // Reset per-stream CUSUM after firing -- standard
                // CUSUM-restart practice; without it the aggregator
                // stays above threshold and re-fires on every cooldown
                // expiry.
                for w in self.per_stream_w.iter_mut() {
                    *w = 0.0;
                }
            }
        }
        out
    }

    pub fn save_state(&self) -> SumCusumAggregatorState {
        SumCusumAggregatorState {
            score_kind: self.score_kind,
            threshold: self.threshold,
            reference: self.reference,
            cooldown: self.cooldown,
            warmup: self.warmup,
            last_emit: self.last_emit,
            step_count: self.step_count,
            per_stream_w: self.per_stream_w.clone(),
        }
    }

    pub fn restore(
        state: SumCusumAggregatorState,
        streams: Vec<S>,
    ) -> Result<Self, String> {
        if streams.is_empty() {
            return Err("restore requires ≥ 1 stream".into());
        }
        if streams.len() != state.per_stream_w.len() {
            return Err(format!(
                "stream count {} ≠ saved per_stream_w len {}",
                streams.len(),
                state.per_stream_w.len()
            ));
        }
        for (i, s) in streams.iter().enumerate() {
            if s.score_kind() != state.score_kind {
                return Err(format!(
                    "stream {i} score_kind {:?} ≠ saved {:?}",
                    s.score_kind(),
                    state.score_kind
                ));
            }
        }
        Ok(Self {
            streams,
            score_kind: state.score_kind,
            threshold: state.threshold,
            reference: state.reference,
            cooldown: state.cooldown,
            warmup: state.warmup,
            last_emit: state.last_emit,
            step_count: state.step_count,
            per_stream_w: state.per_stream_w,
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

#[cfg(test)]
mod tests {
    use super::super::ScoreKind;
    use super::*;

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
        let _ = SumCusumAggregator::new(vec![s1, s2]);
    }

    #[test]
    fn fires_on_dense_change() {
        // d=3. All streams emit BF=5 from step 60 onward (sustained
        // dense evidence past the warmup gate). Default threshold
        // 20.0 + default warmup 50 require a longer tape.
        let mk = |spike_step: usize| {
            let mut v = vec![0.5; 120];
            for s in &mut v[spike_step..] {
                *s = 5.0;
            }
            ScriptedStream::new(v, ScoreKind::BayesFactor)
        };
        let mut agg = SumCusumAggregator::new(vec![mk(60), mk(60), mk(60)]);
        let dummy: Vec<Vec<f64>> = (0..120).map(|_| vec![0.0; 3]).collect();
        let cps = agg.step(&dummy);
        assert!(
            !cps.is_empty(),
            "sum-CUSUM should fire on dense sustained evidence"
        );
        assert!(cps[0].streams.is_empty(), "joint detector → no attribution");
        assert!(cps[0].index >= 60, "fires at or after spike onset");
    }

    #[test]
    fn transient_sparse_spike_blocked_by_threshold() {
        // d=3. Stream 0 emits a single-step BF=5 spike; others sub-
        // neutral throughout. Sum-CUSUM with a high threshold (15.0)
        // should not fire on a transient single-stream blip -- sustained
        // dense evidence is what sum-CUSUM is built for. This documents
        // sum-CUSUM's regime: it lags transient sparse changes that HC
        // would catch immediately. Per Halme-Koivunen 2026 §III the
        // theoretical claim is asymptotic delay; in finite samples
        // threshold dominates.
        let mut hi = vec![0.5; 50];
        hi[10] = 5.0;
        let lo = vec![0.5; 50];
        let mut agg = SumCusumAggregator::new(vec![
            ScriptedStream::new(hi, ScoreKind::BayesFactor),
            ScriptedStream::new(lo.clone(), ScoreKind::BayesFactor),
            ScriptedStream::new(lo, ScoreKind::BayesFactor),
        ])
        .with_threshold(15.0);
        let dummy: Vec<Vec<f64>> = (0..50).map(|_| vec![0.0; 3]).collect();
        let cps = agg.step(&dummy);
        assert!(
            cps.is_empty(),
            "transient sparse spike should not breach τ=15 with sub-neutral peers, got {} CPs",
            cps.len()
        );
    }

    #[test]
    fn save_restore_round_trips() {
        let mk = || ScriptedStream::new(vec![0.7; 20], ScoreKind::BayesFactor);
        let mut agg = SumCusumAggregator::new(vec![mk(), mk(), mk()]);
        let dummy: Vec<Vec<f64>> = (0..10).map(|_| vec![0.0; 3]).collect();
        agg.step(&dummy);

        let state = agg.save_state();
        let json = serde_json::to_string(&state).unwrap();
        let restored: SumCusumAggregatorState = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.step_count, 10);
        assert_eq!(restored.per_stream_w.len(), 3);

        let _ = SumCusumAggregator::restore(restored, vec![mk(), mk(), mk()]).unwrap();
    }

    #[test]
    fn restore_rejects_dim_mismatch() {
        let mk = || ScriptedStream::new(vec![0.7; 20], ScoreKind::BayesFactor);
        let mut agg = SumCusumAggregator::new(vec![mk(), mk(), mk()]);
        agg.step(&[vec![0.0; 3]]);
        let state = agg.save_state();
        // Restore with d=2, expect error.
        let r = SumCusumAggregator::restore(state, vec![mk(), mk()]);
        assert!(r.is_err());
    }
}
