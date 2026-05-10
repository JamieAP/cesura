//! Bridges multistream aggregators (`HcAggregator`,
//! `SumCusumAggregator`) to [`CpDetector`] without touching their public
//! APIs. Pure dispatch wrappers.
//!

use crate::bench::detector::{CpDetector, DetectionResult};
use crate::bench::fixture::Fixture;
use crate::multistream::{HcAggregator, MultiStreamChangePoint, SumCusumAggregator};
use crate::streaming::StreamingDetector;
use crate::ChangePoint;

fn project_with_attribution(
    fires: Vec<MultiStreamChangePoint>,
) -> (Vec<ChangePoint>, Vec<Vec<usize>>) {
    let mut cps = Vec::with_capacity(fires.len());
    let mut attr = Vec::with_capacity(fires.len());
    for m in fires {
        cps.push(ChangePoint {
            index: m.index,
            confidence: m.confidence,
            shift_sigma: 0.0,
        });
        attr.push(m.streams);
    }
    (cps, attr)
}

/// Factory: given a stream count `d`, produce `d` fresh
/// `StreamingDetector`s. Must be `Send + Sync` so adapters can live in
/// trait objects across threads if a future caller wants that.
pub type StreamFactory = Box<dyn Fn(usize) -> Vec<StreamingDetector> + Send + Sync>;

pub fn make_bf_streams_factory() -> StreamFactory {
    Box::new(|d| {
        (0..d)
            .map(|_| StreamingDetector::new(200.0, 250).with_bayes_factor_rule(2.0, 3, 15))
            .collect()
    })
}

pub fn make_map_streams_factory() -> StreamFactory {
    Box::new(|d| (0..d).map(|_| StreamingDetector::new(200.0, 250)).collect())
}

pub struct HcAggregatorAdapter {
    pub streams: StreamFactory,
    pub threshold: f64,
    pub cooldown: Option<usize>,
    pub warmup: Option<usize>,
    pub rank_window: Option<usize>,
    pub persistence: Option<usize>,
    pub label: String,
}

impl HcAggregatorAdapter {
    /// Minimal constructor. Use `with_*` builders to set knobs.
    pub fn new(streams: StreamFactory, threshold: f64, label: impl Into<String>) -> Self {
        Self {
            streams,
            threshold,
            cooldown: None,
            warmup: None,
            rank_window: None,
            persistence: None,
            label: label.into(),
        }
    }

    /// BF-streams convenience matching `aggregator_eval.rs`.
    pub fn with_bf_streams(threshold: f64) -> Self {
        Self::new(make_bf_streams_factory(), threshold, "HC(BF)")
    }

    /// MAP-streams convenience matching `aggregator_eval.rs`.
    pub fn with_map_streams(threshold: f64) -> Self {
        Self::new(make_map_streams_factory(), threshold, "HC(MAP)")
    }

    pub fn cooldown(mut self, c: usize) -> Self {
        self.cooldown = Some(c);
        self
    }
    pub fn warmup(mut self, w: usize) -> Self {
        self.warmup = Some(w);
        self
    }
    pub fn rank_window(mut self, c: usize) -> Self {
        self.rank_window = Some(c);
        self
    }
    pub fn persistence(mut self, n: usize) -> Self {
        self.persistence = Some(n);
        self
    }
    pub fn label(mut self, label: impl Into<String>) -> Self {
        self.label = label.into();
        self
    }
}

impl HcAggregatorAdapter {
    fn run(&self, fix: &Fixture) -> Vec<MultiStreamChangePoint> {
        let streams = (self.streams)(fix.d);
        let mut agg = HcAggregator::new(streams).with_threshold(self.threshold);
        if let Some(c) = self.cooldown {
            agg = agg.with_cooldown(c);
        }
        if let Some(w) = self.warmup {
            agg = agg.with_warmup(w);
        }
        if let Some(c) = self.rank_window {
            agg = agg.with_rank_window(c);
        }
        if let Some(n) = self.persistence {
            agg = agg.with_persistence(n);
        }
        agg.step(&fix.data)
    }
}

impl CpDetector for HcAggregatorAdapter {
    fn name(&self) -> String {
        self.label.clone()
    }

    fn detect(&self, fix: &Fixture) -> Vec<ChangePoint> {
        self.detect_full(fix).cps
    }

    /// HC surfaces per-stream attribution: each fire carries the
    /// streams whose p-value crossed below the HC threshold (sorted
    /// ascending). `attribution[i] = which streams drove cps[i]`.
    fn detect_full(&self, fix: &Fixture) -> DetectionResult {
        let (cps, attribution) = project_with_attribution(self.run(fix));
        DetectionResult { cps, attribution }
    }
}

pub struct SumCusumAggregatorAdapter {
    pub streams: StreamFactory,
    pub threshold: f64,
    pub reference: Option<f64>,
    pub cooldown: Option<usize>,
    pub warmup: Option<usize>,
    pub label: String,
}

impl SumCusumAggregatorAdapter {
    pub fn new(streams: StreamFactory, threshold: f64, label: impl Into<String>) -> Self {
        Self {
            streams,
            threshold,
            reference: None,
            cooldown: None,
            warmup: None,
            label: label.into(),
        }
    }

    pub fn with_bf_streams(threshold: f64) -> Self {
        Self::new(make_bf_streams_factory(), threshold, "SumCusum(BF)")
    }

    pub fn with_map_streams(threshold: f64) -> Self {
        Self::new(make_map_streams_factory(), threshold, "SumCusum(MAP)")
    }

    pub fn reference(mut self, r: f64) -> Self {
        self.reference = Some(r);
        self
    }
    pub fn cooldown(mut self, c: usize) -> Self {
        self.cooldown = Some(c);
        self
    }
    pub fn warmup(mut self, w: usize) -> Self {
        self.warmup = Some(w);
        self
    }
    pub fn label(mut self, label: impl Into<String>) -> Self {
        self.label = label.into();
        self
    }
}

impl SumCusumAggregatorAdapter {
    fn run(&self, fix: &Fixture) -> Vec<MultiStreamChangePoint> {
        let streams = (self.streams)(fix.d);
        let mut agg = SumCusumAggregator::new(streams).with_threshold(self.threshold);
        if let Some(r) = self.reference {
            agg = agg.with_reference(r);
        }
        if let Some(c) = self.cooldown {
            agg = agg.with_cooldown(c);
        }
        if let Some(w) = self.warmup {
            agg = agg.with_warmup(w);
        }
        agg.step(&fix.data)
    }
}

impl CpDetector for SumCusumAggregatorAdapter {
    fn name(&self) -> String {
        self.label.clone()
    }

    fn detect(&self, fix: &Fixture) -> Vec<ChangePoint> {
        self.detect_full(fix).cps
    }

    /// Sum-CUSUM is a joint detector: per-stream attribution is not
    /// separable. Per-CP attribution is therefore always empty here,
    /// matching `MultiStreamChangePoint.streams` from the underlying
    /// aggregator. Documented at the trait surface as "joint detector".
    fn detect_full(&self, fix: &Fixture) -> DetectionResult {
        let (cps, attribution) = project_with_attribution(self.run(fix));
        DetectionResult { cps, attribution }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bench::FixtureRegistry;

    fn synthetic_3sigma() -> Fixture {
        FixtureRegistry::synthetic()
            .into_iter()
            .find(|f| f.name == "noisy_3sigma")
            .expect("noisy_3sigma scenario present")
    }

    #[test]
    fn hc_adapter_parity_with_direct_call() {
        let fix = synthetic_3sigma();

        // Direct path.
        let streams_direct: Vec<StreamingDetector> = (0..fix.d)
            .map(|_| StreamingDetector::new(200.0, 250).with_bayes_factor_rule(2.0, 3, 15))
            .collect();
        let mut hc_direct = HcAggregator::new(streams_direct).with_threshold(3.0);
        let direct: Vec<usize> = hc_direct.step(&fix.data).into_iter().map(|m| m.index).collect();

        // Adapter path.
        let adapter = HcAggregatorAdapter::with_bf_streams(3.0);
        let harness: Vec<usize> = adapter.detect(&fix).into_iter().map(|c| c.index).collect();

        assert_eq!(direct, harness);
    }

    #[test]
    fn sum_cusum_adapter_parity_with_direct_call() {
        let fix = synthetic_3sigma();

        let streams_direct: Vec<StreamingDetector> = (0..fix.d)
            .map(|_| StreamingDetector::new(200.0, 250).with_bayes_factor_rule(2.0, 3, 15))
            .collect();
        let mut agg_direct = SumCusumAggregator::new(streams_direct).with_threshold(20.0);
        let direct: Vec<usize> = agg_direct
            .step(&fix.data)
            .into_iter()
            .map(|m| m.index)
            .collect();

        let adapter = SumCusumAggregatorAdapter::with_bf_streams(20.0);
        let harness: Vec<usize> = adapter.detect(&fix).into_iter().map(|c| c.index).collect();

        assert_eq!(direct, harness);
    }

    #[test]
    fn hc_adapter_surfaces_attribution() {
        // HC must surface non-empty per-stream attribution for at
        // least one fire on a fixture where streams shift heterogeneously.
        // synthetic 3σ tape has d=1 so attribution is trivially [[0]] per
        // fire -- assert that attribution.len() == cps.len() and every
        // entry is non-empty (HC always names ≥1 contributing stream).
        let fix = synthetic_3sigma();
        let adapter = HcAggregatorAdapter::with_bf_streams(3.0);
        let result = adapter.detect_full(&fix);
        assert_eq!(result.cps.len(), result.attribution.len());
        assert!(!result.cps.is_empty(), "HC must fire on synthetic 3σ");
        for (i, streams) in result.attribution.iter().enumerate() {
            assert!(
                !streams.is_empty(),
                "cp[{i}] missing attribution: HC must name ≥1 stream"
            );
        }
    }

    #[test]
    fn sum_cusum_adapter_attribution_empty_by_design() {
        // Sum-CUSUM is a joint detector. attribution[i] is empty per
        // MultiStreamChangePoint contract; aligned len is what we
        // enforce.
        let fix = synthetic_3sigma();
        let adapter = SumCusumAggregatorAdapter::with_bf_streams(20.0);
        let result = adapter.detect_full(&fix);
        assert_eq!(result.cps.len(), result.attribution.len());
        for streams in &result.attribution {
            assert!(streams.is_empty(), "sum-CUSUM is joint; attribution must be empty");
        }
    }

    #[test]
    fn adapter_rebuilds_state_per_call() {
        // Calling `detect` twice on the same fixture must produce
        // identical CPs -- proves the adapter rebuilds aggregator state
        // each call rather than mutating shared state.
        let fix = synthetic_3sigma();
        let adapter = HcAggregatorAdapter::with_bf_streams(3.0);
        let a = adapter.detect(&fix);
        let b = adapter.detect(&fix);
        assert_eq!(
            a.iter().map(|c| c.index).collect::<Vec<_>>(),
            b.iter().map(|c| c.index).collect::<Vec<_>>(),
        );
    }
}
