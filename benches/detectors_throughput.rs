//! Throughput benches for the primary detector configurations.
//!
//! Stationary `N(0,1)` input at lengths {1k, 10k, 100k}. Output is
//! informational, not a CI gate. Run with
//! `cargo bench --features test-utils --bench detectors_throughput`.

use cesura::eval::Rng;
use cesura::focus::FocusDetector;
use cesura::{BocpdDetector, EnsembleDetector, NigAr1};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

const LENGTHS: &[usize] = &[1_000, 10_000, 100_000];

fn stationary(n: usize, seed: u64) -> Vec<f64> {
    let mut rng = Rng::new(seed);
    (0..n).map(|_| rng.normal(0.0, 1.0)).collect()
}

fn bench_detectors(c: &mut Criterion) {
    let mut group = c.benchmark_group("detectors_throughput");
    for &n in LENGTHS {
        let data = stationary(n, 42);
        group.throughput(Throughput::Elements(n as u64));

        group.bench_with_input(BenchmarkId::new("bocpd_nig", n), &data, |b, d| {
            b.iter(|| {
                let det = BocpdDetector::new(200.0, 350);
                let _ = det.detect(d);
            });
        });

        group.bench_with_input(BenchmarkId::new("bocpd_nig_ar1", n), &data, |b, d| {
            b.iter(|| {
                let det = BocpdDetector::with_prior(200.0, 350, NigAr1::default_prior());
                let _ = det.detect(d);
            });
        });

        group.bench_with_input(BenchmarkId::new("ensemble", n), &data, |b, d| {
            b.iter(|| {
                let det = EnsembleDetector::new(200.0, 350);
                let _ = det.detect(d);
            });
        });

        group.bench_with_input(BenchmarkId::new("ensemble_detrend", n), &data, |b, d| {
            b.iter(|| {
                let det = EnsembleDetector::new(200.0, 350).with_auto_detrend(true);
                let _ = det.detect(d);
            });
        });

        group.bench_with_input(BenchmarkId::new("focus_pruned", n), &data, |b, d| {
            b.iter(|| {
                let mut det = FocusDetector::new(8.0).with_pruning();
                let _ = det.detect(d);
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_detectors);
criterion_main!(benches);
