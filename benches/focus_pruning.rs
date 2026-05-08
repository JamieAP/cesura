//! Naive vs functional-pruning FOCuS, stationary N(0,1) at lengths
//! 100 / 1k / 10k / 100k. Output is informational, not a gate.
//!
//! Run with `cargo bench --features test-utils --bench focus_pruning`.

use cesura::eval::Rng;
use cesura::focus::FocusDetector;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

const LENGTHS: &[usize] = &[100, 1_000, 10_000, 100_000];
const THRESHOLD: f64 = 8.0;

fn make_data(n: usize, seed: u64) -> Vec<f64> {
    let mut rng = Rng::new(seed);
    (0..n).map(|_| rng.normal(0.0, 1.0)).collect()
}

fn bench_focus_naive_vs_pruned(c: &mut Criterion) {
    let mut group = c.benchmark_group("focus_inner_loop");
    for &n in LENGTHS {
        let data = make_data(n, 42);
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::new("naive", n), &data, |b, d| {
            b.iter(|| {
                let mut det = FocusDetector::new(THRESHOLD);
                let _ = det.detect(d);
            });
        });
        group.bench_with_input(BenchmarkId::new("pruned", n), &data, |b, d| {
            b.iter(|| {
                let mut det = FocusDetector::new(THRESHOLD).with_pruning();
                let _ = det.detect(d);
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_focus_naive_vs_pruned);
criterion_main!(benches);
