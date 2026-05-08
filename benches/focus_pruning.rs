//! Naive vs functional-pruning FOCuS. Output informational, not a gate.
//!
//! Two regimes:
//! - `stationary` -- `N(0,1)` at lengths 100/1k/10k/100k. Best case
//!   for pruning (segment grows uninterrupted).
//! - `multi_regime` -- alternating ±3σ regimes, 200 obs each, totalling
//!   1k / 10k / 100k. Exercises the deque rebuild path on every CP fire.
//!
//! Run with `cargo bench --features test-utils --bench focus_pruning`.

use cesura::eval::Rng;
use cesura::focus::FocusDetector;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

const STATIONARY_LENGTHS: &[usize] = &[100, 1_000, 10_000, 100_000];
const MULTI_REGIME_LENGTHS: &[usize] = &[1_000, 10_000, 100_000];
const REGIME_LEN: usize = 200;
const THRESHOLD: f64 = 8.0;

fn stationary(n: usize, seed: u64) -> Vec<f64> {
    let mut rng = Rng::new(seed);
    (0..n).map(|_| rng.normal(0.0, 1.0)).collect()
}

/// Alternating ±3σ regimes of length `REGIME_LEN`. Each boundary fires
/// a CP at threshold 8, triggering the pruned mode's rebuild.
fn multi_regime(total: usize, seed: u64) -> Vec<f64> {
    let mut rng = Rng::new(seed);
    let mut data = Vec::with_capacity(total);
    let mut k = 0usize;
    while data.len() < total {
        let mu = if k % 2 == 0 { 0.0 } else { 3.0 };
        let take = REGIME_LEN.min(total - data.len());
        for _ in 0..take {
            data.push(rng.normal(mu, 1.0));
        }
        k += 1;
    }
    data
}

fn run(c: &mut Criterion, group_name: &str, lengths: &[usize], gen: fn(usize, u64) -> Vec<f64>) {
    let mut group = c.benchmark_group(group_name);
    for &n in lengths {
        let data = gen(n, 42);
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

fn bench_focus(c: &mut Criterion) {
    run(c, "focus_stationary", STATIONARY_LENGTHS, stationary);
    run(c, "focus_multi_regime", MULTI_REGIME_LENGTHS, multi_regime);
}

criterion_group!(benches, bench_focus);
criterion_main!(benches);
