//! BOCPD `detect` vs `ConformalCpWrapper::detect` at varying calibration
//! capacities. Output informational, not a gate.
//!
//! Per-CP wrapper cost is `O(cap)` shift on `Vec::insert`/`Vec::remove`
//! against the sorted side-vector. The bench characterises how much
//! that adds at cap ∈ {50, 500, 5000} on a multi-regime fixture.
//!
//! Run with `cargo bench --features test-utils --bench conformal_overhead`.

use cesura::eval::Rng;
use cesura::{BocpdDetector, ConformalCpWrapper};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

const STREAM_LENGTHS: &[usize] = &[10_000, 50_000];
const CAPS: &[usize] = &[50, 500, 5_000];
const REGIME_LEN: usize = 200;
const SHIFT: f64 = 2.0;

fn multi_regime(total: usize, seed: u64) -> Vec<f64> {
    let mut rng = Rng::new(seed);
    let mut data = Vec::with_capacity(total);
    let mut k = 0usize;
    while data.len() < total {
        let mu = if k % 2 == 0 { 0.0 } else { SHIFT };
        let take = REGIME_LEN.min(total - data.len());
        for _ in 0..take {
            data.push(rng.normal(mu, 1.0));
        }
        k += 1;
    }
    data
}

fn bench_conformal(c: &mut Criterion) {
    let mut group = c.benchmark_group("conformal_overhead");
    for &n in STREAM_LENGTHS {
        let data = multi_regime(n, 42);
        group.throughput(Throughput::Elements(n as u64));

        group.bench_with_input(BenchmarkId::new("bocpd_baseline", n), &data, |b, d| {
            b.iter(|| {
                let det = BocpdDetector::new(200.0, REGIME_LEN + 50);
                let _ = det.detect(d);
            });
        });

        for &cap in CAPS {
            let id = BenchmarkId::new(format!("wrapper_cap{cap}"), n);
            group.bench_with_input(id, &data, |b, d| {
                b.iter(|| {
                    let mut wrapper =
                        ConformalCpWrapper::new(BocpdDetector::new(200.0, REGIME_LEN + 50))
                            .with_calibration_capacity(cap);
                    let _ = wrapper.detect(d);
                });
            });
        }
    }
    group.finish();
}

criterion_group!(benches, bench_conformal);
criterion_main!(benches);
