//! Concurrency contract tests.
//!
//! BOCPD is a pure stateless detector (`&self detect`). The streaming variant
//! owns its state with no interior mutability. These tests pin the contract:
//!
//! - Both detector types are `Send + Sync` (compile-time).
//! - Concurrent batch detection from many threads is deterministic and
//!   isolated -- no globals, no thread-local leakage, no shared cache.
//! - Independent streaming detectors evolve without interference.
//!
//! Run: `cargo test --features test-utils --test concurrency`.

use std::sync::Arc;
use std::thread;

use cesura::eval::Rng;
use cesura::streaming::StreamingDetector;
use cesura::BocpdDetector;

fn assert_send_sync<T: Send + Sync>() {}

#[test]
fn detector_types_are_send_sync() {
    // Compile-time: removal of Send/Sync would fail the build, not just the test.
    assert_send_sync::<BocpdDetector>();
    assert_send_sync::<StreamingDetector>();
    assert_send_sync::<cesura::ChangePoint>();
    assert_send_sync::<cesura::streaming::DetectorState>();
}

#[test]
fn parallel_batch_detect_is_deterministic() {
    // Same detector, same input, N threads → all identical outputs.
    // Catches: any future introduction of thread_local, lazy_static, or
    // interior mutability that would silently make detect() non-pure.
    let mut rng = Rng::new(0xC0FFEE);
    let mut data: Vec<f64> = (0..200).map(|_| rng.normal(0.0, 1.0)).collect();
    data.extend((0..200).map(|_| rng.normal(4.0, 1.0)));
    let data = Arc::new(data);
    let det = Arc::new(BocpdDetector::new(200.0, 450));

    let handles: Vec<_> = (0..16)
        .map(|_| {
            let data = Arc::clone(&data);
            let det = Arc::clone(&det);
            thread::spawn(move || {
                det.detect(&data)
                    .into_iter()
                    .map(|cp| (cp.index, cp.confidence.to_bits()))
                    .collect::<Vec<_>>()
            })
        })
        .collect();

    let outputs: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    let first = &outputs[0];
    for (i, out) in outputs.iter().enumerate() {
        assert_eq!(out, first, "thread {i} diverged from thread 0");
    }
}

#[test]
fn parallel_streaming_detectors_are_isolated() {
    // N independent StreamingDetectors, each fed a different signal in its
    // own thread. None should observe any other's state -- i.e., outputs
    // depend only on the signal they were fed.
    //
    // We verify by computing a deterministic reference output single-threaded
    // and matching it from each thread's parallel run.
    let signals: Vec<Vec<f64>> = (0..8u64)
        .map(|seed| {
            let mut rng = Rng::new(seed.wrapping_mul(31) + 1);
            let mut s: Vec<f64> = (0..150).map(|_| rng.normal(0.0, 1.0)).collect();
            s.extend((0..150).map(|_| rng.normal((seed % 5 + 1) as f64, 1.0)));
            s
        })
        .collect();

    // Single-threaded reference.
    let reference: Vec<Vec<(usize, u64)>> = signals
        .iter()
        .map(|s| {
            let mut det = StreamingDetector::new(200.0, 350);
            det.step(s)
                .into_iter()
                .map(|cp| (cp.index, cp.confidence.to_bits()))
                .collect()
        })
        .collect();

    // Multi-threaded.
    let signals = Arc::new(signals);
    let handles: Vec<_> = (0..signals.len())
        .map(|i| {
            let signals = Arc::clone(&signals);
            thread::spawn(move || {
                let mut det = StreamingDetector::new(200.0, 350);
                det.step(&signals[i])
                    .into_iter()
                    .map(|cp| (cp.index, cp.confidence.to_bits()))
                    .collect::<Vec<_>>()
            })
        })
        .collect();

    for (i, h) in handles.into_iter().enumerate() {
        let parallel = h.join().unwrap();
        assert_eq!(
            parallel, reference[i],
            "signal {i}: parallel result diverged from single-threaded reference"
        );
    }
}

// Each from_str returns independent owned buffers. Rust ownership
// prevents two restored detectors from sharing mutable state.
