# bocpd

Bayesian Online Change Point Detection in Rust.

Implements Adams & MacKay (2007) with a Normal-Inverse-Gamma conjugate
prior on the underlying Gaussian. Pure Rust, no `unsafe`, no runtime
dependencies beyond `serde`.

## What it does

Given a stream of `f64` observations, BOCPD maintains a posterior over
*run length* -- how long since the last change point. When the most
probable run length collapses, the index is reported as a change point
with a confidence score and shift magnitude.

Use it for: regime shifts in metrics, sensor faults, structural breaks
in time series. It does not require labelled data, a window size, or a
prior on the post-change distribution.

## API at a glance

```rust
use bocpd::{BocpdDetector, ChangePoint};

let data: Vec<f64> = std::iter::repeat(0.0).take(100)
    .chain(std::iter::repeat(5.0).take(100))
    .collect();

// lambda = expected run length; max_rl ≥ data.len()
let detector = BocpdDetector::new(200.0, 250);
let cps: Vec<ChangePoint> = detector.detect(&data, 0.3);

for cp in &cps {
    println!("t={} conf={:.2} shift={:.2}σ", cp.index, cp.confidence, cp.shift_sigma);
}
```

### Streaming

`StreamingDetector` keeps state between calls and serializes via serde,
so you can checkpoint and resume:

```rust
use bocpd::streaming::StreamingDetector;

let mut det = StreamingDetector::new(200.0, 1024);
for chunk in incoming {
    let new_cps = det.step(chunk, 0.3);
    // ...
}
```

### Detrending

`detrend` provides Welford-based online normalization and a
`detect_with_seasonal_guard` helper that suppresses change points
explainable by a known seasonal period.

## Tuning

- `lambda` -- expected observations between change points. Larger →
  fewer false positives, slower to detect. Must be `> 1.0`.
- `threshold` ∈ `(0, 1)` -- required confidence drop before reporting.
- `max_run_length` -- caps memory and bounds the longest stable regime
  the detector will track. For batch use, set `≥ data.len()`.

Inputs with `< 20` finite samples return an empty result. NaN / non-
finite values are skipped; reported indices map back to the original
input positions.

## Features

- `test-utils` -- exposes the `eval` module (synthetic generators, RNG,
  precision/recall scoring) for downstream test suites.

## References

- Adams, R. P., & MacKay, D. J. C. (2007). *Bayesian Online Changepoint
  Detection.* arXiv:0710.3742.
- Chen, Z., & Wu, Y. (2025). *Post-hoc collective anomaly
  classification.* arXiv:2508.06385.

## License

Apache-2.0. See [LICENSE](LICENSE).
