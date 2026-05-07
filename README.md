# bocpd

Bayesian Online Change Point Detection (Adams & MacKay, 2007) in Rust, with a
Normal-Inverse-Gamma conjugate prior. No `unsafe`. No required runtime deps
beyond `serde`.

Includes:

- Batch detector (`BocpdDetector`) over `&[f64]`.
- Streaming detector (`StreamingDetector`) with serializable state.
- Optional pre-detrending and seasonal-guard helpers (`detrend` module).
- Post-hoc collective anomaly classification per arXiv:2508.06385
  (Chen & Wu, 2025).

## Usage

```rust
use bocpd::{BocpdDetector, ChangePoint};

let data: Vec<f64> = std::iter::repeat(0.0).take(100)
    .chain(std::iter::repeat(5.0).take(100))
    .collect();

let detector = BocpdDetector::new(200.0, 250);
let change_points: Vec<ChangePoint> = detector.detect(&data, 0.3);
```

## References

- Adams, R. P., & MacKay, D. J. C. (2007). *Bayesian Online Changepoint
  Detection.* arXiv:0710.3742.
- Chen, Z., & Wu, Y. (2025). arXiv:2508.06385.

## License

Apache-2.0. See [LICENSE](LICENSE).
