# cesura

> *Caesura -- a deliberate break in a line of verse, where the rhythm changes.*

Bayesian Online Change Point Detection in Rust. Implements Adams &
MacKay (2007) with a Normal-Inverse-Gamma conjugate prior. Pure Rust,
no `unsafe`, no runtime dependencies beyond `serde`.

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
use cesura::{BocpdDetector, ChangePoint};

let data: Vec<f64> = std::iter::repeat(0.0).take(100)
    .chain(std::iter::repeat(5.0).take(100))
    .collect();

// lambda = expected run length; max_rl ≥ data.len()
let detector = BocpdDetector::new(200.0, 250);
let cps: Vec<ChangePoint> = detector.detect(&data);

for cp in &cps {
    println!("t={} conf={:.2} shift={:.2}σ", cp.index, cp.confidence, cp.shift_sigma);
}

// Filter post-hoc by confidence if you want a stricter gate.
let strict: Vec<_> = cps.iter().filter(|c| c.confidence >= 0.5).collect();
```

### Streaming

`StreamingDetector` keeps state between calls and serializes via serde,
so you can checkpoint and resume:

```rust
use cesura::streaming::StreamingDetector;

let mut det = StreamingDetector::new(200.0, 1024);
for chunk in incoming {
    let new_cps = det.step(chunk);
    // ...
}
```

### Detrending

`detrend` provides Welford-based online normalization and a
`detect_with_seasonal_guard` helper that suppresses change points
explainable by a seasonal period. The period can be passed explicitly
or auto-detected via `detrend::dominant_period_via_acf`:

```rust
use cesura::detrend::detect_with_seasonal_guard;

let cps = detect_with_seasonal_guard(&data, None, &detector); // auto-detect
let cps = detect_with_seasonal_guard(&data, Some(60), &detector); // explicit
```

## Tuning

- `lambda` -- expected observations between change points. Larger →
  fewer false positives, slower to detect. Must be `> 1.0`.
- `max_run_length` -- caps memory and bounds the longest stable regime
  the detector will track. For batch use, set `≥ data.len()`.

Inputs with `< 20` finite samples return an empty result. NaN / non-
finite values are skipped; reported indices map back to the original
input positions.

## Features

- `test-utils` -- exposes the `eval` module (synthetic generators, RNG,
  precision/recall scoring) for downstream test suites.
- `robust` -- enables `BocpdDetector::with_beta` for β-divergence
  robust BOCPD (Knoblauch et al. 2018, arXiv:1806.02261). Bounds the
  influence of any single observation on the posterior, so heavy-tailed
  inputs (kurtosis 5-15) no longer produce false alarms on legitimate
  tail events. `β = 0.0` is the default and short-circuits to the
  standard path; `β ∈ [0.05, 0.20]` is a reasonable robust range.

```rust
# #[cfg(feature = "robust")] {
use cesura::BocpdDetector;
let detector = BocpdDetector::new(200.0, 250).with_beta(0.15);
# }
```

If you don't know what β to set, `with_auto_beta(&warmup)` estimates it
from the warmup window's sample excess kurtosis. Near-Gaussian data
(`|k_ex| ≤ 1`) maps to `β = 0` (no-op); heavier tails get a
proportional β capped at 0.20.

```rust
# #[cfg(feature = "robust")] {
use cesura::BocpdDetector;
# let warmup: Vec<f64> = vec![0.0; 300];
let detector = BocpdDetector::new(200.0, 250).with_auto_beta(&warmup);
# }
```

## FOCuS -- frequentist sibling

cesura also ships `FocusDetector`, a Romano-et-al-2023 / Ward-et-al-2024
generalised-likelihood-ratio detector for univariate Gaussian-mean
shifts. It does not compete with BOCPD -- it complements it. Different
assumptions, different operating curve. Use as a parallel sanity-check
when BOCPD fires on a fresh metric and you want an independent vote.

```rust
use cesura::focus::FocusDetector;
let mut det = FocusDetector::new(8.0);
let cps = det.detect(&data);
```

`focus::arl0_calibrate(target)` empirically picks a threshold by
simulation. FOCuS thresholds are NOT comparable to BOCPD confidences;
calibrate each detector independently.

## Joint CP + collective-anomaly detection

`ChenWuDetector` (paper: Chen & Wu 2025, arXiv:2508.06385) emits two
distinct categories from one online recursion: genuine change points
(persistent regime shifts) and collective anomalies (short reverting
deviations). BOCPD on its own emits a start+end CP pair on the same
fixture; the joint detector tells you the difference.

```rust
use cesura::chen_wu::{ChenWuDetector, Detection};

// p0  -- prior change probability per step.
// q0  -- prior anomaly-end probability conditional on an open anomaly.
// Δt  -- maximum collective-anomaly duration.
// λ_a -- anomaly alarm threshold (eq. 12).
// λ_c -- CP alarm threshold (eq. 13).
let det = ChenWuDetector::new(0.1, 0.2, 4, 0.5, 0.5);
for d in det.detect(&data) {
    match d {
        Detection::ChangePoint(cp) =>
            println!("CP at {} (conf {:.2}, {:.1}σ)", cp.index, cp.confidence, cp.shift_sigma),
        Detection::CollectiveAnomaly { start, end, confidence } =>
            println!("anomaly [{}, {}] (conf {:.2})", start, end, confidence),
    }
}
```

Optional builders: `with_search_windows(u_c, u_a)`,
`with_localisation_tolerance(δ)`, `with_min_post_change(n)`,
`with_prior(μ, κ, α, β)`, and (under `feature = "robust"`)
`with_robust(β)` for β-divergence within-regime likelihoods.

For production daemons, `cesura::streaming_chen_wu::StreamingChenWuDetector`
is the online equivalent. Same API surface, plus
`step(&[f64]) -> Vec<Detection>`, `save_state()`, and `restore()`. The
batch and streaming paths share the recursion bit-for-bit -- pinned by
`tests/streaming_chen_wu.rs::matches_batch_detect_paper_section_6_1`.

The detector ships in the default feature set (`feature = "joint-detection"`,
default-on as of 0.8). Disable via `default-features = false` if you
only want the BOCPD path.

## Known limitations

## References

- Adams, R. P., & MacKay, D. J. C. (2007). *Bayesian Online Changepoint
  Detection.* arXiv:0710.3742.
- Page, E. S. (1954). *Continuous Inspection Schemes.* Biometrika 41(1).
  [CUSUM, used as a baseline in `tests/statistical.rs`.]

## License

Apache-2.0. See [LICENSE](LICENSE).
