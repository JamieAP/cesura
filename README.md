# cesura

Research implementations of change-point detection for Rust time series. The library includes Bayesian online change-point detection, robust and multivariate variants, and optional CLI and MCP interfaces.

The batch and streaming APIs have different timing semantics. Batch `detect` normalizes the full input series and uses a window after each candidate change. Its results are retrospective. `StreamingDetector` accepts observations incrementally, but waits for 20 additional valid observations before deciding whether to emit a candidate change. The reported index refers to the earlier trigger, not the time of emission.

## Library use

This example runs both APIs on a synthetic mean shift:

```rust
use cesura::{streaming::StreamingDetector, BocpdDetector};

fn main() {
    let samples: Vec<f64> = (0..400)
        .map(|i| if i < 200 { 0.0 } else { 5.0 })
        .collect();

    let batch = BocpdDetector::new(200.0, 250);
    for point in batch.detect(&samples) {
        println!("batch: index={} score={:.3}", point.index, point.confidence);
    }

    let mut streaming = StreamingDetector::new(200.0, 250);
    for chunk in samples.chunks(32) {
        for point in streaming.step(chunk) {
            println!("stream: index={} score={:.3}", point.index, point.confidence);
        }
    }
}
```

The `canonical` module provides configured multistream aggregators. Choose parameters for the sampling rate and data distribution. Confidence scores and default settings do not establish calibrated probabilities, detection accuracy or financial performance. The algorithms' assumptions and research references are described in the module documentation.

## Features and tools

The default features are `joint-detection`, `robust` and `cli`. Set `default-features = false` to disable them. The optional `mcp` feature builds the local stdio server.

```sh
cargo test --lib
cargo run --example change_points
printf '{"x":1.0}\n' | cargo run --bin cesura -- watch --kind streaming
cargo build --features mcp --bin cesura-mcp
```

`cesura watch` reads JSONL observations from stdin and writes detected changes as JSONL. `cesura info` lists detector kinds and parameters. The MCP server exposes `cesura_feed`, `cesura_snapshot`, `cesura_restore`, `cesura_list_streams` and `cesura_close_stream`.

## Known limitations

- Streaming snapshots omit pending detections and their recent normalization buffer. Restore also resets the pruning cutoff to its default. A snapshot is therefore not a complete checkpoint of detector behavior.
- Streaming history and snapshot size grow with the number of observations, even when the run-length limit is fixed.
- The legacy epoch-window helper `f1_with_margin` can overstate recall when several detections match one event. The main `evaluate` function uses a separate one-to-one matcher.

These are current implementation limits. Validate the relevant detector and evaluation protocol on representative data before relying on their output.

## Evaluation

The repository includes synthetic fixtures and property tests. `test-utils` enables the benchmark harness and optional Polars loaders. Real-data tests need user-supplied files at these locations:

- `data/crypto_macro_1m.parquet`
- `data/indices_1m/`
- `data/hyperliquid_1s/`

Market datasets are not included. Tests that need unavailable data skip or are explicitly ignored, so a passing test run does not establish performance on those datasets. You must have the access and reuse rights for any data you supply.

## License and provenance

[Apache-2.0](LICENSE). [PROVENANCE.md](PROVENANCE.md) records the source revisions and inherited code. Research references appear in the module documentation.
