//! Pre-validate every numeric param before calling cesura library constructors.
//! `StreamingDetector::new(lambda <= 1.0)` panics; the panic corrupts stdout
//! and kills the MCP channel. Reject in this layer before construction.

/// Pre-validation for univariate StreamingDetector / cesura_feed lazy-create.
/// Returns Err(short human-readable message) on rejection so the caller can
/// wrap as McpError::invalid_params or CallToolResult::structured_error.
pub fn validate_streaming(lambda: f64, max_rl: usize) -> Result<(), String> {
    // `!(lambda > 1.0)` rejects NaN (NaN > 1.0 is false). Do not rewrite as
    // `lambda <= 1.0` -- that lets NaN through and the constructor panics.
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    if !(lambda > 1.0) {
        return Err(format!("lambda must be > 1.0, got {lambda}"));
    }
    if max_rl == 0 {
        return Err("max_rl must be > 0".into());
    }
    Ok(())
}

pub fn validate_dm_bocd(d: usize, lambda: f64, max_rl: usize) -> Result<(), String> {
    if d == 0 {
        return Err("d must be > 0".into());
    }
    validate_streaming(lambda, max_rl)
}

pub fn validate_multistream(d: usize, lambda: f64, max_rl: usize) -> Result<(), String> {
    if d == 0 {
        return Err("d must be > 0".into());
    }
    validate_streaming(lambda, max_rl)
}

/// Validate a `with_threshold(t)` argument for SumCusum / HC / FilterTick.
/// `src/streaming.rs:320`, sum_cusum / hc constructors assert `threshold > 0.0`.
pub fn validate_threshold(t: f64) -> Result<(), String> {
    if !t.is_finite() {
        return Err(format!("threshold must be finite, got {t}"));
    }
    if t <= 0.0 {
        return Err(format!("threshold must be > 0.0, got {t}"));
    }
    Ok(())
}

/// Validate `with_persistence(n)` for HC. `src/multistream/hc.rs:275` asserts n >= 1.
pub fn validate_persistence(n: usize) -> Result<(), String> {
    if n == 0 {
        return Err("persistence must be >= 1, got 0".into());
    }
    Ok(())
}

/// Reject empty + over-long client-supplied stream_ids. Empty ids
/// collide with the lazy-create lookup key (`""` cannot be re-targeted
/// from the wire) and unbounded ids let a hostile / buggy client pin
/// arbitrary RAM in `ID_MAP`. 256 chars is generous for human +
/// UUID-shaped ids while bounding worst-case memory.
pub fn validate_stream_id(s: &str) -> Result<(), String> {
    if s.is_empty() {
        return Err("stream_id must be non-empty".into());
    }
    if s.len() > 256 {
        return Err(format!(
            "stream_id must be ≤ 256 chars, got {}",
            s.len()
        ));
    }
    Ok(())
}

pub fn validate_filter_tick(d: usize, k: usize, threshold: f64) -> Result<(), String> {
    if d == 0 {
        return Err("d must be > 0".into());
    }
    if k == 0 || k > d {
        return Err(format!("k must be in [1, d={d}], got {k}"));
    }
    if !threshold.is_finite() {
        return Err("threshold must be finite".into());
    }
    // FilterTickAggregator::new also asserts `threshold > 0.0`; mirror it.
    if threshold <= 0.0 {
        return Err(format!("threshold must be > 0, got {threshold}"));
    }
    Ok(())
}

#[cfg(feature = "joint-detection")]
pub fn validate_chen_wu(
    p0: f64,
    q0: f64,
    delta_t: usize,
    lambda_a: f64,
    lambda_c: f64,
) -> Result<(), String> {
    if !(0.0..1.0).contains(&p0) {
        return Err(format!("p0 must be in [0, 1), got {p0}"));
    }
    if !(0.0..1.0).contains(&q0) {
        return Err(format!("q0 must be in [0, 1), got {q0}"));
    }
    if delta_t < 1 {
        return Err("delta_t must be >= 1".into());
    }
    if !(0.0..1.0).contains(&lambda_a) {
        return Err(format!("lambda_a must be in [0, 1), got {lambda_a}"));
    }
    if !(0.0..1.0).contains(&lambda_c) {
        return Err(format!("lambda_c must be in [0, 1), got {lambda_c}"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_streaming_rejects_low_lambda() {
        let e = validate_streaming(1.0, 250).unwrap_err();
        assert!(e.contains("lambda"));
    }

    #[test]
    fn validate_streaming_accepts_canonical() {
        assert!(validate_streaming(200.0, 250).is_ok());
    }

    #[test]
    fn validate_dm_bocd_rejects_zero_d() {
        let e = validate_dm_bocd(0, 200.0, 250).unwrap_err();
        assert!(e.contains("d"));
    }

    #[test]
    fn validate_multistream_rejects_zero_d() {
        let e = validate_multistream(0, 200.0, 250).unwrap_err();
        assert!(e.contains("d"));
    }

    #[test]
    fn validate_filter_tick_rejects_k_gt_d() {
        let e = validate_filter_tick(3, 4, 0.5).unwrap_err();
        assert!(e.contains("k") || e.contains("d"));
    }

    #[test]
    fn validate_streaming_rejects_zero_max_rl() {
        assert!(validate_streaming(200.0, 0).unwrap_err().contains("max_rl"));
    }

    #[test]
    fn validate_streaming_rejects_nan_lambda() {
        assert!(validate_streaming(f64::NAN, 250).is_err());
    }

    #[test]
    fn validate_filter_tick_rejects_zero_threshold() {
        assert!(validate_filter_tick(4, 2, 0.0).unwrap_err().contains("threshold"));
    }

    #[test]
    fn validate_filter_tick_rejects_non_finite_threshold() {
        assert!(validate_filter_tick(4, 2, f64::INFINITY).unwrap_err().contains("threshold"));
    }

    #[test]
    fn validate_threshold_accepts_positive() {
        assert!(validate_threshold(0.5).is_ok());
        assert!(validate_threshold(10.0).is_ok());
    }

    #[test]
    fn validate_threshold_rejects_zero() {
        let e = validate_threshold(0.0).unwrap_err();
        assert!(e.contains("threshold"));
    }

    #[test]
    fn validate_threshold_rejects_nan() {
        let e = validate_threshold(f64::NAN).unwrap_err();
        assert!(e.contains("finite"));
    }

    #[test]
    fn validate_persistence_rejects_zero() {
        let e = validate_persistence(0).unwrap_err();
        assert!(e.contains("persistence"));
    }

    #[test]
    fn validate_persistence_accepts_one() {
        assert!(validate_persistence(1).is_ok());
    }

    #[test]
    fn validate_stream_id_rejects_empty() {
        let e = validate_stream_id("").unwrap_err();
        assert!(e.contains("non-empty"));
    }

    #[test]
    fn validate_stream_id_rejects_too_long() {
        let s = "x".repeat(257);
        let e = validate_stream_id(&s).unwrap_err();
        assert!(e.contains("256"));
    }

    #[test]
    fn validate_stream_id_accepts_reasonable() {
        assert!(validate_stream_id("abc").is_ok());
        assert!(validate_stream_id(&"x".repeat(256)).is_ok());
    }

    #[cfg(feature = "joint-detection")]
    #[test]
    fn validate_chen_wu_rejects_p0_out_of_range() {
        let e = validate_chen_wu(1.5, 0.05, 20, 0.999, 0.999).unwrap_err();
        assert!(e.contains("p0"));
    }
}
