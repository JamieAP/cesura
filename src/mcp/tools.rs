//! Tool parameter types + `#[tool]`-target impls.
//!
//! Parameter structs define the tool wire format. Feed lazily creates
//! streams and rejects conflicting parameters; the remaining tools
//! snapshot, restore, list, or close registered streams.

use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::str::FromStr;
use std::sync::{Arc, LazyLock, RwLock};

use rmcp::model::CallToolResult;
use rmcp::ErrorData as McpError;
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;
use uuid::Uuid;

use crate::dm_bocd::StreamingDmBocd;
use crate::mcp::registry::Registry;
use crate::mcp::validation;
use crate::multistream::{FilterTickAggregator, HcAggregator, SumCusumAggregator};
use crate::runtime::{DetectorKind, Fire, Obs, ObsShape, Runner, RunnerState, RunnerStatePayload};
use crate::streaming::StreamingDetector;
#[cfg(feature = "joint-detection")]
use crate::streaming_chen_wu::StreamingChenWuDetector;

/// Mapping of client-supplied string `stream_id` → registered entry.
/// Lookup is fully sync (no awaits inside), so a `std::sync::RwLock` is
/// the right primitive -- never hold a `tokio::sync::Mutex` across the
/// kind/params equality checks.
///
/// `expected_d` is recorded at lazy-create time for multistream kinds so
/// every subsequent feed can pre-check `Obs::Vec` length against the
/// aggregator's `d`. Without this guard, an off-by-one observation
/// reaches `hc.rs:288` / `sum_cusum.rs:113`, which `assert_eq!` and
/// panic, corrupting stdout and killing the MCP channel.
#[derive(Debug, Clone)]
struct StreamEntry {
    uuid: Uuid,
    kind: DetectorKind,
    params: serde_json::Value,
    /// Expected observation length for vector-shape kinds. `None` for
    /// scalar kinds.
    expected_d: Option<usize>,
}

static ID_MAP: LazyLock<RwLock<HashMap<String, StreamEntry>>> =
    LazyLock::new(|| RwLock::new(HashMap::new()));

/// Default observation dimension for multistream / multivariate kinds
/// when `params.d` is omitted. Mirrors the canonical 4-stream config
/// used across the cesura test suite. `build_runner` and
/// `expected_d_for` MUST stay in sync on this value -- a mismatch
/// produces correct construction but a wrong shape guard, letting an
/// off-by-one observation reach the inner aggregator and panic.
const DEFAULT_D: usize = 4;

#[derive(Debug, Deserialize, JsonSchema)]
pub struct FeedRequest {
    /// Stream id. Unknown id on first call → `kind` + `params` required and the stream is lazy-created.
    pub stream_id: String,
    /// Detector kind. Required on first call for an unknown stream_id. Ignored thereafter.
    #[serde(default)]
    pub kind: Option<String>,
    /// Detector params. Required on first call for an unknown stream_id. Ignored thereafter.
    #[serde(default)]
    pub params: Option<serde_json::Value>,
    /// One or more observations. Univariate kinds: array of f64. Multistream: array of f64 arrays.
    pub observations: serde_json::Value,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct SnapshotRequest {
    pub stream_id: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct RestoreRequest {
    pub state: serde_json::Value,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct CloseRequest {
    pub stream_id: String,
}

pub async fn cesura_feed_impl(
    streams: &Arc<Registry>,
    req: FeedRequest,
) -> Result<CallToolResult, McpError> {
    validation::validate_stream_id(&req.stream_id)
        .map_err(|e| McpError::invalid_params(e, None))?;

    // 1. Resolve string stream_id → entry under a single write lock.
    //
    //    Why a single write lock (∄ read-then-upgrade): two concurrent
    //    feeds with the same unknown stream_id otherwise both observe
    //    "absent" under read locks, both take the vacant branch, both
    //    call `build_runner` + `streams.insert`, and both register
    //    distinct uuids in `ID_MAP` for the same string id. The second
    //    insert wins and the first stream is orphaned in `streams`
    //    (only surfaces in `cesura_list_streams` as a uuid-shaped
    //    stream_id). Holding the write lock across the entry decision
    //    + insertion eliminates that window.
    //
    //    The lock guard MUST drop before `runner.lock().await` below
    //    (∵ blocking std::sync::RwLockWriteGuard held across an await
    //    risks contention with concurrent feeds + blocks the runtime).
    //    The block returns the StreamEntry by value; the guard goes
    //    out of scope at `};`.
    let entry: StreamEntry = {
        let mut map = ID_MAP.write().expect("ID_MAP poisoned");
        match map.entry(req.stream_id.clone()) {
            Entry::Occupied(occ) => {
                let existing = occ.get();
                // Conflict policy: if caller supplied kind / params on
                // a known id, they must match what we registered. Bare
                // subsequent calls (stream_id + observations only) are
                // the normal shape and pass through.
                if let Some(ref k) = req.kind {
                    match DetectorKind::from_str(k) {
                        Ok(parsed) if parsed == existing.kind => {}
                        Ok(parsed) => {
                            return Ok(CallToolResult::structured_error(json!({
                                "code": "kind_mismatch",
                                "stream_id": req.stream_id,
                                "registered_kind": existing.kind.as_wire(),
                                "requested_kind": parsed.as_wire(),
                            })));
                        }
                        Err(e) => {
                            return Err(McpError::invalid_params(e, None));
                        }
                    }
                }
                if let Some(ref p) = req.params {
                    if p != &existing.params {
                        return Ok(CallToolResult::structured_error(json!({
                            "code": "params_mismatch",
                            "stream_id": req.stream_id,
                            "registered_params": existing.params.clone(),
                            "requested_params": p,
                        })));
                    }
                }
                existing.clone()
            }
            Entry::Vacant(vac) => {
                // Lazy-create path. Both kind and params (at least an
                // empty object) are part of the registered tuple so
                // future calls can be compared bit-for-bit. `?` on
                // build_runner / DetectorKind::from_str drops the write
                // guard before propagating -- no .await in this arm.
                let kind_str = req.kind.clone().ok_or_else(|| {
                    McpError::invalid_params(
                        "stream_id unknown; first call must supply `kind`",
                        Some(json!({"stream_id": req.stream_id})),
                    )
                })?;
                let kind = DetectorKind::from_str(&kind_str)
                    .map_err(|e| McpError::invalid_params(e, None))?;
                let params = req.params.clone().unwrap_or_else(|| json!({}));
                let runner = build_runner(kind, &params)?;
                let expected_d = expected_d_for(kind, &params);
                let uuid = Uuid::new_v4();
                streams.insert(uuid, runner);
                let entry = StreamEntry {
                    uuid,
                    kind,
                    params,
                    expected_d,
                };
                vac.insert(entry.clone());
                entry
            }
        }
    }; // ID_MAP write guard drops here, BEFORE the .await below.

    let uuid = entry.uuid;
    let registered_kind = entry.kind;

    // 2. Locate the runner -- registry drift is an internal-error
    //    signal, not a client bug.
    let runner = streams.get(&uuid).ok_or_else(|| {
        McpError::internal_error(
            "stream_id resolved but Runner missing -- registry / id-map drift",
            Some(json!({"uuid": uuid})),
        )
    })?;

    // 3. Decode observations per registered kind's shape.
    let obs_list = decode_obs(registered_kind, &req.observations).map_err(|e| {
        McpError::invalid_params(
            "failed to decode observations",
            Some(json!({"error": e, "kind": registered_kind.as_wire()})),
        )
    })?;

    // 3a. Length guard for vector observations. hc.rs:288 +
    //     sum_cusum.rs:113 `assert_eq!(obs.len(), d)`; a wrong-d
    //     observation reaches the inner aggregator and panics, killing
    //     the channel. Pre-check against the d recorded at lazy-create.
    if let Some(expected) = entry.expected_d {
        for obs in &obs_list {
            if let Obs::Vec(v) = obs {
                if v.len() != expected {
                    return Ok(CallToolResult::structured_error(json!({
                        "code": "shape_mismatch",
                        "stream_id": req.stream_id,
                        "expected_d": expected,
                        "got_d": v.len(),
                    })));
                }
            }
        }
    }

    // 4. Feed loop under the per-stream Mutex.
    let mut guard = runner.lock().await;
    let mut fires: Vec<Fire> = Vec::new();
    for obs in obs_list {
        fires.extend(guard.feed(obs));
    }
    drop(guard);

    let fires_json = serde_json::to_value(&fires)
        .map_err(|e| McpError::internal_error(format!("fire ser: {e}"), None))?;

    Ok(CallToolResult::structured(json!({
        "fires": fires_json,
        "stream_id": req.stream_id,
    })))
}

/// Extract the expected vector observation length for kinds that take
/// `Obs::Vec`. Mirrors `build_runner`'s `d` parsing -- keep in sync.
fn expected_d_for(kind: DetectorKind, params: &serde_json::Value) -> Option<usize> {
    if kind.obs_shape() != ObsShape::Vector {
        return None;
    }
    let p = params.as_object();
    let d = p
        .and_then(|m| m.get("d"))
        .and_then(|v| v.as_u64())
        .map(|n| n as usize)
        .unwrap_or(DEFAULT_D);
    Some(d)
}

fn build_runner(kind: DetectorKind, params: &serde_json::Value) -> Result<Runner, McpError> {
    let p = params.as_object().cloned().unwrap_or_default();
    let f64_param = |k: &str, default: f64| p.get(k).and_then(|v| v.as_f64()).unwrap_or(default);
    let usize_param = |k: &str, default: usize| {
        p.get(k)
            .and_then(|v| v.as_u64())
            .map(|n| n as usize)
            .unwrap_or(default)
    };

    match kind {
        DetectorKind::Streaming => {
            let lambda = f64_param("lambda", 200.0);
            let max_rl = usize_param("max_rl", 250);
            validation::validate_streaming(lambda, max_rl)
                .map_err(|e| McpError::invalid_params(e, None))?;
            Ok(Runner::Uni(StreamingDetector::new(lambda, max_rl)))
        }
        #[cfg(feature = "joint-detection")]
        DetectorKind::ChenWu => {
            let p0 = f64_param("p0", 0.001);
            let q0 = f64_param("q0", 0.05);
            let delta_t = usize_param("delta_t", 20);
            let lambda_a = f64_param("lambda_a", 0.999);
            let lambda_c = f64_param("lambda_c", 0.999);
            validation::validate_chen_wu(p0, q0, delta_t, lambda_a, lambda_c)
                .map_err(|e| McpError::invalid_params(e, None))?;
            Ok(Runner::UniChenWu(StreamingChenWuDetector::new(
                p0, q0, delta_t, lambda_a, lambda_c,
            )))
        }
        DetectorKind::SumCusum => {
            let d = usize_param("d", DEFAULT_D);
            let lambda = f64_param("lambda", 200.0);
            let max_rl = usize_param("max_rl", 250);
            validation::validate_multistream(d, lambda, max_rl)
                .map_err(|e| McpError::invalid_params(e, None))?;
            let inner: Vec<StreamingDetector> = (0..d)
                .map(|_| StreamingDetector::new(lambda, max_rl))
                .collect();
            let mut agg = SumCusumAggregator::new(inner);
            if let Some(t) = p.get("threshold").and_then(|v| v.as_f64()) {
                validation::validate_threshold(t)
                    .map_err(|e| McpError::invalid_params(e, None))?;
                agg = agg.with_threshold(t);
            }
            Ok(Runner::MultiSumCusum(agg))
        }
        DetectorKind::Hc => {
            let d = usize_param("d", DEFAULT_D);
            let lambda = f64_param("lambda", 200.0);
            let max_rl = usize_param("max_rl", 250);
            validation::validate_multistream(d, lambda, max_rl)
                .map_err(|e| McpError::invalid_params(e, None))?;
            let inner: Vec<StreamingDetector> = (0..d)
                .map(|_| StreamingDetector::new(lambda, max_rl))
                .collect();
            let mut agg = HcAggregator::new(inner);
            if let Some(t) = p.get("threshold").and_then(|v| v.as_f64()) {
                validation::validate_threshold(t)
                    .map_err(|e| McpError::invalid_params(e, None))?;
                agg = agg.with_threshold(t);
            }
            if let Some(n) = p.get("persistence").and_then(|v| v.as_u64()) {
                let n = n as usize;
                validation::validate_persistence(n)
                    .map_err(|e| McpError::invalid_params(e, None))?;
                agg = agg.with_persistence(n);
            }
            Ok(Runner::MultiHc(agg))
        }
        DetectorKind::FilterTick => {
            let d = usize_param("d", DEFAULT_D);
            let k = usize_param("k", 2);
            let threshold = f64_param("threshold", 0.5);
            validation::validate_filter_tick(d, k, threshold)
                .map_err(|e| McpError::invalid_params(e, None))?;
            Ok(Runner::MultiFilterTick(FilterTickAggregator::new(
                d, k, threshold,
            )))
        }
        DetectorKind::DmBocd => {
            let d = usize_param("d", DEFAULT_D);
            let lambda = f64_param("lambda", 200.0);
            let max_rl = usize_param("max_rl", 250);
            validation::validate_dm_bocd(d, lambda, max_rl)
                .map_err(|e| McpError::invalid_params(e, None))?;
            Ok(Runner::DmBocd(StreamingDmBocd::new(d, lambda, max_rl)))
        }
    }
}

fn decode_obs(kind: DetectorKind, raw: &serde_json::Value) -> Result<Vec<Obs>, String> {
    let arr = raw.as_array().ok_or("observations must be an array")?;
    match kind.obs_shape() {
        ObsShape::Scalar => arr
            .iter()
            .map(|v| {
                v.as_f64()
                    .map(Obs::F64)
                    .ok_or_else(|| format!("scalar expected, got {v}"))
            })
            .collect(),
        ObsShape::Vector => arr
            .iter()
            .map(|v| {
                let inner = v.as_array().ok_or("vector observation expected")?;
                inner
                    .iter()
                    .map(|x| x.as_f64().ok_or_else(|| format!("non-numeric: {x}")))
                    .collect::<Result<Vec<f64>, String>>()
                    .map(Obs::Vec)
            })
            .collect(),
    }
}

pub async fn cesura_snapshot_impl(
    streams: &Arc<Registry>,
    req: SnapshotRequest,
) -> Result<CallToolResult, McpError> {
    validation::validate_stream_id(&req.stream_id)
        .map_err(|e| McpError::invalid_params(e, None))?;
    let uuid = {
        let map = ID_MAP.read().expect("ID_MAP poisoned");
        match map.get(&req.stream_id) {
            Some(entry) => entry.uuid,
            None => {
                return Err(McpError::resource_not_found(
                    format!("stream_id {} not found", req.stream_id),
                    Some(json!({"stream_id": req.stream_id})),
                ));
            }
        }
    };
    let runner = streams.get(&uuid).ok_or_else(|| {
        McpError::internal_error(
            "stream_id resolved but Runner missing -- registry / id-map drift",
            Some(json!({"uuid": uuid})),
        )
    })?;
    let guard = runner.lock().await;
    let state = match guard.save() {
        Ok(s) => s,
        Err(e) => {
            return Ok(CallToolResult::structured_error(json!({
                "code": "snapshot_unsupported",
                "stream_id": req.stream_id,
                "kind": guard.kind(),
                "reason": e,
            })));
        }
    };
    let state_json = serde_json::to_value(&state)
        .map_err(|e| McpError::internal_error(format!("state ser: {e}"), None))?;
    Ok(CallToolResult::structured(json!({
        "stream_id": req.stream_id,
        "state": state_json,
    })))
}

pub async fn cesura_restore_impl(
    streams: &Arc<Registry>,
    req: RestoreRequest,
) -> Result<CallToolResult, McpError> {
    // Mirror cesura_snapshot's `snapshot_unsupported` for dm-bocd. Without
    // this, Runner::restore returns a generic "kind/payload mismatch or
    // unsupported kind" string; a stable `restore_unsupported` code gives
    // clients a clean discriminator paired with snapshot's response.
    if let Some("dm-bocd") = req.state.pointer("/kind").and_then(|v| v.as_str()) {
        return Ok(CallToolResult::structured_error(json!({
            "code": "restore_unsupported",
            "kind": "dm-bocd",
            "reason": "StreamingDmBocd does not yet implement save_state/restore (deferred per src/dm_bocd.rs:869)",
        })));
    }
    let state: RunnerState = serde_json::from_value(req.state.clone()).map_err(|e| {
        McpError::invalid_params(
            format!("state failed to deserialise as RunnerState: {e}"),
            Some(json!({"state_preview": preview(&req.state)})),
        )
    })?;
    // Extract expected_d from the payload BEFORE Runner::restore consumes
    // it. For multistream variants the d is either streams.len() (sum-cusum
    // / hc) or state.d (filter-tick). Keeps the cesura_feed shape-mismatch
    // guard correct for restored streams.
    let expected_d = expected_d_from_payload(&state.payload);
    let runner = Runner::restore(state)
        .map_err(|e| McpError::invalid_params(format!("restore: {e}"), None))?;
    let kind = runner.kind_enum();

    let new_str_id = format!("restored-{}", Uuid::new_v4());
    let uuid = Uuid::new_v4();
    streams.insert(uuid, runner);
    ID_MAP.write().expect("ID_MAP poisoned").insert(
        new_str_id.clone(),
        StreamEntry {
            uuid,
            kind,
            params: json!({}),
            expected_d,
        },
    );
    Ok(CallToolResult::structured(json!({
        "stream_id": new_str_id,
        "kind": kind.as_wire(),
    })))
}

/// Extract `d` from a restored `RunnerStatePayload` for the shape guard.
/// Univariate variants ⇒ `None`. Multistream variants carry d either as
/// `streams.len()` (sum-cusum / hc) or as a struct field (filter-tick).
fn expected_d_from_payload(payload: &RunnerStatePayload) -> Option<usize> {
    match payload {
        RunnerStatePayload::Uni(_) => None,
        #[cfg(feature = "joint-detection")]
        RunnerStatePayload::UniChenWu(_) => None,
        RunnerStatePayload::MultiSumCusum { streams, .. } => Some(streams.len()),
        RunnerStatePayload::MultiHc { streams, .. } => Some(streams.len()),
        RunnerStatePayload::MultiFilterTick(s) => Some(s.d),
    }
}

/// Truncate a JSON value's string form to ~200 codepoints for error
/// payloads. Char-boundary-safe: `&s[..n]` panics if `n` lands inside a
/// multi-byte sequence -- a panic on the error-rendering path would
/// corrupt stdout and kill the MCP channel, which is the exact failure
/// mode this layer exists to prevent.
fn preview(v: &serde_json::Value) -> String {
    let s = v.to_string();
    match s.char_indices().nth(200) {
        Some((i, _)) => format!("{}…", &s[..i]),
        None => s,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preview_handles_multi_byte_utf8() {
        // 100 codepoints of CJK = 300 bytes (each char is 3 bytes in UTF-8).
        // Byte 200 lands mid-codepoint; raw-byte slicing would panic here.
        let big = serde_json::json!("数据".repeat(100));
        let p = preview(&big);
        assert!(p.ends_with('…'), "expected truncation marker, got: {p}");
        assert!(
            p.is_char_boundary(p.len()),
            "truncation broke a char boundary"
        );
        // Short ASCII input passes through unchanged (JSON-encoded form).
        let small = serde_json::json!("hello");
        assert_eq!(preview(&small), "\"hello\"");
    }
}

pub async fn cesura_list_streams_impl(
    streams: &Arc<Registry>,
) -> Result<CallToolResult, McpError> {
    // Snapshot the reverse map (Uuid → client-supplied stream_id) with a
    // brief read lock, then drop the guard before awaiting registry.list.
    let reverse: HashMap<Uuid, String> = {
        let id_map = ID_MAP.read().expect("ID_MAP poisoned");
        id_map
            .iter()
            .map(|(stream_id, entry)| (entry.uuid, stream_id.clone()))
            .collect()
    };

    let summary = streams.list().await;
    let listing: Vec<serde_json::Value> = summary
        .into_iter()
        .map(|row| {
            // Orphan fallback: a uuid registered in the registry but absent
            // from ID_MAP (e.g. TOCTOU between insert paths) is still surfaced,
            // using the Uuid string as a stand-in stream_id.
            let stream_id = reverse
                .get(&row.id)
                .cloned()
                .unwrap_or_else(|| row.id.to_string());
            json!({
                "stream_id": stream_id,
                "uuid": row.id,
                "kind": row.kind,
            })
        })
        .collect();

    Ok(CallToolResult::structured(json!({
        "streams": listing,
    })))
}

pub async fn cesura_close_stream_impl(
    streams: &Arc<Registry>,
    req: CloseRequest,
) -> Result<CallToolResult, McpError> {
    validation::validate_stream_id(&req.stream_id)
        .map_err(|e| McpError::invalid_params(e, None))?;
    // Drop ID_MAP guard before the registry call to keep the critical
    // section tight. `was_present` ⇐ both ID_MAP had an entry ∧ registry
    // confirmed removal. Absent id → ok still true (idempotent), but
    // was_present=false surfaces the client bug.
    let entry = ID_MAP
        .write()
        .expect("ID_MAP poisoned")
        .remove(&req.stream_id);
    let was_present = match entry {
        Some(entry) => streams.close(&entry.uuid),
        None => false,
    };

    Ok(CallToolResult::structured(json!({
        "ok": true,
        "was_present": was_present,
        "stream_id": req.stream_id,
    })))
}
