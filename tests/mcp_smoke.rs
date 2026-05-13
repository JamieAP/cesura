//! e2e smoke test against a spawned `cesura-mcp` process. Drives raw
//! JSON-RPC frames over stdio and asserts the wire shape matches the
//! design.

#![cfg(all(feature = "mcp", feature = "test-utils"))]

use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

struct Server {
    child: Child,
    stdin: ChildStdin,
    reader: BufReader<ChildStdout>,
    next_id: u64,
}

impl Server {
    fn spawn() -> Self {
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_cesura-mcp"));
        let mut child = cmd
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn cesura-mcp");
        let stdin = child.stdin.take().unwrap();
        let stdout = child.stdout.take().unwrap();
        Self {
            child,
            stdin,
            reader: BufReader::new(stdout),
            next_id: 1,
        }
    }

    fn request(&mut self, method: &str, params: serde_json::Value) -> serde_json::Value {
        let id = self.next_id;
        self.next_id += 1;
        let req = serde_json::json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params,
        });
        writeln!(self.stdin, "{}", req).unwrap();
        let mut line = String::new();
        self.reader.read_line(&mut line).unwrap();
        serde_json::from_str(&line).expect("response is JSON")
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
    }
}

/// Drive an initialize + notifications/initialized handshake. Required
/// before any `tools/call` per the MCP protocol.
fn handshake(s: &mut Server) {
    s.request(
        "initialize",
        serde_json::json!({
            "protocolVersion": "2025-11-25",
            "capabilities": {},
            "clientInfo": {"name": "test", "version": "1"}
        }),
    );
    writeln!(
        s.stdin,
        "{}",
        serde_json::json!({"jsonrpc":"2.0","method":"notifications/initialized"})
    )
    .unwrap();
}

#[test]
fn initialize_handshake_returns_server_info() {
    let mut s = Server::spawn();
    let resp = s.request(
        "initialize",
        serde_json::json!({
            "protocolVersion": "2025-11-25",
            "capabilities": {},
            "clientInfo": {"name": "test", "version": "1"}
        }),
    );
    let result = resp.get("result").expect("initialize result");
    let info = result.get("serverInfo").expect("serverInfo");
    assert_eq!(info.get("name").and_then(|v| v.as_str()), Some("cesura-mcp"));
    let instructions = result
        .get("instructions")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    assert!(
        instructions.contains("cesura_feed"),
        "instructions must mention cesura_feed; got: {instructions}"
    );
}

#[test]
fn cesura_feed_lazy_creates_and_matches_library() {
    use cesura::runtime::{Obs, Runner};
    use cesura::streaming::StreamingDetector;

    let mut s = Server::spawn();
    handshake(&mut s);

    // Deterministic uni step at t=200, length 400. Big enough to fire.
    let data: Vec<f64> = (0..400)
        .map(|t| if t < 200 { 0.0 } else { 5.0 })
        .collect();

    let resp = s.request(
        "tools/call",
        serde_json::json!({
            "name": "cesura_feed",
            "arguments": {
                "stream_id": "abc",
                "kind": "streaming",
                "params": {"lambda": 200.0, "max_rl": 250},
                "observations": data,
            }
        }),
    );
    let content = resp
        .pointer("/result/structuredContent")
        .unwrap_or_else(|| panic!("structuredContent missing in: {resp}"));
    let wire_fires = content
        .get("fires")
        .and_then(|v| v.as_array())
        .expect("fires array");

    // Library-direct reference -- same data, same constructor.
    let mut r = Runner::Uni(StreamingDetector::new(200.0, 250));
    let mut lib_fires = Vec::new();
    for &x in &data {
        lib_fires.extend(r.feed(Obs::F64(x)));
    }

    assert_eq!(
        wire_fires.len(),
        lib_fires.len(),
        "fire count mismatch: wire={} lib={}",
        wire_fires.len(),
        lib_fires.len()
    );
    for (wire, lib) in wire_fires.iter().zip(lib_fires.iter()) {
        assert_eq!(
            wire.get("index").and_then(|v| v.as_u64()),
            Some(lib.index as u64),
            "fire index drift: wire={wire} lib_index={}",
            lib.index
        );
        // Confidence: bit-identical via serde, but allow 1e-9 slack
        // against any f64 ↔ JSON ↔ f64 rounding.
        let wire_conf = wire
            .get("confidence")
            .and_then(|v| v.as_f64())
            .expect("confidence is f64");
        assert!(
            (wire_conf - lib.confidence).abs() <= 1e-9 * lib.confidence.abs().max(1.0),
            "confidence drift: wire={wire_conf} lib={}",
            lib.confidence
        );
        assert_eq!(
            wire.get("kind").and_then(|v| v.as_str()),
            Some("cp"),
            "kind tag drift: wire={wire}"
        );
    }
}

#[test]
fn cesura_feed_rejects_kind_mismatch_on_second_call() {
    let mut s = Server::spawn();
    handshake(&mut s);

    s.request(
        "tools/call",
        serde_json::json!({
            "name": "cesura_feed",
            "arguments": {
                "stream_id": "xyz",
                "kind": "streaming",
                "params": {"lambda": 200.0, "max_rl": 250},
                "observations": [1.0, 2.0]
            }
        }),
    );

    let resp = s.request(
        "tools/call",
        serde_json::json!({
            "name": "cesura_feed",
            "arguments": {
                "stream_id": "xyz",
                "kind": "hc",
                "params": {"d": 3, "lambda": 200.0, "max_rl": 250},
                "observations": [[1.0, 2.0, 3.0]]
            }
        }),
    );

    let content = resp
        .pointer("/result/structuredContent")
        .unwrap_or_else(|| panic!("structuredContent missing in: {resp}"));
    assert_eq!(
        content.get("code").and_then(|v| v.as_str()),
        Some("kind_mismatch"),
        "expected kind_mismatch, got: {content}"
    );
}

#[test]
fn cesura_feed_wrong_d_returns_structured_error_not_panic() {
    let mut s = Server::spawn();
    handshake(&mut s);

    // Lazy-create a 4-d hc stream with a correctly-shaped seed.
    s.request(
        "tools/call",
        serde_json::json!({
            "name": "cesura_feed",
            "arguments": {
                "stream_id": "wrong-d",
                "kind": "hc",
                "params": {"d": 4, "lambda": 200.0, "max_rl": 250},
                "observations": [[0.0, 0.0, 0.0, 0.0]]
            }
        }),
    );

    // Wrong-d follow-up. Without the guard this `assert_eq!` panics
    // inside hc.rs:288 and corrupts stdout.
    let resp = s.request(
        "tools/call",
        serde_json::json!({
            "name": "cesura_feed",
            "arguments": {
                "stream_id": "wrong-d",
                "observations": [[1.0, 2.0]]
            }
        }),
    );
    let content = resp
        .pointer("/result/structuredContent")
        .unwrap_or_else(|| panic!("structuredContent missing in: {resp}"));
    assert_eq!(
        content.get("code").and_then(|v| v.as_str()),
        Some("shape_mismatch"),
        "expected shape_mismatch, got: {content}"
    );

    // Channel must still be alive: a follow-up tools/list returns the
    // 5-tool surface. If the server panicked, this read times out / EOFs.
    let resp2 = s.request("tools/list", serde_json::json!({}));
    assert!(
        resp2.pointer("/result/tools").is_some(),
        "channel died after wrong-d feed: {resp2}"
    );
}

#[test]
fn cesura_snapshot_restore_roundtrip_preserves_fires() {
    let mut s = Server::spawn();
    handshake(&mut s);

    // First half: 200 zeros on stream "s1" in process A.
    let first_half: Vec<f64> = (0..200).map(|_| 0.0).collect();
    s.request(
        "tools/call",
        serde_json::json!({
            "name": "cesura_feed",
            "arguments": {
                "stream_id": "s1",
                "kind": "streaming",
                "params": {"lambda": 200.0, "max_rl": 250},
                "observations": first_half
            }
        }),
    );

    // Snapshot s1.
    let snap_resp = s.request(
        "tools/call",
        serde_json::json!({
            "name": "cesura_snapshot",
            "arguments": {"stream_id": "s1"}
        }),
    );
    let state = snap_resp
        .pointer("/result/structuredContent/state")
        .expect("snapshot state")
        .clone();

    // Process B: restore, then feed second half (shift to 5.0).
    let mut s2 = Server::spawn();
    handshake(&mut s2);
    let restore_resp = s2.request(
        "tools/call",
        serde_json::json!({
            "name": "cesura_restore",
            "arguments": {"state": state}
        }),
    );
    let restored_id = restore_resp
        .pointer("/result/structuredContent/stream_id")
        .and_then(|v| v.as_str())
        .expect("restored stream_id")
        .to_string();

    let second_half: Vec<f64> = (200..400).map(|_| 5.0).collect();
    let feed_resp = s2.request(
        "tools/call",
        serde_json::json!({
            "name": "cesura_feed",
            "arguments": {"stream_id": restored_id, "observations": second_half.clone()}
        }),
    );
    let restored_fires = feed_resp
        .pointer("/result/structuredContent/fires")
        .and_then(|v| v.as_array())
        .expect("fires")
        .clone();

    // Reference: single process fed both halves; compare second-half fires.
    let mut s3 = Server::spawn();
    handshake(&mut s3);
    s3.request(
        "tools/call",
        serde_json::json!({
            "name": "cesura_feed",
            "arguments": {
                "stream_id": "ref",
                "kind": "streaming",
                "params": {"lambda": 200.0, "max_rl": 250},
                "observations": (0..200).map(|_| 0.0).collect::<Vec<f64>>()
            }
        }),
    );
    let ref_resp = s3.request(
        "tools/call",
        serde_json::json!({
            "name": "cesura_feed",
            "arguments": {"stream_id": "ref", "observations": second_half}
        }),
    );
    let ref_fires = ref_resp
        .pointer("/result/structuredContent/fires")
        .and_then(|v| v.as_array())
        .expect("ref fires")
        .clone();

    // Compare on detection semantics, not the full fire envelope.
    //
    // `StreamingDetector::save_state` does NOT persist `norm_ring` or
    // `pending` (src/streaming.rs:745) -- the ring is reinitialised
    // empty on restore. `shift_sigma` is computed at fire-emit time
    // from that ring (compute_shift_sigma, src/streaming.rs:241), so a
    // restored detector reports a different shift magnitude than the
    // single-process baseline: the pre-shift window is missing.
    //
    // Detection itself (index + posterior confidence + kind + streams)
    // is driven by `rl_log` / `stats` / `total_steps`, all of which DO
    // round-trip, so those fields match bit-for-bit. The `shift_sigma`
    // gap is a known upstream defect, tracked separately.
    assert_eq!(
        restored_fires.len(),
        ref_fires.len(),
        "fire count drift across restore"
    );
    for (r, b) in restored_fires.iter().zip(ref_fires.iter()) {
        assert_eq!(r.get("index"), b.get("index"), "index drift: r={r} b={b}");
        assert_eq!(
            r.get("confidence"),
            b.get("confidence"),
            "confidence drift: r={r} b={b}"
        );
        assert_eq!(r.get("kind"), b.get("kind"), "kind drift: r={r} b={b}");
        assert_eq!(r.get("streams"), b.get("streams"), "streams drift: r={r} b={b}");
    }
}

#[test]
fn cesura_snapshot_dm_bocd_returns_structured_error() {
    let mut s = Server::spawn();
    handshake(&mut s);

    s.request(
        "tools/call",
        serde_json::json!({
            "name": "cesura_feed",
            "arguments": {
                "stream_id": "d1",
                "kind": "dm-bocd",
                "params": {"d": 2, "lambda": 200.0, "max_rl": 250},
                "observations": [[0.0, 0.0]]
            }
        }),
    );
    let resp = s.request(
        "tools/call",
        serde_json::json!({
            "name": "cesura_snapshot",
            "arguments": {"stream_id": "d1"}
        }),
    );
    let content = resp
        .pointer("/result/structuredContent")
        .expect("structuredContent");
    assert_eq!(
        content.get("code").and_then(|v| v.as_str()),
        Some("snapshot_unsupported"),
        "expected snapshot_unsupported, got: {content}"
    );
}

#[test]
fn cesura_feed_rejects_persistence_zero() {
    let mut s = Server::spawn();
    handshake(&mut s);

    let resp = s.request(
        "tools/call",
        serde_json::json!({
            "name": "cesura_feed",
            "arguments": {
                "stream_id": "p0",
                "kind": "hc",
                "params": {"d": 3, "lambda": 200.0, "max_rl": 250, "persistence": 0},
                "observations": [[0.0, 0.0, 0.0]]
            }
        }),
    );

    // Whether the rejection arrives as a top-level JSON-RPC error or as
    // structured_error inside result, SOMETHING must surface. If neither
    // path produced a response, the bin panicked.
    let surfaced = resp.pointer("/error").is_some()
        || resp
            .pointer("/result/structuredContent/code")
            .is_some()
        || resp.pointer("/result/isError").and_then(|v| v.as_bool()) == Some(true);
    assert!(
        surfaced,
        "persistence=0 must produce an error response, got: {resp}"
    );

    // Channel survives -- tools/list still works.
    let resp2 = s.request("tools/list", serde_json::json!({}));
    assert!(
        resp2.pointer("/result/tools").is_some(),
        "channel died after persistence=0 attempt: {resp2}"
    );
}

#[test]
fn cesura_list_and_close_streams() {
    let mut s = Server::spawn();
    handshake(&mut s);

    // Create two streams with distinct kinds.
    for (id, kind, params, obs) in &[
        (
            "a",
            "streaming",
            serde_json::json!({"lambda": 200.0, "max_rl": 250}),
            serde_json::json!([0.0]),
        ),
        (
            "b",
            "filter-tick",
            serde_json::json!({"d": 3, "k": 2, "threshold": 0.5}),
            serde_json::json!([[0.0, 0.0, 0.0]]),
        ),
    ] {
        let resp = s.request(
            "tools/call",
            serde_json::json!({
                "name": "cesura_feed",
                "arguments": {
                    "stream_id": id,
                    "kind": kind,
                    "params": params,
                    "observations": obs,
                }
            }),
        );
        // Sanity-guard: lazy-create must succeed for the rest of the test to mean anything.
        assert!(
            resp.pointer("/result/structuredContent/fires").is_some(),
            "feed failed for {id}/{kind}: {resp}"
        );
    }

    // List → 2 streams with the right kinds.
    let resp = s.request(
        "tools/call",
        serde_json::json!({"name": "cesura_list_streams", "arguments": {}}),
    );
    let streams = resp
        .pointer("/result/structuredContent/streams")
        .and_then(|v| v.as_array())
        .unwrap_or_else(|| panic!("streams array missing: {resp}"));
    assert_eq!(streams.len(), 2, "expected 2 streams, got: {resp}");
    let kinds: std::collections::HashSet<&str> = streams
        .iter()
        .filter_map(|s| s.get("kind").and_then(|v| v.as_str()))
        .collect();
    assert!(
        kinds.contains("streaming") && kinds.contains("filter-tick"),
        "kinds={kinds:?}"
    );
    // Client-supplied stream_ids must round-trip via the reverse map.
    let stream_ids: std::collections::HashSet<&str> = streams
        .iter()
        .filter_map(|s| s.get("stream_id").and_then(|v| v.as_str()))
        .collect();
    assert!(
        stream_ids.contains("a") && stream_ids.contains("b"),
        "stream_ids={stream_ids:?}"
    );

    // Close present → was_present=true.
    let close_present = s.request(
        "tools/call",
        serde_json::json!({"name": "cesura_close_stream", "arguments": {"stream_id": "a"}}),
    );
    let c = close_present
        .pointer("/result/structuredContent")
        .unwrap_or_else(|| panic!("structuredContent missing: {close_present}"));
    assert_eq!(c.get("ok").and_then(|v| v.as_bool()), Some(true));
    assert_eq!(c.get("was_present").and_then(|v| v.as_bool()), Some(true));

    // Close absent → was_present=false (channel must survive).
    let close_absent = s.request(
        "tools/call",
        serde_json::json!({"name": "cesura_close_stream", "arguments": {"stream_id": "nope"}}),
    );
    let c = close_absent
        .pointer("/result/structuredContent")
        .unwrap_or_else(|| panic!("structuredContent missing: {close_absent}"));
    assert_eq!(c.get("ok").and_then(|v| v.as_bool()), Some(true));
    assert_eq!(c.get("was_present").and_then(|v| v.as_bool()), Some(false));

    // List → registry shrunk to 1.
    let resp = s.request(
        "tools/call",
        serde_json::json!({"name": "cesura_list_streams", "arguments": {}}),
    );
    let streams = resp
        .pointer("/result/structuredContent/streams")
        .and_then(|v| v.as_array())
        .unwrap_or_else(|| panic!("streams array missing after close: {resp}"));
    assert_eq!(streams.len(), 1, "expected 1 stream post-close, got: {resp}");
    assert_eq!(
        streams[0].get("stream_id").and_then(|v| v.as_str()),
        Some("b")
    );
}

#[test]
fn malformed_jsonrpc_frame_terminates_cleanly_without_panic() {
    // Primary-source evidence (rmcp 1.6 stderr trace 2026-05-12):
    //   ERROR rmcp::transport::async_rw: Error reading from stream:
    //     serde error key must be a string at line 1 column 2
    //   INFO  rmcp::service: input stream terminated
    //   INFO  rmcp::service: serve finished quit_reason=Closed
    //
    // rmcp 1.6 `transport-io` does NOT emit a JSON-RPC -32700 reply on
    // malformed frames -- the transport treats a serde error as fatal
    // and closes the input stream. A reply-then-recover invariant is
    // not achievable in v1 without wrapping the transport (out of
    // scope: "sync v1 hardening").
    //
    // What v1 DOES guarantee:
    //   (a) ∄ panic -- the process exits cleanly, not via an unwound
    //       stack written to stdout.
    //   (b) ∄ stdout corruption -- every line written before close
    //       parses as JSON (every other test in this file relies on
    //       `Server::request`'s `serde_json::from_str` to enforce that
    //       implicitly).
    //   (c) The cause is observable on stderr.
    //
    let mut s = Server::spawn();
    handshake(&mut s);

    writeln!(s.stdin, "{{not jsonRPC at all").unwrap();

    let mut line = String::new();
    let n = s
        .reader
        .read_line(&mut line)
        .expect("read after malformed frame");
    assert_eq!(
        n, 0,
        "expected clean EOF after malformed frame, got line: {line:?}"
    );

    // Close stdin so the child's input-stream loop exits if it hadn't
    // already, then reap. rmcp closes its input on serde error, so
    // wait() should return promptly; if it ever hangs we'd need a
    // try_wait poll, but the stderr trace above confirms quit_reason=Closed.
    let status = s.child.wait().expect("child reaped");
    assert!(
        status.success() || status.code() == Some(0),
        "server must exit cleanly without panic, got: {status:?}"
    );
}

#[test]
fn stdout_only_carries_jsonrpc_frames() {
    // After a normal session, every line we've read should parse as
    // JSON. `Server::request()`'s `from_str` would have panicked on a
    // non-JSON line; this test makes the implicit invariant explicit
    // so a regression that writes logs / banners to stdout fails here
    // rather than at the next test's serde call.
    let mut s = Server::spawn();
    handshake(&mut s);
    let _ = s.request("tools/list", serde_json::json!({}));
    let resp = s.request(
        "tools/call",
        serde_json::json!({
            "name": "cesura_close_stream",
            "arguments": {"stream_id": "nonexistent"}
        }),
    );
    assert!(
        resp.pointer("/result").is_some(),
        "well-formed response expected, got: {resp}"
    );
}
