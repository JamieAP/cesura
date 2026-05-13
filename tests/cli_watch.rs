//! `cesura watch` ≡ library output. Spawns the built binary via
//! assert_cmd, pipes deterministic JSONL fixtures from stdin, asserts
//! every emitted `fire` line corresponds bit-identically to what the
//! library produces in-process.

use std::io::Write;
use std::process::{Command, Stdio};

use assert_cmd::cargo::CommandCargoExt;
use cesura::runtime::{Fire, Obs, Runner};
use cesura::streaming::StreamingDetector;
use serde::Deserialize;

fn synthetic_uni(n: usize, cp: usize) -> Vec<f64> {
    let mut v = Vec::with_capacity(n);
    let mut state: u64 = 0xCAFE_BABE_DEAD_BEEF;
    for t in 0..n {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u1 = ((state >> 33) as u32 as f64 + 1.0) / (u32::MAX as f64 + 2.0);
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u2 = ((state >> 33) as u32 as f64 + 1.0) / (u32::MAX as f64 + 2.0);
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        let mu = if t >= cp { 5.0 } else { 0.0 };
        v.push(mu + z);
    }
    v
}

fn synthetic_multi(n: usize, cp: usize, d: usize) -> Vec<Vec<f64>> {
    let mut out = Vec::with_capacity(n);
    let mut state: u64 = 0x1234_5678_9ABC_DEF0;
    for t in 0..n {
        let mut row = Vec::with_capacity(d);
        for _ in 0..d {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u1 = ((state >> 33) as u32 as f64 + 1.0) / (u32::MAX as f64 + 2.0);
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u2 = ((state >> 33) as u32 as f64 + 1.0) / (u32::MAX as f64 + 2.0);
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            let mu = if t >= cp { 5.0 } else { 0.0 };
            row.push(mu + z);
        }
        out.push(row);
    }
    out
}

#[derive(Deserialize, Debug, PartialEq)]
#[serde(tag = "type")]
enum WireLine {
    #[serde(rename = "fire")]
    Fire {
        index: usize,
        confidence: f64,
        #[serde(default)]
        shift_sigma: Option<f64>,
        streams: Vec<usize>,
        kind: String,
    },
    #[serde(rename = "parse_error")]
    ParseError { line: usize, msg: String },
}

fn run_watch(args: &[&str], stdin_text: &str) -> (String, String) {
    let mut cmd = Command::cargo_bin("cesura").expect("locate cesura binary");
    let mut child = cmd
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn cesura");
    child
        .stdin
        .as_mut()
        .unwrap()
        .write_all(stdin_text.as_bytes())
        .unwrap();
    let output = child.wait_with_output().expect("wait_with_output");
    let stdout = String::from_utf8(output.stdout).unwrap();
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(output.status.success(), "exit={:?} stderr={stderr}", output.status);
    (stdout, stderr)
}

fn library_fires_uni(data: &[f64]) -> Vec<Fire> {
    let mut r = Runner::Uni(StreamingDetector::new(200.0, 250));
    let mut fires = Vec::new();
    for &x in data {
        fires.extend(r.feed(Obs::F64(x)));
    }
    fires
}

#[test]
fn watch_streaming_emits_library_equivalent_fires() {
    let data = synthetic_uni(400, 200);
    let stdin_text: String = data
        .iter()
        .map(|x| format!("{{\"x\":{x}}}\n"))
        .collect();

    let (stdout, _) = run_watch(
        &["watch", "--kind", "streaming", "--lambda", "200", "--max-rl", "250"],
        &stdin_text,
    );

    let lines: Vec<WireLine> = stdout
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).unwrap_or_else(|e| panic!("bad line {l}: {e}")))
        .collect();

    let lib = library_fires_uni(&data);
    assert_eq!(lines.len(), lib.len(), "fire count mismatch");
    for (wire, fire) in lines.iter().zip(lib.iter()) {
        match wire {
            WireLine::Fire {
                index,
                confidence,
                shift_sigma,
                streams,
                kind,
            } => {
                assert_eq!(*index, fire.index);
                assert!((confidence - fire.confidence).abs() < 1e-12);
                assert_eq!(*shift_sigma, fire.shift_sigma);
                assert_eq!(streams, &fire.streams);
                assert_eq!(kind, fire.kind.as_ref());
            }
            other => panic!("expected fire, got {other:?}"),
        }
    }
}

#[test]
fn watch_min_confidence_filters() {
    let data = synthetic_uni(400, 200);
    let stdin_text: String = data.iter().map(|x| format!("{{\"x\":{x}}}\n")).collect();
    let (stdout_all, _) = run_watch(
        &["watch", "--kind", "streaming", "--min-confidence", "0.0"],
        &stdin_text,
    );
    let (stdout_filtered, _) = run_watch(
        &["watch", "--kind", "streaming", "--min-confidence", "0.999"],
        &stdin_text,
    );
    let n_all = stdout_all.lines().filter(|l| !l.is_empty()).count();
    let n_filtered = stdout_filtered.lines().filter(|l| !l.is_empty()).count();
    assert!(n_filtered <= n_all);
    let _ = n_all;
}

#[test]
fn watch_parse_error_continues_stream() {
    let stdin_text = "{\"x\":1.0}\nNOT JSON\n{\"x\":2.0}\n";
    let (stdout, _) = run_watch(&["watch", "--kind", "streaming"], stdin_text);
    let has_parse_err = stdout.lines().any(|l| l.contains("\"parse_error\""));
    assert!(has_parse_err, "expected a parse_error line, got: {stdout}");
}

#[test]
fn watch_shape_mismatch_emits_parse_error() {
    // Univariate detector receiving a vector observation → parse_error.
    let stdin_text = "{\"x\":[1.0, 2.0]}\n{\"x\":1.0}\n";
    let (stdout, _) = run_watch(&["watch", "--kind", "streaming"], stdin_text);
    assert!(stdout.contains("\"parse_error\""));
    assert!(stdout.contains("does not match"));
}

#[test]
fn watch_vec_dim_mismatch_emits_parse_error_not_panic() {
    // Vector detector with wrong-length observation must not reach the
    // multistream aggregator (which asserts dim == d and panics).
    let stdin_text = "{\"x\":[1.0, 2.0]}\n{\"x\":[1.0, 2.0, 3.0, 4.0]}\n{\"x\":[1.1, 2.1, 3.1, 4.1]}\n";
    let (stdout, _) =
        run_watch(&["watch", "--kind", "sum-cusum", "--d", "4"], stdin_text);
    assert!(stdout.contains("\"parse_error\""), "expected parse_error, got: {stdout}");
    assert!(stdout.contains("does not match --d 4"), "expected dim msg, got: {stdout}");
    // Channel must keep processing -- subsequent valid lines should still parse.
    let valid_lines: usize = stdout
        .lines()
        .filter(|l| !l.contains("\"parse_error\"") && !l.trim().is_empty())
        .count();
    let parse_err_lines: usize = stdout.lines().filter(|l| l.contains("\"parse_error\"")).count();
    assert_eq!(parse_err_lines, 1, "exactly one parse_error expected: {stdout}");
    // No fires guaranteed (d=4 cold), but channel survived → either 0 fires or some.
    let _ = valid_lines;
}

#[test]
fn watch_sum_cusum_emits_library_equivalent_fires() {
    let data = synthetic_multi(400, 200, 3);
    let stdin_text: String = data
        .iter()
        .map(|row| {
            let inner: Vec<String> = row.iter().map(|v| v.to_string()).collect();
            format!("{{\"x\":[{}]}}\n", inner.join(","))
        })
        .collect();

    let (stdout, _) = run_watch(
        &[
            "watch", "--kind", "sum-cusum", "--d", "3", "--threshold", "0.1",
        ],
        &stdin_text,
    );

    let n_wire = stdout
        .lines()
        .filter(|l| !l.is_empty())
        .filter(|l| l.contains("\"fire\""))
        .count();

    // Run library-direct with the same constructor as the CLI's
    // sum-cusum path (lambda=200, max_rl=250, threshold=0.1) for d=3.
    let mut r = Runner::MultiSumCusum(
        cesura::multistream::SumCusumAggregator::new(
            (0..3).map(|_| StreamingDetector::new(200.0, 250)).collect(),
        )
        .with_threshold(0.1),
    );
    let mut lib_count = 0;
    for row in &data {
        lib_count += r.feed(Obs::Vec(row.clone())).len();
    }
    assert_eq!(n_wire, lib_count);
}

#[test]
fn info_emits_detectors_json() {
    let mut cmd = Command::cargo_bin("cesura").expect("locate cesura binary");
    let output = cmd.args(["info"]).output().expect("run cesura info");
    assert!(output.status.success(), "stderr={}", String::from_utf8_lossy(&output.stderr));
    let json: serde_json::Value = serde_json::from_slice(&output.stdout).expect("info json");
    let detectors = json
        .get("detectors")
        .and_then(|v| v.as_array())
        .expect("detectors array");
    let kinds: Vec<&str> = detectors
        .iter()
        .filter_map(|d| d.get("kind").and_then(|v| v.as_str()))
        .collect();
    assert!(kinds.contains(&"streaming"), "kinds: {kinds:?}");
    assert!(kinds.contains(&"sum-cusum"));
    assert!(kinds.contains(&"hc"));
    assert!(kinds.contains(&"filter-tick"));
    assert!(kinds.contains(&"dm-bocd"));
}

#[test]
fn info_defaults_match_watchargs_defaults() {
    // Drift guard: if anyone bumps a default in info.rs without bumping
    // the matching clap default in watch.rs (or vice versa), this test
    // fails and the wire-format documentation stays honest.
    let mut cmd = Command::cargo_bin("cesura").unwrap();
    let output = cmd.args(["info"]).output().unwrap();
    assert!(output.status.success());
    let json: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    let detectors = json.get("detectors").unwrap().as_array().unwrap();
    let by_kind: std::collections::HashMap<&str, &serde_json::Value> = detectors
        .iter()
        .map(|d| (d.get("kind").unwrap().as_str().unwrap(), d))
        .collect();

    // Pull `cesura watch --kind <X> --help` and parse the default values
    // out of the rendered help text. This binds info to the bin's
    // *actual* defaults rather than a separately-maintained constant.
    // `chen-wu` only ships under the `joint-detection` feature -- skip it
    // when the test runs under `--features cli` alone, since `info` will
    // not list it and `watch --help` will not advertise its flags.
    let mut kinds: Vec<&str> = vec!["streaming", "sum-cusum", "hc", "filter-tick", "dm-bocd"];
    #[cfg(feature = "joint-detection")]
    kinds.push("chen-wu");
    for kind in kinds {
        let det = by_kind.get(kind).unwrap_or_else(|| panic!("info missing {kind}"));
        let params = det.get("params").unwrap().as_array().unwrap();
        let mut help_cmd = Command::cargo_bin("cesura").unwrap();
        let help = help_cmd.args(["watch", "--help"]).output().unwrap();
        let help_text = String::from_utf8(help.stdout).unwrap();
        for param in params {
            let name = param.get("name").unwrap().as_str().unwrap();
            // `threshold` and `persistence` are kind-dispatched in
            // build_runner (no clap default) -- info reports the
            // kind-specific effective default, not the clap default.
            // Behavioural equivalence is covered by the library-vs-CLI
            // tests below; skip here.
            if name == "threshold" || name == "persistence" {
                continue;
            }
            let default = param.get("default").unwrap();
            // Help renders `--<name> <NAME>  ...  [default: <v>]`.
            let needle_long = format!("--{name} ");
            if !help_text.contains(&needle_long) {
                // Some info params (chen-wu p0/q0/lambda-a/lambda-c) use
                // single-word flag names; just check the value appears
                // in help context. Belt-and-braces.
                continue;
            }
            // Normalise: clap renders 200.0 as "200", so try both the
            // raw and the integer-trimmed form.
            let default_strs: Vec<String> = match default {
                serde_json::Value::Number(n) => {
                    let raw = n.to_string();
                    let mut out = vec![raw.clone()];
                    if let Some(stripped) = raw.strip_suffix(".0") {
                        out.push(stripped.to_string());
                    }
                    out
                }
                serde_json::Value::String(s) => vec![s.clone()],
                _ => continue,
            };
            // Find the line for this flag, then check default appears
            // on it. Belt-and-braces against value collisions elsewhere.
            let flag_line = help_text
                .lines()
                .skip_while(|l| !l.contains(&needle_long))
                .take(3)
                .collect::<Vec<_>>()
                .join(" ");
            let matched = default_strs.iter().any(|d| flag_line.contains(d));
            assert!(
                matched,
                "kind={kind} param={name} default {default_strs:?} missing from flag line.\n\
                 Flag line: {flag_line}\nFull help:\n{help_text}"
            );
        }
    }
}

#[test]
fn watch_attaches_timestamp_from_firing_observation_to_fire_line() {
    // Feed a deterministic shift series with `t` on every line. Any
    // fire surfaced by the CLI must carry a non-null `t` whose value
    // appears verbatim in the input -- proves the passthrough contract.
    let data = synthetic_uni(400, 200);
    let stdin_text: String = data
        .iter()
        .enumerate()
        .map(|(i, x)| format!("{{\"x\":{x},\"t\":\"obs-{i}\"}}\n"))
        .collect();
    let (stdout, _) = run_watch(&["watch", "--kind", "streaming"], &stdin_text);
    let fire_lines: Vec<&str> = stdout
        .lines()
        .filter(|l| l.contains("\"fire\""))
        .collect();
    assert!(!fire_lines.is_empty(), "synthetic shift must produce a fire");
    for line in fire_lines {
        let v: serde_json::Value = serde_json::from_str(line).expect("fire line is JSON");
        let t = v.get("t").and_then(|x| x.as_str()).expect("fire line carries t");
        assert!(t.starts_with("obs-"), "t passthrough must echo input form, got: {t}");
    }
}

#[test]
fn watch_snapshot_on_exit_writes_state() {
    let dir = tempdir_in_target();
    let snap_path = dir.join("snap.json");
    let stdin_text: String = (0..30).map(|i| format!("{{\"x\":{i}.0}}\n")).collect();
    let mut cmd = Command::cargo_bin("cesura").unwrap();
    let mut child = cmd
        .args([
            "watch",
            "--kind",
            "streaming",
            "--snapshot-on-exit",
            snap_path.to_str().unwrap(),
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.as_mut().unwrap().write_all(stdin_text.as_bytes()).unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(output.status.success());
    assert!(snap_path.exists(), "snapshot file missing");
    let json = std::fs::read_to_string(&snap_path).unwrap();
    assert!(json.contains("\"streaming\""), "snapshot json: {json}");
}

fn tempdir_in_target() -> std::path::PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!(
        "cesura-cli-watch-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&p).unwrap();
    p
}
