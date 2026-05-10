//!
//!
//! The forbidden names are also forbidden as `const KNOWN_EVENTS` /
//! `static PREPARE_*` shadows, since those carry the same dedup
//! contract.

use std::fs;
use std::path::{Path, PathBuf};

const FORBIDDEN_FN_NAMES: &[&str] = &[
    "count_event_hits",
    "count_far_cps",
    "f1_with_margin",
    "parse_iso_date",
    "per_event_hits",
    "align_assets_3",
    "load_real_csv",
    "load_btc_csv",
    "prepare_real_csvs",
];

const FORBIDDEN_CONST_NAMES: &[&str] = &[
    "KNOWN_EVENTS",
    "BTC_PARQUET",
];

fn collect_rs_files(root: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(root) else {
        return;
    };
    for entry in entries.flatten() {
        let p = entry.path();
        if p.is_dir() {
            collect_rs_files(&p, out);
        } else if p.extension().map(|e| e == "rs").unwrap_or(false) {
            out.push(p);
        }
    }
}

#[test]
fn examples_have_no_dedup_helpers() {
    let mut files = Vec::new();
    collect_rs_files(Path::new("examples"), &mut files);
    let mut violations: Vec<String> = Vec::new();
    for f in files {
        let src = match fs::read_to_string(&f) {
            Ok(s) => s,
            Err(_) => continue,
        };
        for name in FORBIDDEN_FN_NAMES {
            let needle_pub = format!("pub fn {name}");
            let needle_priv = format!("fn {name}(");
            if src.contains(&needle_pub) || src.contains(&needle_priv) {
                violations.push(format!("{}: defines `fn {name}`", f.display()));
            }
        }
        for name in FORBIDDEN_CONST_NAMES {
            // Word-boundary match -- `const BTC_PARQUET:` (typed) and
            // `const BTC_PARQUET ` (whitespace) flag exact redefinitions
            // without false-matching `const BTC_PARQUET_PATH` aliases that
            // re-export bench's canonical constant.
            let needle_typed = format!("const {name}:");
            let needle_space = format!("const {name} ");
            if src.contains(&needle_typed) || src.contains(&needle_space) {
                violations.push(format!("{}: defines `const {name}`", f.display()));
            }
        }
    }
    assert!(
        violations.is_empty(),
        "legacy helpers must live exactly once in src/bench/, not examples/:\n  {}",
        violations.join("\n  ")
    );
}
