//! DashMap<Uuid, Arc<tokio::sync::Mutex<Runner>>> stream registry.
//!
//! Outer DashMap shards provide sharded registry access. Inner per-stream
//! Mutex = serialised state mutations for one stream. Two streams can
//! be fed concurrently.
//!
//! DashMap accessors hold a bucket lock.
//! Clone the inner `Arc<Mutex<Runner>>` out before `.await` anywhere;
//! holding a DashMap iter / get guard across `.await` deadlocks.
//!
//! `list()` awaits each per-stream Mutex sequentially -- latency
//! scales with the slowest concurrent feed. Acceptable at MCP cadence;
//! revisit with `try_lock` + busy markers if a multi-tenant deployment
//! needs sub-feed-latency listing.

use std::sync::Arc;

use dashmap::DashMap;
use tokio::sync::Mutex;
use uuid::Uuid;

use crate::runtime::Runner;

#[derive(Default)]
pub struct Registry {
    inner: DashMap<Uuid, Arc<Mutex<Runner>>>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct StreamSummary {
    pub id: Uuid,
    pub kind: String,
}

impl Registry {
    pub fn insert(&self, id: Uuid, runner: Runner) {
        self.inner.insert(id, Arc::new(Mutex::new(runner)));
    }

    /// Return the inner Arc so callers can drop the DashMap accessor
    /// BEFORE `.lock().await`. Returning the accessor directly would
    /// invite the held-across-await deadlock.
    pub fn get(&self, id: &Uuid) -> Option<Arc<Mutex<Runner>>> {
        self.inner.get(id).map(|r| r.value().clone())
    }

    /// Remove a stream from the lookup table. Returns true if the id was
    /// present. Any in-flight operation holding a cloned `Arc<Mutex<Runner>>`
    /// completes against the now-orphaned Mutex, then drops naturally --
    /// `close` does NOT block on in-flight work.
    pub fn close(&self, id: &Uuid) -> bool {
        self.inner.remove(id).is_some()
    }

    /// List every stream. Clones Arcs out before any await so callers
    /// can safely await per-stream locks without holding bucket locks.
    pub async fn list(&self) -> Vec<StreamSummary> {
        let entries: Vec<(Uuid, Arc<Mutex<Runner>>)> = self
            .inner
            .iter()
            .map(|r| (*r.key(), r.value().clone()))
            .collect();
        let mut out = Vec::with_capacity(entries.len());
        for (id, runner) in entries {
            let guard = runner.lock().await;
            out.push(StreamSummary {
                id,
                kind: guard.kind().to_string(),
            });
        }
        out
    }

    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.inner.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::runtime::Runner;
    use crate::streaming::StreamingDetector;
    use uuid::Uuid;

    #[tokio::test]
    async fn registry_insert_and_lookup() {
        let reg = Registry::default();
        let id = Uuid::new_v4();
        let runner = Runner::Uni(StreamingDetector::new(200.0, 250));
        reg.insert(id, runner);
        let entry = reg.get(&id).expect("present");
        let guard = entry.lock().await;
        assert_eq!(guard.kind(), "streaming");
    }

    #[tokio::test]
    async fn registry_close_returns_presence_flag() {
        let reg = Registry::default();
        let id = Uuid::new_v4();
        assert!(!reg.close(&id), "absent id returns false");
        let runner = Runner::Uni(StreamingDetector::new(200.0, 250));
        reg.insert(id, runner);
        assert!(reg.close(&id), "present id returns true");
        assert!(!reg.close(&id), "double close returns false");
    }

    #[tokio::test]
    async fn registry_list_yields_summary_per_stream() {
        let reg = Registry::default();
        let id = Uuid::new_v4();
        reg.insert(id, Runner::Uni(StreamingDetector::new(200.0, 250)));
        let listing = reg.list().await;
        assert_eq!(listing.len(), 1);
        assert_eq!(listing[0].id, id);
        assert_eq!(listing[0].kind, "streaming");
    }
}
