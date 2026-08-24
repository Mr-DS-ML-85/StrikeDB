//! Query / reducer router — the cost-based, ANN-aware planner + tiered memory.
//!
//! The key architectural fix: vector distance is a *cost-aware operator*, so
//! filtered similarity search is planned, not bolted on. Given a predicate's
//! estimated selectivity the planner chooses:
//!   * PreFilter  (filter-then-scan)  — when the predicate is very selective
//!   * PostFilter (index-then-filter) — when the predicate is loose
//! This is the pgvector gap ("no filtered-ANN planning") closed.

pub mod tiered;

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::RwLock;
use storage::Engine;
use views::{Tables, VectorIndex};

pub use tiered::TieredMemory;

#[derive(Debug, PartialEq)]
pub enum AnnPlan {
    /// Filter first (scan matching rows), then rank by vector distance.
    PreFilter,
    /// Query the ANN index wide, then drop rows failing the predicate.
    PostFilter,
}

pub struct Router {
    engine: Arc<Engine>,
    tables: Tables,
    vectors_default: Arc<VectorIndex>,
    vectors_ns: RwLock<HashMap<String, Arc<VectorIndex>>>,
}

/// A planned RAG hit: the row id, its cosine distance, and the joined row.
#[derive(Debug)]
pub struct RagHit {
    pub id: u64,
    pub distance: f32,
    pub row: views::Row,
}

impl Router {
    pub fn new(engine: Arc<Engine>) -> Self {
        let tables = Tables::new(Arc::clone(&engine));
        let vectors_default = Arc::new(VectorIndex::open(Arc::clone(&engine)));
        Self { engine, tables, vectors_default, vectors_ns: RwLock::new(HashMap::new()) }
    }

    pub fn tables(&self) -> &Tables {
        &self.tables
    }
    pub fn vectors(&self) -> Arc<VectorIndex> {
        Arc::clone(&self.vectors_default)
    }
    /// Return (or create) the VectorIndex for a specific namespace.
    /// Each namespace gets its own HNSW graph and dim, so different
    /// namespaces can hold vectors of different dimensionalities.
    pub fn vectors_ns(&self, name: &str) -> Arc<VectorIndex> {
        let mut ns = self.vectors_ns.write().unwrap();
        ns.entry(name.to_string())
            .or_insert_with(|| Arc::new(VectorIndex::open_ns(Arc::clone(&self.engine), name.to_string())))
            .clone()
    }

    /// Enumerate all open namespaces as `(name, element_count)` pairs.
    /// The default (non-namespaced) index only appears once it has been
    /// opened; namespaces created via `vectors_ns` are always listed.
    pub fn namespaces(&self) -> Vec<(String, usize)> {
        let ns = self.vectors_ns.read().unwrap();
        let mut out: Vec<(String, usize)> = ns
            .iter()
            .map(|(k, v)| (k.clone(), v.len()))
            .collect();
        out.sort_by(|a, b| a.0.cmp(&b.0));
        out
    }

    /// Push every open namespace (and the default index) to the device.
    /// Called when compute mode switches to a GPU mode, so indexes that were
    /// opened while `CpuOnly` — e.g. restored from WAL at startup — get a
    /// GPU copy without a rebuild.
    pub fn upload_all_to_gpu(&self) {
        // Materialize every PERSISTED namespace before uploading. After a
        // restart the ns map is RAM-empty (graphs are lazily rebuilt), so an
        // immediate GPU.MODE used to upload nothing useful while payloads sat
        // recoverable on disk — first searches then silently ran CPU-side.
        // Scan `vec:` keys, derive distinct namespace names (default and the
        // reserved __ltm__ memory namespace excluded), and force-open each.
        let mut names: Vec<String> = Vec::new();
        let snap = self.engine.snapshot();
        for (key, _) in self.engine.scan_prefix(b"vec:", snap) {
            if key.len() <= 5 {
                continue;
            }
            let rest = &key[4..];
            // default-index keys are exactly "vec:" + 8-byte id
            if rest.len() == 8 {
                continue;
            }
            if let Some(pos) = rest.iter().rposition(|&c| c == b':') {
                let ns = &rest[..pos];
                if ns.is_empty() || ns == b"__ltm__" {
                    continue;
                }
                if let Ok(s) = std::str::from_utf8(ns) {
                    names.push(s.to_string());
                }
            }
        }
        names.sort();
        names.dedup();
        for name in &names {
            self.vectors_ns(name);
        }
        self.vectors_default.upload_to_gpu();
        let ns = self.vectors_ns.read().unwrap();
        for v in ns.values() {
            v.upload_to_gpu();
        }
    }

    /// Full FLUSHALL: wipe durable engine state (WAL + snapshot backed up by
    /// atomic rename first) and every in-RAM vector graph. Tables, timeseries
    /// and agent memory are pure engine views over the same keyspace, so the
    /// engine wipe covers them. Returns the WAL backup path for reporting.
    pub fn flush_all_with_backup(&self) -> std::io::Result<String> {
        let backup = self.engine.flushall_with_backup()?;
        self.vectors_default.reset_ram();
        {
            let mut ns = self.vectors_ns.write().unwrap();
            for v in ns.values() {
                v.reset_ram();
            }
            // Drop the namespace entries entirely: the next VSEARCHNS/VADDNS on
            // a name recreates an empty index via `vectors_ns`, which now scans
            // an empty keyspace.
            ns.clear();
        }
        Ok(backup)
    }

    /// Cost model: choose a plan from estimated selectivity (fraction passing filter).
    /// Very selective predicate => pre-filter is cheaper (few candidates to rank).
    /// Loose predicate => let the ANN index prune, then post-filter.
    pub fn plan_ann(&self, selectivity: f32) -> AnnPlan {
        if selectivity <= 0.05 {
            AnnPlan::PreFilter
        } else {
            AnnPlan::PostFilter
        }
    }

    /// RAG-as-a-query: "k most similar rows in `table` matching col == val,
    /// joined against the row", executed under ONE plan and ONE snapshot.
    ///
    /// `id_of` maps a table pk to the vector id (they share the id space here).
    pub fn rag_search(
        &self,
        table: &str,
        filter_col: &str,
        filter_val: &[u8],
        query: &[f32],
        k: usize,
    ) -> Vec<RagHit> {
        // Estimate selectivity from a cheap scan (real planner uses histograms).
        let total = self.tables.scan(table).len().max(1);
        let matching = self.tables.filter_eq(table, filter_col, filter_val);
        let selectivity = matching.len() as f32 / total as f32;

        match self.plan_ann(selectivity) {
            AnnPlan::PreFilter => {
                // Rank only the matching rows by vector distance.
                let mut hits: Vec<RagHit> = matching
                    .into_iter()
                    .filter_map(|(pk, row)| {
                        let id: u64 = pk.parse().ok()?;
                        let v = self.vectors().get_vector(id)?;
                        let d = cosine_dist(query, &v);
                        Some(RagHit { id, distance: d, row })
                    })
                    .collect();
                hits.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
                hits.truncate(k);
                hits
            }
            AnnPlan::PostFilter => {
                // Ask the ANN index for candidates whose pk passes the filter.
                let allowed: std::collections::HashSet<u64> = matching
                    .iter()
                    .filter_map(|(pk, _)| pk.parse::<u64>().ok())
                    .collect();
                self.vectors()
                    .search_filtered(query, k, |id| allowed.contains(&id))
                    .into_iter()
                    .filter_map(|(id, d)| {
                        let row = self.tables.get(table, &id.to_string())?;
                        Some(RagHit { id, distance: d, row })
                    })
                    .collect()
            }
        }
    }

    pub fn engine(&self) -> &Arc<Engine> {
        &self.engine
    }
}

fn cosine_dist(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        2.0
    } else {
        1.0 - dot / (na * nb)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn eng() -> Arc<Engine> {
        let dir = std::env::temp_dir().join(format!("dbstrike_router_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        use std::time::{SystemTime, UNIX_EPOCH};
        let n = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        Engine::open(dir.join(format!("rt_{n}.wal"))).unwrap()
    }

    #[test]
    fn planner_switches_on_selectivity() {
        let r = Router::new(eng());
        assert_eq!(r.plan_ann(0.01), AnnPlan::PreFilter);
        assert_eq!(r.plan_ann(0.5), AnnPlan::PostFilter);
    }

    #[test]
    fn rag_join_end_to_end() {
        let r = Router::new(eng());
        // 3 tickets; only "open" ones should come back, ranked by similarity.
        let mk = |pk: &str, status: &str| {
            let mut row = views::Row::new();
            row.insert("status".into(), status.as_bytes().to_vec());
            r.tables().upsert("tickets", pk, row).unwrap();
        };
        mk("1", "open");
        mk("2", "closed");
        mk("3", "open");
        r.vectors().insert(1, vec![1.0, 0.0]).unwrap();
        r.vectors().insert(2, vec![0.95, 0.05]).unwrap(); // closest but closed
        r.vectors().insert(3, vec![0.8, 0.2]).unwrap();

        let hits = r.rag_search("tickets", "status", b"open", &[1.0, 0.0], 5);
        // ticket 2 excluded despite being closest; 1 and 3 returned
        assert!(hits.iter().all(|h| h.id != 2));
        assert_eq!(hits[0].id, 1);
    }
}
