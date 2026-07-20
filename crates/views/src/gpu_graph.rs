//! GPU-native graph construction via CAGRA approach.
//!
//! Instead of building HNSW segments (which copy all_f32 + all_i8 per shard),
//! this builds a flat CSR graph directly on GPU:
//!   1. Upload INT8 vectors to GPU once
//!   2. Compute kNN graph on GPU (batch distance, serialized)
//!   3. Merge kNN graphs into flat CSR
//!   4. No HNSW segments = no data copies = minimal RAM
//!
//! Memory: only N×dim bytes (INT8 vectors) on GPU + N×k×4 bytes (kNN graph).

use std::sync::Arc;

/// Build a kNN graph on GPU and return as flat CSR adjacency list.
/// Returns: Vec<Vec<usize>> where graph[i] = list of k nearest neighbors of node i.
///
/// Memory: N vectors uploaded once to GPU. kNN output is N×k integers.
/// No HNSW segments, no all_f32 copies.
pub fn gpu_build_flat_knn(
    data: &[f32],
    perm: &[usize],
    dim: usize,
    n: usize,
    k: usize,
) -> Vec<Vec<usize>> {
    // Upload all vectors as INT8 to GPU once
    let mut vectors_i8: Vec<i8> = Vec::with_capacity(n * dim);
    for i in 0..n {
        let true_row = perm[i];
        for d in 0..dim {
            vectors_i8.push((data[true_row * dim + d] * 127.0) as i8);
        }
    }

    // GPU: compute kNN graph (batch distance on GPU, serialized)
    eprintln!("[GPU] Building flat kNN graph on GPU: {n} vectors, dim={dim}, k={k}");
    match gpu::gpu_build_knn_graph(&vectors_i8, n, dim, k) {
        Some(knn) => {
            eprintln!("[GPU] kNN graph built on GPU: {n} × {k}");
            knn
        }
        None => {
            eprintln!("[GPU] GPU kNN failed, falling back to CPU brute-force");
            cpu_build_flat_knn(&vectors_i8, n, dim, k)
        }
    }
}

/// CPU fallback: brute-force kNN for small datasets.
fn cpu_build_flat_knn(vectors: &[i8], n: usize, dim: usize, k: usize) -> Vec<Vec<usize>> {
    let mut graph = vec![vec![0usize; k]; n];
    for i in 0..n {
        let mut dists: Vec<(i32, usize)> = (0..n).filter(|&j| j != i).map(|j| {
            let mut dot = 0i32;
            for d in 0..dim {
                dot += vectors[i * dim + d] as i32 * vectors[j * dim + d] as i32;
            }
            (dot, j)
        }).collect();
        dists.select_nth_unstable_by(k, |a, b| b.0.cmp(&a.0));
        dists.truncate(k);
        graph[i] = dists.into_iter().map(|(_, j)| j).collect();
    }
    graph
}

/// Merge kNN graphs from all shards into one flat CSR graph.
/// Uses the perm array to remap shard-local indices to global indices.
pub fn merge_flat_knn_graphs(
    shard_graphs: Vec<Vec<Vec<usize>>>,
    perm: &[usize],
    ranges: &[(usize, usize)],
    n: usize,
    k: usize,
) -> Vec<Vec<usize>> {
    let mut global_graph = vec![vec![0usize; k]; n];
    for (si, (lo, hi)) in ranges.iter().enumerate() {
        let shard = &shard_graphs[si];
        for local_i in 0..shard.len() {
            let global_i = perm[lo + local_i];
            // Remap shard-local neighbor indices to global indices
            global_graph[global_i] = shard[local_i].iter()
                .map(|&local_j| perm[lo + local_j])
                .collect();
            // Pad with self-loops if fewer than k neighbors
            while global_graph[global_i].len() < k {
                global_graph[global_i].push(global_i);
            }
        }
    }
    global_graph
}

/// Build flat CSR graph (N × k × i32) from merged kNN graph.
pub fn flat_knn_to_csr(graph: &[Vec<usize>], n: usize, k: usize) -> Vec<i32> {
    let mut csr = vec![-1i32; n * k];
    for (i, neighbors) in graph.iter().enumerate() {
        for (j, &nbr) in neighbors.iter().enumerate().take(k) {
            csr[i * k + j] = nbr as i32;
        }
        // Pad with self-loops
        for j in neighbors.len()..k {
            csr[i * k + j] = i as i32;
        }
    }
    csr
}
