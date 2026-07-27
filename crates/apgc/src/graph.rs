/// Graph utility functions for APGC.

use crate::precision::{Edge, PrecisionLevel};

/// Merge two graphs by interleaving edges from each.
pub fn merge_graphs(g1_edges: &[Vec<Edge>], g2_edges: &[Vec<Edge>], k: usize) -> Vec<Vec<Edge>> {
    let n = g1_edges.len().max(g2_edges.len());
    let mut merged = Vec::with_capacity(n);
    for i in 0..n {
        let mut combined = Vec::new();
        if i < g1_edges.len() {
            combined.extend(g1_edges[i].iter().cloned().take(k / 2));
        }
        if i < g2_edges.len() {
            combined.extend(g2_edges[i].iter().cloned().take(k / 2));
        }
        combined.truncate(k);
        merged.push(combined);
    }
    merged
}

/// Count edges per precision level.
pub fn precision_counts(edges: &[Vec<Edge>]) -> (usize, usize, usize, usize, usize) {
    let mut fp32 = 0;
    let mut fp16 = 0;
    let mut bf16 = 0;
    let mut fp8 = 0;
    let mut int8 = 0;
    for node_edges in edges {
        for e in node_edges {
            match e.precision {
                PrecisionLevel::Fp32 => fp32 += 1,
                PrecisionLevel::Fp16 => fp16 += 1,
                PrecisionLevel::Bf16 => bf16 += 1,
                PrecisionLevel::Fp8 => fp8 += 1,
                PrecisionLevel::Int8 => int8 += 1,
            }
        }
    }
    (fp32, fp16, bf16, fp8, int8)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PrecisionLevel;

    #[test]
    fn test_merge_graphs() {
        let g1 = vec![
            vec![Edge { from: 0, to: 1, distance: 0.5, precision: PrecisionLevel::Fp32 }],
        ];
        let g2 = vec![
            vec![Edge { from: 0, to: 2, distance: 0.3, precision: PrecisionLevel::Fp16 }],
        ];
        let merged = merge_graphs(&g1, &g2, 4);
        assert_eq!(merged[0].len(), 2);
    }
}
