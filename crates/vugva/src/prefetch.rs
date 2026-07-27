/// Lookahead Prefetcher — predicts upcoming chunks from traversal history.

use std::collections::VecDeque;
use crate::chunk::ChunkId;

/// Tracks recent chunk access patterns and predicts upcoming chunks.
pub struct LookaheadTracker {
    /// Circular buffer of recent chunk accesses
    history: VecDeque<ChunkId>,
    /// Max history size
    history_size: usize,
    /// How many chunks ahead to predict
    window: usize,
}

impl LookaheadTracker {
    pub fn new(history_size: usize, window: usize) -> Self {
        Self {
            history: VecDeque::with_capacity(history_size),
            history_size,
            window,
        }
    }

    /// Record a chunk access.
    pub fn record_access(&mut self, chunk_id: ChunkId) {
        if self.history.len() >= self.history_size {
            self.history.pop_front();
        }
        self.history.push_back(chunk_id);
    }

    /// Predict upcoming chunk IDs based on access pattern.
    /// Simple pattern: look at distance between recent accesses.
    /// Returns chunk IDs that should be prefetched.
    pub fn predict(&self) -> Vec<ChunkId> {
        if self.history.len() < 3 {
            return Vec::new();
        }

        let mut predictions = Vec::new();
        let len = self.history.len();

        // Look at the last few accesses to find a pattern
        // Simple approach: if accesses are sequential (id, id+1, id+2),
        // predict the next few
        if len >= 2 {
            let last = *self.history.back().unwrap();
            let prev = *self.history.get(len - 2).unwrap();

            // Detect sequential pattern (small positive delta)
            if last > prev && (last - prev) < 1000 {
                let step = last - prev;
                for i in 1..=self.window {
                    predictions.push(last + step * i as u64);
                }
            }
            // Detect stride pattern (constant gap)
            else if len >= 3 {
                let prev2 = *self.history.get(len - 3).unwrap();
                if last > prev && prev > prev2 {
                    let step1 = last - prev;
                    let step2 = prev - prev2;
                    if step1 == step2 {
                        for i in 1..=self.window {
                            predictions.push(last + step1 * i as u64);
                        }
                    }
                }
            }
        }

        predictions
    }

    /// Get recent history.
    pub fn history(&self) -> Vec<ChunkId> {
        self.history.iter().copied().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sequential_prediction() {
        let mut tracker = LookaheadTracker::new(64, 4);
        tracker.record_access(100);
        tracker.record_access(101);
        tracker.record_access(102);
        let predictions = tracker.predict();
        assert_eq!(predictions, vec![103, 104, 105, 106]);
    }

    #[test]
    fn test_stride_prediction() {
        let mut tracker = LookaheadTracker::new(64, 3);
        tracker.record_access(100);
        tracker.record_access(110);
        tracker.record_access(120);
        let predictions = tracker.predict();
        assert_eq!(predictions, vec![130, 140, 150]);
    }

    #[test]
    fn test_no_prediction_short_history() {
        let tracker = LookaheadTracker::new(64, 4);
        assert!(tracker.predict().is_empty());
    }
}
