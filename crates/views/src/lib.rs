//! Views — relational tables, KV, vectors, and time-series, all expressed as
//! key conventions over the single storage substrate. No view owns its own
//! engine; they share one MVCC + WAL substrate.

pub mod kv;
pub mod table;
pub mod timeseries;
pub mod vector;

pub use kv::Kv;
pub use table::{Row, Tables};
pub use timeseries::TimeSeries;
pub use vector::VectorIndex;
