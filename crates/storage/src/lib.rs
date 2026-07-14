//! DB-Strike unified storage substrate.
//!
//! One MVCC + WAL engine. Relational tables, KV, vectors, time-series and the
//! CDC/pub-sub log are all *views* over this single substrate — never separate
//! engines bolted together.

pub mod crc;
pub mod engine;
pub mod value;
pub mod wal;

pub use engine::{Engine, Key, Mutation, Subscriber, Txn, TxnError, Version};
pub use value::Value;
