//! Distribution / consensus primitives — pure Rust.
//!
//! Two building blocks the architecture calls for:
//!   * Hybrid Logical Clock (HLC): causal ordering across regions without
//!     atomic-clock hardware (Spanner-free causal consistency).
//!   * CRDTs: conflict-free convergence for counters/presence/session state,
//!     the classic Redis-territory data given up to gain coordination-free
//!     active-active writes.
//!
//! Strong per-key consensus (Raft) is a larger effort; the HLC + CRDT pair here
//! is the tunable-consistency substrate those ledgers would build on.

pub mod crdt;
pub mod hlc;

pub use crdt::{GCounter, LwwRegister, PnCounter};
pub use hlc::Hlc;
