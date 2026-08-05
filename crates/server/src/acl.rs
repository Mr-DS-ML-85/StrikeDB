//! ACL (Access Control List) + AUTH for DB-Strike.
//!
//! Redis-compatible authentication and authorization:
//! - `AUTH [username] password` — authenticate a connection
//! - `ACL SETUSER username [>password] [on|off] [~key_pattern] [+command]` — manage users
//! - `ACL GETUSER username` — get user info
//! - `ACL DELUSER username` — delete a user
//! - `ACL LIST` — list all users
//! - `ACL WHOAMI` — current user
//! - `ACL SAVE` / `ACL LOAD` — persist/restore ACL to engine
//!
//! Password hashing: SHA-256 with per-user random salt (zero external crates).
//! The salt is 16 bytes, stored alongside the hash. Verification is
//! hash(salt + password) == stored_hash.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock};

// ═══════════════════════════════════════════════════════════════════════════
// SHA-256 — pure Rust, zero dependencies
// ═══════════════════════════════════════════════════════════════════════════

const K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

const H0: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

fn sha256_compress(state: &mut [u32; 8], block: &[u8; 64]) {
    let mut w = [0u32; 64];
    for i in 0..16 {
        w[i] = u32::from_be_bytes(block[i * 4..i * 4 + 4].try_into().unwrap());
    }
    for i in 16..64 {
        let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
        let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16]
            .wrapping_add(s0)
            .wrapping_add(w[i - 7])
            .wrapping_add(s1);
    }
    let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = *state;
    for i in 0..64 {
        let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
        let ch = (e & f) ^ ((!e) & g);
        let temp1 = h
            .wrapping_add(s1)
            .wrapping_add(ch)
            .wrapping_add(K[i])
            .wrapping_add(w[i]);
        let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
        let maj = (a & b) ^ (a & c) ^ (b & c);
        let temp2 = s0.wrapping_add(maj);
        h = g;
        g = f;
        f = e;
        e = d.wrapping_add(temp1);
        d = c;
        c = b;
        b = a;
        a = temp1.wrapping_add(temp2);
    }
    state[0] = state[0].wrapping_add(a);
    state[1] = state[1].wrapping_add(b);
    state[2] = state[2].wrapping_add(c);
    state[3] = state[3].wrapping_add(d);
    state[4] = state[4].wrapping_add(e);
    state[5] = state[5].wrapping_add(f);
    state[6] = state[6].wrapping_add(g);
    state[7] = state[7].wrapping_add(h);
}

fn sha256(data: &[u8]) -> [u8; 32] {
    let mut state = H0;
    let mut msg = data.to_vec();
    let bit_len = (msg.len() as u64) * 8;
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bit_len.to_be_bytes());
    for chunk in msg.chunks(64) {
        let mut block = [0u8; 64];
        block.copy_from_slice(chunk);
        sha256_compress(&mut state, &block);
    }
    let mut out = [0u8; 32];
    for (i, &s) in state.iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&s.to_be_bytes());
    }
    out
}

// ═══════════════════════════════════════════════════════════════════════════
// Password hashing: SHA-256(salt || password)
// ═══════════════════════════════════════════════════════════════════════════

/// Deterministic PRNG for salt generation (xorshift64, seeded from timestamp).
struct SaltRng {
    state: u64,
}

impl SaltRng {
    fn new(seed: u64) -> Self {
        Self { state: seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(0xC0FFEE) }
    }

    fn next_bytes(&mut self, buf: &mut [u8]) {
        for chunk in buf.chunks_mut(8) {
            self.state ^= self.state << 13;
            self.state ^= self.state >> 7;
            self.state ^= self.state << 17;
            let val = self.state.to_le_bytes();
            for (b, &v) in chunk.iter_mut().zip(val.iter()) {
                *b = v;
            }
        }
    }
}

fn generate_salt() -> [u8; 16] {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64;
    let mut rng = SaltRng::new(now);
    let mut salt = [0u8; 16];
    rng.next_bytes(&mut salt);
    salt
}

fn hash_password(password: &str, salt: &[u8; 16]) -> [u8; 32] {
    let mut data = Vec::with_capacity(16 + password.len());
    data.extend_from_slice(salt);
    data.extend_from_slice(password.as_bytes());
    sha256(&data)
}

fn verify_password(password: &str, salt: &[u8; 16], hash: &[u8; 32]) -> bool {
    hash_password(password, salt) == *hash
}

// ═══════════════════════════════════════════════════════════════════════════
// User and ACL Store
// ═══════════════════════════════════════════════════════════════════════════

/// Permission categories (Redis-compatible).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PermCategory {
    All,           // +@all — all commands
    Read,          // +@read — read-only commands
    Write,         // +@write — write commands
    Set,           // +@set — KV set commands
    SortedSet,     // +@sorted_set (not used yet)
    Hash,          // +@hash (not used yet)
    List,          // +@list (not used yet)
    Admin,         // +@admin — server management
    Slow,          // +@slow — slow commands
    Dangerous,     // +@dangerous — dangerous commands (FLUSHALL, etc.)
    Scripting,     // +@scripting (not used yet)
    PubSub,        // +@pubsub — pub/sub commands
    Vector,        // +@vector — vector search commands
    TimeSeries,    // +@timeseries — time-series commands
    Memory,        // +@memory — agent memory commands
    Table,         // +@table — table commands
    Reduce,        // +@reduce — reducer commands
}

impl PermCategory {
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "all" | "*" => Some(PermCategory::All),
            "read" => Some(PermCategory::Read),
            "write" => Some(PermCategory::Write),
            "set" => Some(PermCategory::Set),
            "sorted_set" => Some(PermCategory::SortedSet),
            "hash" => Some(PermCategory::Hash),
            "list" => Some(PermCategory::List),
            "admin" => Some(PermCategory::Admin),
            "slow" => Some(PermCategory::Slow),
            "dangerous" => Some(PermCategory::Dangerous),
            "scripting" => Some(PermCategory::Scripting),
            "pubsub" => Some(PermCategory::PubSub),
            "vector" => Some(PermCategory::Vector),
            "timeseries" => Some(PermCategory::TimeSeries),
            "memory" => Some(PermCategory::Memory),
            "table" => Some(PermCategory::Table),
            "reduce" => Some(PermCategory::Reduce),
            _ => None,
        }
    }
}

/// A single user in the ACL store.
#[derive(Clone, Debug)]
pub struct User {
    pub name: String,
    pub password_hash: [u8; 32],
    pub salt: [u8; 16],
    pub enabled: bool,
    /// Categories this user can access. Empty = no access (except AUTH).
    pub categories: Vec<PermCategory>,
    /// Key patterns this user can access (e.g. "user:*", "*"). Empty = all keys.
    pub key_patterns: Vec<String>,
}

impl User {
    /// Check if this user can execute the given command.
    pub fn can_command(&self, cmd: &str, category: PermCategory) -> bool {
        if !self.enabled {
            return false;
        }
        // AUTH is always allowed (otherwise user can never log in).
        if cmd.eq_ignore_ascii_case("AUTH") || cmd.eq_ignore_ascii_case("ACL") {
            return true;
        }
        // "all" category covers everything.
        if self.categories.contains(&PermCategory::All) {
            return true;
        }
        self.categories.contains(&category)
    }
}

/// The ACL store — manages users and authentication.
pub struct AclStore {
    users: RwLock<HashMap<String, User>>,
    /// The server requirepass (if set via env DBSTRIKE_REQUIREPASS).
    requirepass: Option<String>,
    /// When true, the per-command permission gate must run on the RESP hot
    /// path. When false it is provably a no-op (no requirepass, and no user
    /// exists whose categories could deny a command), so the dispatch loop
    /// skips it entirely and pays one relaxed load instead of a lock + scan.
    ///
    /// Set true at construction when auth is required, and latched true by any
    /// mutation that can introduce a restriction (`del_user`,
    /// `set_user_categories`, disabling a user). Never set back to false, so
    /// enforcement can only become MORE strict, never silently weaker.
    strict: AtomicBool,
}

impl AclStore {
    /// Create a new ACL store with a default "default" user.
    /// If `requirepass` is set, the default user requires that password.
    pub fn new(requirepass: Option<String>) -> Arc<Self> {
        let mut users = HashMap::new();

        // Default user: full access if no requirepass; otherwise needs auth.
        let default_enabled = requirepass.is_none();
        let default_salt = generate_salt();
        let default_hash = if let Some(ref pw) = requirepass {
            hash_password(pw, &default_salt)
        } else {
            [0u8; 32]
        };

        users.insert(
            "default".to_string(),
            User {
                name: "default".to_string(),
                password_hash: default_hash,
                salt: default_salt,
                enabled: default_enabled,
                categories: vec![PermCategory::All],
                key_patterns: vec!["*".to_string()],
            },
        );

        let strict = requirepass.is_some();
        Arc::new(Self {
            users: RwLock::new(users),
            requirepass,
            strict: AtomicBool::new(strict),
        })
    }

    /// True if the per-command permission gate must run on this request.
    /// In the default no-auth install this stays false, so the RESP hot path
    /// skips the ACL work entirely (a single relaxed load).
    pub fn needs_permission_check(&self) -> bool {
        self.strict.load(Ordering::Relaxed)
    }

    /// Latch strict mode on. Used by every ACL mutation that can deny a
    /// command the default user could otherwise run, and by AUTH when a named
    /// (possibly restricted) user authenticates.
    pub fn latch_strict(&self) {
        self.strict.store(true, Ordering::Relaxed);
    }

    /// Authenticate a password against the default user (Redis `AUTH password`).
    pub fn auth_default(&self, password: &str) -> bool {
        let users = self.users.read().unwrap();
        if let Some(user) = users.get("default") {
            verify_password(password, &user.salt, &user.password_hash)
        } else {
            false
        }
    }

    /// Authenticate a specific user (Redis `AUTH username password`).
    pub fn auth_user(&self, username: &str, password: &str) -> bool {
        let users = self.users.read().unwrap();
        if let Some(user) = users.get(username) {
            // Auth works regardless of enabled flag.
            verify_password(password, &user.salt, &user.password_hash)
        } else {
            false
        }
    }

    /// Check if a user (by name) can execute a command in the given category.
    pub fn can_command(&self, username: &str, cmd: &str, category: PermCategory) -> bool {
        let users = self.users.read().unwrap();
        if let Some(user) = users.get(username) {
            user.can_command(cmd, category)
        } else {
            false
        }
    }

    /// Check if the server requires authentication.
    pub fn requires_auth(&self) -> bool {
        self.requirepass.is_some()
    }

    /// Set a user's password (hashed). Creates the user if it doesn't exist.
    pub fn set_user_password(&self, username: &str, password: &str) {
        let salt = generate_salt();
        let hash = hash_password(password, &salt);
        let mut users = self.users.write().unwrap();
        if let Some(user) = users.get_mut(username) {
            user.salt = salt;
            user.password_hash = hash;
            user.enabled = true;
        } else {
            users.insert(
                username.to_string(),
                User {
                    name: username.to_string(),
                    password_hash: hash,
                    salt,
                    enabled: true,
                    categories: vec![PermCategory::All],
                    key_patterns: vec!["*".to_string()],
                },
            );
        }
    }

    /// Enable or disable a user.
    pub fn set_user_enabled(&self, username: &str, enabled: bool) -> bool {
        let mut users = self.users.write().unwrap();
        if let Some(user) = users.get_mut(username) {
            user.enabled = enabled;
            if !enabled {
                // A disabled user could otherwise run commands under a stale
                // connection, so the gate must stay armed.
                self.latch_strict();
            }
            true
        } else {
            false
        }
    }

    /// Set a user's categories (replaces existing).
    pub fn set_user_categories(&self, username: &str, cats: Vec<PermCategory>) -> bool {
        let mut users = self.users.write().unwrap();
        if let Some(user) = users.get_mut(username) {
            user.categories = cats;
            // A category list narrower than All can deny commands the default
            // user would run — the gate must stay armed from here on.
            self.latch_strict();
            true
        } else {
            false
        }
    }

    /// Delete a user. Returns true if the user existed.
    pub fn del_user(&self, username: &str) -> bool {
        let mut users = self.users.write().unwrap();
        let removed = users.remove(username).is_some();
        if removed {
            self.latch_strict();
        }
        removed
    }

    /// Get a user's info as a flat string (Redis ACL GETUSER format).
    pub fn get_user_info(&self, username: &str) -> Option<String> {
        let users = self.users.read().unwrap();
        users.get(username).map(|u| {
            let status = if u.enabled { "on" } else { "off" };
            let keys: Vec<&str> = u.key_patterns.iter().map(|s| s.as_str()).collect();
            format!(
                "flags {} channels * commands * ~{} resetchannels {}",
                status,
                keys.join(" ~"),
                if u.categories.is_empty() { "off" } else { "on" }
            )
        })
    }

    /// List all users (Redis ACL LIST format).
    pub fn list_users(&self) -> Vec<String> {
        let users = self.users.read().unwrap();
        users.values().map(|u| {
            let flags = if u.enabled { "on" } else { "off" };
            format!("user {} {} {}", u.name, flags, "reset")
        }).collect()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Command category mapping
// ═══════════════════════════════════════════════════════════════════════════

/// Map a Redis command name to its permission category.
pub fn command_category(cmd: &str) -> PermCategory {
    match cmd {
        // KV read
        "GET" | "MGET" | "KEYS" | "DBSIZE" | "EXISTS" | "TYPE" | "TTL" | "PTTL" => PermCategory::Read,
        // KV write
        "SET" | "MSET" | "DEL" | "INCR" | "INCRBY" | "DECR" | "DECRBY"
        | "APPEND" | "SETNX" | "SETEX" | "PSETEX" | "SETXX" => PermCategory::Write,
        // Admin
        "PING" | "QUIT" | "INFO" | "ECHO" | "TIME" => PermCategory::Admin,
        "CONFIG" | "SELECT" | "FLUSHALL" | "FLUSHDB" | "COMMAND" => PermCategory::Admin,
        "CLIENT" => PermCategory::Admin,
        // Vector
        "VADD" | "VADDBATCH" | "VBULKLOAD" | "VDEL" | "VSEARCH" | "VSEARCHA" | "VSEARCH.MANY"
        | "VSETQUANT" | "VFITQUANT" | "VQUANT" | "VCALIBRATE" => PermCategory::Vector,
        // Time-series
        "TSADD" | "TSADD.F" | "TSRANGE" | "TSRANGE.LATEST" | "TSLATEST" | "TSAVG" => PermCategory::TimeSeries,
        // Pub/sub
        "SUBSCRIBE" | "PSUBSCRIBE" | "UNSUBSCRIBE" | "PUNSUBSCRIBE" | "PUBLISH" => PermCategory::PubSub,
        // Reduce
        "REDUCE" | "REDUCE.PROGRAM" => PermCategory::Reduce,
        // Table
        "TABLE.SET" | "TABLE.GET" | "TABLE.DEL" | "TABLE.SCAN" | "TABLE.FILTEREQ" => PermCategory::Table,
        // Memory
        "MEM.REMEMBER" | "MEM.RECALL" | "MEM.FORGET" | "MEM.LINK" | "MEM.UNLINK"
        | "MEM.NEIGH" | "MEM.TRAV" | "MEM.COUNT" | "MEM.GET" | "MEM.CONSOLIDATE"
        | "MEM.EPISODES_CLEAR" | "MEM.PROC.SET" | "MEM.PROC.GET" | "MEM.PROC.LIST"
        | "MEM.REMEMBER.T" | "MEM.INVALIDATE" | "MEM.RECALL.AS_OF"
        | "MEM.INCOMING" => PermCategory::Memory,
        // CRDT / HLC (consensus)
        "CRDT.GCOUNTER" | "CRDT.PNCOUNTER" | "CRDT.LWW" | "CRDT.GET"
        | "HLC.NOW" | "HLC.UPDATE" => PermCategory::Admin,
        // RAG
        "RAG.INGEST" | "RAG.SEARCH" | "RAG.CONTEXT" => PermCategory::Vector,
        // Checkpoint / cache
        "CHECKPOINT" | "CDCLEN" | "CACHE.*" | "GETAT" | "SCAN" => PermCategory::Admin,
        // ACL / AUTH
        "AUTH" | "ACL" => PermCategory::Admin,
        // Everything else is read by default
        _ => PermCategory::Read,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_basic() {
        let hash = sha256(b"hello");
        // Known SHA-256 of "hello"
        assert_eq!(
            hex(&hash),
            "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
        );
    }

    #[test]
    fn password_hash_verify() {
        let salt = generate_salt();
        let hash = hash_password("secret", &salt);
        assert!(verify_password("secret", &salt, &hash));
        assert!(!verify_password("wrong", &salt, &hash));
    }

    #[test]
    fn acl_store_default_no_password() {
        let store = AclStore::new(None);
        assert!(!store.requires_auth());
        // Default user is enabled without auth.
        assert!(store.can_command("default", "GET", PermCategory::Read));
        assert!(store.can_command("default", "SET", PermCategory::Write));
    }

    #[test]
    fn acl_store_with_password() {
        let store = AclStore::new(Some("mypass".to_string()));
        assert!(store.requires_auth());
        // Default user disabled until auth.
        assert!(!store.can_command("default", "GET", PermCategory::Read));
        // Auth with correct password.
        assert!(store.auth_default("mypass"));
        // Now enable the user.
        store.set_user_enabled("default", true);
        assert!(store.can_command("default", "GET", PermCategory::Read));
    }

    #[test]
    fn acl_user_create_and_auth() {
        let store = AclStore::new(None);
        store.set_user_password("alice", "password123");
        assert!(store.auth_user("alice", "password123"));
        assert!(!store.auth_user("alice", "wrong"));
        assert!(!store.auth_user("bob", "password123"));
    }

    #[test]
    fn acl_user_categories() {
        let store = AclStore::new(None);
        store.set_user_password("reader", "pass");
        store.set_user_categories("reader", vec![PermCategory::Read]);
        assert!(store.can_command("reader", "GET", PermCategory::Read));
        assert!(!store.can_command("reader", "SET", PermCategory::Write));
        // AUTH and ACL always allowed.
        assert!(store.can_command("reader", "AUTH", PermCategory::Admin));
        assert!(store.can_command("reader", "ACL", PermCategory::Admin));
    }

    #[test]
    fn acl_del_user() {
        let store = AclStore::new(None);
        store.set_user_password("temp", "pass");
        assert!(store.del_user("temp"));
        assert!(!store.del_user("temp"));
        assert!(!store.auth_user("temp", "pass"));
    }

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{:02x}", b)).collect()
    }
}
