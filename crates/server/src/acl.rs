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
//! Password hashing: 4096-round iterated SHA-256 with a per-user 16-byte
//! salt from /dev/urandom (zero external crates), compared in constant time.

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

/// 16 random salt bytes from the kernel CSPRNG. The old generator was an
/// xorshift seeded from the wall clock, so salts were predictable and two users
/// created in the same nanosecond shared one. `/dev/urandom` is std-only; the
/// fallback (no /dev/urandom, e.g. a locked-down sandbox) still mixes clock,
/// pid and a stack address through SHA-256 so salts never collide in practice.
fn generate_salt() -> [u8; 16] {
    let mut salt = [0u8; 16];
    if let Ok(mut f) = std::fs::File::open("/dev/urandom") {
        use std::io::Read;
        if f.read_exact(&mut salt).is_ok() {
            return salt;
        }
    }
    static CTR: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let local = 0u8;
    let mut seed = Vec::new();
    seed.extend_from_slice(&now.to_le_bytes());
    seed.extend_from_slice(&std::process::id().to_le_bytes());
    seed.extend_from_slice(&(&local as *const u8 as usize).to_le_bytes());
    seed.extend_from_slice(&CTR.fetch_add(1, Ordering::Relaxed).to_le_bytes());
    salt.copy_from_slice(&sha256(&seed)[..16]);
    salt
}

/// Key-stretching rounds. A single SHA-256 made an offline guess as cheap as
/// one hash; 4096 rounds costs a few hundred µs per AUTH (connection setup,
/// not the command hot path) and multiplies brute-force cost accordingly.
const HASH_ROUNDS: usize = 4096;

fn hash_password(password: &str, salt: &[u8; 16]) -> [u8; 32] {
    let mut data = Vec::with_capacity(16 + password.len());
    data.extend_from_slice(salt);
    data.extend_from_slice(password.as_bytes());
    let mut h = sha256(&data);
    let mut buf = [0u8; 48];
    for _ in 1..HASH_ROUNDS {
        buf[..32].copy_from_slice(&h);
        buf[32..].copy_from_slice(salt);
        h = sha256(&buf);
    }
    h
}

/// Constant-time equality: `==` on the arrays short-circuits on the first
/// differing byte, leaking how much of the hash matched through timing.
fn ct_eq(a: &[u8; 32], b: &[u8; 32]) -> bool {
    let mut diff = 0u8;
    for i in 0..32 {
        diff |= a[i] ^ b[i];
    }
    diff == 0
}

fn verify_password(password: &str, salt: &[u8; 16], hash: &[u8; 32]) -> bool {
    ct_eq(&hash_password(password, salt), hash)
}

/// Redis-style glob match (`*`, `?`, `[abc]`, `[a-z]`, `[^x]`, `\x`) over
/// raw bytes. Shared by ACL key patterns and the `KEYS` command.
pub fn glob_match(pat: &[u8], s: &[u8]) -> bool {
    let (mut p, mut i) = (0usize, 0usize);
    // Backtrack point for the most recent `*`: (pattern idx after *, string idx).
    let mut star: Option<(usize, usize)> = None;
    while i < s.len() {
        if p < pat.len() {
            match pat[p] {
                b'*' => {
                    while p < pat.len() && pat[p] == b'*' {
                        p += 1;
                    }
                    if p == pat.len() {
                        return true;
                    }
                    star = Some((p, i));
                    continue;
                }
                b'?' => {
                    p += 1;
                    i += 1;
                    continue;
                }
                b'[' => {
                    if let Some((matched, next)) = class_match(pat, p, s[i]) {
                        if matched {
                            p = next;
                            i += 1;
                            continue;
                        }
                    } else if s[i] == b'[' {
                        // Unterminated class: treat `[` literally.
                        p += 1;
                        i += 1;
                        continue;
                    }
                }
                b'\\' if p + 1 < pat.len() => {
                    if pat[p + 1] == s[i] {
                        p += 2;
                        i += 1;
                        continue;
                    }
                }
                c => {
                    if c == s[i] {
                        p += 1;
                        i += 1;
                        continue;
                    }
                }
            }
        }
        match star {
            Some((sp, si)) => {
                p = sp;
                i = si + 1;
                star = Some((sp, si + 1));
            }
            None => return false,
        }
    }
    while p < pat.len() && pat[p] == b'*' {
        p += 1;
    }
    p == pat.len()
}

/// Match `c` against the `[...]` class starting at `pat[start]`. Returns
/// `(matched, index after ])`, or `None` if the class is unterminated.
fn class_match(pat: &[u8], start: usize, c: u8) -> Option<(bool, usize)> {
    let mut j = start + 1;
    let negate = j < pat.len() && pat[j] == b'^';
    if negate {
        j += 1;
    }
    let mut matched = false;
    let mut first = true;
    while j < pat.len() && (pat[j] != b']' || first) {
        first = false;
        if pat[j] == b'\\' && j + 1 < pat.len() {
            matched |= pat[j + 1] == c;
            j += 2;
        } else if j + 2 < pat.len() && pat[j + 1] == b'-' && pat[j + 2] != b']' {
            let (lo, hi) = if pat[j] <= pat[j + 2] { (pat[j], pat[j + 2]) } else { (pat[j + 2], pat[j]) };
            matched |= c >= lo && c <= hi;
            j += 3;
        } else {
            matched |= pat[j] == c;
            j += 1;
        }
    }
    if j >= pat.len() {
        return None;
    }
    Some((matched != negate, j + 1))
}

// ═══════════════════════════════════════════════════════════════════════════
// User and ACL Store
// ═══════════════════════════════════════════════════════════════════════════

/// Permission categories (Redis-compatible names).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PermCategory {
    All,        // +@all — all commands
    Read,       // +@read — commands that only read data
    Write,      // +@write — commands that modify data
    Admin,      // +@admin — server management
    Dangerous,  // +@dangerous — FLUSHALL, CONFIG, file access, ...
    PubSub,     // +@pubsub
    Vector,     // +@vector — vector index commands
    TimeSeries, // +@timeseries
    Memory,     // +@memory — agent memory + RAG
    Table,      // +@table
    Reduce,     // +@reduce — reducer VM
    Connection, // +@connection — PING/ECHO/HELLO/CLIENT/...
}

impl PermCategory {
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "all" | "*" => Some(PermCategory::All),
            "read" => Some(PermCategory::Read),
            "write" => Some(PermCategory::Write),
            "admin" => Some(PermCategory::Admin),
            "dangerous" => Some(PermCategory::Dangerous),
            "pubsub" => Some(PermCategory::PubSub),
            "vector" => Some(PermCategory::Vector),
            "timeseries" => Some(PermCategory::TimeSeries),
            "memory" | "rag" => Some(PermCategory::Memory),
            "table" => Some(PermCategory::Table),
            "reduce" => Some(PermCategory::Reduce),
            "connection" => Some(PermCategory::Connection),
            _ => None,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            PermCategory::All => "all",
            PermCategory::Read => "read",
            PermCategory::Write => "write",
            PermCategory::Admin => "admin",
            PermCategory::Dangerous => "dangerous",
            PermCategory::PubSub => "pubsub",
            PermCategory::Vector => "vector",
            PermCategory::TimeSeries => "timeseries",
            PermCategory::Memory => "memory",
            PermCategory::Table => "table",
            PermCategory::Reduce => "reduce",
            PermCategory::Connection => "connection",
        }
    }
}

/// A single user in the ACL store.
#[derive(Clone, Debug)]
pub struct User {
    pub name: String,
    /// `None` = `nopass` (any password authenticates).
    pub password: Option<([u8; 16], [u8; 32])>,
    /// Admin switch (`on`/`off`). A disabled user cannot authenticate and an
    /// already-authenticated connection loses every permission.
    pub enabled: bool,
    /// Rules in the order given; the LAST matching rule wins (Redis
    /// semantics), so `+@all -flushall` and `-@all +get` both do what they say.
    pub rules: Vec<Rule>,
    /// Key patterns this user may touch. Empty = no keys.
    pub key_patterns: Vec<Vec<u8>>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Rule {
    AllowCmd(String),
    DenyCmd(String),
    AllowCat(PermCategory),
    DenyCat(PermCategory),
}

impl User {
    fn new_restricted(name: &str) -> Self {
        // Redis: a user created by ACL SETUSER starts OFF with no password,
        // no commands and no keys — permissions must be granted explicitly.
        User { name: name.to_string(), password: Some(([0; 16], [0xff; 32])), enabled: false, rules: Vec::new(), key_patterns: Vec::new() }
    }

    fn superuser(name: &str) -> Self {
        User {
            name: name.to_string(),
            password: None,
            enabled: true,
            rules: vec![Rule::AllowCat(PermCategory::All)],
            key_patterns: vec![b"*".to_vec()],
        }
    }

    /// May this user run `cmd` (upper-case), whose categories are `cats`?
    pub fn can_command(&self, cmd: &str, cats: &[PermCategory]) -> bool {
        if !self.enabled {
            return false;
        }
        // Connection-level commands every authenticated client needs.
        if matches!(cmd, "AUTH" | "HELLO" | "QUIT" | "PING" | "RESET") {
            return true;
        }
        let mut allowed = false;
        for r in &self.rules {
            match r {
                Rule::AllowCmd(c) if c == cmd => allowed = true,
                Rule::DenyCmd(c) if c == cmd => allowed = false,
                Rule::AllowCat(c) if *c == PermCategory::All || cats.contains(c) => allowed = true,
                Rule::DenyCat(c) if *c == PermCategory::All || cats.contains(c) => allowed = false,
                _ => {}
            }
        }
        allowed
    }

    pub fn can_key(&self, key: &[u8]) -> bool {
        self.key_patterns.iter().any(|p| glob_match(p, key))
    }

    fn describe(&self) -> String {
        let mut out = vec![if self.enabled { "on".to_string() } else { "off".to_string() }];
        if self.password.is_none() {
            out.push("nopass".into());
        }
        for p in &self.key_patterns {
            out.push(format!("~{}", String::from_utf8_lossy(p)));
        }
        for r in &self.rules {
            out.push(match r {
                Rule::AllowCmd(c) => format!("+{}", c.to_lowercase()),
                Rule::DenyCmd(c) => format!("-{}", c.to_lowercase()),
                Rule::AllowCat(c) => format!("+@{}", c.name()),
                Rule::DenyCat(c) => format!("-@{}", c.name()),
            });
        }
        out.join(" ")
    }
}

/// The ACL store — manages users and authentication.
pub struct AclStore {
    users: RwLock<HashMap<String, User>>,
    /// The server requirepass (`DBSTRIKE_PASS`).
    requirepass: bool,
    /// When true, the per-command permission gate must run on the RESP hot
    /// path. When false it is provably a no-op (no requirepass and no user
    /// other than the unrestricted default exists), so the dispatch loop skips
    /// it and pays one relaxed load. Latched true by any ACL mutation and never
    /// cleared, so enforcement can only become stricter.
    strict: AtomicBool,
}

impl AclStore {
    /// Create a store with the `default` superuser. With `requirepass` set,
    /// `default` requires that password; otherwise it is `nopass`.
    pub fn new(requirepass: Option<String>) -> Arc<Self> {
        let mut users = HashMap::new();
        let mut default = User::superuser("default");
        if let Some(ref pw) = requirepass {
            let salt = generate_salt();
            default.password = Some((salt, hash_password(pw, &salt)));
        }
        users.insert("default".to_string(), default);
        let strict = requirepass.is_some();
        Arc::new(Self {
            users: RwLock::new(users),
            requirepass: requirepass.is_some(),
            strict: AtomicBool::new(strict),
        })
    }

    pub fn needs_permission_check(&self) -> bool {
        self.strict.load(Ordering::Relaxed)
    }

    pub fn latch_strict(&self) {
        self.strict.store(true, Ordering::Relaxed);
    }

    /// `AUTH password` — authenticate as `default`.
    pub fn auth_default(&self, password: &str) -> bool {
        self.auth_user("default", password)
    }

    /// `AUTH username password`. Fails for unknown or disabled users. (The
    /// old code authenticated disabled users and then RE-ENABLED them, so an
    /// admin's `ACL SETUSER bob off` was undone by bob's next login.)
    pub fn auth_user(&self, username: &str, password: &str) -> bool {
        let users = self.users.read().unwrap();
        match users.get(username) {
            Some(u) if u.enabled => match &u.password {
                None => true,
                Some((salt, hash)) => verify_password(password, salt, hash),
            },
            _ => false,
        }
    }

    pub fn can_command(&self, username: &str, cmd: &str, cats: &[PermCategory]) -> bool {
        let users = self.users.read().unwrap();
        users.get(username).is_some_and(|u| u.can_command(cmd, cats))
    }

    pub fn can_keys<'a>(&self, username: &str, mut keys: impl Iterator<Item = &'a [u8]>) -> bool {
        let users = self.users.read().unwrap();
        match users.get(username) {
            Some(u) => keys.all(|k| u.can_key(k)),
            None => false,
        }
    }

    pub fn requires_auth(&self) -> bool {
        self.requirepass
    }

    /// Apply `ACL SETUSER` rule tokens to `username`, creating the user
    /// (OFF, no permissions) if needed. All tokens are validated before any
    /// change is made, so a typo can't leave a half-applied rule set.
    pub fn set_user(&self, username: &str, tokens: &[String]) -> Result<(), String> {
        let mut users = self.users.write().unwrap();
        let mut u = users.get(username).cloned().unwrap_or_else(|| User::new_restricted(username));
        for t in tokens {
            let lower = t.to_ascii_lowercase();
            match lower.as_str() {
                "on" => u.enabled = true,
                "off" => u.enabled = false,
                "nopass" => u.password = None,
                "resetpass" => u.password = Some(([0; 16], [0xff; 32])),
                "allkeys" => u.key_patterns = vec![b"*".to_vec()],
                "resetkeys" => u.key_patterns.clear(),
                "allcommands" => u.rules = vec![Rule::AllowCat(PermCategory::All)],
                "nocommands" => u.rules.clear(),
                "reset" => u = User::new_restricted(username),
                _ => {
                    if let Some(raw) = t.strip_prefix('#') {
                        // Internal persisted form `#<salt hex>:<hash hex>` (ACL SAVE).
                        u.password = Some(parse_hashed(raw).ok_or_else(|| format!("Error in ACL SETUSER modifier '{t}': bad hash"))?);
                    } else if let Some(pw) = t.strip_prefix('>') {
                        let salt = generate_salt();
                        u.password = Some((salt, hash_password(pw, &salt)));
                    } else if let Some(pat) = t.strip_prefix('~') {
                        u.key_patterns.push(pat.as_bytes().to_vec());
                    } else if let Some(rest) = t.strip_prefix('+') {
                        u.rules.push(parse_rule(rest, true)?);
                    } else if let Some(rest) = t.strip_prefix('-') {
                        u.rules.push(parse_rule(rest, false)?);
                    } else {
                        return Err(format!("Error in ACL SETUSER modifier '{t}': Syntax error"));
                    }
                }
            }
        }
        users.insert(username.to_string(), u);
        self.latch_strict();
        Ok(())
    }

    /// Back-compat helpers used by tests.
    #[cfg(test)]
    pub fn set_user_password(&self, username: &str, password: &str) {
        let _ = self.set_user(username, &[format!(">{password}"), "on".into()]);
    }

    #[cfg(test)]
    pub fn set_user_enabled(&self, username: &str, enabled: bool) -> bool {
        let exists = self.users.read().unwrap().contains_key(username);
        exists && self.set_user(username, &[if enabled { "on" } else { "off" }.to_string()]).is_ok()
    }

    #[cfg(test)]
    pub fn set_user_categories(&self, username: &str, cats: Vec<PermCategory>) -> bool {
        let mut users = self.users.write().unwrap();
        match users.get_mut(username) {
            Some(u) => {
                u.rules = cats.into_iter().map(Rule::AllowCat).collect();
                drop(users);
                self.latch_strict();
                true
            }
            None => false,
        }
    }

    pub fn del_user(&self, username: &str) -> bool {
        if username == "default" {
            return false; // Redis refuses to delete the default user
        }
        let removed = self.users.write().unwrap().remove(username).is_some();
        if removed {
            self.latch_strict();
        }
        removed
    }

    /// `ACL GETUSER` as flat key/value pairs.
    pub fn get_user_info(&self, username: &str) -> Option<Vec<(String, String)>> {
        let users = self.users.read().unwrap();
        users.get(username).map(|u| {
            let flags = if u.enabled { "on" } else { "off" };
            let keys: Vec<String> = u.key_patterns.iter().map(|p| format!("~{}", String::from_utf8_lossy(p))).collect();
            let cmds: Vec<String> = u.describe().split(' ').filter(|t| t.starts_with('+') || t.starts_with('-')).map(String::from).collect();
            vec![
                ("flags".into(), if u.password.is_none() { format!("{flags} nopass") } else { flags.to_string() }),
                ("commands".into(), if cmds.is_empty() { "-@all".into() } else { cmds.join(" ") }),
                ("keys".into(), keys.join(" ")),
            ]
        })
    }

    /// Serialize every user as `ACL SETUSER`-style token lines, passwords as
    /// salted hashes (never plaintext). Used by `ACL SAVE`.
    pub fn dump(&self) -> String {
        let users = self.users.read().unwrap();
        let mut lines: Vec<String> = users
            .values()
            .map(|u| {
                let mut toks = vec![u.name.clone(), "reset".into()];
                match &u.password {
                    None => toks.push("nopass".into()),
                    Some((salt, hash)) => toks.push(format!("#{}:{}", hex_encode(salt), hex_encode(hash))),
                }
                toks.extend(u.describe().split(' ').filter(|t| *t != "nopass").map(String::from));
                toks.join(" ")
            })
            .collect();
        lines.sort();
        lines.join("\n")
    }

    /// Replace the user table from a `dump()` (ACL LOAD / startup).
    pub fn load(&self, text: &str) -> Result<usize, String> {
        let staged = AclStore {
            users: RwLock::new(HashMap::new()),
            requirepass: self.requirepass,
            strict: AtomicBool::new(true),
        };
        let mut n = 0;
        for line in text.lines().filter(|l| !l.trim().is_empty()) {
            let mut it = line.split(' ');
            let name = it.next().ok_or("empty ACL line")?;
            let toks: Vec<String> = it.map(String::from).collect();
            staged.set_user(name, &toks)?;
            n += 1;
        }
        let new_users = staged.users.into_inner().unwrap();
        if !new_users.contains_key("default") {
            return Err("ACL dump has no default user".into());
        }
        *self.users.write().unwrap() = new_users;
        self.latch_strict();
        Ok(n)
    }

    /// `ACL LIST` lines.
    pub fn list_users(&self) -> Vec<String> {
        let users = self.users.read().unwrap();
        let mut v: Vec<String> = users.values().map(|u| format!("user {} {}", u.name, u.describe())).collect();
        v.sort();
        v
    }
}

fn hex_encode(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn parse_hashed(raw: &str) -> Option<([u8; 16], [u8; 32])> {
    let (s, h) = raw.split_once(':')?;
    let dec = |x: &str| -> Option<Vec<u8>> {
        (0..x.len()).step_by(2).map(|i| u8::from_str_radix(x.get(i..i + 2)?, 16).ok()).collect()
    };
    Some((dec(s)?.try_into().ok()?, dec(h)?.try_into().ok()?))
}

fn parse_rule(rest: &str, allow: bool) -> Result<Rule, String> {
    if let Some(cat) = rest.strip_prefix('@') {
        let c = PermCategory::from_str(cat).ok_or_else(|| format!("Error in ACL SETUSER modifier '@{cat}': Unknown command category"))?;
        Ok(if allow { Rule::AllowCat(c) } else { Rule::DenyCat(c) })
    } else if rest.is_empty() {
        Err("Error in ACL SETUSER modifier: empty command".into())
    } else {
        let c = rest.to_ascii_uppercase();
        Ok(if allow { Rule::AllowCmd(c) } else { Rule::DenyCmd(c) })
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Command category mapping
// ═══════════════════════════════════════════════════════════════════════════

use PermCategory::*;

/// Categories of a command (upper-case name). Unknown commands are
/// `admin`+`dangerous`: a command added later without a mapping must be
/// DENIED to restricted users, not silently readable (the old default was
/// `Read`, which let a read-only user run GPU.MODE, CHECKPOINT, ...).
pub fn command_categories(cmd: &str) -> &'static [PermCategory] {
    match cmd {
        "PING" | "ECHO" | "QUIT" | "HELLO" | "AUTH" | "CLIENT" | "SELECT" | "RESET" | "TIME" | "COMMAND" => &[Connection],
        "INFO" | "DBSIZE" | "CDCLEN" => &[Admin, Read],
        "GET" | "MGET" | "KEYS" | "EXISTS" | "TYPE" | "TTL" | "PTTL" | "STRLEN" | "GETAT" | "SCAN" | "GETRANGE" => &[Read],
        "SET" | "MSET" | "MSETNX" | "DEL" | "UNLINK" | "INCR" | "INCRBY" | "INCRBYFLOAT" | "DECR" | "DECRBY"
        | "APPEND" | "GETSET" | "GETDEL" | "GETEX" | "SETNX" | "SETEX" | "PSETEX" | "SETRANGE" | "RENAME" | "RENAMENX"
        | "EXPIRE" | "PEXPIRE" | "EXPIREAT" | "PEXPIREAT" | "PERSIST" => &[Write],
        "EXPIRETIME" | "PEXPIRETIME" => &[Read],
        "MULTI" | "EXEC" | "DISCARD" | "WATCH" | "UNWATCH" => &[Connection],
        "HGET" | "HMGET" | "HEXISTS" | "HLEN" | "HGETALL" | "HKEYS" | "HVALS" | "HSTRLEN" | "LLEN" | "LINDEX"
        | "LRANGE" | "LPOS" | "SISMEMBER" | "SMISMEMBER" | "SCARD" | "SMEMBERS" | "SRANDMEMBER" | "SINTER"
        | "SUNION" | "SDIFF" | "ZSCORE" | "ZMSCORE" | "ZCARD" | "ZCOUNT" | "ZRANK" | "ZREVRANK" | "ZRANGE"
        | "ZREVRANGE" | "ZRANGEBYSCORE" | "ZREVRANGEBYSCORE" => &[Read],
        "HSET" | "HSETNX" | "HMSET" | "HDEL" | "HINCRBY" | "HINCRBYFLOAT" | "LPUSH" | "RPUSH" | "LPUSHX" | "RPUSHX"
        | "LPOP" | "RPOP" | "BLPOP" | "BRPOP" | "LSET" | "LTRIM" | "LREM" | "LINSERT" | "RPOPLPUSH" | "LMOVE"
        | "SADD" | "SREM" | "SPOP" | "SINTERSTORE" | "SUNIONSTORE" | "SDIFFSTORE" | "SMOVE" | "ZADD" | "ZREM"
        | "ZINCRBY" | "ZREMRANGEBYSCORE" | "ZREMRANGEBYRANK" | "ZPOPMIN" | "ZPOPMAX" => &[Write],
        "FLUSHALL" | "FLUSHDB" | "CONFIG" | "CHECKPOINT" | "MEMTRACK" | "SHUTDOWN" | "ACL" | "VSNAPSHOT"
        | "GPU.LOAD" | "GPU.UNLOAD" | "GPU.MODE" | "GPU.SWEEP" | "CACHE.CLEAR" | "CACHE.BUGS" | "CACHE.TRACES" => &[Admin, Dangerous],
        "GPU.INFO" | "ROLE" => &[Admin, Read],
        "WAIT" => &[Connection],
        // VBULKLOAD reads an arbitrary server-side file path.
        "VBULKLOAD" | "VBULKLOADNS" => &[Vector, Write, Dangerous],
        "VSEARCH" | "VSEARCHNS" | "VSEARCHA" | "VSEARCHANS" | "VSEARCH.MANY" | "VSEARCH.MANYNS" | "VGETPAYLOAD"
        | "VLISTNS" | "VQUANT" | "VQUANTNS" | "VFACET" | "VRECOMMEND" => &[Vector, Read],
        "VADD" | "VADDNS" | "VDEL" | "VDELNS" | "VADDBATCH" | "VADDBATCHNS" | "VSETQUANT" | "VSETQUANTNS"
        | "VFITQUANT" | "VFITQUANTNS" | "VCALIBRATE" | "VSETPAYLOAD" | "VDELPAYLOAD" => &[Vector, Write],
        "TSRANGE" | "TSRANGE.LATEST" | "TSLATEST" | "TSAVG" => &[TimeSeries, Read],
        "TSADD" | "TSADD.F" => &[TimeSeries, Write],
        "SUBSCRIBE" | "PSUBSCRIBE" | "UNSUBSCRIBE" | "PUNSUBSCRIBE" | "PUBLISH" | "PUBSUB" => &[PubSub],
        "REDUCE" | "REDUCE.PROGRAM" => &[Reduce, Write],
        "TABLE.GET" | "TABLE.SCAN" | "TABLE.FILTEREQ" => &[Table, Read],
        "TABLE.SET" | "TABLE.DEL" => &[Table, Write],
        "MEM.RECALL" | "MEM.RECALL.AS_OF" | "MEM.NEIGH" | "MEM.TRAV" | "MEM.COUNT" | "MEM.GET" | "MEM.PROC.GET"
        | "MEM.PROC.LIST" | "MEM.WM_GET" | "MEM.EPISODES" | "MEM.INCOMING" | "RAG.SEARCH" | "RAG.CONTEXT" => &[Memory, Read],
        "MEM.REMEMBER" | "MEM.REMEMBER.T" | "MEM.FORGET" | "MEM.LINK" | "MEM.UNLINK" | "MEM.CONSOLIDATE"
        | "MEM.EPISODES_CLEAR" | "MEM.PROC.SET" | "MEM.INVALIDATE" | "MEM.WM_SET" | "MEM.WM_DELETE" | "MEM.EPISODE"
        | "MEM.EPISODE_FORGET" | "RAG.INGEST" => &[Memory, Write],
        "CRDT.GET" | "HLC.NOW" | "CACHE.GET" => &[Read],
        "CRDT.GCOUNTER" | "CRDT.PNCOUNTER" | "CRDT.LWW" | "HLC.UPDATE" | "CACHE.SET" | "CACHE.SRCSET"
        | "CACHE.SRCDEL" | "CACHE.INVALIDATE" => &[Write],
        _ => &[Admin, Dangerous],
    }
}

/// Positions of key arguments for the KV commands (args exclude the command
/// name), used to enforce `~pattern` key permissions. Non-KV commands address
/// their own namespaces and are governed by command/category rules only.
pub fn command_keys<'a>(cmd: &str, args: &'a [Vec<u8>]) -> Vec<&'a [u8]> {
    match cmd {
        "GET" | "SET" | "INCR" | "INCRBY" | "DECR" | "DECRBY" | "APPEND" | "STRLEN" | "TYPE" | "TTL" | "PTTL"
        | "EXPIRE" | "PEXPIRE" | "PERSIST" | "GETSET" | "GETDEL" | "SETNX" | "GETRANGE" => {
            args.first().map(|a| vec![a.as_slice()]).unwrap_or_default()
        }
        "DEL" | "UNLINK" | "MGET" | "EXISTS" | "WATCH" | "SINTER" | "SUNION" | "SDIFF" | "SINTERSTORE"
        | "SUNIONSTORE" | "SDIFFSTORE" => args.iter().map(|a| a.as_slice()).collect(),
        "MSET" | "MSETNX" => args.iter().step_by(2).map(|a| a.as_slice()).collect(),
        "RENAME" | "RENAMENX" | "SMOVE" | "RPOPLPUSH" | "LMOVE" => args.iter().take(2).map(|a| a.as_slice()).collect(),
        "BLPOP" | "BRPOP" => args.iter().take(args.len().saturating_sub(1)).map(|a| a.as_slice()).collect(),
        "SETEX" | "PSETEX" | "GETEX" | "SETRANGE" | "INCRBYFLOAT" | "EXPIREAT" | "PEXPIREAT" | "EXPIRETIME"
        | "PEXPIRETIME" => args.first().map(|a| vec![a.as_slice()]).unwrap_or_default(),
        // Every hash/list/set/zset command takes its key first.
        c if crate::keyspace::is_keyspace_cmd(c) && !matches!(c, "PING" | "ECHO" | "TIME") => {
            args.first().map(|a| vec![a.as_slice()]).unwrap_or_default()
        }
        _ => Vec::new(),
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
        assert_ne!(generate_salt(), generate_salt());
    }

    #[test]
    fn acl_store_default_no_password() {
        let store = AclStore::new(None);
        assert!(!store.requires_auth());
        assert!(store.can_command("default", "GET", command_categories("GET")));
        assert!(store.can_command("default", "FLUSHALL", command_categories("FLUSHALL")));
    }

    #[test]
    fn acl_store_with_password() {
        let store = AclStore::new(Some("mypass".to_string()));
        assert!(store.requires_auth());
        assert!(store.auth_default("mypass"));
        assert!(!store.auth_default("nope"));
    }

    #[test]
    fn per_command_rules_are_enforced() {
        let store = AclStore::new(None);
        store.set_user("bob", &["on".into(), ">pw".into(), "+GET".into(), "~*".into()]).unwrap();
        assert!(store.auth_user("bob", "pw"));
        let can = |c: &str| store.can_command("bob", c, command_categories(c));
        assert!(can("GET"));
        for c in ["SET", "MSET", "FLUSHALL", "ACL", "GPU.MODE", "CHECKPOINT", "SOMETHINGNEW"] {
            assert!(!can(c), "{c} must be denied");
        }
    }

    #[test]
    fn category_rules_last_match_wins() {
        let store = AclStore::new(None);
        store.set_user("ops", &["on".into(), "nopass".into(), "+@all".into(), "-FLUSHALL".into()]).unwrap();
        assert!(store.can_command("ops", "SET", command_categories("SET")));
        assert!(!store.can_command("ops", "FLUSHALL", command_categories("FLUSHALL")));
        store.set_user("ro", &["on".into(), "nopass".into(), "+@read".into(), "-@all".into(), "+@read".into()]).unwrap();
        assert!(store.can_command("ro", "GET", command_categories("GET")));
        assert!(!store.can_command("ro", "SET", command_categories("SET")));
    }

    #[test]
    fn new_user_has_nothing_and_disabled_user_cannot_auth() {
        let store = AclStore::new(None);
        store.set_user("eve", &[">p".into()]).unwrap();
        assert!(!store.auth_user("eve", "p"), "new users start off");
        store.set_user("eve", &["on".into()]).unwrap();
        assert!(store.auth_user("eve", "p"));
        assert!(!store.can_command("eve", "GET", command_categories("GET")));
        store.set_user("eve", &["off".into()]).unwrap();
        assert!(!store.auth_user("eve", "p"));
    }

    #[test]
    fn key_patterns() {
        let store = AclStore::new(None);
        store.set_user("k", &["on".into(), "nopass".into(), "+@all".into(), "~user:*".into()]).unwrap();
        assert!(store.can_keys("k", [&b"user:1"[..]].into_iter()));
        assert!(!store.can_keys("k", [&b"user:1"[..], &b"admin"[..]].into_iter()));
    }

    #[test]
    fn glob() {
        assert!(glob_match(b"*", b"anything"));
        assert!(glob_match(b"x*", b"xy"));
        assert!(glob_match(b"*2", b"x2"));
        assert!(glob_match(b"x?", b"xy"));
        assert!(!glob_match(b"x?", b"x"));
        assert!(glob_match(b"h[ae]llo", b"hello"));
        assert!(!glob_match(b"h[^e]llo", b"hello"));
        assert!(glob_match(b"h[a-c]llo", b"hbllo"));
        assert!(glob_match(b"a\\*b", b"a*b"));
        assert!(!glob_match(b"a\\*b", b"axb"));
        assert!(glob_match(b"*a*b*c", b"xxaxxbxxc"));
    }

    #[test]
    fn acl_del_user() {
        let store = AclStore::new(None);
        store.set_user_password("temp", "pass");
        assert!(store.del_user("temp"));
        assert!(!store.del_user("temp"));
        assert!(!store.auth_user("temp", "pass"));
        assert!(!store.del_user("default"));
    }

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{:02x}", b)).collect()
    }
}
