//! LGM: the tier membrane — query-frequency-driven placement across T0/T1/T2.
//!
//! The pool's original policy was a fixed idle timeout: demote anything not
//! touched for five seconds, promote anything whose lifetime access count
//! crosses a constant. That has two failure modes which matter under real query
//! skew. A page hammered during startup and never touched again keeps a high
//! `access_count` forever and looks permanently hot, because a raw counter has
//! no notion of *recently*. And a page touched every four seconds never demotes
//! however cold it is relative to its peers, because the timeout is absolute
//! rather than comparative.
//!
//! The membrane replaces both with a score. Every page carries an exponentially
//! weighted access rate that decays when it is not used, pages are ranked
//! against each other rather than against a constant, and VRAM is filled from
//! the top of that ranking down to a high-water mark.
//!
//! ## Why a separate module
//!
//! The policy is deliberately one pure function, [`TierPolicy::score`], over a
//! small observation struct. LGM §3.3 replaces it with a learned model; keeping
//! the decision behind a single call with no access to the pool means that swap
//! touches nothing else. It also means the policy is testable without a GPU.
//!
//! ## What this is not
//!
//! It is frequency-driven, not learned. No gradient, no network, nothing
//! trained. Calling it "learned tiering" would be the same category of
//! overclaim this project has already had to retract once.

/// Decay applied to a page's rate each sweep.
///
/// 0.75 gives a half-life of about 2.4 sweeps: recent enough that a shifting
/// workload is followed within a few sweeps, slow enough that one idle sweep
/// does not evict a page that is genuinely in use. Sweeps are the unit rather
/// than wall-clock because migration cost is what the decay trades against, and
/// that is paid per sweep.
pub const RATE_DECAY: f64 = 0.75;

/// Fraction of VRAM the membrane will fill with hot pages.
///
/// Not 1.0: promotion needs somewhere to land, and a pool at exactly capacity
/// makes every promotion wait for an eviction first, which serialises the thing
/// the prefetcher exists to overlap.
pub const VRAM_HIGH_WATER: f64 = 0.85;

/// A page must beat the incumbent by this factor to displace it.
///
/// Without hysteresis two pages either side of the boundary trade places every
/// sweep, and each trade is a full PCIe round trip in both directions. The
/// migration then costs more than the placement saves — the classic thrash.
/// 1.5 was chosen so that a page has to be meaningfully hotter, not marginally
/// hotter, to justify moving bytes.
pub const HYSTERESIS: f64 = 1.5;

/// What the policy is allowed to see about a page.
///
/// Deliberately narrow. The policy cannot reach into the pool, so it cannot
/// accidentally depend on allocation order, pointer identity, or anything else
/// that is not a property of the workload.
#[derive(Debug, Clone, Copy)]
pub struct PageObs {
    /// Decayed access rate — see [`Membrane::observe`].
    pub rate: f64,
    /// Size in bytes. Larger pages must earn their residence.
    pub bytes: usize,
    /// Whether the page is currently VRAM-resident, for hysteresis.
    pub resident: bool,
    /// Pinned pages are never demoted.
    pub pinned: bool,
}

/// The scoring policy. LGM §3.4's argmax, with fixed coefficients.
#[derive(Debug, Clone, Copy)]
pub struct TierPolicy {
    /// Weight on access rate.
    pub alpha: f64,
    /// Weight on the size penalty — how much residence is discounted per MiB.
    pub beta: f64,
}

impl Default for TierPolicy {
    fn default() -> Self {
        // alpha dominates: frequency is the signal, size is a tie-breaker
        // between pages of similar heat rather than a first-order term.
        TierPolicy {
            alpha: 1.0,
            beta: 0.05,
        }
    }
}

impl TierPolicy {
    /// Value of keeping this page in VRAM. Higher wins.
    ///
    /// Size enters as a penalty per MiB rather than a divisor: a page twice the
    /// size should need to be somewhat hotter to justify residence, but not
    /// twice as hot, because the cost of *holding* it scales with size while
    /// the benefit of a hit does not.
    pub fn score(&self, obs: &PageObs) -> f64 {
        if obs.pinned {
            return f64::INFINITY;
        }
        let mib = obs.bytes as f64 / (1024.0 * 1024.0);
        // Incumbency multiplies the *benefit* term, never the net score.
        //
        // Scaling the net is wrong and not subtly so: for a page large enough
        // that the size penalty exceeds its rate the net is negative, and
        // multiplying a negative by 1.5 makes it smaller — so the bonus
        // becomes a penalty and residents are evicted preferentially. At the
        // default beta a 100 MiB page crosses into negative territory at any
        // rate below 5.0, which is most of them, so this was not an edge case.
        //
        // Scaling the rate keeps the intent — a challenger must be 1.5× as hot
        // to displace an incumbent — and is monotone in rate regardless of
        // sign.
        let benefit = self.alpha * obs.rate * if obs.resident { HYSTERESIS } else { 1.0 };
        benefit - self.beta * mib
    }
}

/// Per-page decayed access rates, keyed by VMT name.
#[derive(Debug, Default)]
pub struct Membrane {
    rates: std::collections::HashMap<String, f64>,
    /// Accesses seen since the last [`Membrane::decay`].
    hits: std::collections::HashMap<String, u64>,
    /// Sweeps completed, for diagnostics and tests.
    sweeps: u64,
}

impl Membrane {
    /// Empty membrane.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record one access to `name`.
    ///
    /// Counted rather than applied immediately: a page touched fifty times in
    /// one sweep window should score above one touched once, and folding each
    /// hit into the EMA separately would let a burst saturate the rate and then
    /// decay from a ceiling, losing the distinction.
    pub fn observe(&mut self, name: &str) {
        *self.hits.entry(name.to_string()).or_insert(0) += 1;
    }

    /// Fold the window's hits into each rate and start a new window.
    ///
    /// Every *known* page decays, not only those with hits — otherwise a page
    /// that stops being queried keeps its rate forever and never yields its
    /// VRAM, which is precisely the staleness the raw counter had.
    pub fn decay(&mut self) {
        for (name, rate) in self.rates.iter_mut() {
            let h = self.hits.get(name).copied().unwrap_or(0) as f64;
            *rate = *rate * RATE_DECAY + h * (1.0 - RATE_DECAY);
        }
        // Pages seen for the first time in this window.
        for (name, &h) in self.hits.iter() {
            self.rates
                .entry(name.clone())
                .or_insert_with(|| h as f64 * (1.0 - RATE_DECAY));
        }
        self.hits.clear();
        self.sweeps += 1;
    }

    /// Current decayed rate for `name`, or 0.0 if never observed.
    pub fn rate(&self, name: &str) -> f64 {
        self.rates.get(name).copied().unwrap_or(0.0)
    }

    /// Sweeps completed.
    pub fn sweeps(&self) -> u64 {
        self.sweeps
    }

    /// Drop a page's history — call when the page is deallocated, or its rate
    /// keeps decaying forever in a map that only grows.
    pub fn forget(&mut self, name: &str) {
        self.rates.remove(name);
        self.hits.remove(name);
    }

    /// Rank `candidates` by score, hottest first, and return the prefix that
    /// fits `vram_budget` bytes.
    ///
    /// Returns `(keep, evict)`. The split is by *rank against each other*, not
    /// against a threshold: what matters is which pages are the best use of a
    /// fixed amount of VRAM, and an absolute threshold cannot express that
    /// because it does not know how many pages are competing.
    pub fn plan(
        &self,
        policy: &TierPolicy,
        candidates: &[(String, PageObs)],
        vram_budget: usize,
    ) -> (Vec<String>, Vec<String>) {
        let mut ranked: Vec<(f64, &String, &PageObs)> = candidates
            .iter()
            .map(|(n, o)| (policy.score(o), n, o))
            .collect();
        // Descending by score; ties broken by name so a plan is deterministic
        // and a test cannot pass or fail on HashMap iteration order.
        ranked.sort_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.1.cmp(b.1))
        });

        let cap = (vram_budget as f64 * VRAM_HIGH_WATER) as usize;
        let (mut keep, mut evict) = (Vec::new(), Vec::new());
        let mut used = 0usize;
        for (_, name, obs) in ranked {
            if obs.pinned || used + obs.bytes <= cap {
                used += obs.bytes;
                keep.push(name.clone());
            } else {
                evict.push(name.clone());
            }
        }
        (keep, evict)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn obs(rate: f64, mb: usize, resident: bool) -> PageObs {
        PageObs {
            rate,
            bytes: mb << 20,
            resident,
            pinned: false,
        }
    }

    /// A page that stops being queried must lose its claim on VRAM.
    ///
    /// This is the failure the raw `access_count` had: a lifetime counter only
    /// grows, so a page hammered at startup outranks a page in active use
    /// forever. The rate has to actually fall.
    #[test]
    fn an_idle_page_decays_below_an_active_one() {
        let mut m = Membrane::new();
        for _ in 0..10 {
            m.observe("startup");
        }
        m.decay();
        let peak = m.rate("startup");
        assert!(peak > 0.0);

        // "startup" goes quiet; "steady" is used lightly but continuously.
        for _ in 0..6 {
            m.observe("steady");
            m.decay();
        }
        assert!(
            m.rate("startup") < m.rate("steady"),
            "an idle page ({:.3}) must fall below a continuously used one \
             ({:.3}) — otherwise a startup burst pins VRAM forever",
            m.rate("startup"),
            m.rate("steady"),
        );
        assert!(m.rate("startup") < peak * 0.2, "decay must be substantial");
    }

    /// The membrane must follow a workload that moves.
    #[test]
    fn the_membrane_follows_a_shifting_workload() {
        let mut m = Membrane::new();
        for _ in 0..8 {
            for _ in 0..5 {
                m.observe("A");
            }
            m.decay();
        }
        assert!(m.rate("A") > m.rate("B"));

        // Traffic moves to B entirely.
        for _ in 0..8 {
            for _ in 0..5 {
                m.observe("B");
            }
            m.decay();
        }
        assert!(
            m.rate("B") > m.rate("A"),
            "membrane must follow the workload: A={:.3} B={:.3}",
            m.rate("A"),
            m.rate("B")
        );
    }

    /// Hot pages are kept, cold ones evicted, and the budget is respected.
    #[test]
    fn plan_keeps_the_hottest_pages_within_budget() {
        let m = Membrane::new();
        let p = TierPolicy::default();
        let cands = vec![
            ("hot".to_string(), obs(10.0, 100, false)),
            ("warm".to_string(), obs(5.0, 100, false)),
            ("cold".to_string(), obs(0.1, 100, false)),
        ];
        // 250 MB budget × 0.85 high-water = 212 MB → two 100 MB pages fit.
        let (keep, evict) = m.plan(&p, &cands, 250 << 20);
        assert_eq!(keep, vec!["hot".to_string(), "warm".to_string()]);
        assert_eq!(evict, vec!["cold".to_string()]);
    }

    /// A pinned page is never evicted, even past the budget.
    #[test]
    fn pinned_pages_survive_any_budget() {
        let m = Membrane::new();
        let p = TierPolicy::default();
        let mut pinned = obs(0.0, 500, false);
        pinned.pinned = true;
        let cands = vec![
            ("pinned".to_string(), pinned),
            ("hot".to_string(), obs(99.0, 100, false)),
        ];
        let (keep, _) = m.plan(&p, &cands, 64 << 20);
        assert!(
            keep.contains(&"pinned".to_string()),
            "a pinned page must be kept regardless of budget — the framework \
             holds a live pointer into it"
        );
    }

    /// Two pages of nearly equal heat must not trade places.
    ///
    /// Each trade is a full PCIe round trip in both directions, so a policy
    /// that flips on a marginal difference spends more on migration than
    /// placement saves. The incumbent has to be beaten by a margin.
    #[test]
    fn hysteresis_prevents_boundary_thrash() {
        let m = Membrane::new();
        let p = TierPolicy::default();
        // `incumbent` is resident and marginally colder than the challenger.
        let cands = vec![
            ("incumbent".to_string(), obs(1.00, 100, true)),
            ("challenger".to_string(), obs(1.15, 100, false)),
        ];
        let (keep, evict) = m.plan(&p, &cands, 120 << 20); // room for one
        assert_eq!(
            keep,
            vec!["incumbent".to_string()],
            "a marginally hotter challenger must not displace a resident page"
        );
        assert_eq!(evict, vec!["challenger".to_string()]);

        // A decisively hotter challenger should win.
        let cands = vec![
            ("incumbent".to_string(), obs(1.00, 100, true)),
            ("challenger".to_string(), obs(5.00, 100, false)),
        ];
        let (keep, _) = m.plan(&p, &cands, 120 << 20);
        assert_eq!(
            keep,
            vec!["challenger".to_string()],
            "hysteresis must not become a permanent incumbency lock"
        );
    }

    /// Forgetting a deallocated page must actually drop it.
    #[test]
    fn forget_drops_history() {
        let mut m = Membrane::new();
        m.observe("gone");
        m.decay();
        assert!(m.rate("gone") > 0.0);
        m.forget("gone");
        assert_eq!(m.rate("gone"), 0.0);
    }

    /// Plans must not depend on hash iteration order.
    #[test]
    fn plan_is_deterministic_under_ties() {
        let m = Membrane::new();
        let p = TierPolicy::default();
        let cands: Vec<(String, PageObs)> = ["d", "a", "c", "b"]
            .iter()
            .map(|n| (n.to_string(), obs(1.0, 10, false)))
            .collect();
        let first = m.plan(&p, &cands, 25 << 20);
        for _ in 0..8 {
            assert_eq!(m.plan(&p, &cands, 25 << 20), first);
        }
    }
}
