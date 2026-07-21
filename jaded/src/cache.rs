//! Bounded LRU response cache.
//!
//! Keyed by a hash of the request fields that determine the output. On a hit
//! the daemon replays the cached response text as a single token frame instead
//! of running the model. This is a *response* cache, distinct from the KV-cache
//! prefix reuse done during prefill.

use ovata_infer_protocol::InferenceRequest;
use std::collections::{HashMap, VecDeque};

pub struct PromptCache {
    // the actual store: request-hash -> (generated text, tokens used to make it)
    map: HashMap<u64, (String, u64)>,
    // insertion order of the keys, oldest at the front. This is the "LRU" book-
    // keeping: we evict from the front when we hit capacity. (Insertion-order,
    // not true access-order — we don't reshuffle on a hit.)
    order: VecDeque<u64>,
    // hard cap on how many entries we keep; 0 disables caching entirely
    max_entries: usize,
}

impl PromptCache {
    pub fn new(max_entries: usize) -> Self {
        // start empty; the map and order grow together as we insert
        PromptCache {
            map: HashMap::new(),
            order: VecDeque::new(),
            max_entries,
        }
    }

    /// Hash the request fields that determine the generated output.
    pub fn key(req: &InferenceRequest) -> u64 {
        use std::hash::{Hash, Hasher};
        // fold every output-affecting field into one 64-bit hash. Two requests
        // with the same hash are treated as the same query. We deliberately hash
        // ONLY the fields that change the generated text — fields like `trust`
        // (provenance only) or the *_only probe flags are not included.
        let mut h = std::collections::hash_map::DefaultHasher::new();
        req.prompt.hash(&mut h);
        req.model.hash(&mut h);
        req.max_tokens.hash(&mut h);
        req.grammar.hash(&mut h);
        req.anchor.hash(&mut h);
        req.stop_anchor.hash(&mut h);
        // keep_anchors changes whether the stop-anchor boundary appears in the
        // output, so it must distinguish cache entries.
        req.keep_anchors.hash(&mut h);
        // collapse the accumulated state into the final key
        h.finish()
    }

    pub fn get(&self, key: u64) -> Option<&(String, u64)> {
        // straight lookup; None means a cache miss (caller then runs the model)
        self.map.get(&key)
    }

    pub fn insert(&mut self, key: u64, text: String, tokens_used: u64) {
        // bail out in two no-op cases: caching is disabled (cap 0), or we already
        // have this key. We keep the FIRST value for a key rather than overwriting
        // (see the reinsert test) — simpler and avoids churning the order queue.
        if self.max_entries == 0 || self.map.contains_key(&key) {
            return;
        }
        // if we're at capacity, make room by evicting the oldest entry: pull the
        // front key off the order queue and drop it from the map too.
        if self.map.len() >= self.max_entries {
            if let Some(oldest) = self.order.pop_front() {
                self.map.remove(&oldest);
            }
        }
        // now there's room: store the value and record this key as newest (back)
        self.map.insert(key, (text, tokens_used));
        self.order.push_back(key);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn miss_then_hit() {
        let mut c = PromptCache::new(4);
        assert!(c.get(7).is_none());
        c.insert(7, "hello".into(), 3);
        assert_eq!(c.get(7), Some(&("hello".to_owned(), 3)));
    }

    #[test]
    fn zero_capacity_never_stores() {
        let mut c = PromptCache::new(0);
        c.insert(1, "x".into(), 1);
        assert!(c.get(1).is_none());
    }

    #[test]
    fn evicts_oldest_when_full() {
        let mut c = PromptCache::new(2);
        c.insert(1, "a".into(), 1);
        c.insert(2, "b".into(), 1);
        c.insert(3, "c".into(), 1); // evicts key 1 (oldest)
        assert!(c.get(1).is_none());
        assert!(c.get(2).is_some());
        assert!(c.get(3).is_some());
    }

    #[test]
    fn reinsert_of_existing_key_is_ignored() {
        let mut c = PromptCache::new(2);
        c.insert(1, "first".into(), 1);
        c.insert(1, "second".into(), 9);
        assert_eq!(c.get(1), Some(&("first".to_owned(), 1)));
    }

    #[test]
    fn key_is_stable_and_field_sensitive() {
        let base = InferenceRequest {
            prompt: "hi".into(),
            model: "m".into(),
            max_tokens: 10,
            grammar: None,
            anchor: None,
            stop_anchor: None,
            count_only: false,
            stats_only: false,
            health_only: false,
            keep_anchors: false,
            trust: 0,
            rlm: false,
        };
        let same = base.clone();
        assert_eq!(PromptCache::key(&base), PromptCache::key(&same));

        let mut diff = base.clone();
        diff.prompt = "bye".into();
        assert_ne!(PromptCache::key(&base), PromptCache::key(&diff));
    }
}
