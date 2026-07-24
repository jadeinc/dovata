// moe_policy.cpp — implementation of the per-layer residency + LRU cache policy.
// See moe_policy.h. Pure host C++; no CUDA/ggml.

#include "moe_policy.h"

#include <cassert>

namespace moe {

LayerResidency::LayerResidency(int n_experts, int pin_count, int cache_cap, const int* order)
    : n_experts_(n_experts),
      pin_count_(pin_count),
      cache_cap_(cache_cap),
      cache_used_(0),
      expert_slot_(n_experts, -1),
      slot_expert_(pin_count + cache_cap, -1),
      lru_next_(pin_count + cache_cap, -1),
      lru_prev_(pin_count + cache_cap, -1),
      lru_front_(-1),
      lru_back_(-1) {
    assert(pin_count >= 0 && cache_cap >= 0);
    assert(pin_count + cache_cap <= n_experts);
    // Pin the top-H experts into the leading slots; they never move or evict.
    for (int i = 0; i < pin_count_; ++i) {
        int e = order[i];
        assert(e >= 0 && e < n_experts_);
        slot_expert_[i] = e;
        expert_slot_[e] = i;
    }
    // Cache slots [H, H+C) start empty.
}

void LayerResidency::lru_unlink(int slot) {
    int p = lru_prev_[slot];
    int n = lru_next_[slot];
    if (p != -1) lru_next_[p] = n; else lru_front_ = n;
    if (n != -1) lru_prev_[n] = p; else lru_back_ = p;
    lru_prev_[slot] = lru_next_[slot] = -1;
}

void LayerResidency::lru_push_front(int slot) {
    lru_prev_[slot] = -1;
    lru_next_[slot] = lru_front_;
    if (lru_front_ != -1) lru_prev_[lru_front_] = slot;
    lru_front_ = slot;
    if (lru_back_ == -1) lru_back_ = slot;
}

void LayerResidency::touch(int expert_id) {
    int slot = expert_slot_[expert_id];
    // Pinned (slot < pin_count_) and cold (slot == -1) experts have no LRU state.
    if (slot < pin_count_) return;
    lru_unlink(slot);
    lru_push_front(slot);
}

Admission LayerResidency::admit(int expert_id) {
    assert(!is_resident(expert_id));
    int slot;
    int evicted = -1;
    if (cache_used_ < cache_cap_) {
        // Use the next free cache slot (slots are filled front-to-back initially).
        slot = pin_count_ + cache_used_;
        ++cache_used_;
    } else {
        // Cache full: evict the LRU victim at the back of the list.
        slot = lru_back_;
        assert(slot >= pin_count_);
        evicted = slot_expert_[slot];
        expert_slot_[evicted] = -1;
        lru_unlink(slot);
    }
    slot_expert_[slot] = expert_id;
    expert_slot_[expert_id] = slot;
    lru_push_front(slot); // newly admitted = most-recently-used
    return Admission{slot, evicted};
}

int LayerResidency::plan_step(const int32_t* selected, int k, int32_t* out_slots,
                              std::vector<Fetch>& fetches) {
    fetches.clear();
    for (int i = 0; i < k; ++i) {
        int e = selected[i];
        if (is_resident(e)) {
            // hit: keep it resident (and refresh LRU recency for cache entries)
            touch(e);
            out_slots[i] = slot_of(e);
        } else {
            // miss: admit into a cache slot; the caller pages its weights in
            Admission a = admit(e);
            fetches.push_back(Fetch{e, a.slot});
            out_slots[i] = a.slot;
        }
    }
    // Note: a single token can select the same expert at most once (top-k is a
    // set), so no in-step duplicate-admit hazard. Distinct cold experts that map
    // to the same victim are handled in admission order (LRU), which is correct
    // because each admit() updates the maps before the next is considered.
    return static_cast<int>(fetches.size());
}

void LayerResidency::clear_cache() {
    for (int s = pin_count_; s < pin_count_ + cache_cap_; ++s) {
        int e = slot_expert_[s];
        if (e != -1) expert_slot_[e] = -1;
        slot_expert_[s] = -1;
        lru_next_[s] = lru_prev_[s] = -1;
    }
    cache_used_ = 0;
    lru_front_ = lru_back_ = -1;
}

} // namespace moe
