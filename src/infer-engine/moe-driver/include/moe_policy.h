// moe_policy.h — per-layer MoE expert residency + LRU cache policy.
//
// PURE host C++: no CUDA, no ggml, no llama internals. This is the bug-prone
// bookkeeping that decides which experts live in VRAM (pinned or cached) and
// which are cold (host RAM, computed on CPU this step). It is split out so it
// compiles and unit-tests on any platform (including macOS) without the GPU
// backend — mirroring the pure-core philosophy of the dovata library.
//
// Layout of the resident (VRAM) expert tensor for one layer:
//
//     slot:  0 .. H-1     H .. H+C-1
//            [ pinned  ]  [  LRU cache  ]
//
// Pinned slots hold the H globally-hottest experts (from the usage-index
// permutation) and are never evicted. Cache slots are filled on demand as cold
// experts are touched, evicting the least-recently-used cache slot when full.
// Everything not resident is "cold": computed on CPU for the current token and
// (by the driver) asynchronously promoted into a cache slot for next time.

#ifndef JADE_MOE_POLICY_H
#define JADE_MOE_POLICY_H

#include <cstdint>
#include <vector>

namespace moe {

// Result of admitting a cold expert into the cache.
struct Admission {
    int slot;        // resident slot the expert now occupies (>= pin_count)
    int evicted;     // expert id evicted to make room, or -1 if a free slot was used
};

class LayerResidency {
public:
    // n_experts : total experts in the layer (e.g. 128)
    // pin_count : H, number of pinned slots (top-H of `order`)
    // cache_cap : C, number of LRU cache slots
    // order     : length-n_experts permutation, hottest-first (expert_order[layer]);
    //             order[0..H) are pinned. Must be a valid permutation of [0,n_experts).
    LayerResidency(int n_experts, int pin_count, int cache_cap, const int* order);

    int n_experts() const { return n_experts_; }
    int pin_count() const { return pin_count_; }
    int cache_cap() const { return cache_cap_; }
    int resident_cap() const { return pin_count_ + cache_cap_; }

    // The expert id occupying a resident slot, or -1 if the cache slot is empty.
    int expert_at(int slot) const { return slot_expert_[slot]; }

    // Resident slot holding `expert_id`, or -1 if the expert is cold.
    int slot_of(int expert_id) const { return expert_slot_[expert_id]; }
    bool is_resident(int expert_id) const { return expert_slot_[expert_id] >= 0; }
    bool is_pinned(int expert_id) const {
        int s = expert_slot_[expert_id];
        return s >= 0 && s < pin_count_;
    }

    // Mark a resident expert as used this step (no-op for pinned / cold experts).
    // Refreshes LRU recency so it survives the next eviction.
    void touch(int expert_id);

    // Bring a cold expert into a cache slot, evicting the LRU cache entry if the
    // cache is full. Precondition: !is_resident(expert_id). The admitted expert
    // becomes the most-recently-used. Returns the slot used and any evicted expert.
    Admission admit(int expert_id);

    // Reset the cache to empty (pinned experts stay resident). Used between
    // independent sequences if the driver chooses to drop a context's working set.
    void clear_cache();

    // Steady-state diagnostics for tests / logging.
    int cache_occupancy() const { return cache_used_; }

    // ---- per-token planning (the brain the ggml custom-op callback runs) ----
    // A cold expert that must be copied host->VRAM into `slot` this step.
    struct Fetch {
        int expert_id;
        int slot;
    };

    // Classify one token's `k` selected experts for one layer:
    //   * resident picks are touched (LRU refresh),
    //   * cold picks are admitted (evicting LRU), recorded in `fetches` so the
    //     caller can copy their weights host->VRAM into the returned slot.
    // Writes the resident slot of each selected expert into `out_slots[0..k)` —
    // these are the remapped ids fed to the single GPU mul_mat_id. After this
    // call every selected expert is resident, so the matmul never misses.
    // `fetches` is cleared then filled. Returns the number of cold fetches.
    int plan_step(const int32_t* selected, int k, int32_t* out_slots, std::vector<Fetch>& fetches);

private:
    void lru_push_front(int slot);   // mark `slot` most-recently-used
    void lru_unlink(int slot);

    int n_experts_;
    int pin_count_;
    int cache_cap_;
    int cache_used_;                 // number of occupied cache slots

    std::vector<int> expert_slot_;   // expert id -> resident slot, or -1 (cold)
    std::vector<int> slot_expert_;   // resident slot -> expert id, or -1 (empty)

    // Intrusive LRU doubly-linked list over CACHE slots only, indexed by slot.
    // Heads/tails use sentinel -1. front = most-recently-used, back = LRU victim.
    std::vector<int> lru_next_;      // size resident_cap (pinned entries unused)
    std::vector<int> lru_prev_;
    int lru_front_;
    int lru_back_;
};

} // namespace moe

#endif // JADE_MOE_POLICY_H
