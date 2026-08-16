// moe_policy_test.cpp — standalone unit test for LayerResidency.
// Pure host C++; builds and runs on macOS:
//   c++ -std=c++17 -I include src/moe_policy.cpp test/moe_policy_test.cpp -o /tmp/moe_policy_test && /tmp/moe_policy_test

#include "moe_policy.h"
#include "moe_driver.h"

#include <cassert>
#include <cstdio>
#include <numeric>
#include <vector>

using moe::Admission;
using moe::LayerResidency;

static int failures = 0;
#define CHECK(cond)                                                            \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::printf("  FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);       \
            ++failures;                                                        \
        }                                                                      \
    } while (0)

// order = identity over [0, n): expert i has rank i (i.e. expert i is i-th hottest).
static std::vector<int> identity_order(int n) {
    std::vector<int> o(n);
    std::iota(o.begin(), o.end(), 0);
    return o;
}

static void test_pinned_always_resident() {
    auto order = identity_order(16);
    LayerResidency r(16, 4, 4, order.data()); // H=4, C=4
    for (int e = 0; e < 4; ++e) {
        CHECK(r.is_resident(e));
        CHECK(r.is_pinned(e));
        CHECK(r.slot_of(e) == e); // pinned mapped to leading slots in order
    }
    for (int e = 4; e < 16; ++e) {
        CHECK(!r.is_resident(e)); // cold until admitted
    }
    CHECK(r.cache_occupancy() == 0);
}

static void test_admit_fills_then_evicts_lru() {
    auto order = identity_order(16);
    LayerResidency r(16, 2, 2, order.data()); // H=2 (experts 0,1), C=2

    // Fill the two cache slots with experts 10 then 11.
    Admission a10 = r.admit(10);
    CHECK(a10.evicted == -1);
    CHECK(a10.slot == 2); // first cache slot
    Admission a11 = r.admit(11);
    CHECK(a11.evicted == -1);
    CHECK(a11.slot == 3);
    CHECK(r.cache_occupancy() == 2);
    CHECK(r.is_resident(10) && r.is_resident(11));

    // Cache full. Admitting 12 evicts the LRU victim. 10 was used before 11,
    // so 10 is the LRU and should be evicted.
    Admission a12 = r.admit(12);
    CHECK(a12.evicted == 10);
    CHECK(!r.is_resident(10));
    CHECK(r.is_resident(12));
    CHECK(r.slot_of(12) == a10.slot); // reuses the slot 10 vacated
}

static void test_touch_refreshes_recency() {
    auto order = identity_order(16);
    LayerResidency r(16, 0, 2, order.data()); // no pins, C=2

    r.admit(5); // slot 0 (MRU)
    r.admit(6); // slot 1 (MRU), 5 now LRU
    r.touch(5); // 5 becomes MRU again -> 6 is now LRU

    Admission a = r.admit(7); // should evict 6, not 5
    CHECK(a.evicted == 6);
    CHECK(r.is_resident(5));
    CHECK(!r.is_resident(6));
    CHECK(r.is_resident(7));
}

static void test_touch_pinned_is_noop() {
    auto order = identity_order(16);
    LayerResidency r(16, 1, 1, order.data()); // pin expert 0, C=1
    r.touch(0);       // pinned: must not corrupt LRU
    r.admit(9);       // slot 1
    r.touch(0);       // still no-op
    Admission a = r.admit(8); // evicts 9 (only cache entry)
    CHECK(a.evicted == 9);
    CHECK(r.is_pinned(0) && r.is_resident(0));
    CHECK(r.is_resident(8));
}

static void test_clear_cache_keeps_pins() {
    auto order = identity_order(16);
    LayerResidency r(16, 2, 3, order.data());
    r.admit(10);
    r.admit(11);
    r.admit(12);
    CHECK(r.cache_occupancy() == 3);
    r.clear_cache();
    CHECK(r.cache_occupancy() == 0);
    CHECK(!r.is_resident(10) && !r.is_resident(11) && !r.is_resident(12));
    CHECK(r.is_pinned(0) && r.is_pinned(1)); // pins survive
    // Cache usable again after clear.
    Admission a = r.admit(13);
    CHECK(a.evicted == -1 && a.slot == 2);
}

// Simulate a short routed stream and assert the resident set never exceeds H+C
// and that every "hit" expert is genuinely resident at use time.
static void test_stream_invariants() {
    auto order = identity_order(128);
    LayerResidency r(128, 16, 48, order.data());
    // pseudo-random routed ids (deterministic LCG), top-8 per "token"
    uint32_t s = 12345;
    auto next = [&]() { s = s * 1664525u + 1013904223u; return (int)(s % 128); };
    for (int tok = 0; tok < 200; ++tok) {
        for (int k = 0; k < 8; ++k) {
            int e = next();
            if (r.is_resident(e)) {
                r.touch(e); // hit
            } else {
                Admission a = r.admit(e); // miss -> promote
                CHECK(a.slot >= 16 && a.slot < 64); // landed in a cache slot
            }
            CHECK(r.is_resident(e)); // resident immediately after handling
        }
        CHECK(r.cache_occupancy() <= 48);
    }
}

static void test_plan_step_hits_and_misses() {
    auto order = identity_order(128);
    LayerResidency r(128, 16, 48, order.data()); // pinned 0..15
    std::vector<moe::LayerResidency::Fetch> fetches;
    int out[8];

    // Selection: 3 pinned (hits), 1 cold (miss).
    int sel1[4] = {0, 5, 15, 100};
    int n = r.plan_step(sel1, 4, out, fetches);
    CHECK(n == 1);                       // exactly one cold fetch
    CHECK(fetches[0].expert_id == 100);
    CHECK(fetches[0].slot >= 16);        // landed in a cache slot
    // every selected expert is now resident and out_slots points at it
    for (int i = 0; i < 4; ++i) {
        CHECK(r.is_resident(sel1[i]));
        CHECK(out[i] == r.slot_of(sel1[i]));
    }

    // Re-selecting expert 100 next step is now a HIT (was paged in).
    int sel2[2] = {100, 7};
    n = r.plan_step(sel2, 2, out, fetches);
    CHECK(n == 0);                       // no fetches: 100 cached, 7 pinned
    CHECK(out[0] == r.slot_of(100));
}

// Warm-up behaviour: on a repeating working set, the miss rate must decay to ~0
// once the set fits in pin+cache (the property that makes paging cheap on long
// generations — the whole premise of the smart engine).
static void test_plan_step_working_set_warms_up() {
    auto order = identity_order(128);
    LayerResidency r(128, 16, 48, order.data()); // resident cap 64
    std::vector<moe::LayerResidency::Fetch> fetches;
    int out[8];
    // A fixed working set of 40 cold experts (ids 64..103), 8 picked per step,
    // cycling — fits in the 48-slot cache, so after the first pass misses -> 0.
    const int ws[40] = {
        64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,
        84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103};
    int total_fetches = 0, late_fetches = 0;
    for (int step = 0; step < 40; ++step) {
        int sel[8];
        for (int j = 0; j < 8; ++j) sel[j] = ws[(step * 8 + j) % 40];
        int n = r.plan_step(sel, 8, out, fetches);
        total_fetches += n;
        if (step >= 5) late_fetches += n; // after the working set is resident
    }
    CHECK(total_fetches == 40);  // each of the 40 distinct experts paged exactly once
    CHECK(late_fetches == 0);    // steady state: zero misses once warm
    CHECK(r.cache_occupancy() == 40);
}

// Exercise the public C-ABI lifecycle (the surface llama.cpp + Rust bind to).
static void test_c_abi_init_and_split() {
    CHECK(!moe_driver_active()); // inactive before init

    // disabled config -> stays inactive, no crash
    moe_driver_config off{};
    off.enabled = false;
    std::vector<int> order(2 * 8);
    for (int l = 0; l < 2; ++l)
        for (int e = 0; e < 8; ++e) order[l * 8 + e] = (e + l) % 8; // per-layer permutation
    CHECK(moe_driver_init(&off, 2, 8, 2, order.data()) != 0);
    CHECK(!moe_driver_active());

    // enabled: pin=2, cache=2, 2 layers, 8 experts
    moe_driver_config on{};
    on.enabled = true;
    on.pin = 2;
    on.cache = 2;
    CHECK(moe_driver_init(&on, 2, 8, 2, order.data()) == 0);
    CHECK(moe_driver_active());

    const int32_t* layer_order = nullptr;
    int32_t resident = 0;
    CHECK(moe_driver_plan_split(1, 8, &layer_order, &resident));
    CHECK(resident == 4); // pin+cache
    CHECK(layer_order != nullptr);
    CHECK(layer_order[0] == order[1 * 8 + 0]); // layer-1 ranking echoed back
    CHECK(!moe_driver_plan_split(5, 8, &layer_order, &resident)); // out-of-range layer

    // reject impossible resident set (pin+cache > n_experts)
    moe_driver_config bad{};
    bad.enabled = true;
    bad.pin = 6;
    bad.cache = 6;
    moe_driver_free();
    CHECK(moe_driver_init(&bad, 2, 8, 2, order.data()) != 0);
    CHECK(!moe_driver_active());

    moe_driver_free();
    CHECK(!moe_driver_active()); // idempotent free
}

int main() {
    std::printf("moe_policy_test:\n");
    test_pinned_always_resident();
    test_admit_fills_then_evicts_lru();
    test_touch_refreshes_recency();
    test_touch_pinned_is_noop();
    test_clear_cache_keeps_pins();
    test_stream_invariants();
    test_plan_step_hits_and_misses();
    test_plan_step_working_set_warms_up();
    test_c_abi_init_and_split();
    if (failures == 0) {
        std::printf("  OK — all checks passed\n");
        return 0;
    }
    std::printf("  %d check(s) FAILED\n", failures);
    return 1;
}
