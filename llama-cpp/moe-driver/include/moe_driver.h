// moe_driver.h — C-ABI surface for the index-driven MoE expert-offload driver.
//
// This is the ONLY header the vendored llama.cpp seam includes. It speaks pure
// ggml + C types (no llama internal types, no C++), so the driver stays
// decoupled from llama internals and survives upstream rebases. All real logic
// (residency policy, paging, CUDA copy engine, dual graph emission) lives in the
// moe-driver library; llama.cpp only delegates to these entry points when the
// driver is active.
//
// Lifetime: moe_driver_init() once per model load (after tensors are known) ...
// moe_driver_build_ffn() per layer during graph build ... moe_driver_free() at
// context teardown. When the driver is inactive every seam call is a no-op /
// passthrough and llama.cpp runs stock.

#ifndef JADE_MOE_DRIVER_H
#define JADE_MOE_DRIVER_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct ggml_tensor;
struct ggml_context;

// Configuration passed from jaded (via the ffi/api crates). Mirrors MoeOffload
// in jaded/src/config.rs.
struct moe_driver_config {
    bool        enabled;       // master switch; false => all seam calls passthrough
    int32_t     pin;           // H: pinned experts per layer
    int32_t     cache;         // C: LRU cache slots per layer
    const char* order_path;    // path to expert_order asset (hottest-first per layer)
};

// True once a successful init has made the driver active. The seam guards every
// call site with this so a disabled/failed driver leaves stock behavior intact.
bool moe_driver_active(void);

// Initialize from config + model facts. `expert_order_flat` is the validated
// per-layer ranking (row-major [n_layers*n_experts], hottest-first), already
// parsed by the Rust side (jaded::moe::ExpertOrder) so the C++ stays JSON-free.
// Allocates the per-layer LayerResidency state. Returns 0 on success; non-zero
// leaves the driver inactive (caller falls back to stock). Never aborts.
int moe_driver_init(const struct moe_driver_config* cfg,
                    int32_t n_layers,
                    int32_t n_experts,
                    int32_t n_expert_used,
                    const int32_t* expert_order_flat);

// Load-time placement: given a layer's full stacked expert tensor name and its
// expert dimension, report how to split it. Out-params give the permutation
// (hottest-first expert ids) and the resident count (pin+cache) so the loader
// can build the resident (VRAM) tensor and keep the cold remainder in host RAM
// as the page source. Returns false (no split) when the driver is inactive.
bool moe_driver_plan_split(int32_t layer,
                           int32_t n_experts,
                           const int32_t** out_order,   // borrowed, len n_experts
                           int32_t*        out_resident_count);

// Graph hook (minimal seam): given a layer's HOST-resident expert tensors (the
// full [..,n_expert] weights kept in host RAM by the CPU buft override) and the
// router's `selected_experts`, return the driver's VRAM **resident** expert
// tensors (H+C experts) plus a **remapped** id tensor, so the caller can run the
// stock mul_mat_id path against them unchanged.
//
// Internally, on first call per layer it lazily allocates the resident tensors
// in a CUDA buffer and copies the pinned (top-H) experts in; every call emits a
// ggml_map_custom op on `selected_experts` whose callback runs the residency
// policy, pages cold experts host->VRAM cache slots (ggml_backend_tensor_set),
// and writes the resident slot-ids into `out_ids`.
//
// Returns false if inactive (caller keeps the stock tensors). The out tensors
// are valid for the lifetime of the graph context `ctx`.
bool moe_driver_remap(struct ggml_context* ctx,
                      int32_t              layer,
                      struct ggml_tensor*  host_gate,        // [n_embd, n_ff, n_expert]
                      struct ggml_tensor*  host_up,
                      struct ggml_tensor*  host_down,
                      struct ggml_tensor*  selected_experts, // [n_expert_used, n_tokens] i32
                      struct ggml_tensor** out_gate,
                      struct ggml_tensor** out_up,
                      struct ggml_tensor** out_down,
                      struct ggml_tensor** out_ids);

// CUDA on-stream pager, invoked from ggml-cuda's GGML_OP_CUSTOM dispatch (so it
// stays on the CUDA backend — no per-layer graph split). Reads the selected
// expert ids (device), runs the residency policy on the host, pages cold experts
// from the pageable host pool into VRAM cache slots via cudaMemcpyAsync on
// `stream`, and writes the resident slot-ids to `out_slots` (device). `stream`
// is a cudaStream_t passed as void* to keep this header CUDA-free.
void moe_driver_page_cuda(int32_t layer,
                          const int32_t* selected_experts_dev,
                          int32_t*       out_slots_dev,
                          int32_t        n_expert_used,
                          int32_t        n_tokens,
                          void*          stream);

// Cumulative paging counters since init (for the latency study / logging):
// total cold misses paged, cache hits, and bytes copied host->VRAM.
struct moe_driver_stats {
    uint64_t misses;
    uint64_t hits;
    uint64_t bytes_paged;
};
void moe_driver_get_stats(struct moe_driver_stats* out);

// Release all driver state. Safe to call when inactive.
void moe_driver_free(void);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // JADE_MOE_DRIVER_H
