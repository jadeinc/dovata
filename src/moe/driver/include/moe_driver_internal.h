// moe_driver_internal.h — shared state between the host orchestration
// (moe_driver.cpp) and the ggml graph hook (graph.cpp). NOT part of the public
// C-ABI; ggml/CUDA-free so the host side stays portable.

#ifndef JADE_MOE_DRIVER_INTERNAL_H
#define JADE_MOE_DRIVER_INTERNAL_H

#include "moe_driver.h"
#include "moe_policy.h"

#include <cstdint>
#include <vector>

namespace moe {

// Per-layer GPU/host handles, populated lazily by graph.cpp on the first
// moe_driver_remap() call for the layer. All ggml/CUDA types are opaque void*
// here so the host core (moe_driver.cpp) stays ggml-free; graph.cpp casts them.
struct LayerTensors {
    bool  allocated = false;       // resident tensors built + pin region copied
    // Resident VRAM expert tensors (H+C experts), the mul_mat_id targets.
    void* resident_gate = nullptr; // ggml_tensor*
    void* resident_up   = nullptr;
    void* resident_down = nullptr;
    // Host page source: the full [..,n_expert] expert tensors kept in host RAM
    // by the CPU buft override (ggml_tensor*). Read when paging cold experts.
    void* host_gate = nullptr;
    void* host_up   = nullptr;
    void* host_down = nullptr;
    // Raw data pointers (set by graph.cpp at alloc), so the CUDA pager (page.cu)
    // can issue cudaMemcpyAsync without depending on ggml types.
    void* dev_gate = nullptr;  // resident VRAM data (gate)
    void* dev_up   = nullptr;
    void* dev_down = nullptr;
    void* host_gate_data = nullptr; // host page-source data (full experts)
    void* host_up_data   = nullptr;
    void* host_down_data = nullptr;
    // Per-expert byte stride (nb[2]) in each projection = the H2D copy size.
    size_t bytes_gate = 0;
    size_t bytes_up   = 0;
    size_t bytes_down = 0;
    // Driver-owned ggml context + CUDA buffer backing the resident tensors.
    void* ggml_ctx    = nullptr;   // ggml_context*
    void* ggml_buffer = nullptr;   // ggml_backend_buffer_t
};

// Process-wide driver state. One per loaded model.
struct Driver {
    int32_t n_layers = 0;
    int32_t n_experts = 0;
    int32_t n_expert_used = 0;
    int32_t pin = 0;
    int32_t cache = 0;

    std::vector<int32_t> order;             // row-major [n_layers * n_experts]
    std::vector<LayerResidency> residency;  // one per layer (pin/cache slots)
    std::vector<LayerTensors> tensors;      // one per layer (set by load hook)
    moe_driver_stats stats{0, 0, 0};
};

// The active driver, or nullptr when inactive.
Driver* driver();

} // namespace moe

#endif // JADE_MOE_DRIVER_INTERNAL_H
