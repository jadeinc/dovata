// moe_stub.cpp — inert host implementation of the MoE expert-offload C-ABI.
//
// The real driver (moe_driver.cpp + graph.cpp + page.cu) is compiled into the
// ggml-cuda backend and is therefore present ONLY in CUDA builds. MoE offload is
// a CUDA-only feature, but the seam that calls it (`build_moe_ffn` in
// llama-graph.cpp) and dovata's `moe_driver_init` reference these symbols
// unconditionally — so a non-CUDA build (`--features inference` without `cuda`,
// or any non-CUDA backend) would fail to link with undefined references.
//
// This file provides no-op definitions for the full surface so those builds
// link. Every entry reports "inactive": moe_driver_init returns non-zero, so
// moe_driver_active() stays false and the seam runs the stock path unchanged.
// It is compiled into libllama ONLY when GGML_CUDA is off (see
// src/CMakeLists.txt), so it never collides with the real driver.
//
// Keep this in lockstep with moe_driver.h — a symbol added there must be stubbed
// here, or the non-CUDA build breaks again.

#include "moe_driver.h"

extern "C" {

bool moe_driver_active(void) {
    return false;
}

int moe_driver_init(const struct moe_driver_config* /*cfg*/,
                    int32_t /*n_layers*/,
                    int32_t /*n_experts*/,
                    int32_t /*n_expert_used*/,
                    const int32_t* /*expert_order_flat*/) {
    // Non-zero => caller (dovata) logs "MoE driver init failed" and runs stock.
    return 1;
}

bool moe_driver_plan_split(int32_t /*layer*/,
                           int32_t /*n_experts*/,
                           const int32_t** /*out_order*/,
                           int32_t* /*out_resident_count*/) {
    return false;
}

bool moe_driver_remap(struct ggml_context* /*ctx*/,
                      int32_t              /*layer*/,
                      struct ggml_tensor*  /*host_gate*/,
                      struct ggml_tensor*  /*host_up*/,
                      struct ggml_tensor*  /*host_down*/,
                      struct ggml_tensor*  /*selected_experts*/,
                      struct ggml_tensor** /*out_gate*/,
                      struct ggml_tensor** /*out_up*/,
                      struct ggml_tensor** /*out_down*/,
                      struct ggml_tensor** /*out_ids*/) {
    return false;
}

void moe_driver_page_cuda(int32_t /*layer*/,
                          const int32_t* /*selected_experts_dev*/,
                          int32_t*       /*out_slots_dev*/,
                          int32_t        /*n_expert_used*/,
                          int32_t        /*n_tokens*/,
                          void*          /*stream*/) {
}

void moe_driver_get_stats(struct moe_driver_stats* out) {
    if (out) {
        out->misses = 0;
        out->hits = 0;
        out->bytes_paged = 0;
    }
}

void moe_driver_free(void) {
}

} // extern "C"
