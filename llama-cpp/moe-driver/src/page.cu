// page.cu — CUDA on-stream expert pager.
//
// Invoked from ggml-cuda's GGML_OP_CUSTOM dispatch (so the whole op stays on the
// CUDA backend — no per-layer CPU/GPU backend split, which was the decode
// slowdown). It classifies the token's selected experts on the host (the LRU
// policy is cheap), pages cold experts host->VRAM cache slots via cudaMemcpyAsync
// on the op's stream, and writes the resident slot-ids back to the device for the
// subsequent mul_mat_id.
//
// Copies are from the pageable host expert pool (no pinning — the A10G test box
// only has 15 GB RAM, less than the ~16 GB pool). Pageable async H2D is
// effectively synchronous w.r.t. the host, which is fine for correctness; the
// win is eliminating the ~96 backend-split syncs/token, not async overlap.

#include "moe_driver.h"
#include "moe_driver_internal.h"

#include <cuda_runtime.h>
#include <cstdint>
#include <vector>

using namespace moe;

extern "C" void moe_driver_page_cuda(int32_t layer,
                                     const int32_t* sel_dev,
                                     int32_t*       out_dev,
                                     int32_t        n_used,
                                     int32_t        n_tokens,
                                     void*          stream_v) {
    Driver* d = driver();
    if (!d || layer < 0 || layer >= d->n_layers) return;
    cudaStream_t stream = static_cast<cudaStream_t>(stream_v);
    LayerResidency& res = d->residency[layer];
    LayerTensors&   lt  = d->tensors[layer];
    const int n = n_used * n_tokens;

    // D2H the selected ids (tiny) so the host policy can classify them. This sync
    // is the one stall per layer — far cheaper than the 2 backend transitions the
    // CPU map_custom op caused.
    static thread_local std::vector<int32_t> h_sel, h_slots;
    h_sel.resize(n);
    h_slots.resize(n);
    cudaMemcpyAsync(h_sel.data(), sel_dev, n * sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    std::vector<LayerResidency::Fetch> fetches;
    for (int t = 0; t < n_tokens; ++t) {
        res.plan_step(h_sel.data() + (size_t) t * n_used, n_used,
                      h_slots.data() + (size_t) t * n_used, fetches);
        for (const auto& f : fetches) {
            const size_t e = (size_t) f.expert_id;
            const size_t s = (size_t) f.slot;
            cudaMemcpyAsync((char*) lt.dev_gate + s * lt.bytes_gate,
                            (char*) lt.host_gate_data + e * lt.bytes_gate,
                            lt.bytes_gate, cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync((char*) lt.dev_up + s * lt.bytes_up,
                            (char*) lt.host_up_data + e * lt.bytes_up,
                            lt.bytes_up, cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync((char*) lt.dev_down + s * lt.bytes_down,
                            (char*) lt.host_down_data + e * lt.bytes_down,
                            lt.bytes_down, cudaMemcpyHostToDevice, stream);
            d->stats.misses++;
            d->stats.bytes_paged += lt.bytes_gate + lt.bytes_up + lt.bytes_down;
        }
        d->stats.hits += (uint64_t) n_used - fetches.size();
    }

    // H2D the remapped slot-ids for the subsequent mul_mat_id. Pageable H2D is
    // host-synchronous, so h_slots is safe to reuse on the next call.
    cudaMemcpyAsync(out_dev, h_slots.data(), n * sizeof(int32_t), cudaMemcpyHostToDevice, stream);
}
