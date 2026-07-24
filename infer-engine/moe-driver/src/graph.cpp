// graph.cpp — ggml graph hook for index-driven MoE paging.
//
// Compiled into the ggml-cuda target. moe_driver_remap (the build_moe_ffn seam
// entry) returns the driver's VRAM resident expert tensors (H+C experts) + a
// remapped slot-id tensor; the caller runs the stock mul_mat_id path over them.
//
// The remap+paging is emitted as a GGML_OP_CUSTOM node so it runs on the CUDA
// backend (ggml-cuda dispatches it to moe_driver_page_cuda, in page.cu) — NOT a
// ggml_map_custom CPU op, which split the graph at every layer (~96 CPU/GPU
// syncs/token) and tanked decode throughput. The custom op's fun is never called
// on CUDA; userdata carries the layer index to the dispatch.

#include "moe_driver.h"
#include "moe_driver_internal.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-cuda.h"

#include <cstdint>
#include <vector>

using namespace moe;

namespace {

// Userdata carried in the custom op's params → read by the ggml-cuda dispatch.
struct CallbackCtx { int layer; };
std::vector<CallbackCtx> g_cb_ctx;

// Never invoked on the CUDA backend (the op is dispatched to moe_driver_page_cuda).
// Present only because ggml_custom_4d requires a function pointer.
void noop_custom(ggml_tensor*, int, int, void*) {}

// Allocate this layer's resident tensors (H+C experts) in a CUDA buffer and copy
// the pinned top-H experts in. One-time per layer.
bool ensure_resident(int layer, ggml_tensor* hg, ggml_tensor* hu, ggml_tensor* hd) {
    Driver* d = driver();
    LayerTensors& lt = d->tensors[layer];
    if (lt.allocated) return true;

    const int64_t resident = d->pin + d->cache;
    ggml_init_params ip{};
    ip.mem_size   = ggml_tensor_overhead() * 4 + 256;
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;
    ggml_context* gctx = ggml_init(ip);
    if (!gctx) return false;

    ggml_tensor* rg = ggml_new_tensor_3d(gctx, hg->type, hg->ne[0], hg->ne[1], resident);
    ggml_tensor* ru = ggml_new_tensor_3d(gctx, hu->type, hu->ne[0], hu->ne[1], resident);
    ggml_tensor* rd = ggml_new_tensor_3d(gctx, hd->type, hd->ne[0], hd->ne[1], resident);

    ggml_backend_buffer_t buf =
        ggml_backend_alloc_ctx_tensors_from_buft(gctx, ggml_backend_cuda_buffer_type(0));
    if (!buf) { ggml_free(gctx); return false; }

    // Copy the pinned experts into slots 0..pin-1 (matching LayerResidency's
    // pinned-slot mapping). Per-expert stride = nb[2].
    const int32_t* order = d->order.data() + (size_t) layer * d->n_experts;
    for (int h = 0; h < d->pin; ++h) {
        const int e = order[h];
        ggml_backend_tensor_set(rg, (const char*) hg->data + (size_t) e * hg->nb[2],
                                (size_t) h * rg->nb[2], rg->nb[2]);
        ggml_backend_tensor_set(ru, (const char*) hu->data + (size_t) e * hu->nb[2],
                                (size_t) h * ru->nb[2], ru->nb[2]);
        ggml_backend_tensor_set(rd, (const char*) hd->data + (size_t) e * hd->nb[2],
                                (size_t) h * rd->nb[2], rd->nb[2]);
    }

    lt.resident_gate = rg; lt.resident_up = ru; lt.resident_down = rd;
    lt.host_gate = hg; lt.host_up = hu; lt.host_down = hd;
    lt.dev_gate = rg->data; lt.dev_up = ru->data; lt.dev_down = rd->data;
    lt.host_gate_data = hg->data; lt.host_up_data = hu->data; lt.host_down_data = hd->data;
    lt.bytes_gate = rg->nb[2]; lt.bytes_up = ru->nb[2]; lt.bytes_down = rd->nb[2];
    lt.ggml_ctx = gctx; lt.ggml_buffer = buf;
    lt.allocated = true;
    return true;
}

} // namespace

extern "C" bool moe_driver_remap(
        struct ggml_context* ctx,
        int32_t layer,
        struct ggml_tensor* host_gate,
        struct ggml_tensor* host_up,
        struct ggml_tensor* host_down,
        struct ggml_tensor* selected_experts,
        struct ggml_tensor** out_gate,
        struct ggml_tensor** out_up,
        struct ggml_tensor** out_down,
        struct ggml_tensor** out_ids) {
    Driver* d = driver();
    if (!d || layer < 0 || layer >= d->n_layers) return false;
    if (!ensure_resident(layer, host_gate, host_up, host_down)) return false;

    if ((int) g_cb_ctx.size() < d->n_layers) g_cb_ctx.resize(d->n_layers);
    g_cb_ctx[layer].layer = layer;

    LayerTensors& lt = d->tensors[layer];
    *out_gate = (ggml_tensor*) lt.resident_gate;
    *out_up   = (ggml_tensor*) lt.resident_up;
    *out_down = (ggml_tensor*) lt.resident_down;

    // GGML_OP_CUSTOM node → ggml-cuda dispatches it to moe_driver_page_cuda on
    // the CUDA stream (no backend split). Output mirrors selected_experts (i32).
    ggml_tensor* args[1] = { selected_experts };
    *out_ids = ggml_custom_4d(ctx, GGML_TYPE_I32,
                              selected_experts->ne[0], selected_experts->ne[1], 1, 1,
                              args, 1, noop_custom, 1, &g_cb_ctx[layer]);
    return true;
}
