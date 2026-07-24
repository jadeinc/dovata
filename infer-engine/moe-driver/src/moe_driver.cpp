// moe_driver.cpp — host-side orchestration behind the C-ABI (moe_driver.h).
//
// This file is ggml/CUDA-FREE: it owns the driver's lifetime, config, per-layer
// residency state, and stats, and exposes the planning entry the graph hook
// calls. The actual graph emission + host->VRAM paging (moe_driver_build_ffn)
// lives in graph.cpp, which is compiled only inside the llama.cpp build where
// ggml is available; this file builds and unit-tests on any platform.

#include "moe_driver.h"
#include "moe_driver_internal.h"

#include <cstring>
#include <memory>
#include <vector>

namespace moe {

// Single process-wide driver instance (one model per daemon). Null = inactive.
std::unique_ptr<Driver> g_driver;

Driver* driver() { return g_driver.get(); }

} // namespace moe

using namespace moe;

extern "C" {

bool moe_driver_active(void) {
    return g_driver != nullptr;
}

int moe_driver_init(const struct moe_driver_config* cfg,
                    int32_t n_layers,
                    int32_t n_experts,
                    int32_t n_expert_used,
                    const int32_t* expert_order_flat) {
    // Defensive: a disabled config or missing inputs leaves the driver inactive
    // (the seam then runs the stock path) rather than failing the model load.
    if (!cfg || !cfg->enabled || !expert_order_flat) {
        return 1;
    }
    if (n_layers <= 0 || n_experts <= 0 || n_expert_used <= 0) {
        return 2;
    }
    const int32_t resident = cfg->pin + cfg->cache;
    if (cfg->pin < 0 || cfg->cache < 0 || resident <= 0 || resident > n_experts) {
        return 3;
    }

    auto drv = std::make_unique<Driver>();
    drv->n_layers = n_layers;
    drv->n_experts = n_experts;
    drv->n_expert_used = n_expert_used;
    drv->pin = cfg->pin;
    drv->cache = cfg->cache;
    drv->stats = moe_driver_stats{0, 0, 0};

    // Build per-layer residency from the ranking. We copy the order so the
    // driver doesn't depend on the caller keeping the buffer alive.
    drv->order.assign(expert_order_flat, expert_order_flat + (size_t)n_layers * n_experts);
    drv->residency.reserve(n_layers);
    for (int32_t l = 0; l < n_layers; ++l) {
        const int32_t* layer_order = drv->order.data() + (size_t)l * n_experts;
        drv->residency.emplace_back(n_experts, cfg->pin, cfg->cache, layer_order);
    }
    // Per-layer GPU/host handles, populated lazily by the graph hook. MUST be
    // sized here — moe_driver_remap indexes d->tensors[layer] (segfaults if empty).
    drv->tensors.resize(n_layers);

    g_driver = std::move(drv);
    return 0;
}

bool moe_driver_plan_split(int32_t layer,
                           int32_t n_experts,
                           const int32_t** out_order,
                           int32_t* out_resident_count) {
    Driver* d = driver();
    if (!d || layer < 0 || layer >= d->n_layers || n_experts != d->n_experts) {
        return false;
    }
    if (out_order) {
        *out_order = d->order.data() + (size_t)layer * d->n_experts;
    }
    if (out_resident_count) {
        *out_resident_count = d->pin + d->cache;
    }
    return true;
}

void moe_driver_get_stats(struct moe_driver_stats* out) {
    if (!out) return;
    Driver* d = driver();
    *out = d ? d->stats : moe_driver_stats{0, 0, 0};
}

void moe_driver_free(void) {
    g_driver.reset();
}

} // extern "C"
