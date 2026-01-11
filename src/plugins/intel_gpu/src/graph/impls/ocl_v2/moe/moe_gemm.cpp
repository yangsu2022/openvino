// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// clang-format off
#include "moe_gemm_gen_micro.hpp"
// clang-format on

#include "moe_gemm.hpp"
#include "moe_expert_weight_manager.hpp"

#include "../primitive_ocl_base.hpp"
#include "../utils/jitter.hpp"
#include "../utils/kernel_generator.hpp"
#include "common_utils/dispatch_utils.hpp"
#include "common_utils/jitter.hpp"
#include "moe_gemm_base.hpp"
#include "moe_gemm_inst.h"
#include "ocl_v2/utils/fused_ops_jitter.hpp"

#include <iostream>

// OTD Debug logging
#define MOE_OTD_LOG(msg) std::cout << "[MOE-OTD] " << msg << std::endl

namespace ov::intel_gpu::ocl {
namespace {

// Global OTD weight manager (shared across all moe_gemm instances)
static std::mutex g_otd_manager_mutex;
static std::unique_ptr<MoEExpertWeightManager> g_otd_manager;

MoEExpertWeightManager* get_or_create_otd_manager(cldnn::engine& engine, const MoEOTDConfig& config) {
    std::lock_guard<std::mutex> lock(g_otd_manager_mutex);
    if (!g_otd_manager || !g_otd_manager->is_enabled()) {
        g_otd_manager = std::make_unique<MoEExpertWeightManager>(engine, config);
        g_otd_manager->initialize();
    }
    return g_otd_manager.get();
}

#ifdef ENABLE_ONEDNN_FOR_GPU
inline bool is_prefill_stage(const RuntimeParams& params) {
    const auto target_seq_len = params.input_layouts[0].get_partial_shape()[0];
    const auto num_offsets = params.input_layouts[3].get_partial_shape()[0];
    if (num_offsets.is_dynamic())
        return false;
    if (target_seq_len.is_dynamic())
        return false;
    return (target_seq_len.get_length() / num_offsets.get_length()) > 1;
}
#endif

class MoEGemmImpl : public PrimitiveImplOCL {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::ocl::MoEGemmImpl)
#ifdef ENABLE_ONEDNN_FOR_GPU
    static constexpr bool prefill = true;

    Stage::Ptr regular_micro_single_token = make_stage<MoEGemmMicroGenerator>(!prefill);
    Stage::Ptr regular_micro_multi_tokens = make_stage<MoEGemmMicroGenerator>(prefill);
#endif

    // OTD state
    bool m_otd_enabled = false;
    MoEOTDConfig m_otd_config;
    uint32_t m_otd_layer_idx = 0;
    bool m_otd_is_up_projection = true;

    explicit MoEGemmImpl() : PrimitiveImplOCL(MoEGemm::get_type_info_static()) {
        MOE_OTD_LOG("MoEGemmImpl: Default constructor called (WARNING: OTD config NOT initialized!)");
    }
    explicit MoEGemmImpl(const RuntimeParams& impl_param) : MoEGemmImpl() {
        auto params = impl_param;
        MOE_OTD_LOG("XXXCREATING IMPL XXX this=" << (void*)this << ", dynamic=" << params.is_dynamic());
        GPU_DEBUG_TRACE_DETAIL << "create stages for dynamic = " << params.is_dynamic() << "\n";

        // Initialize OTD config from primitive descriptor
        auto desc = params.typed_desc<moe_gemm>();
        MOE_OTD_LOG("MoEGemmImpl: Reading descriptor - otd_enabled=" << desc->otd_enabled
                    << ", otd_layer_idx=" << desc->otd_layer_idx
                    << ", otd_is_up_projection=" << desc->otd_is_up_projection
                    << ", otd_weights_path='" << desc->otd_weights_path << "'"
                    << ", otd_num_experts=" << desc->otd_num_experts);
        if (desc->otd_enabled) {
            m_otd_enabled = true;
            m_otd_config.enabled = true;
            m_otd_config.weights_path = desc->otd_weights_path;
            m_otd_config.resident_experts = desc->otd_resident_experts;
            m_otd_layer_idx = desc->otd_layer_idx;
            m_otd_is_up_projection = desc->otd_is_up_projection;
            MOE_OTD_LOG("MoEGemmImpl: OTD enabled this=" << (void*)this << ", layer=" << m_otd_layer_idx 
                        << ", projection=" << (m_otd_is_up_projection ? "up" : "down")
                        << ", weights_path=" << m_otd_config.weights_path
                        << ", resident_experts=" << m_otd_config.resident_experts);
            GPU_DEBUG_TRACE_DETAIL << "MoEGemmImpl: OTD enabled for layer " << m_otd_layer_idx 
                                   << " (" << (m_otd_is_up_projection ? "up" : "down") << ")\n";
        } else {
            MOE_OTD_LOG("MoEGemmImpl: OTD not enabled for this layer");
        }

#ifdef ENABLE_ONEDNN_FOR_GPU
        add_stage(regular_micro_multi_tokens, params);
        add_stage(regular_micro_single_token, params);
#endif
    }

    [[nodiscard]] std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<MoEGemmImpl>(this);
    }

    // Serialize OTD state for cloning/caching
    void save(cldnn::BinaryOutputBuffer& ob) const override {
        PrimitiveImplOCL::save(ob);
        ob << m_otd_enabled;
        ob << m_otd_config.enabled;
        ob << m_otd_config.weights_path;
        ob << m_otd_config.resident_experts;
        ob << m_otd_layer_idx;
        ob << m_otd_is_up_projection;
        MOE_OTD_LOG("MoEGemmImpl::save: Serialized OTD state - layer=" << m_otd_layer_idx 
                    << ", enabled=" << m_otd_enabled);
    }

    void load(cldnn::BinaryInputBuffer& ib) override {
        PrimitiveImplOCL::load(ib);
        ib >> m_otd_enabled;
        ib >> m_otd_config.enabled;
        ib >> m_otd_config.weights_path;
        ib >> m_otd_config.resident_experts;
        ib >> m_otd_layer_idx;
        ib >> m_otd_is_up_projection;
        MOE_OTD_LOG("MoEGemmImpl::load: Deserialized OTD state - layer=" << m_otd_layer_idx 
                    << ", enabled=" << m_otd_enabled);
    }

    void update_rt_params(const primitive_inst& instance) override {
        if (m_rt_params == nullptr) {
            m_rt_params = std::make_unique<MoEGemmRuntimeParams>();
        }
        update_stages_flags(instance);
        auto rtp = static_cast<MoEGemmRuntimeParams*>(m_rt_params.get());
        rtp->num_actually_used_experts = instance.get_input_layout(moe_gemm::MoEGemmInputIdx::EXPERTS_IDS).get_shape()[0];
        GPU_DEBUG_TRACE_DETAIL << "moe_gemm :: num_actually_used_experts = " << rtp->num_actually_used_experts << "\n";
    }

    void update(primitive_inst& inst, const kernel_impl_params& impl_params) override {
        PrimitiveImplOCL::update(inst, impl_params);
        inst.update_shape_info_tensor(impl_params);
        update_rt_params(inst);
    }

    /// @brief Load required experts from disk if OTD is enabled
    void load_otd_experts_if_needed(primitive_inst& instance) {
        MOE_OTD_LOG(">>> load_otd_experts_if_needed: ENTRY - layer=" << m_otd_layer_idx 
                    << ", projection=" << (m_otd_is_up_projection ? "up" : "down") 
                    << ", otd_enabled=" << m_otd_enabled);
        
        if (!m_otd_enabled) {
            MOE_OTD_LOG("<<< load_otd_experts_if_needed: EXIT (OTD not enabled)");
            return;
        }

        MOE_OTD_LOG("load_otd_experts_if_needed: Layer " << m_otd_layer_idx 
                    << " (" << (m_otd_is_up_projection ? "up" : "down") << ")");

        auto& engine = instance.get_network().get_engine();
        auto& stream = instance.get_network().get_stream();
        MOE_OTD_LOG("load_otd_experts_if_needed: [1] Got engine and stream");
        
        // Get or create the OTD manager
        MOE_OTD_LOG("load_otd_experts_if_needed: [2] Calling get_or_create_otd_manager()...");
        auto* otd_manager = get_or_create_otd_manager(engine, m_otd_config);
        MOE_OTD_LOG("load_otd_experts_if_needed: [3] otd_manager=" << (void*)otd_manager);
        
        if (!otd_manager || !otd_manager->is_enabled()) {
            MOE_OTD_LOG("load_otd_experts_if_needed: [ERROR] OTD manager not available!");
            MOE_OTD_LOG("<<< load_otd_experts_if_needed: EXIT (no manager)");
            GPU_DEBUG_TRACE_DETAIL << "MoEGemmImpl: OTD manager not available\n";
            return;
        }
        MOE_OTD_LOG("load_otd_experts_if_needed: [4] Manager is enabled");

        // Read expert IDs from the input tensor
        MOE_OTD_LOG("load_otd_experts_if_needed: [5] Getting experts_ids memory...");
        auto experts_ids_mem = instance.dep_memory_ptr(moe_gemm::MoEGemmInputIdx::EXPERTS_IDS);
        MOE_OTD_LOG("load_otd_experts_if_needed: [6] experts_ids_mem=" << (void*)experts_ids_mem.get());
        
        if (!experts_ids_mem) {
            MOE_OTD_LOG("load_otd_experts_if_needed: [ERROR] No experts_ids memory!");
            MOE_OTD_LOG("<<< load_otd_experts_if_needed: EXIT (no experts_ids)");
            return;
        }

        MOE_OTD_LOG("load_otd_experts_if_needed: [7] Reading expert IDs from memory...");
        std::vector<int32_t> expert_ids;
        {
            cldnn::mem_lock<int32_t, cldnn::mem_lock_type::read> lock(experts_ids_mem, stream);
            auto shape = experts_ids_mem->get_layout().get_shape();
            size_t num_experts = shape[0];
            MOE_OTD_LOG("load_otd_experts_if_needed: [8] num_experts=" << num_experts);
            expert_ids.reserve(num_experts);
            for (size_t i = 0; i < num_experts; ++i) {
                expert_ids.push_back(lock[i]);
            }
        }
        MOE_OTD_LOG("load_otd_experts_if_needed: [9] Read " << expert_ids.size() << " expert IDs");

        MOE_OTD_LOG("load_otd_experts_if_needed: [10] Loading " << expert_ids.size() 
                    << " experts for layer " << m_otd_layer_idx);
        GPU_DEBUG_TRACE_DETAIL << "MoEGemmImpl: Loading " << expert_ids.size() << " experts for layer " 
                               << m_otd_layer_idx << " (" << (m_otd_is_up_projection ? "up" : "down") << ")\n";

        // Load the required experts
        MOE_OTD_LOG("load_otd_experts_if_needed: [11] Calling otd_manager->load_experts()...");
        otd_manager->load_experts(m_otd_layer_idx, expert_ids, stream, m_otd_is_up_projection);
        MOE_OTD_LOG("load_otd_experts_if_needed: [12] otd_manager->load_experts() RETURNED");
        MOE_OTD_LOG("<<< load_otd_experts_if_needed: EXIT (success)");
    }

    [[nodiscard]] event::ptr execute(const std::vector<event::ptr>& events, primitive_inst& instance) override {
#ifdef ENABLE_ONEDNN_FOR_GPU
        // CRITICAL FIX: Re-initialize OTD config from descriptor if not set
        // This handles cases where impl is created through alternative paths (e.g., dynamic shapes)
        const auto& params = *instance.get_impl_params();
        if (!m_otd_enabled) {
            auto desc = params.typed_desc<moe_gemm>();
            if (desc->otd_enabled) {
                MOE_OTD_LOG("XXXEXECUTING XXX: Re-initializing OTD config from descriptor!");
                m_otd_enabled = true;
                m_otd_config.enabled = true;
                m_otd_config.weights_path = desc->otd_weights_path;
                m_otd_config.resident_experts = desc->otd_resident_experts;
                m_otd_layer_idx = desc->otd_layer_idx;
                m_otd_is_up_projection = desc->otd_is_up_projection;
                MOE_OTD_LOG("XXXEXECUTING XXX: OTD re-initialized - layer=" << m_otd_layer_idx 
                            << ", projection=" << (m_otd_is_up_projection ? "up" : "down"));
            }
        }

        // Load OTD experts if needed
        MOE_OTD_LOG("EXECUTE: [A] About to call load_otd_experts_if_needed()...");
        load_otd_experts_if_needed(instance);
        MOE_OTD_LOG("EXECUTE: [B] load_otd_experts_if_needed() RETURNED");

        MOE_OTD_LOG("EXECUTE: [C] Checking prefill stage...");
        bool is_prefill = is_prefill_stage(params);
        MOE_OTD_LOG("EXECUTE: [D] is_prefill=" << is_prefill);
        MOE_OTD_LOG("XXXEXECUTING XXX this=" << (void*)this
                    << ", layer=" << m_otd_layer_idx 
                    << ", is_prefill=" << is_prefill 
                    << ", OTD=" << m_otd_enabled);
        MOE_OTD_LOG("EXECUTE: [E] Calling update_rt_params()...");
        update_rt_params(instance);
        MOE_OTD_LOG("EXECUTE: [F] update_rt_params() RETURNED");
        
        if (is_prefill) {
            if (has_stage(regular_micro_multi_tokens)) {
                MOE_OTD_LOG("EXECUTE: [G] Using prefill micro_multi_tokens stage, calling execute_stage()...");
                GPU_DEBUG_TRACE_DETAIL << "Execute prefill micro_multi_tokens stage" << std::endl;
                auto result = execute_stage(events, instance, regular_micro_multi_tokens);
                MOE_OTD_LOG("EXECUTE: [H] execute_stage() RETURNED");
                return result;
            } else {
                OPENVINO_THROW("Prefill stage is not available");
            }
        } else {
            MOE_OTD_LOG("EXECUTE: [G] Using single_token stage, calling execute_stage()...");
            auto result = execute_stage(events, instance, regular_micro_single_token);
            MOE_OTD_LOG("EXECUTE: [H] execute_stage() RETURNED");
            return result;
        }
#else
        OPENVINO_THROW("moe_gemm is only supported on systolic platforms.");
#endif
        return nullptr;
    }
};
}  // namespace

std::unique_ptr<primitive_impl> MoEGemm::create_impl(const program_node& node, const RuntimeParams& params) const {
    assert(node.is_type<moe_gemm>());
    
    // CRITICAL FIX: Ensure runtime params contain OTD configuration from node's primitive
    // During execution, params may be reconstructed without OTD fields, so we must use node's descriptor
    auto node_prim = node.as<moe_gemm>().get_primitive();
    
    // Create new RuntimeParams with the node's primitive descriptor (which has OTD config)
    auto updated_params = params;
    updated_params.desc = node_prim;  // Replace params descriptor with node's descriptor
    
    MOE_OTD_LOG("MoEGemm::create_impl: Using node's descriptor with OTD config - otd_enabled=" 
                << node_prim->otd_enabled << ", layer=" << node_prim->otd_layer_idx
                << ", projection=" << (node_prim->otd_is_up_projection ? "up" : "down"));
    
    return std::make_unique<MoEGemmImpl>(updated_params);
}
}  // namespace ov::intel_gpu::ocl

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::moe_gemm)
BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::ocl::MoEGemmImpl)
