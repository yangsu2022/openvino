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
#include "intel_gpu/runtime/debug_configuration.hpp"

#include <iostream>
#include <cstdlib>

// OTD Debug logging - controlled by OV_MOE_OTD_DEBUG environment variable
// Set OV_MOE_OTD_DEBUG=1 to enable MOE OTD debug output, unset or 0 to disable
static bool moe_otd_debug_enabled() {
    static bool initialized = false;
    static bool enabled = false;
    if (!initialized) {
        const char* env = std::getenv("OV_MOE_OTD_DEBUG");
        enabled = env && (std::string(env) == "1" || std::string(env) == "true");
        initialized = true;
    }
    return enabled;
}
#define MOE_OTD_LOG(msg) if (moe_otd_debug_enabled()) std::cout << "[MOE-OTD] " << msg << std::endl

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
    
    // OTD buffer cache - stores the loaded weight buffer for use during kernel execution
    // This is set by load_otd_experts_if_needed() and used by get_arguments()
    mutable cldnn::memory::ptr m_otd_weight_buffer = nullptr;

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

    /// @brief Override get_arguments to replace weight input with OTD buffer when OTD is enabled
    /// This is the CRITICAL connection between the OTD loaded weights and kernel execution
    [[nodiscard]] cldnn::kernel_arguments_data get_arguments(const cldnn::primitive_inst& instance) const override {
        // Get base arguments from parent class
        cldnn::kernel_arguments_data args = PrimitiveImplOCL::get_arguments(instance);
        
        // If OTD is enabled and we have a cached weight buffer, replace the weight input
        if (m_otd_enabled && m_otd_weight_buffer) {
            MOE_OTD_LOG("get_arguments: OTD OVERRIDE - Replacing weight input with OTD buffer");
            MOE_OTD_LOG("get_arguments:   Layer=" << m_otd_layer_idx 
                        << ", projection=" << (m_otd_is_up_projection ? "up" : "down"));
            MOE_OTD_LOG("get_arguments:   Original weight buffer: " << (void*)args.inputs[moe_gemm::MoEGemmInputIdx::WEIGHT].get()
                        << " (size=" << (args.inputs[moe_gemm::MoEGemmInputIdx::WEIGHT] ? args.inputs[moe_gemm::MoEGemmInputIdx::WEIGHT]->size() : 0) << ")");
            MOE_OTD_LOG("get_arguments:   OTD weight buffer: " << (void*)m_otd_weight_buffer.get()
                        << " (size=" << m_otd_weight_buffer->size() << ")");
            
            // Replace the weight input with the OTD buffer containing loaded experts
            args.inputs[moe_gemm::MoEGemmInputIdx::WEIGHT] = m_otd_weight_buffer;
            
            MOE_OTD_LOG("get_arguments: Weight input REPLACED with OTD buffer successfully");
        }
        
        return args;
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
        std::vector<int32_t> expert_ids_raw;  // May contain duplicates
        std::vector<int32_t> expert_ids;      // Unique expert IDs
        std::vector<int32_t> global_expert_ids;  // Global expert IDs (layer_idx * 32 + local_id)
        {
            cldnn::mem_lock<int32_t, cldnn::mem_lock_type::read> lock(experts_ids_mem, stream);
            auto shape = experts_ids_mem->get_layout().get_shape();
            size_t num_expert_selections = shape[0];  // Total number of expert selections (tokens × top-k)
            MOE_OTD_LOG("load_otd_experts_if_needed: [8] Total expert selections (raw)=" << num_expert_selections);
            expert_ids_raw.reserve(num_expert_selections);
            for (size_t i = 0; i < num_expert_selections; ++i) {
                expert_ids_raw.push_back(lock[i]);
            }
            
            // Remove duplicates to get unique expert IDs
            std::unordered_set<int32_t> unique_set(expert_ids_raw.begin(), expert_ids_raw.end());
            expert_ids.assign(unique_set.begin(), unique_set.end());
            
            // Convert local expert IDs to global IDs for statistical analysis
            // Global expert ID = layer_idx * 32 + local_expert_id
            int32_t global_offset = m_otd_layer_idx * 32;
            global_expert_ids.reserve(expert_ids.size());
            for (int32_t local_id : expert_ids) {
                global_expert_ids.push_back(global_offset + local_id);
            }
        }
        MOE_OTD_LOG("load_otd_experts_if_needed: [9] Total selections=" << expert_ids_raw.size() 
                    << ", Unique experts=" << expert_ids.size() 
                    << " (dedup ratio: " << (expert_ids_raw.size() / (float)expert_ids.size()) << "x)");
        
        // Log all unique expert IDs (sorted for better readability)
        // Show both local IDs (used by model) and global IDs (for statistical analysis)
        std::vector<int32_t> sorted_local_ids = expert_ids;
        std::sort(sorted_local_ids.begin(), sorted_local_ids.end());
        
        std::vector<int32_t> sorted_global_ids = global_expert_ids;
        std::sort(sorted_global_ids.begin(), sorted_global_ids.end());
        
        std::ostringstream local_ids_str, global_ids_str;
        local_ids_str << "[";
        global_ids_str << "[";
        for (size_t i = 0; i < sorted_local_ids.size(); ++i) {
            if (i > 0) {
                local_ids_str << ", ";
                global_ids_str << ", ";
            }
            local_ids_str << sorted_local_ids[i];
            global_ids_str << sorted_global_ids[i];
        }
        local_ids_str << "]";
        global_ids_str << "]";
        MOE_OTD_LOG("load_otd_experts_if_needed: [9.1] Local expert IDs (used by model): " << local_ids_str.str());
        MOE_OTD_LOG("load_otd_experts_if_needed: [9.2] Global expert IDs (for statistics): " << global_ids_str.str());

        MOE_OTD_LOG("load_otd_experts_if_needed: [10] Loading " << global_expert_ids.size() 
                    << " experts for layer " << m_otd_layer_idx);
        GPU_DEBUG_TRACE_DETAIL << "MoEGemmImpl: Loading " << global_expert_ids.size() << " experts for layer " 
                               << m_otd_layer_idx << " (" << (m_otd_is_up_projection ? "up" : "down") << ")\n";

        // CRITICAL FIX: Pass GLOBAL expert IDs (not local) to load_experts()
        // This ensures LRU cache keys are unique across all layers
        // Previously: Layer 0 expert_id=5 and Layer 1 expert_id=5 would collide
        // Now: Layer 0 uses global_id=5, Layer 1 uses global_id=37 (32+5)
        MOE_OTD_LOG("load_otd_experts_if_needed: [11] Calling otd_manager->load_experts() with GLOBAL expert IDs...");
        otd_manager->load_experts(m_otd_layer_idx, global_expert_ids, stream, m_otd_is_up_projection);
        MOE_OTD_LOG("load_otd_experts_if_needed: [12] otd_manager->load_experts() RETURNED");
        
        // CRITICAL FIX: Get the LAYER-SPECIFIC weight buffer (subbuffer view)
        // This returns a subbuffer starting at layer_idx * 32 * expert_size,
        // allowing the kernel to access weights using local expert IDs (0-31)
        m_otd_weight_buffer = otd_manager->get_weight_buffer_for_layer(m_otd_layer_idx, m_otd_is_up_projection);
        MOE_OTD_LOG("load_otd_experts_if_needed: [13] Cached OTD weight buffer for layer " << m_otd_layer_idx 
                    << ": " << (void*)m_otd_weight_buffer.get()
                    << " (size=" << (m_otd_weight_buffer ? m_otd_weight_buffer->size() : 0) << " bytes)");
        
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
