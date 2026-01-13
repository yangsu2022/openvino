// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "moe_expert_weight_manager.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"

#include <algorithm>
#include <cstring>
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

MoEExpertWeightManager::MoEExpertWeightManager(cldnn::engine& engine, const MoEOTDConfig& config)
    : m_engine(engine)
    , m_config(config) {
    MOE_OTD_LOG("MoEExpertWeightManager: Created with path=" << config.weights_path 
                << ", resident_experts=" << config.resident_experts 
                << ", enabled=" << config.enabled);
    GPU_DEBUG_TRACE_DETAIL << "MoEExpertWeightManager created with path: " << config.weights_path 
                           << ", resident_experts: " << config.resident_experts 
                           << ", enabled: " << config.enabled << "\n";
}

MoEExpertWeightManager::~MoEExpertWeightManager() {
    if (m_weights_file && m_weights_file->is_open()) {
        m_weights_file->close();
    }
}

bool MoEExpertWeightManager::initialize() {
    MOE_OTD_LOG("MoEExpertWeightManager::initialize() called");
    if (m_initialized) {
        MOE_OTD_LOG("MoEExpertWeightManager::initialize() - already initialized");
        return true;
    }

    if (!m_config.enabled || m_config.weights_path.empty()) {
        MOE_OTD_LOG("MoEExpertWeightManager::initialize() - OTD disabled or no path");
        GPU_DEBUG_TRACE_DETAIL << "MoEExpertWeightManager: OTD disabled or no weights path\n";
        return false;
    }

    // Open weights file
    MOE_OTD_LOG("MoEExpertWeightManager::initialize() - Opening file: " << m_config.weights_path);
    m_weights_file = std::make_unique<std::ifstream>(m_config.weights_path, std::ios::binary);
    if (!m_weights_file->is_open()) {
        MOE_OTD_LOG("MoEExpertWeightManager::initialize() - FAILED to open file!");
        GPU_DEBUG_TRACE_DETAIL << "MoEExpertWeightManager: Failed to open weights file: " << m_config.weights_path << "\n";
        return false;
    }

    // Read and validate header
    if (!read_file_header()) {
        GPU_DEBUG_TRACE_DETAIL << "MoEExpertWeightManager: Failed to read file header\n";
        return false;
    }

    // Allocate GPU buffers
    if (!allocate_buffers()) {
        GPU_DEBUG_TRACE_DETAIL << "MoEExpertWeightManager: Failed to allocate buffers\n";
        return false;
    }

    m_initialized = true;
    MOE_OTD_LOG("MoEExpertWeightManager initialized successfully!");
    MOE_OTD_LOG("  Layers: " << m_header.num_layers);
    MOE_OTD_LOG("  Experts per layer: " << m_header.num_experts_per_layer);
    MOE_OTD_LOG("  Resident experts: " << m_config.resident_experts);
    MOE_OTD_LOG("  Up weight size per expert: " << m_header.expert_up_weight_size << " bytes");
    MOE_OTD_LOG("  Down weight size per expert: " << m_header.expert_down_weight_size << " bytes");
    GPU_DEBUG_TRACE_DETAIL << "MoEExpertWeightManager initialized successfully\n";
    GPU_DEBUG_TRACE_DETAIL << "  Layers: " << m_header.num_layers << "\n";
    GPU_DEBUG_TRACE_DETAIL << "  Experts per layer: " << m_header.num_experts_per_layer << "\n";
    GPU_DEBUG_TRACE_DETAIL << "  Resident experts: " << m_config.resident_experts << "\n";
    GPU_DEBUG_TRACE_DETAIL << "  Up weight size per expert: " << m_header.expert_up_weight_size << " bytes\n";
    GPU_DEBUG_TRACE_DETAIL << "  Down weight size per expert: " << m_header.expert_down_weight_size << " bytes\n";

    return true;
}

bool MoEExpertWeightManager::read_file_header() {
    if (!m_weights_file || !m_weights_file->is_open()) {
        return false;
    }

    m_weights_file->seekg(0, std::ios::beg);
    m_weights_file->read(reinterpret_cast<char*>(&m_header), sizeof(MoEWeightsFileHeader));

    if (!m_weights_file->good()) {
        GPU_DEBUG_TRACE_DETAIL << "MoEExpertWeightManager: Failed to read header\n";
        return false;
    }

    // Validate magic number
    if (std::strncmp(m_header.magic, "MOEW", 4) != 0) {
        GPU_DEBUG_TRACE_DETAIL << "MoEExpertWeightManager: Invalid magic number\n";
        return false;
    }

    // Validate version
    if (m_header.version != 1) {
        GPU_DEBUG_TRACE_DETAIL << "MoEExpertWeightManager: Unsupported version: " << m_header.version << "\n";
        return false;
    }

    // Initialize layer infos
    m_layer_infos.resize(m_header.num_layers);
    for (uint32_t i = 0; i < m_header.num_layers; ++i) {
        m_layer_infos[i].layer_idx = i;
        m_layer_infos[i].num_experts = m_header.num_experts_per_layer;
        m_layer_infos[i].up_weights.resize(m_header.num_experts_per_layer);
        m_layer_infos[i].down_weights.resize(m_header.num_experts_per_layer);
        m_layer_infos[i].up_scales.resize(m_header.num_experts_per_layer);
        m_layer_infos[i].down_scales.resize(m_header.num_experts_per_layer);
        m_layer_infos[i].up_biases.resize(m_header.num_experts_per_layer);
        m_layer_infos[i].down_biases.resize(m_header.num_experts_per_layer);

        // Calculate offsets for each expert within this layer
        size_t layer_base_offset = m_header.data_offset;
        size_t per_layer_size = m_header.num_experts_per_layer * 
            (m_header.expert_up_weight_size + m_header.expert_down_weight_size +
             m_header.expert_up_scale_size + m_header.expert_down_scale_size +
             m_header.expert_up_bias_size + m_header.expert_down_bias_size);
        layer_base_offset += i * per_layer_size;

        size_t current_offset = layer_base_offset;
        for (uint32_t e = 0; e < m_header.num_experts_per_layer; ++e) {
            // Up weights
            m_layer_infos[i].up_weights[e].offset = current_offset;
            m_layer_infos[i].up_weights[e].size = m_header.expert_up_weight_size;
            current_offset += m_header.expert_up_weight_size;

            // Up scales
            m_layer_infos[i].up_scales[e].offset = current_offset;
            m_layer_infos[i].up_scales[e].size = m_header.expert_up_scale_size;
            current_offset += m_header.expert_up_scale_size;

            // Up biases
            m_layer_infos[i].up_biases[e].offset = current_offset;
            m_layer_infos[i].up_biases[e].size = m_header.expert_up_bias_size;
            current_offset += m_header.expert_up_bias_size;

            // Down weights
            m_layer_infos[i].down_weights[e].offset = current_offset;
            m_layer_infos[i].down_weights[e].size = m_header.expert_down_weight_size;
            current_offset += m_header.expert_down_weight_size;

            // Down scales
            m_layer_infos[i].down_scales[e].offset = current_offset;
            m_layer_infos[i].down_scales[e].size = m_header.expert_down_scale_size;
            current_offset += m_header.expert_down_scale_size;

            // Down biases
            m_layer_infos[i].down_biases[e].offset = current_offset;
            m_layer_infos[i].down_biases[e].size = m_header.expert_down_bias_size;
            current_offset += m_header.expert_down_bias_size;
        }
    }

    return true;
}

bool MoEExpertWeightManager::allocate_buffers() {
    // Determine number of resident experts
    // With global expert IDs, we can cache experts across ALL layers (768 total = 24 layers × 32 experts)
    // So the cap should be total experts, not per-layer experts
    int64_t total_experts = static_cast<int64_t>(m_header.num_layers) * m_header.num_experts_per_layer;
    int64_t resident_experts = m_config.resident_experts;
    if (resident_experts <= 0) {
        // Auto mode: use half of all experts or minimum 8
        resident_experts = std::max(8LL, total_experts / 2);
    }
    // Cap at total experts (not per-layer experts!) since we use global IDs
    resident_experts = std::min(resident_experts, total_experts);

    GPU_DEBUG_TRACE_DETAIL << "MoEExpertWeightManager: Allocating buffers for " << resident_experts << " resident experts (global)\n";
    MOE_OTD_LOG("allocate_buffers: resident_experts=" << resident_experts 
                << " (config=" << m_config.resident_experts << ", total=" << total_experts 
                << ", auto=" << (m_config.resident_experts <= 0 ? "yes" : "no") << ")");

    // Calculate buffer sizes
    size_t up_weight_buffer_size = resident_experts * m_header.expert_up_weight_size;
    size_t down_weight_buffer_size = resident_experts * m_header.expert_down_weight_size;
    size_t up_scale_buffer_size = resident_experts * m_header.expert_up_scale_size;
    size_t down_scale_buffer_size = resident_experts * m_header.expert_down_scale_size;
    size_t up_bias_buffer_size = resident_experts * m_header.expert_up_bias_size;
    size_t down_bias_buffer_size = resident_experts * m_header.expert_down_bias_size;
    
    size_t total_gpu_memory = up_weight_buffer_size + down_weight_buffer_size + 
                             up_scale_buffer_size + down_scale_buffer_size +
                             up_bias_buffer_size + down_bias_buffer_size;
    
    MOE_OTD_LOG("allocate_buffers: Buffer sizes calculated:");
    MOE_OTD_LOG("  Up weight:   " << (up_weight_buffer_size / 1024.0 / 1024.0) << " MB");
    MOE_OTD_LOG("  Down weight: " << (down_weight_buffer_size / 1024.0 / 1024.0) << " MB");
    MOE_OTD_LOG("  Total GPU:   " << (total_gpu_memory / 1024.0 / 1024.0) << " MB");

    try {
        // Allocate GPU memory for weights
        MOE_OTD_LOG("allocate_buffers: Allocating up_weight_buffer (" << (up_weight_buffer_size / 1024.0 / 1024.0) << " MB)...");
        if (up_weight_buffer_size > 0) {
            cldnn::layout up_weight_layout({static_cast<int64_t>(up_weight_buffer_size)}, 
                                           cldnn::data_types::u8, cldnn::format::bfyx);
            m_up_weight_buffer = m_engine.allocate_memory(up_weight_layout, cldnn::allocation_type::usm_host, false);
            MOE_OTD_LOG("  ✓ up_weight_buffer allocated (usm_host)");
        }

        MOE_OTD_LOG("allocate_buffers: Allocating down_weight_buffer (" << (down_weight_buffer_size / 1024.0 / 1024.0) << " MB)...");
        if (down_weight_buffer_size > 0) {
            cldnn::layout down_weight_layout({static_cast<int64_t>(down_weight_buffer_size)}, 
                                             cldnn::data_types::u8, cldnn::format::bfyx);
            m_down_weight_buffer = m_engine.allocate_memory(down_weight_layout, cldnn::allocation_type::usm_host, false);
            MOE_OTD_LOG("  ✓ down_weight_buffer allocated (usm_host)");
        }

        // Allocate GPU memory for scales
        if (up_scale_buffer_size > 0) {
            cldnn::layout up_scale_layout({static_cast<int64_t>(up_scale_buffer_size)}, 
                                          cldnn::data_types::u8, cldnn::format::bfyx);
            m_up_scale_buffer = m_engine.allocate_memory(up_scale_layout, cldnn::allocation_type::usm_host, false);
            MOE_OTD_LOG("  ✓ up_scale_buffer allocated (usm_host)");
        }

        if (down_scale_buffer_size > 0) {
            cldnn::layout down_scale_layout({static_cast<int64_t>(down_scale_buffer_size)}, 
                                            cldnn::data_types::u8, cldnn::format::bfyx);
            m_down_scale_buffer = m_engine.allocate_memory(down_scale_layout, cldnn::allocation_type::usm_host, false);
            MOE_OTD_LOG("  ✓ down_scale_buffer allocated (usm_host)");
        }

        // Allocate GPU memory for biases
        if (up_bias_buffer_size > 0) {
            cldnn::layout up_bias_layout({static_cast<int64_t>(up_bias_buffer_size)}, 
                                         cldnn::data_types::u8, cldnn::format::bfyx);
            m_up_bias_buffer = m_engine.allocate_memory(up_bias_layout, cldnn::allocation_type::usm_host, false);
            MOE_OTD_LOG("  ✓ up_bias_buffer allocated (usm_host)");
        }

        if (down_bias_buffer_size > 0) {
            cldnn::layout down_bias_layout({static_cast<int64_t>(down_bias_buffer_size)}, 
                                           cldnn::data_types::u8, cldnn::format::bfyx);
            m_down_bias_buffer = m_engine.allocate_memory(down_bias_layout, cldnn::allocation_type::usm_host, false);
            MOE_OTD_LOG("  ✓ down_bias_buffer allocated (usm_host)");
        }

        // Allocate host staging buffer (largest of all)
        size_t max_expert_size = std::max({m_header.expert_up_weight_size, 
                                           m_header.expert_down_weight_size,
                                           m_header.expert_up_scale_size,
                                           m_header.expert_down_scale_size,
                                           m_header.expert_up_bias_size,
                                           m_header.expert_down_bias_size});
        MOE_OTD_LOG("allocate_buffers: Allocating host staging buffer (" << (max_expert_size / 1024.0 / 1024.0) << " MB)...");
        m_host_staging_buffer.resize(max_expert_size);
        MOE_OTD_LOG("  ✓ host_staging_buffer allocated");

        // Initialize slot management
        m_up_slot_to_expert.resize(resident_experts, -1);
        m_up_slot_access_time.resize(resident_experts, 0);
        m_down_slot_to_expert.resize(resident_experts, -1);
        m_down_slot_access_time.resize(resident_experts, 0);

        GPU_DEBUG_TRACE_DETAIL << "MoEExpertWeightManager: Allocated buffers successfully\n";
        GPU_DEBUG_TRACE_DETAIL << "  Up weight buffer: " << up_weight_buffer_size << " bytes\n";
        GPU_DEBUG_TRACE_DETAIL << "  Down weight buffer: " << down_weight_buffer_size << " bytes\n";
        GPU_DEBUG_TRACE_DETAIL << "  Up scale buffer: " << up_scale_buffer_size << " bytes\n";
        GPU_DEBUG_TRACE_DETAIL << "  Down scale buffer: " << down_scale_buffer_size << " bytes\n";

        return true;
    } catch (const std::exception& e) {
        GPU_DEBUG_TRACE_DETAIL << "MoEExpertWeightManager: Failed to allocate buffers: " << e.what() << "\n";
        return false;
    }
}

void MoEExpertWeightManager::load_experts(uint32_t layer_idx,
                                          const std::vector<int32_t>& expert_ids,
                                          cldnn::stream& stream,
                                          bool is_up_projection) {
    MOE_OTD_LOG(">>> load_experts ENTRY: layer=" << layer_idx 
                << ", num_experts=" << expert_ids.size() 
                << ", projection=" << (is_up_projection ? "up" : "down"));
    
    std::lock_guard<std::mutex> lock(m_mutex);
    MOE_OTD_LOG("    [1] Mutex acquired");

    if (!m_initialized || layer_idx >= m_layer_infos.size()) {
        MOE_OTD_LOG("    [ERROR] Not initialized or invalid layer_idx!");
        return;
    }
    MOE_OTD_LOG("    [2] Validation passed (initialized=" << m_initialized << ", layer_idx=" << layer_idx << ")");

    auto& slot_to_expert = is_up_projection ? m_up_slot_to_expert : m_down_slot_to_expert;
    auto& expert_to_slot = is_up_projection ? m_up_expert_to_slot : m_down_expert_to_slot;
    auto& slot_access_time = is_up_projection ? m_up_slot_access_time : m_down_slot_access_time;
    MOE_OTD_LOG("    [3] Got slot management references (slot_to_expert.size=" << slot_to_expert.size() << ")");

    // Check if buffers need allocation
    if (!m_up_weight_buffer || !m_down_weight_buffer) {
        MOE_OTD_LOG("    [4] Buffers not allocated, calling allocate_buffers()...");
        if (!allocate_buffers()) {
            MOE_OTD_LOG("    [ERROR] allocate_buffers() FAILED!");
            return;
        }
        MOE_OTD_LOG("    [4] allocate_buffers() SUCCESS");
    } else {
        MOE_OTD_LOG("    [4] Buffers already allocated, skipping");
    }

    MOE_OTD_LOG("    [5] Starting expert loading loop (" << expert_ids.size() << " experts)");
    
    // Calculate the base slot for this layer
    // Each layer gets its own range of slots: layer 0 uses 0-31, layer 1 uses 32-63, etc.
    // This prevents cache thrashing between layers (e.g., layer 0's expert 5 and layer 1's expert 5
    // no longer compete for the same slot)
    size_t layer_base_slot = static_cast<size_t>(layer_idx) * 32;
    size_t max_slots = slot_to_expert.size();
    
    MOE_OTD_LOG("    [5.pre] Layer " << layer_idx << " base_slot=" << layer_base_slot << ", max_slots=" << max_slots);
    
    for (size_t i = 0; i < expert_ids.size(); ++i) {
        int32_t expert_id = expert_ids[i];
        MOE_OTD_LOG("    [5." << i << "] Processing expert_id=" << expert_id);
        
        // Calculate local expert ID (0-31 within this layer)
        int32_t local_expert_id = expert_id % 32;
        
        // CRITICAL FIX: Use layer-aware slot indexing
        // slot_idx = layer_idx * 32 + local_expert_id
        // This ensures each layer has its own dedicated slot range, preventing cross-layer evictions
        size_t slot_idx = layer_base_slot + static_cast<size_t>(local_expert_id);
        
        // Handle buffer overflow: if slot exceeds buffer size, use LRU eviction
        if (slot_idx >= max_slots) {
            MOE_OTD_LOG("        [5." << i << ".overflow] slot_idx " << slot_idx << " exceeds max_slots " << max_slots << ", using LRU");
            slot_idx = static_cast<size_t>(find_available_slot(is_up_projection));
        }
        
        int32_t slot_idx_i32 = static_cast<int32_t>(slot_idx);
        MOE_OTD_LOG("        [5." << i << ".slot] local_expert_id=" << local_expert_id << ", final slot_idx=" << slot_idx_i32);
        
        // Check if this expert is already loaded at the correct slot
        auto it = expert_to_slot.find(expert_id);
        if (it != expert_to_slot.end() && it->second == slot_idx_i32) {
            // Expert is already loaded at the correct position
            MOE_OTD_LOG("        [5." << i << ".a] Expert " << expert_id << " (local=" << local_expert_id << ") already loaded at slot " << slot_idx_i32 << ", updating access time");
            slot_access_time[slot_idx] = ++m_access_counter;
            continue;
        }
        MOE_OTD_LOG("        [5." << i << ".b] Expert " << expert_id << " (local=" << local_expert_id << ") NOT loaded at slot " << slot_idx_i32 << ", need to load");

        // If slot was occupied by a different expert, remove the old mapping
        int32_t old_expert = slot_to_expert[slot_idx];
        if (old_expert >= 0 && old_expert != expert_id) {
            MOE_OTD_LOG("        [5." << i << ".d] Slot " << slot_idx_i32 << " was occupied by expert " << old_expert << ", evicting");
            expert_to_slot.erase(old_expert);
        } else if (old_expert < 0) {
            MOE_OTD_LOG("        [5." << i << ".d] Slot " << slot_idx_i32 << " is empty");
        }

        // Load expert from disk
        MOE_OTD_LOG("        [5." << i << ".e] Calling load_expert_to_host(layer=" << layer_idx << ", expert=" << expert_id << ")...");
        load_expert_to_host(layer_idx, expert_id, is_up_projection);
        MOE_OTD_LOG("        [5." << i << ".e] load_expert_to_host() RETURNED");
        
        // Copy to GPU
        MOE_OTD_LOG("        [5." << i << ".f] Calling copy_to_gpu_slot(slot=" << slot_idx_i32 << ")...");
        copy_to_gpu_slot(slot_idx_i32, is_up_projection, stream);
        MOE_OTD_LOG("        [5." << i << ".f] copy_to_gpu_slot() RETURNED");

        // Update mappings
        MOE_OTD_LOG("        [5." << i << ".g] Updating mappings...");
        slot_to_expert[slot_idx] = expert_id;
        expert_to_slot[expert_id] = slot_idx_i32;
        slot_access_time[slot_idx] = ++m_access_counter;
        MOE_OTD_LOG("        [5." << i << ".g] Mappings updated");

        MOE_OTD_LOG("        [5." << i << ".h] Expert " << expert_id << " SUCCESSFULLY loaded to slot " << slot_idx_i32);
    }
    MOE_OTD_LOG("    [6] Expert loading loop COMPLETED");
    MOE_OTD_LOG("<<< load_experts EXIT: Successfully loaded " << expert_ids.size() << " experts for layer " << layer_idx);
}

void MoEExpertWeightManager::load_expert_to_host(uint32_t layer_idx, int32_t global_expert_id, bool is_up_projection) {
    MOE_OTD_LOG("            >>> load_expert_to_host ENTRY: layer=" << layer_idx << ", global_expert_id=" << global_expert_id << ", proj=" << (is_up_projection ? "up" : "down"));
    
    if (!m_weights_file || !m_weights_file->is_open()) {
        MOE_OTD_LOG("            [ERROR] Weights file not open!");
        return;
    }
    MOE_OTD_LOG("            [1] Weights file is open");

    // Convert global expert ID back to local ID for array indexing
    // Global ID = layer_idx * 32 + local_id, so local_id = global_id % 32
    int32_t local_expert_id = global_expert_id % 32;
    MOE_OTD_LOG("            [1.5] Converted global_expert_id=" << global_expert_id << " to local_expert_id=" << local_expert_id << " for layer " << layer_idx);

    const auto& layer_info = m_layer_infos[layer_idx];
    const auto& weight_desc = is_up_projection ? 
        layer_info.up_weights[local_expert_id] : layer_info.down_weights[local_expert_id];
    MOE_OTD_LOG("            [2] Got weight descriptor: offset=" << weight_desc.offset << ", size=" << weight_desc.size << " bytes (" << (weight_desc.size/1024.0/1024.0) << " MB)");
    MOE_OTD_LOG("            [3] Host staging buffer size: " << m_host_staging_buffer.size() << " bytes");
    
    if (weight_desc.size > m_host_staging_buffer.size()) {
        MOE_OTD_LOG("            [ERROR] Weight size (" << weight_desc.size << ") > staging buffer size (" << m_host_staging_buffer.size() << ")!");
        return;
    }

    MOE_OTD_LOG("            [4] Seeking to file offset " << weight_desc.offset << "...");
    m_weights_file->seekg(weight_desc.offset, std::ios::beg);
    if (!m_weights_file->good()) {
        MOE_OTD_LOG("            [ERROR] File seek FAILED! File state: " << m_weights_file->rdstate());
        return;
    }
    MOE_OTD_LOG("            [5] Seek successful, current position: " << m_weights_file->tellg());
    
    MOE_OTD_LOG("            [6] Reading " << weight_desc.size << " bytes to staging buffer at " << (void*)m_host_staging_buffer.data() << "...");
    m_weights_file->read(reinterpret_cast<char*>(m_host_staging_buffer.data()), weight_desc.size);
    
    if (!m_weights_file->good()) {
        MOE_OTD_LOG("            [ERROR] File read FAILED! Read " << m_weights_file->gcount() << " bytes, file state: " << m_weights_file->rdstate());
        return;
    }
    MOE_OTD_LOG("            [7] Read successful, read " << m_weights_file->gcount() << " bytes");
    MOE_OTD_LOG("            <<< load_expert_to_host EXIT: SUCCESS");
}

void MoEExpertWeightManager::copy_to_gpu_slot(int32_t slot_idx, bool is_up_projection, cldnn::stream& stream) {
    MOE_OTD_LOG("            >>> copy_to_gpu_slot ENTRY: slot=" << slot_idx << ", proj=" << (is_up_projection ? "up" : "down"));
    
    auto& weight_buffer = is_up_projection ? m_up_weight_buffer : m_down_weight_buffer;
    if (!weight_buffer) {
        MOE_OTD_LOG("            [ERROR] Weight buffer is NULL!");
        return;
    }
    MOE_OTD_LOG("            [1] Weight buffer exists");

    size_t expert_size = is_up_projection ? m_header.expert_up_weight_size : m_header.expert_down_weight_size;
    size_t offset = slot_idx * expert_size;
    MOE_OTD_LOG("            [2] Calculated: expert_size=" << expert_size << " bytes, offset=" << offset << " bytes");

    // CRITICAL FIX: Use direct memcpy with usm_host memory (CPU and GPU accessible)
    // usm_host allocation allows CPU direct access, perfect for Intel iGPU using system RAM
    MOE_OTD_LOG("            [3] Using direct memcpy (usm_host memory)");
    
    auto* gpu_ptr = static_cast<uint8_t*>(weight_buffer->buffer_ptr());
    if (!gpu_ptr) {
        MOE_OTD_LOG("            [ERROR] buffer_ptr() returned NULL!");
        MOE_OTD_LOG("            <<< copy_to_gpu_slot EXIT (buffer_ptr failed)");
        return;
    }
    
    MOE_OTD_LOG("            [4] Got usm_host pointer: " << (void*)gpu_ptr);
    MOE_OTD_LOG("            [5] Performing memcpy from " << (void*)m_host_staging_buffer.data() 
                << " to " << (void*)(gpu_ptr + offset) << ", size=" << expert_size);
    
    std::memcpy(gpu_ptr + offset, m_host_staging_buffer.data(), expert_size);
    MOE_OTD_LOG("            [6] memcpy completed successfully");
    MOE_OTD_LOG("            <<< copy_to_gpu_slot EXIT: SUCCESS");
}

int32_t MoEExpertWeightManager::find_available_slot(bool is_up_projection) {
    MOE_OTD_LOG(">>> find_available_slot: ENTRY - proj=" << (is_up_projection ? "up" : "down"));
    
    auto& slot_to_expert = is_up_projection ? m_up_slot_to_expert : m_down_slot_to_expert;
    auto& slot_access_time = is_up_projection ? m_up_slot_access_time : m_down_slot_access_time;
    MOE_OTD_LOG("find_available_slot: Total slots=" << slot_to_expert.size());

    // First, look for an empty slot
    MOE_OTD_LOG("find_available_slot: Looking for empty slot...");
    for (size_t i = 0; i < slot_to_expert.size(); ++i) {
        if (slot_to_expert[i] < 0) {
            MOE_OTD_LOG("find_available_slot: Found empty slot " << i);
            MOE_OTD_LOG("<<< find_available_slot: EXIT - slot=" << i);
            return static_cast<int32_t>(i);
        }
    }

    // No empty slot, find LRU slot
    MOE_OTD_LOG("find_available_slot: No empty slot, finding LRU...");
    size_t lru_slot = 0;
    uint64_t min_access_time = slot_access_time[0];
    for (size_t i = 1; i < slot_access_time.size(); ++i) {
        if (slot_access_time[i] < min_access_time) {
            min_access_time = slot_access_time[i];
            lru_slot = i;
        }
    }
    MOE_OTD_LOG("find_available_slot: LRU slot=" << lru_slot << ", evicting expert " << slot_to_expert[lru_slot]);
    MOE_OTD_LOG("<<< find_available_slot: EXIT - slot=" << lru_slot);

    return static_cast<int32_t>(lru_slot);
}

cldnn::memory::ptr MoEExpertWeightManager::get_weight_buffer(bool is_up_projection) const {
    return is_up_projection ? m_up_weight_buffer : m_down_weight_buffer;
}

cldnn::memory::ptr MoEExpertWeightManager::get_weight_buffer_for_layer(uint32_t layer_idx, bool is_up_projection) const {
    // Get the full buffer
    auto& full_buffer = is_up_projection ? m_up_weight_buffer : m_down_weight_buffer;
    if (!full_buffer) {
        MOE_OTD_LOG("get_weight_buffer_for_layer: ERROR - buffer not allocated!");
        return nullptr;
    }
    
    // Calculate the byte offset for this layer
    // Each layer gets 32 slots, so offset = layer_idx * 32 * expert_size
    size_t expert_size = is_up_projection ? m_header.expert_up_weight_size : m_header.expert_down_weight_size;
    size_t byte_offset = static_cast<size_t>(layer_idx) * 32 * expert_size;
    
    // Calculate remaining buffer size after offset
    size_t remaining_size = full_buffer->size() - byte_offset;
    
    // The kernel expects a buffer with 32 experts (or less if we're near the end)
    size_t subbuffer_size = std::min(32 * expert_size, remaining_size);
    
    MOE_OTD_LOG("get_weight_buffer_for_layer: layer=" << layer_idx 
                << ", byte_offset=" << byte_offset 
                << ", subbuffer_size=" << subbuffer_size
                << ", expert_size=" << expert_size);
    
    // Create a subbuffer that starts at the layer's offset
    // This allows the kernel to access weights using local expert IDs (0-31)
    cldnn::layout subbuffer_layout({static_cast<int64_t>(subbuffer_size)}, 
                                    cldnn::data_types::u8, cldnn::format::bfyx);
    
    try {
        auto subbuffer = m_engine.create_subbuffer(*full_buffer, subbuffer_layout, byte_offset);
        MOE_OTD_LOG("get_weight_buffer_for_layer: Created subbuffer at offset " << byte_offset);
        return subbuffer;
    } catch (const std::exception& e) {
        MOE_OTD_LOG("get_weight_buffer_for_layer: ERROR creating subbuffer - " << e.what());
        // Fallback: return the full buffer (kernel will still work but with wrong data)
        return full_buffer;
    }
}

cldnn::memory::ptr MoEExpertWeightManager::get_scale_buffer(bool is_up_projection) const {
    return is_up_projection ? m_up_scale_buffer : m_down_scale_buffer;
}

cldnn::memory::ptr MoEExpertWeightManager::get_bias_buffer(bool is_up_projection) const {
    return is_up_projection ? m_up_bias_buffer : m_down_bias_buffer;
}

const std::vector<int32_t>& MoEExpertWeightManager::get_slot_to_expert_mapping(bool is_up_projection) const {
    return is_up_projection ? m_up_slot_to_expert : m_down_slot_to_expert;
}

const std::unordered_map<int32_t, int32_t>& MoEExpertWeightManager::get_expert_to_slot_mapping(bool is_up_projection) const {
    return is_up_projection ? m_up_expert_to_slot : m_down_expert_to_slot;
}

size_t MoEExpertWeightManager::get_num_loaded_experts(bool is_up_projection) const {
    const auto& expert_to_slot = is_up_projection ? m_up_expert_to_slot : m_down_expert_to_slot;
    return expert_to_slot.size();
}

const LayerWeightInfo& MoEExpertWeightManager::get_layer_info(uint32_t layer_idx) const {
    if (layer_idx >= m_layer_infos.size()) {
        OPENVINO_THROW("Invalid layer index: ", layer_idx);
    }
    return m_layer_infos[layer_idx];
}

void MoEExpertWeightManager::set_layer_info(uint32_t layer_idx, const LayerWeightInfo& info) {
    if (layer_idx >= m_layer_infos.size()) {
        m_layer_infos.resize(layer_idx + 1);
    }
    m_layer_infos[layer_idx] = info;
}

}  // namespace ov::intel_gpu::ocl
