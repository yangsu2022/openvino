// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/runtime/memory.hpp"
#include "intel_gpu/runtime/engine.hpp"
#include "intel_gpu/runtime/stream.hpp"
#include "intel_gpu/runtime/layout.hpp"

#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <fstream>
#include <memory>
#include <mutex>

namespace ov::intel_gpu::ocl {

/// @brief File header for MoE weights binary file
struct MoEWeightsFileHeader {
    char magic[4] = {'M', 'O', 'E', 'W'};  // Magic number "MOEW"
    uint32_t version = 1;
    uint32_t num_layers = 0;
    uint32_t num_experts_per_layer = 0;
    uint64_t expert_up_weight_size = 0;      // Size of up-projection weight per expert (bytes)
    uint64_t expert_down_weight_size = 0;    // Size of down-projection weight per expert (bytes)
    uint64_t expert_up_scale_size = 0;       // Size of up-projection scale per expert (bytes)
    uint64_t expert_down_scale_size = 0;     // Size of down-projection scale per expert (bytes)
    uint64_t expert_up_bias_size = 0;        // Size of up-projection bias per expert (bytes)
    uint64_t expert_down_bias_size = 0;      // Size of down-projection bias per expert (bytes)
    uint64_t data_offset = 0;                // Offset to the start of weight data
    uint64_t reserved[8] = {0};              // Reserved for future use
};

/// @brief Describes weight tensor layout for one expert
struct ExpertWeightDesc {
    size_t offset = 0;      // Offset in the file
    size_t size = 0;        // Size in bytes
    cldnn::layout layout;   // Tensor layout
};

/// @brief Layer-level weight information
struct LayerWeightInfo {
    uint32_t layer_idx = 0;
    uint32_t num_experts = 0;
    std::vector<ExpertWeightDesc> up_weights;      // Per-expert up-projection weights
    std::vector<ExpertWeightDesc> down_weights;    // Per-expert down-projection weights
    std::vector<ExpertWeightDesc> up_scales;       // Per-expert up-projection scales
    std::vector<ExpertWeightDesc> down_scales;     // Per-expert down-projection scales
    std::vector<ExpertWeightDesc> up_biases;       // Per-expert up-projection biases
    std::vector<ExpertWeightDesc> down_biases;     // Per-expert down-projection biases
};

/// @brief Configuration for MoEExpertWeightManager
struct MoEOTDConfig {
    std::string weights_path;          // Path to the weights file on disk (SSD)
    int64_t resident_experts = 0;      // Number of experts to keep in GPU memory
    bool enabled = false;              // Whether OTD is enabled
};

/// @brief Manages MoE expert weights with Offload-To-Disk (OTD) capability
///
/// This class manages the loading and caching of MoE expert weights from
/// disk (SSD) to GPU memory. It maintains a buffer in GPU memory that can
/// hold a limited number of experts, and dynamically loads/unloads experts
/// based on runtime requirements.
class MoEExpertWeightManager {
public:
    /// @brief Construct a new MoE Expert Weight Manager
    /// @param engine GPU engine for memory allocation
    /// @param config OTD configuration
    MoEExpertWeightManager(cldnn::engine& engine, const MoEOTDConfig& config);
    
    /// @brief Destructor
    ~MoEExpertWeightManager();

    /// @brief Initialize the manager by reading file header and allocating buffers
    /// @return true if initialization succeeded
    bool initialize();

    /// @brief Check if OTD is enabled and properly initialized
    bool is_enabled() const { return m_initialized && m_config.enabled; }

    /// @brief Load specified experts into GPU memory buffer
    /// @param layer_idx The layer index (0-based)
    /// @param expert_ids List of expert IDs to load
    /// @param stream GPU stream for async operations
    /// @param is_up_projection true for up-projection, false for down-projection
    void load_experts(uint32_t layer_idx,
                      const std::vector<int32_t>& expert_ids,
                      cldnn::stream& stream,
                      bool is_up_projection);

    /// @brief Get the GPU memory buffer containing loaded expert weights
    /// @param is_up_projection true for up-projection, false for down-projection
    /// @return Pointer to GPU memory
    cldnn::memory::ptr get_weight_buffer(bool is_up_projection) const;

    /// @brief Get the GPU memory buffer containing loaded expert scales
    cldnn::memory::ptr get_scale_buffer(bool is_up_projection) const;

    /// @brief Get the GPU memory buffer containing loaded expert biases
    cldnn::memory::ptr get_bias_buffer(bool is_up_projection) const;

    /// @brief Get mapping from buffer slot index to expert ID
    /// @param is_up_projection true for up-projection, false for down-projection
    const std::vector<int32_t>& get_slot_to_expert_mapping(bool is_up_projection) const;

    /// @brief Get mapping from expert ID to buffer slot index
    /// @param is_up_projection true for up-projection, false for down-projection
    const std::unordered_map<int32_t, int32_t>& get_expert_to_slot_mapping(bool is_up_projection) const;

    /// @brief Get the number of experts currently loaded
    size_t get_num_loaded_experts(bool is_up_projection) const;

    /// @brief Get the file header information
    const MoEWeightsFileHeader& get_file_header() const { return m_header; }

    /// @brief Get layer weight information
    const LayerWeightInfo& get_layer_info(uint32_t layer_idx) const;

    /// @brief Set layer weight information (used during file creation)
    void set_layer_info(uint32_t layer_idx, const LayerWeightInfo& info);

private:
    /// @brief Read file header and validate
    bool read_file_header();

    /// @brief Calculate buffer sizes and allocate GPU memory
    bool allocate_buffers();

    /// @brief Load a single expert's weights from disk to host buffer
    void load_expert_to_host(uint32_t layer_idx, int32_t expert_id, bool is_up_projection);

    /// @brief Copy host buffer to GPU memory slot
    void copy_to_gpu_slot(int32_t slot_idx, bool is_up_projection, cldnn::stream& stream);

    /// @brief Find an available slot (or evict least recently used)
    int32_t find_available_slot(bool is_up_projection);

    cldnn::engine& m_engine;
    MoEOTDConfig m_config;
    MoEWeightsFileHeader m_header;
    
    bool m_initialized = false;
    std::unique_ptr<std::ifstream> m_weights_file;
    std::mutex m_mutex;

    // Per-layer weight information
    std::vector<LayerWeightInfo> m_layer_infos;

    // GPU memory buffers for up-projection
    cldnn::memory::ptr m_up_weight_buffer;
    cldnn::memory::ptr m_up_scale_buffer;
    cldnn::memory::ptr m_up_bias_buffer;

    // GPU memory buffers for down-projection
    cldnn::memory::ptr m_down_weight_buffer;
    cldnn::memory::ptr m_down_scale_buffer;
    cldnn::memory::ptr m_down_bias_buffer;

    // Host staging buffers for async loading
    std::vector<uint8_t> m_host_staging_buffer;

    // Slot management for up-projection
    std::vector<int32_t> m_up_slot_to_expert;     // slot_idx -> expert_id (-1 if empty)
    std::unordered_map<int32_t, int32_t> m_up_expert_to_slot;  // expert_id -> slot_idx
    std::vector<uint64_t> m_up_slot_access_time;  // For LRU eviction

    // Slot management for down-projection
    std::vector<int32_t> m_down_slot_to_expert;
    std::unordered_map<int32_t, int32_t> m_down_expert_to_slot;
    std::vector<uint64_t> m_down_slot_access_time;

    // Access time counter for LRU
    uint64_t m_access_counter = 0;
};

}  // namespace ov::intel_gpu::ocl
