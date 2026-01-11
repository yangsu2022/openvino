// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"
#include "intel_gpu/op/moe_compressed.hpp"

namespace cldnn {

/// @brief    gemm for moe_pattern which selectively executes experts.
/// @details
/// This primitive implements the GEMM operation for the Mixture of Experts (MoE) pattern,
/// allowing for efficient execution of a subset of experts based on the input data.
/// @param input         Input data tensor.
/// @param weight        Weights tensor containing expert weights.
/// @param experts_ids   Tensor containing the IDs of the experts that are actually used at each time.
/// @param inputs_offset_per_expert   Tensor containing the offsets information of inputs per expert.
/// @param input_tokens_lens   Tensor containing the lengths of input tokens used by each expert.
/// @param num_experts_per_token  Number of experts per token selected by router.
struct moe_gemm : public primitive_base<moe_gemm> {
    CLDNN_DECLARE_PRIMITIVE(moe_gemm)

    enum MoEGemmInputIdx {
        // required
        INPUT = 0,
        WEIGHT = 1,
        EXPERTS_IDS = 2,
        INPUT_OFFSET_PER_EXPERT = 3,
        INPUT_TOKENS_LENS = 4,
        // optional
        BIAS = 5,
        WEIGHT_SCALE = 6,
        WEIGHT_ZP = 7
    };

    moe_gemm() : primitive_base("", {}) {}

    /// @brief Constructs moe_gemm primitive.
    ///
    moe_gemm(const primitive_id& id,
             const std::vector<input_info>& inputs,
             const ov::intel_gpu::op::MOECompressed::Config& moe_config)
          : primitive_base(id, inputs),
            num_experts_per_token(static_cast<int32_t>(moe_config.top_k)),
            has_batch_dim(moe_config.has_batch_dim) {}

    bool has_bias = false;
    int32_t num_experts_per_token = 0;
    bool has_batch_dim = true;

    // OTD (Offload-To-Disk) parameters
    bool otd_enabled = false;                // Whether OTD is enabled for this primitive
    std::string otd_weights_path;            // Path to the weights file on disk
    int64_t otd_resident_experts = 0;        // Number of experts to keep in GPU memory
    uint32_t otd_layer_idx = 0;              // Layer index for OTD weight lookup
    bool otd_is_up_projection = true;        // True for up-projection, false for down-projection
    size_t otd_expert_weight_size = 0;       // Size of each expert's weight in bytes
    size_t otd_expert_scale_size = 0;        // Size of each expert's scale in bytes
    size_t otd_expert_bias_size = 0;         // Size of each expert's bias in bytes
    uint32_t otd_num_experts = 0;            // Total number of experts

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, has_bias);
        seed = hash_combine(seed, num_experts_per_token);
        seed = hash_combine(seed, has_batch_dim);
        seed = hash_combine(seed, otd_enabled);
        seed = hash_combine(seed, otd_layer_idx);
        seed = hash_combine(seed, otd_is_up_projection);
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;
        auto rhs_casted = downcast<const moe_gemm>(rhs);
        return has_bias == rhs_casted.has_bias &&
               num_experts_per_token == rhs_casted.num_experts_per_token &&
               has_batch_dim == rhs_casted.has_batch_dim &&
               otd_enabled == rhs_casted.otd_enabled &&
               otd_layer_idx == rhs_casted.otd_layer_idx &&
               otd_is_up_projection == rhs_casted.otd_is_up_projection;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<moe_gemm>::save(ob);
        ob << has_bias;
        ob << num_experts_per_token;
        ob << has_batch_dim;
        ob << otd_enabled;
        ob << otd_weights_path;
        ob << otd_resident_experts;
        ob << otd_layer_idx;
        ob << otd_is_up_projection;
        ob << otd_expert_weight_size;
        ob << otd_expert_scale_size;
        ob << otd_expert_bias_size;
        ob << otd_num_experts;
        // Log OTD serialization
        if (otd_enabled) {
            std::cerr << "[MOE-OTD] moe_gemm::save: Serializing OTD config - enabled=" << otd_enabled 
                      << ", layer=" << otd_layer_idx 
                      << ", projection=" << (otd_is_up_projection ? "up" : "down") 
                      << ", path=" << otd_weights_path << std::endl;
        }
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<moe_gemm>::load(ib);
        ib >> has_bias;
        ib >> num_experts_per_token;
        ib >> has_batch_dim;
        ib >> otd_enabled;
        ib >> otd_weights_path;
        ib >> otd_resident_experts;
        ib >> otd_layer_idx;
        ib >> otd_is_up_projection;
        ib >> otd_expert_weight_size;
        ib >> otd_expert_scale_size;
        ib >> otd_expert_bias_size;
        ib >> otd_num_experts;
        // Log OTD deserialization
        if (otd_enabled) {
            std::cerr << "[MOE-OTD] moe_gemm::load: Deserialized OTD config - enabled=" << otd_enabled 
                      << ", layer=" << otd_layer_idx 
                      << ", projection=" << (otd_is_up_projection ? "up" : "down") 
                      << ", path=" << otd_weights_path << std::endl;
        }
    }
};
}
