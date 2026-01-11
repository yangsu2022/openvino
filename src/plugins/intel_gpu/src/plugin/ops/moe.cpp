// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "openvino/op/constant.hpp"
#include "openvino/op/moe.hpp"
#include "intel_gpu/op/moe_compressed.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/op/moe_3gemm_fused_compressed.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/primitives/moe_3gemm_fused_compressed.hpp"
#include "intel_gpu/primitives/moe_gemm.hpp"
#include "intel_gpu/primitives/moe_mask_gen.hpp"
#include <intel_gpu/primitives/moe_scatter_reduction.hpp>
#include <intel_gpu/primitives/moe_gather.hpp>
#include <intel_gpu/primitives/swiglu.hpp>
#include <intel_gpu/primitives/eltwise.hpp>
#include "intel_gpu/runtime/internal_properties.hpp"

#include <limits>
#include <iostream>

// OTD Debug logging
#define MOE_OTD_LOG(msg) std::cout << "[MOE-OTD] " << msg << std::endl

namespace ov {
namespace op {
namespace internal {
using MOE3GemmFusedCompressed = ov::intel_gpu::op::MOE3GemmFusedCompressed;
using MOECompressed = ov::intel_gpu::op::MOECompressed;
}  // namespace internal
}  // namespace op
}  // namespace ov

namespace ov::intel_gpu {
using namespace cldnn;

static void CreateMOE3GemmFusedCompressedOp(ProgramBuilder& p, const std::shared_ptr<ov::intel_gpu::op::MOE3GemmFusedCompressed>& op) {
    auto inputs = p.GetInputInfo(op);
    const auto& config = op->get_config();
    ///   0: hidden_states - input tensor with hidden representations
    ///   1: routing_weights - [num_seq, num_experts] routing weights for all experts
    ///   2: w0_weight - expert weights for first projection,
    ///                  shape [num_experts, inter_size, group_num, group_size]
    ///   3: w0_scale - expert scale for first projection for compressed experts,
    ///                  shape [num_experts, inter_size, group_num, 1]
    ///   4: w0_zp - expert zp for first projection for compressed experts,
    ///                  shape [num_experts, inter_size, group_num, 1]
    ///   5: w1_weight - expert weights for second projection,
    ///                  shape [num_experts, inter_size, group_num, group_size]
    ///   6: w1_scale - expert scale for second projection for compressed experts,
    ///                  shape [num_experts, inter_size, group_num, 1]
    ///   7: w1_zp - expert zp for second projection for compressed experts,
    ///                  shape [num_experts, inter_size, group_num, 1]
    ///   8: w2_weight - expert weights for final projection,
    ///                  shape [num_experts, hidden_size, group_num, group_size]
    ///   9: w2_scale - expert scale for final projection for compressed experts,
    ///                  shape [num_experts, hidden_size, group_num, 1]
    ///   10: w2_zp - expert zp for final projection for compressed experts,
    ///                  shape [num_experts, hidden_size, group_num, 1]
    validate_inputs_count(op, {11});

    const std::string layerName = layer_type_name_ID(op);
    const cldnn::moe_3gemm_fused_compressed moe(layerName, inputs, config);

    p.add_primitive(*op, moe);
}

static void CreateMOECompressedOp(ProgramBuilder& p, const std::shared_ptr<ov::op::internal::MOECompressed>& op) {
    MOE_OTD_LOG("CreateMOECompressedOp: Entry point");
    auto inputs = p.GetInputInfo(op);
    auto& config = op->get_config();
    MOE_OTD_LOG("CreateMOECompressedOp: num_expert=" << config.num_expert << ", hidden_size=" << config.hidden_size);
    MOE_OTD_LOG("CreateMOECompressedOp: inter_size=" << config.inter_size << ", group_size=" << config.group_size);
    MOE_OTD_LOG("CreateMOECompressedOp: top_k=" << config.top_k << ", expert_type=" << static_cast<int>(config.expert_type));
    std::vector<cldnn::input_info> input_infos;
    for (const auto& input : inputs) {
        input_infos.push_back(cldnn::input_info(input));
    }
    if (config.expert_type == ov::op::internal::MOE::Expert_type::GEMM3_SWIGLU) {
        // Create GEMM3_SWIGLU specific primitives
        //   0: hidden_states - input tensor with hidden representations
        //   1: routing_weights - [num_experts, ...] normalized weights for selected experts
        //      (input to final multiplication)
        //   2: router_topk_output_indices - [..., topk] indices of selected top-k experts
        //   3: w0_weight - expert weights for first projection,
        //   shape [num_experts, inter_size, group_num, group_size]
        //   4: w0_scale - expert scale for first projection for compressed experts,
        //   shape [num_experts, inter_size, group_num, 1]
        //   5: w0_zp - expert zp for first projection for compressed experts,
        //   shape [num_experts, inter_size, group_num, 1]
        //   6: w1_weight - expert weights for second projection,
        //   shape [num_experts, inter_size, group_num, group_size]
        //   7: w1_scale - expert scale for second projection for compressed experts,
        //   shape [num_experts, inter_size, group_num, 1]
        //   8: w1_zp - expert zp for second projection for compressed experts,
        //   shape [num_experts, inter_size, group_num, 1]
        //   9: w2_weight - expert weights for final projection,
        //   shape [num_experts, hidden_size, group_num, group_size]
        //   10: w2_scale - expert scale for final projection for compressed experts,
        //   shape [num_experts, hidden_size, group_num, 1]
        //   11: w2_zp - expert zp for final projection for compressed experts,
        //   shape [num_experts, hidden_size, group_num, 1]

        // Use moe_3gemm_fused_compressed to replace it.
    } else  {
        // Create GEMM2_BIAS_SWIGLU_CLAMP specific primitives
        MOE_OTD_LOG("CreateMOECompressedOp: Processing GEMM2_BIAS_SWIGLU_CLAMP expert type");
        // input0 : input {#tokens, hidden_size}
        // input1 : topk_weight {#tokens, num_experts_per_token}
        // input2 : topk_idx {#tokens, num_experts_per_token}
        // input3 : compressed_weights_input_up {#experts, ofm, num_groups, group_size}
        // input4 : scale_input_up {#experts, ofm, num_groups, 1}
        // input5 : bias_up {#experts, 1, ofm}
        // input6 : compressed_weights_input_down {#experts, ofm, num_groups, group_size}
        // input7 : scale_input_down {#experts, ofm, num_groups, 1}
        // input8 : bias_down {#experts, 1, ofm}
        // moe_mask_gen
        // moe_gather
        // moe_gemm_up + bias
        // swiglu_with_clamp
        // moe_gemm_down + bias
        // moe_scatter_reduce

        // Read OTD configuration from execution config
        const auto& exec_config = p.get_config();
        bool otd_enabled = exec_config.get_property(ov::intel_gpu::moe_otd_enabled.name()).as<bool>();
        std::string otd_weights_path = exec_config.get_property(ov::intel_gpu::moe_weights_path.name()).as<std::string>();
        int64_t otd_resident_experts = exec_config.get_property(ov::intel_gpu::moe_resident_experts.name()).as<int64_t>();

        MOE_OTD_LOG("CreateMOECompressedOp: OTD config - enabled=" << otd_enabled 
                    << ", weights_path=" << otd_weights_path 
                    << ", resident_experts=" << otd_resident_experts);

        // Static layer counter for tracking MoE layer index
        static thread_local uint32_t moe_layer_counter = 0;
        uint32_t current_layer_idx = moe_layer_counter++;
        MOE_OTD_LOG("CreateMOECompressedOp: Current layer index = " << current_layer_idx);

        std::string prim_name_base = layer_type_name_ID(op);
        auto  moe_mask_gen_name = prim_name_base + "_moe_mask_gen";
        auto  moe_mask_gen_reshape_name = prim_name_base + "_moe_mask_gen_reshape";
        auto  moe_gather_name = prim_name_base + "_moe_gather";
        auto  moe_bias_up_name = prim_name_base + "_moe_bias_up";
        auto  moe_gemm_up_name = prim_name_base + "_moe_gemm_up";
        auto  moe_swiglu_name = prim_name_base + "_moe_swiglu";
        auto  moe_gemm_down_name = prim_name_base + "_moe_gemm_down";
        auto  moe_bias_down_name = prim_name_base + "_moe_bias_down";
        auto  moe_scatter_reduce_name = prim_name_base + "_moe_scatter_reduce";
        auto moe_mask_gen_prim = cldnn::moe_mask_gen(moe_mask_gen_name,
                                                     input_infos[2],  // topk indices
                                                     static_cast<int32_t>(config.num_expert),
                                                     static_cast<int32_t>(config.top_k));
        p.add_primitive(*op, moe_mask_gen_prim);
        MOE_OTD_LOG("CreateMOECompressedOp: Created moe_mask_gen primitive: " << moe_mask_gen_name);
        auto moe_mask_gen_reshape_prim =
            cldnn::moe_mask_gen_reshape(moe_mask_gen_reshape_name,
                                        input_info(moe_mask_gen_prim, moe_mask_gen::MoEMaskGenOutputIdx::TOKENS_PER_EXPERT),
                                        input_info(moe_mask_gen_prim, moe_mask_gen::MoEMaskGenOutputIdx::EXPERTS_INFO_START_IDX),
                                        input_info(moe_mask_gen_prim, moe_mask_gen::MoEMaskGenOutputIdx::EXPERTS_ID),
                                        input_info(moe_mask_gen_prim, moe_mask_gen::MoEMaskGenOutputIdx::TOKENS_LENS_PER_EXPERT),
                                        input_info(moe_mask_gen_prim, moe_mask_gen::MoEMaskGenOutputIdx::NUM_ACTUALLY_USED_EXPERTS));
        p.add_primitive(*op, moe_mask_gen_reshape_prim);
        MOE_OTD_LOG("CreateMOECompressedOp: Created moe_mask_gen_reshape primitive: " << moe_mask_gen_reshape_name);
        auto moe_gather_prim = cldnn::moe_gather(moe_gather_name,
                                                 input_infos[0],  // input
                                                 input_info(moe_mask_gen_reshape_name, moe_mask_gen_reshape::MoEMaskGenReshapeOutputIdx::TOKENS_PER_EXPERT),
                                                 config);
        p.add_primitive(*op, moe_gather_prim);
        MOE_OTD_LOG("CreateMOECompressedOp: Created moe_gather primitive: " << moe_gather_name);
        std::vector<cldnn::input_info> moe_gemm_up_inputs = {
            input_info(moe_gather_name),  // topk_weight
            input_infos[3],  // compressed_weights_input_up
            input_info(moe_mask_gen_reshape_name, moe_mask_gen_reshape::MoEMaskGenReshapeOutputIdx::EXPERTS_ID),
            input_info(moe_mask_gen_reshape_name, moe_mask_gen_reshape::MoEMaskGenReshapeOutputIdx::EXPERTS_INFO_START_IDX),
            input_info(moe_mask_gen_reshape_name, moe_mask_gen_reshape::MoEMaskGenReshapeOutputIdx::TOKENS_LENS_PER_EXPERT)
        };
        size_t down_idx = 0;
        if (config.has_zp) {
            moe_gemm_up_inputs.push_back(input_infos[6]);  // bias_up
            moe_gemm_up_inputs.push_back(input_infos[4]);  // scale_input_up
            moe_gemm_up_inputs.push_back(input_infos[5]);  // zp_input_up
            down_idx = 7;
        } else {
            moe_gemm_up_inputs.push_back(input_infos[5]);  // bias_up
            moe_gemm_up_inputs.push_back(input_infos[4]);  // scale_input_up
            down_idx = 6;
        }
        auto moe_gemm_up = cldnn::moe_gemm(moe_gemm_up_name, moe_gemm_up_inputs, config);
        moe_gemm_up.has_bias = true;

        // Configure OTD for up-projection
        if (otd_enabled && !otd_weights_path.empty()) {
            moe_gemm_up.otd_enabled = true;
            moe_gemm_up.otd_weights_path = otd_weights_path;
            moe_gemm_up.otd_resident_experts = otd_resident_experts;
            moe_gemm_up.otd_layer_idx = current_layer_idx;
            moe_gemm_up.otd_is_up_projection = true;
            moe_gemm_up.otd_num_experts = static_cast<uint32_t>(config.num_expert);
            MOE_OTD_LOG("CreateMOECompressedOp: OTD enabled for moe_gemm_up, layer=" << current_layer_idx);
            MOE_OTD_LOG("CreateMOECompressedOp: moe_gemm_up primitive fields set:" 
                        << " otd_enabled=" << moe_gemm_up.otd_enabled
                        << ", otd_layer_idx=" << moe_gemm_up.otd_layer_idx
                        << ", otd_is_up_projection=" << moe_gemm_up.otd_is_up_projection
                        << ", otd_weights_path=" << moe_gemm_up.otd_weights_path
                        << ", otd_num_experts=" << moe_gemm_up.otd_num_experts);
            GPU_DEBUG_TRACE_DETAIL << "MOE OTD enabled for layer " << current_layer_idx << " up-projection\n";
        }

        p.add_primitive(*op, moe_gemm_up);
        MOE_OTD_LOG("CreateMOECompressedOp: Created moe_gemm_up primitive: " << moe_gemm_up_name);

        // gpt-oss swiglu pattern
        // config.expert_alpha : clamp_max
        // config.expert_beta : swish_beta which is slightly different from usual swiglu pattern
        // - Applied clamp
        // - Added one for up value
        // - Gate stride is 1 (not splitting to half and half)
        // - config.expert_alpha : clamp_max
        // - config.expert_beta : swish_beta
        // TODO : update for each new pattern
        auto moe_swiglu_prim = cldnn::swiglu(moe_swiglu_name,
                                             input_info(moe_gemm_up_name),
                                             2, // axis
                                             2, // glu_stride
                                             ov::op::internal::GLU::GluType::Swish,
                                             0,                    // gate idx
                                             -config.expert_alpha, // clamp_min
                                             config.expert_alpha,  // clamp_max
                                             config.expert_beta,   // swish beta
                                             1.0f,                 // up_add_val
                                             cldnn::tensor());
        p.add_primitive(*op, moe_swiglu_prim);
        MOE_OTD_LOG("CreateMOECompressedOp: Created swiglu primitive: " << moe_swiglu_name);
        std::vector<cldnn::input_info> moe_gemm_down_inputs = {
            input_info(moe_swiglu_name),
            input_infos[down_idx],  // compressed_weights_input_down
            input_info(moe_mask_gen_reshape_name, moe_mask_gen_reshape::MoEMaskGenReshapeOutputIdx::EXPERTS_ID),
            input_info(moe_mask_gen_reshape_name, moe_mask_gen_reshape::MoEMaskGenReshapeOutputIdx::EXPERTS_INFO_START_IDX),
            input_info(moe_mask_gen_reshape_name, moe_mask_gen_reshape::MoEMaskGenReshapeOutputIdx::TOKENS_LENS_PER_EXPERT),
        };

        if (config.has_zp) {
            moe_gemm_down_inputs.push_back(input_infos[down_idx + 3]);  // bias_up
            moe_gemm_down_inputs.push_back(input_infos[down_idx + 1]);  // scale_input_up
            moe_gemm_down_inputs.push_back(input_infos[down_idx + 2]);  // zp_input_up
        } else {
            moe_gemm_down_inputs.push_back(input_infos[down_idx + 2]);  // bias_up
            moe_gemm_down_inputs.push_back(input_infos[down_idx + 1]);  // scale_input_up
        }

        auto moe_gemm_down = cldnn::moe_gemm(moe_gemm_down_name, moe_gemm_down_inputs, config);
        moe_gemm_down.has_bias = true;

        // Configure OTD for down-projection
        if (otd_enabled && !otd_weights_path.empty()) {
            moe_gemm_down.otd_enabled = true;
            moe_gemm_down.otd_weights_path = otd_weights_path;
            moe_gemm_down.otd_resident_experts = otd_resident_experts;
            moe_gemm_down.otd_layer_idx = current_layer_idx;
            moe_gemm_down.otd_is_up_projection = false;
            moe_gemm_down.otd_num_experts = static_cast<uint32_t>(config.num_expert);
            MOE_OTD_LOG("CreateMOECompressedOp: OTD enabled for moe_gemm_down, layer=" << current_layer_idx);
            MOE_OTD_LOG("CreateMOECompressedOp: moe_gemm_down primitive fields set:" 
                        << " otd_enabled=" << moe_gemm_down.otd_enabled
                        << ", otd_layer_idx=" << moe_gemm_down.otd_layer_idx
                        << ", otd_is_up_projection=" << moe_gemm_down.otd_is_up_projection
                        << ", otd_weights_path=" << moe_gemm_down.otd_weights_path
                        << ", otd_num_experts=" << moe_gemm_down.otd_num_experts);
            GPU_DEBUG_TRACE_DETAIL << "MOE OTD enabled for layer " << current_layer_idx << " down-projection\n";
        }

        p.add_primitive(*op, moe_gemm_down);
        MOE_OTD_LOG("CreateMOECompressedOp: Created moe_gemm_down primitive: " << moe_gemm_down_name);
        auto moe_scatter_reduce_prim = cldnn::moe_scatter_reduction(moe_scatter_reduce_name,
                input_info(moe_gemm_down_name),
                input_infos[2],
                input_infos[1],
                input_info(moe_mask_gen_reshape_name, moe_mask_gen_reshape::MoEMaskGenReshapeOutputIdx::TOKENS_PER_EXPERT),
                input_info(moe_mask_gen_reshape_name, moe_mask_gen_reshape::MoEMaskGenReshapeOutputIdx::EXPERTS_INFO_START_IDX),
                input_info(moe_mask_gen_reshape_name, moe_mask_gen_reshape::MoEMaskGenReshapeOutputIdx::TOKENS_LENS_PER_EXPERT),
                input_info(moe_mask_gen_reshape_name, moe_mask_gen_reshape::MoEMaskGenReshapeOutputIdx::EXPERTS_ID),
                config);
        p.add_primitive(*op, moe_scatter_reduce_prim);
        MOE_OTD_LOG("CreateMOECompressedOp: Created moe_scatter_reduction primitive: " << moe_scatter_reduce_name);
        MOE_OTD_LOG("CreateMOECompressedOp: Layer " << current_layer_idx << " primitives creation completed");
    }
}
REGISTER_FACTORY_IMPL(internal, MOE3GemmFusedCompressed);
REGISTER_FACTORY_IMPL(internal, MOECompressed);

}  // namespace ov::intel_gpu
