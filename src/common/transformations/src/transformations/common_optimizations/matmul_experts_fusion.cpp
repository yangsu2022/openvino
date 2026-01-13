// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/matmul_experts_fusion.hpp"

#include "itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/minimum.hpp"
#include "openvino/op/moe.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scatter_elements_update.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/swish.hpp"
#include "openvino/op/tile.hpp"
#include "openvino/op/topk.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"
#include "openvino/util/env_util.hpp"

#include <iostream>
#include <cstdlib>

// MoE OTD Debug logging - controlled by OV_MOE_OTD_DEBUG environment variable
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

using namespace ov::pass;
ov::pass::FuseVectorizedMOE2GEMM::FuseVectorizedMOE2GEMM() {
    MATCHER_SCOPE(FuseVectorizedMOE2GEMM);
    MOE_OTD_LOG("FuseVectorizedMOE2GEMM: Registering transformation pass");

    auto experts_input = pattern::wrap_type<ov::op::v1::Reshape>({pattern::any_input(), pattern::any_input()});
    auto tile = pattern::wrap_type<ov::op::v0::Tile>({experts_input, pattern::any_input()});
    auto after_tile_reshape = pattern::wrap_type<ov::op::v1::Reshape>({tile, pattern::any_input()});
    auto gate_up_matmul = pattern::wrap_type<ov::op::v0::MatMul>({after_tile_reshape, pattern::any_input()},
                                                                 {{"transpose_a", false}, {"transpose_b", true}});
    auto gate_up_add = pattern::wrap_type<ov::op::v1::Add>({gate_up_matmul, pattern::any_input()});

    // Branch 1: Slice_1 -> Clamp -> Add_1
    auto slice1 = pattern::wrap_type<ov::op::v8::Slice>(
        {gate_up_add, pattern::any_input(), pattern::any_input(), pattern::any_input(), pattern::any_input()});
    auto clamp = pattern::wrap_type<ov::op::v0::Clamp>({slice1});
    auto add1 = pattern::wrap_type<ov::op::v1::Add>({clamp, pattern::wrap_const()});

    // Branch 2: Slice_2 -> Minimum_1 -> Swish
    auto slice2 = pattern::wrap_type<ov::op::v8::Slice>(
        {gate_up_add, pattern::any_input(), pattern::any_input(), pattern::any_input(), pattern::any_input()});
    auto minimum1 = pattern::wrap_type<ov::op::v1::Minimum>({slice2, pattern::wrap_const()});
    auto swish_beta = pattern::wrap_const();
    auto swish = pattern::wrap_type<ov::op::v4::Swish>({minimum1, swish_beta});

    // Join: Multiply_2
    auto multiply2 = pattern::wrap_type<ov::op::v1::Multiply>({add1, swish});

    // Down projection
    auto down_proj_matmul = pattern::wrap_type<ov::op::v0::MatMul>({multiply2, pattern::any_input()},
                                                                   {{"transpose_a", false}, {"transpose_b", true}});
    auto down_proj_add = pattern::wrap_type<ov::op::v1::Add>({down_proj_matmul, pattern::wrap_const()});
    auto end_reshape = pattern::wrap_type<ov::op::v1::Reshape>({down_proj_add, pattern::any_input()});

    // Routing weights/mask
    auto router_topk_indices = pattern::any_input();
    auto scatter_elements_update = pattern::wrap_type<ov::op::v12::ScatterElementsUpdate>(
        {pattern::any_input(), router_topk_indices, pattern::any_input(), pattern::any_input()});

    auto router_transpose = pattern::wrap_type<ov::op::v1::Transpose>({scatter_elements_update, pattern::any_input()});
    auto router_reshape = pattern::wrap_type<ov::op::v1::Reshape>({router_transpose, pattern::any_input()});
    auto unsqueeze_routing_weights = pattern::wrap_type<ov::op::v0::Unsqueeze>({router_reshape, pattern::any_input()});

    auto mul3 = pattern::wrap_type<ov::op::v1::Multiply>({end_reshape, unsqueeze_routing_weights});
    auto reduce_sum = pattern::wrap_type<ov::op::v1::ReduceSum>({mul3, pattern::any_input()}, {{"keep_dims", false}});
    auto moe_pattern = reduce_sum;

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto& pm = m.get_pattern_value_map();

        MOE_OTD_LOG("FuseVectorizedMOE2GEMM: Pattern matched for node: " << m.get_match_root()->get_friendly_name());

        if (transformation_callback(m.get_match_root())) {
            MOE_OTD_LOG("FuseVectorizedMOE2GEMM: Transformation callback returned false, skipping");
            return false;
        }

        auto experts_input_node = pm.at(experts_input).get_node()->input_value(0);

        auto routing_weights_node = pm.at(unsqueeze_routing_weights).get_node_shared_ptr();
        auto gate_up_weight = pm.at(gate_up_matmul).get_node()->input_value(1).get_node_shared_ptr();
        auto gate_up_bias_node = pm.at(gate_up_add).get_node()->input_value(1).get_node_shared_ptr();
        auto down_proj_weight = pm.at(down_proj_matmul).get_node()->input_value(1).get_node_shared_ptr();
        auto down_proj_bias_node = pm.at(down_proj_add).get_node()->input_value(1).get_node_shared_ptr();
        auto topk_indices_node = pm.at(scatter_elements_update).get_node()->input_value(1);

        // Log weight shapes
        MOE_OTD_LOG("FuseVectorizedMOE2GEMM: gate_up_weight shape = " << gate_up_weight->get_output_partial_shape(0));
        MOE_OTD_LOG("FuseVectorizedMOE2GEMM: down_proj_weight shape = " << down_proj_weight->get_output_partial_shape(0));

        ov::OutputVector moe_inputs = {experts_input_node,
                                       routing_weights_node,
                                       topk_indices_node,
                                       gate_up_weight,
                                       gate_up_bias_node,
                                       down_proj_weight,
                                       down_proj_bias_node};

        ov::op::internal::MOE::Config config;

        // Extract expert_beta from Swish beta attribute
        auto swish_beta_const = ov::as_type_ptr<ov::op::v0::Constant>(pm.at(swish_beta).get_node_shared_ptr());
        auto swish_beta_const_val = swish_beta_const->cast_vector<float>()[0];
        config.expert_beta = swish_beta_const_val;

        // Extract expert_alpha from Clamp max attribute
        if (auto clamp_op = ov::as_type_ptr<ov::op::v0::Clamp>(pm.at(clamp).get_node_shared_ptr())) {
            config.expert_alpha = static_cast<float>(clamp_op->get_max());
        }

        // Set expert_type
        config.expert_type = ov::op::internal::MOE::Expert_type::GEMM2_BIAS_SWIGLU_CLAMP;

        MOE_OTD_LOG("FuseVectorizedMOE2GEMM: Creating MOE node with expert_type=GEMM2_BIAS_SWIGLU_CLAMP, alpha=" 
                    << config.expert_alpha << ", beta=" << config.expert_beta);

        auto moe = std::make_shared<ov::op::internal::MOE>(moe_inputs, config);
        moe->set_friendly_name(m.get_match_root()->get_friendly_name());
        ov::copy_runtime_info(m.get_matched_nodes(), moe);
        ov::replace_node(m.get_match_root(), moe);

        MOE_OTD_LOG("FuseVectorizedMOE2GEMM: Successfully fused to MOE node: " << moe->get_friendly_name());

        register_new_node(moe);
        return true;
    };

    auto matcher = std::make_shared<pattern::Matcher>(moe_pattern, matcher_name);
    this->register_matcher(matcher, callback);
}

ov::pass::FuseVectorizedMOE3GEMM::FuseVectorizedMOE3GEMM() {
    MATCHER_SCOPE(FuseVectorizedMOE3GEMM);

    auto experts_input = pattern::wrap_type<ov::op::v1::Reshape>({pattern::any_input(), pattern::any_input()});
    auto tile = pattern::wrap_type<ov::op::v0::Tile>({experts_input, pattern::any_input()});
    auto after_tile_reshape = pattern::wrap_type<ov::op::v1::Reshape>({tile, pattern::any_input()});

    // First GEMM (activation gate)
    auto gate_matmul = pattern::wrap_type<ov::op::v0::MatMul>({after_tile_reshape, pattern::any_input()},
                                                              {{"transpose_a", false}, {"transpose_b", true}});
    auto swish = pattern::wrap_type<ov::op::v4::Swish>({gate_matmul});
    // Second GEMM (up_projection)
    auto up_matmul = pattern::wrap_type<ov::op::v0::MatMul>({after_tile_reshape, pattern::any_input()},
                                                            {{"transpose_a", false}, {"transpose_b", true}});
    // Join: Multiply (SwiGLU)
    auto swiglu = pattern::wrap_type<ov::op::v1::Multiply>({swish, up_matmul});

    // Third GEMM (down_projection)
    auto down_matmul = pattern::wrap_type<ov::op::v0::MatMul>({swiglu, pattern::any_input()},
                                                              {{"transpose_a", false}, {"transpose_b", true}});
    auto end_reshape = pattern::wrap_type<ov::op::v1::Reshape>({down_matmul, pattern::any_input()});

    // Routing weights/mask
    auto router_topk_indices = pattern::any_input();
    auto scatter_elements_update = pattern::wrap_type<ov::op::v12::ScatterElementsUpdate>(
        {pattern::any_input(), router_topk_indices, pattern::any_input(), pattern::any_input()});
    auto router_transpose = pattern::wrap_type<ov::op::v1::Transpose>({scatter_elements_update, pattern::any_input()});
    auto router_reshape = pattern::wrap_type<ov::op::v1::Reshape>({router_transpose, pattern::any_input()});
    auto unsqueeze_routing_weights = pattern::wrap_type<ov::op::v0::Unsqueeze>({router_reshape, pattern::any_input()});

    auto mul3 = pattern::wrap_type<ov::op::v1::Multiply>({end_reshape, unsqueeze_routing_weights});
    auto reduce_sum = pattern::wrap_type<ov::op::v1::ReduceSum>({mul3, pattern::any_input()}, {{"keep_dims", false}});
    auto moe_pattern = reduce_sum;

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto& pm = m.get_pattern_value_map();

        if (transformation_callback(m.get_match_root())) {
            return false;
        }

        auto experts_input_node = pm.at(experts_input).get_node()->input_value(0);
        auto routing_weights_node = pm.at(unsqueeze_routing_weights).get_node_shared_ptr();
        auto gate_weight = pm.at(gate_matmul).get_node()->input_value(1).get_node_shared_ptr();
        auto up_weight = pm.at(up_matmul).get_node()->input_value(1).get_node_shared_ptr();
        auto down_weight = pm.at(down_matmul).get_node()->input_value(1).get_node_shared_ptr();
        auto topk_indices_node = pm.at(scatter_elements_update).get_node()->input_value(1);

        ov::OutputVector moe_inputs = {
            experts_input_node,
            routing_weights_node,
            topk_indices_node,
            gate_weight,
            up_weight,
            down_weight,
        };

        ov::op::internal::MOE::Config config;
        config.expert_type = ov::op::internal::MOE::Expert_type::GEMM3_SWIGLU;
        // Extract expert_beta if Swish has beta input provided
        if (auto swish_op = ov::as_type_ptr<ov::op::v4::Swish>(pm.at(swish).get_node_shared_ptr())) {
            if (swish_op->get_input_size() > 1) {
                if (auto swish_beta_const =
                        ov::as_type_ptr<ov::op::v0::Constant>(swish_op->get_input_node_shared_ptr(1))) {
                    config.expert_beta = swish_beta_const->cast_vector<float>()[0];
                }
            }
        }

        auto moe = std::make_shared<ov::op::internal::MOE>(moe_inputs, config);
        moe->set_friendly_name(m.get_match_root()->get_friendly_name());
        ov::copy_runtime_info(m.get_matched_nodes(), moe);
        ov::replace_node(m.get_match_root(), moe);

        register_new_node(moe);
        return true;
    };

    auto matcher = std::make_shared<pattern::Matcher>(moe_pattern, matcher_name);
    this->register_matcher(matcher, callback);
}
