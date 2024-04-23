// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/clean_repeated_ptr_shifts.hpp"

#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/loop_manager.hpp"
#include "snippets/snippets_isa.hpp"
#include "snippets/itt.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

bool CleanRepeatedDataPointerShifts::reuse_increments(const LoopManagerPtr& loop_manager, const ExpressionPtr& loop_end_expr) {
    const auto loop_end = ov::as_type_ptr<op::LoopEnd>(loop_end_expr->get_node());
    if (!loop_end)
        return false;

    const auto& loop_connectors = loop_end_expr->get_input_port_connectors();
    const auto input_count = loop_end->get_input_num();
    const auto output_count = loop_end->get_output_num();

    std::set<size_t> resetting_data_indexes;
    std::set<size_t> buffers_ids;
    // We count expressions only on inputs of Loop because we can only read from the same data but not write to the same data.
    //       Parameter
    //        |    |
    //    Load_0  Load_1
    std::set<ExpressionPtr> read_data_exprs;
    for (size_t i = 0; i < input_count; ++i) {
        const auto& parent_output = loop_connectors[i]->get_source().get_expr();
        if (const auto buffer = ov::as_type_ptr<op::Buffer>(parent_output->get_node())) {
            // If Buffer is missed in set, Just save - it's first meeting
            if (buffers_ids.count(buffer->get_id()) == 0) {
                buffers_ids.insert(buffer->get_id());
            } else {
                // The Buffer with the same ID is in set - need to add this Buffer idx to set of Buffers for resetting
                resetting_data_indexes.insert(i);
            }
        } else {
            // Remember the current expression if missed
            if (read_data_exprs.count(parent_output) == 0) {
                read_data_exprs.insert(parent_output);
            } else {
                // Otherwise we have several Load-semantic expressions which read from the same data.
                // Have to zero ptr increments and finalization offsets for all expression except one.
                resetting_data_indexes.insert(i);
            }
        }
    }
    for (size_t i = 0; i < output_count; ++i) {
        const auto consumer_inputs = loop_connectors[input_count + i]->get_consumers();
        size_t buffer_count = 0;
        size_t loop_count = 0;
        for (const auto& consumer_input : consumer_inputs) {
            const auto& child_node = consumer_input.get_expr()->get_node();
            if (const auto buffer = ov::as_type_ptr<op::Buffer>(child_node)) {
                buffer_count++;
                // If Buffer is missed in set, Just save - it's first meeting
                if (buffers_ids.count(buffer->get_id()) == 0) {
                    buffers_ids.insert(buffer->get_id());
                } else {
                    // The Buffer with the same ID is in set - need to add this Buffer idx to set of Buffers for resetting
                    resetting_data_indexes.insert(input_count + i);
                }
            } else if (ov::is_type<op::LoopEnd>(child_node)) {
                loop_count++;
            }
        }
        if (buffer_count > 0) {
            OPENVINO_ASSERT((buffer_count == 1) && (buffer_count + loop_count == consumer_inputs.size()),
                            "Loop output must have not more than 1 Buffer");
        }
    }

    if (resetting_data_indexes.empty())
        return false;

    const auto loop_info = loop_manager->get_loop_info(loop_end->get_id());

    // TODO [133463]: We have to update LoopEnd and LoopInfo since the both entities must be valid.
    //                To avoid the both changes, we have to insert Loop ops to LinearIR in the end of pipeline.
    auto loop_entries = loop_info->get_entry_points();
    auto loop_exits = loop_info->get_exit_points();
    auto new_is_incremented = loop_end->get_is_incremented();
    if (const auto loop_end_dynamic = ov::as_type_ptr<op::LoopEndDynamic>(loop_end_expr->get_node())) {
        for (auto idx_to_drop : resetting_data_indexes) {
            new_is_incremented[idx_to_drop] = false;
            auto& loop_port = idx_to_drop < input_count ? loop_entries[idx_to_drop] : loop_exits[idx_to_drop - input_count];
            loop_port.is_incremented = false;
        }
    } else if (const auto loop_end_static = ov::as_type_ptr<op::LoopEndStatic>(loop_end_expr->get_node())) {
        auto new_ptr_increments = loop_end_static->get_ptr_increments();
        auto new_finalization_offsets = loop_end_static->get_finalization_offsets();
        for (auto idx_to_drop : resetting_data_indexes) {
            new_ptr_increments[idx_to_drop] = 0;
            new_finalization_offsets[idx_to_drop] = 0;
            new_is_incremented[idx_to_drop] = false;
            auto& loop_port = idx_to_drop < input_count ? loop_entries[idx_to_drop] : loop_exits[idx_to_drop - input_count];
            loop_port.ptr_increment = 0;
            loop_port.finalization_offset = 0;
            loop_port.is_incremented = false;
        }
        loop_end_static->set_ptr_increments(new_ptr_increments);
        loop_end_static->set_finalization_offsets(new_finalization_offsets);
    }
    loop_end->set_is_incremented(new_is_incremented);
    loop_info->set_entry_points(loop_entries);
    loop_info->set_exit_points(loop_exits);
    return true;
}

bool CleanRepeatedDataPointerShifts::run(lowered::LinearIR& linear_ir, lowered::LinearIR::constExprIt begin, lowered::LinearIR::constExprIt end) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::CleanRepeatedDataPointerShifts")
    bool modified = false;

    const auto& loop_manager = linear_ir.get_loop_manager();
    for (auto expr_it = begin; expr_it != end; ++expr_it) {
        const auto& expr = *expr_it;
        const auto& node = expr->get_node();
        if (ov::is_type<op::LoopEnd>(node)) {
            modified |= reuse_increments(loop_manager, expr);
        }
    }

    return modified;
}

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
