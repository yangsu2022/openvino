# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
set(CMAKE_CXX_COMPILER dpcpp)
set(CMAKE_CXX_STANDARD 17)
set(TARGET_NAME "hello_nv12_input_classification_oneAPI")

ie_add_sample(NAME ${TARGET_NAME}
              SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp"
              DEPENDENCIES format_reader ie_samples_utils)
              
target_link_libraries(${TARGET_NAME} PRIVATE openvino::runtime OpenCL sycl)