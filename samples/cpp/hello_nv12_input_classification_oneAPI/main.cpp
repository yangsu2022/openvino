// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// for openclinterop part by yangsu
#include <CL/opencl.h>
#include <CL/sycl.hpp>
// #include <CL/sycl/backend/opencl.hpp>

//--- for remote plugin by yangsu
// #pragma once
// #include <openvino/runtime/core.hpp>
#include <openvino/openvino.hpp>
#include <openvino/runtime/intel_gpu/ocl/ocl.hpp>
// #include <openvino/runtime/intel_gpu/ocl/va.hpp>

//--- end

// copy from oneAPI sample sepia-filter
#include "device_selector.hpp"
// dpc_common.hpp can be found in the dev-utilities include folder.
// e.g., $ONEAPI_ROOT/dev-utilities/<version>/include/dpc_common.hpp
#include "dpc_common.hpp"

// stb/*.h files can be found in the dev-utilities include folder.
// e.g., $ONEAPI_ROOT/dev-utilities/<version>/include/stb/*.h
#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

#include <sys/stat.h>
#include <chrono>
#include <cassert>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#ifdef _WIN32
#include "samples/os/windows/w_dirent.h"
#else
#include <dirent.h>
#endif

// clang-format off
#include "openvino/openvino.hpp"

#include "samples/args_helper.hpp"
#include "samples/common.hpp"
#include "samples/slog.hpp"
#include "samples/classification_results.h"
#include "format_reader_ptr.h"
// clang-format on

constexpr auto N_TOP_RESULTS = 3;

using namespace std;
using namespace ov::preprocess;
using namespace sycl;

// ---
// Few useful acronyms.
constexpr auto sycl_read = access::mode::read;
constexpr auto sycl_write = access::mode::write;
// constexpr auto sycl_global_buffer = access::target::global_buffer;

static void ReportTime(const string &msg, event e)
{
    cl_ulong time_start =
        e.get_profiling_info<info::event_profiling::command_start>();

    cl_ulong time_end =
        e.get_profiling_info<info::event_profiling::command_end>();

    double elapsed = (time_end - time_start) / 1e6;
    cout << msg << elapsed << " milliseconds\n";
}

__attribute__((always_inline)) static void ApplyFilter(uint8_t *src_image,
                                                       uint8_t *dst_image,
                                                       int i)
{
    i *= 3;
    float temp;
    temp = (0.393f * src_image[i]) + (0.769f * src_image[i + 1]) +
           (0.189f * src_image[i + 2]);
    dst_image[i] = temp > 255 ? 255 : temp;
    temp = (0.349f * src_image[i]) + (0.686f * src_image[i + 1]) +
           (0.168f * src_image[i + 2]);
    dst_image[i + 1] = temp > 255 ? 255 : temp;
    temp = (0.272f * src_image[i]) + (0.534f * src_image[i + 1]) +
           (0.131f * src_image[i + 2]);
    dst_image[i + 2] = temp > 255 ? 255 : temp;
}

// --- end

// test for kernel
/*
    buffer sycl_image_buf(image, range(img_size));

    // get Buffer from the host image data
    // Using these events to time command group execution
    event e2;
    e2 = sycl_queue.submit([&](auto &h) {

        accessor image_acc(sycl_image_buf, h);

        h.parallel_for(range<1>(num_pixels), [=](auto i) {
            // test kernel
            image_acc[i]++;
            });
        });
    host_accessor host_accessor(sycl_image_buf);
    sycl_queue.wait_and_throw();
*/

/**
 * @brief The entry point of the OpenVINO Runtime sample application
 */
int main(int argc, char *argv[])
{
    // -------- Read image names --------
    int img_width, img_height, channels;
    uint8_t *image = stbi_load(argv[2], &img_width, &img_height, &channels, 0);
    if (image == NULL)
    {
        cout << "Error in loading the image\n";
        exit(1);
    }
    cout << "Loaded image with a width of " << img_width << ", a height of "
         << img_height << " and " << channels << " channels\n";
    size_t input_width = img_width;
    size_t input_height = img_height;
    size_t num_pixels = img_width * img_height;
    size_t img_size = img_width * img_height * channels;

    uint8_t *image_output = new uint8_t[img_size];
    uint8_t *image_output_ref = new uint8_t[img_size];

    memset(image_output, 0, img_size * sizeof(uint8_t));
    memset(image_output_ref, 0, img_size * sizeof(uint8_t));
    // try
    //     {
    // -------- Get OpenVINO runtime version --------
    slog::info << ov::get_openvino_version() << slog::endl;

    // -------- Parsing and validation input arguments --------
    if (argc != 5)
    {
        std::cout << "Usage : " << argv[0] << " <path_to_model> <path_to_image> <image_size> <device_name> <iteraton_number>"
                  << std::endl;
        return EXIT_FAILURE;
    }

    const std::string model_path{argv[1]};
    const std::string image_path{argv[2]};
    const std::string device_name{argv[3]};
    const std::string iter_num_string{argv[4]};
    auto iter_num = stoi(iter_num_string);

    size_t batch = 1;

    // Read labels from file (e.x. AlexNet.labels)
    std::string labelFileName = fileNameNoExt(model_path) + ".labels";
    std::vector<std::string> labels;

    std::ifstream inputFile;
    inputFile.open(labelFileName, std::ios::in);
    if (inputFile.is_open())
    {
        std::string strLine;
        while (std::getline(inputFile, strLine))
        {
            trim(strLine);
            labels.push_back(strLine);
        }
    }

    // -------- Step 1. Initialize OpenVINO Runtime Core ---------
    ov::Core core;
    core.set_property(ov::cache_dir("~/cache"));
    // -------- Step 2. Read a model --------
    slog::info << "Loading model files: " << model_path << slog::endl;
    std::shared_ptr<ov::Model> model = core.read_model(model_path);
    // printInputAndOutputsInfo(*model);

    OPENVINO_ASSERT(model->inputs().size() == 1, "Sample supports models with 1 input only");
    OPENVINO_ASSERT(model->outputs().size() == 1, "Sample supports models with 1 output only");

    std::string input_tensor_name = model->input().get_any_name();
    std::string output_tensor_name = model->output().get_any_name();

    // -------- Step 3. Configure preprocessing  --------

    PrePostProcessor ppp = PrePostProcessor(model);

    // 1) Select input with 'input_tensor_name' tensor name
    InputInfo &input_info = ppp.input(input_tensor_name);
    // 2) Set input type
    // - as 'u8' precision
    // - set color format to NV12 (single plane)
    // - static spatial dimensions for resize preprocessing operation
    input_info.tensor()
        .set_element_type(ov::element::u8)
        // .set_color_format(ColorFormat::NV12_SINGLE_PLANE)
        .set_color_format(ColorFormat::RGB)
        .set_spatial_static_shape(input_height, input_width);
    // 3) Pre-processing steps:
    //    a) Convert to 'float'. This is to have color conversion more accurate
    //    b) Convert to BGR: Assumes that model accepts images in BGR format. For RGB, change it manually
    //    c) Resize image from tensor's dimensions to model ones
    input_info.preprocess()
        .convert_element_type(ov::element::f32)
        // .convert_color(ColorFormat::RGB)
        .resize(ResizeAlgorithm::RESIZE_LINEAR);
    // 4) Set model data layout (Assuming model accepts images in NCHW layout)
    input_info.model().set_layout("NCHW");

    // 5) Apply preprocessing to an input with 'input_tensor_name' name of loaded model
    model = ppp.build();

    // yangsu
    // read image via SYCL and send the input into OV
    MyDeviceSelector selector;

    auto prop_list = property_list{property::queue::enable_profiling()};
    queue sycl_queue(selector, dpc_common::exception_handler, prop_list);
    std::cout << "Running on "
              << sycl_queue.get_device().get_info<info::device::name>()
              << std::endl;
    // Creation of RemoteContext from Native Handle
    auto cl_queue = get_native<backend::opencl>(sycl_queue);
    auto remote_context = ov::intel_gpu::ocl::ClContext(core, cl_queue);

    const auto t0 = std::chrono::high_resolution_clock::now();

    // -------- Step 4. Loading a model to the device --------
    ov::CompiledModel compiled_model = core.compile_model(model, remote_context);

    // ov::CompiledModel compiled_model = core.compile_model(model, device_name);
    std::cout << "---Load model";
    const auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << " - " << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << "ms\n";

    // -------- Step 5. Create an infer request --------
    ov::InferRequest infer_request = compiled_model.create_infer_request();

    std::cout << "---Create an infer request";
    const auto t2 = std::chrono::high_resolution_clock::now();

    std::cout << " - " << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() * 0.001 << "ms\n";
    // -------- Step 6. Prepare input data  --------

    // get ov::Tensor

    // ov::Tensor input_tensor{ov::element::u8, {batch, input_height, input_width, 3}, image_data.get()};
    // original: nv12 yuv input image
    // model input (shape={1,450,300,1})
    // ov::Tensor input_tensor{ov::element::u8, {batch, input_height * 3 / 2, input_width, 1}, image_data.get()};

    cout << "Use remote tensor API and set_tensor\n";

    int i = 0;
    while (i < iter_num)
    {
        std::cout << "No " << i + 1 << ". do inf: \n";

        const auto t_k_start = std::chrono::high_resolution_clock::now();

        buffer image_buf(image, range(img_size));
        buffer image_buf_out(image_output, range(img_size));
        // buffer<int> image_buf_out{range(img_size)};

        std::cout << "---sycl buffer ";
        const auto t_buffer = std::chrono::high_resolution_clock::now();
        std::cout << " - " << std::chrono::duration_cast<std::chrono::microseconds>(t_buffer - t_k_start).count() * 0.001 << "ms\n";

        event e_filter;
        e_filter = sycl_queue.submit([&](auto &h)
                                     {

            accessor image_acc(image_buf, h, read_only);
            accessor image_out_acc(image_buf_out, h, write_only);

            h.parallel_for(range<1>(num_pixels), [=](auto i) {
                ApplyFilter(image_acc.get_pointer(), image_out_acc.get_pointer(), i);
            }); });
        sycl_queue.wait_and_throw();

        std::cout << "---sycl filter total time";
        const auto t_k_end = std::chrono::high_resolution_clock::now();
        std::cout << " - " << std::chrono::duration_cast<std::chrono::microseconds>(t_k_end - t_k_start).count() * 0.001 << "ms\n";

        // report execution times:
        ReportTime("---kernel time: ", e_filter);

        // -------- Step 6. Set input tensor  --------
        // -------- remote tensor API --------

        // auto cl_buffers = get_native<backend::opencl>(sycl_image_buf);
        // auto cl_buffers = get_native<backend::opencl>(image_buf);
        auto cl_buffers = get_native<backend::opencl>(image_buf_out);

        auto remote_tensor = remote_context.create_tensor(ov::element::u8, {batch, input_height, input_width, 3}, cl_buffers);
        infer_request.set_tensor(input_tensor_name, remote_tensor);
        std::cout << "---Set tensor";
        const auto t3 = std::chrono::high_resolution_clock::now();
        // std::cout << " - " << std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count() * 0.001 << "ms\n";
        std::cout << " - " << std::chrono::duration_cast<std::chrono::microseconds>(t3 - t_k_end).count() * 0.001 << "ms\n";

        // -------- Step 7. Do inference --------
        // Running the request synchronously
        // infer_request.infer();
        // async
        infer_request.start_async();
        infer_request.wait();

        std::cout << "---Run infer req";
        const auto t4 = std::chrono::high_resolution_clock::now();
        std::cout << " - " << std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count() * 0.001 << "ms\n";
        // const auto start_time = std::chrono::high_resolution_clock::now();
        // const auto end_time = std::chrono::high_resolution_clock::now();
        // std::cout << "Execution time: "
        //         << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() << "ms\n\n";

        // -------- Step 8. Process output --------

        ov::Tensor output = infer_request.get_tensor(output_tensor_name);

        std::cout << "---get tensor";
        const auto t5 = std::chrono::high_resolution_clock::now();
        std::cout << " - " << std::chrono::duration_cast<std::chrono::microseconds>(t5 - t4).count() * 0.001 << "ms\n";

        // Print classification results
        ClassificationResult classification_result(output, {image_path}, batch, N_TOP_RESULTS, labels);
        classification_result.show();

        std::cout << "---sum of inference";
        std::cout << " - " << std::chrono::duration_cast<std::chrono::microseconds>(t5 - t_k_start).count() * 0.001 << "ms\n";

        i++;
    }

    // yangsu

    // } catch (const std::exception &ex) {
    //     std::cerr << ex.what() << std::endl;

    //     return EXIT_FAILURE;
    // }
    // return EXIT_SUCCESS;

    // check
    // get reference result
    for (size_t i = 0; i < num_pixels; i++)
    {
        ApplyFilter(image, image_output_ref, i);
    }

    stbi_write_png("image_out.png", img_width, img_height, channels,
                   image_output, img_width * channels);
    stbi_write_png("image.png", img_width, img_height, channels,
                   image, img_width * channels);
    stbi_write_png("image_ref.png", img_width, img_height, channels, image_output_ref, img_width * channels);
    stbi_image_free(image);
    delete[] image_output;
    delete[] image_output_ref;
}