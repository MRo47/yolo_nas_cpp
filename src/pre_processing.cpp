#include "yolo_nas_cpp/pre_processing.hpp"

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>

namespace yolo_nas_cpp
{

cv::Size parse_cv_size(const json & shape_arr, const std::string & param_name = "output_shape")
{
  if (!shape_arr.is_array() || shape_arr.size() != 2) {
    throw std::runtime_error(
      "'" + param_name + "' must be a JSON array with exactly 2 elements [height, width].");
  }
  // JSON stores [height, width], cv::Size constructor takes (width, height)
  return cv::Size(shape_arr[1].get<int>(), shape_arr[0].get<int>());
}

std::unique_ptr<PreProcessingStep> PreProcessingStep::create_from_json(
  const std::string & step_name, const json & params)
{
  if (step_name == "StandardizeImage") {
    return std::make_unique<StandardizeImage>(params);
  } else if (step_name == "NormalizeImage") {
    return std::make_unique<NormalizeImage>(params);
  } else if (step_name == "DetectionCenterPadding") {
    return std::make_unique<DetectionCenterPadding>(params);
  } else if (step_name == "DetectionBottomRightPadding") {
    return std::make_unique<DetectionBottomRightPadding>(params);
  } else if (step_name == "ImagePermute") {
    return std::make_unique<ImagePermute>(params);
  } else if (step_name == "DetectionLongestMaxSizeRescale") {
    return std::make_unique<DetectionLongestMaxSizeRescale>(params);
  } else if (step_name == "DetectionRescale") {
    return std::make_unique<DetectionRescale>(params);
  } else {
    throw std::runtime_error("Unknown pre-processing step type requested: " + step_name);
  }
}

// PreProcessing implementation
PreProcessing::PreProcessing(const json & config)
{
  if (!config.is_array()) {
    throw std::runtime_error("Expected 'pre_processing' config to be a JSON array.");
  }

  for (const auto & step_config : config) {
    if (!step_config.is_object() || step_config.size() != 1) {
      throw std::runtime_error(
        "Each step in 'pre_processing' must be an object with exactly one key (the step name).");
    }

    auto it = step_config.begin();
    const std::string & step_name = it.key();
    const json & params = it.value();

    try {
      processing_steps_.emplace_back(
        PreProcessingStep::create_from_json(
          step_name, params));  // Move the created step into the vector
      std::cout << "Successfully added preprocessing step: " << step_name << std::endl;

    } catch (const std::exception & e) {
      throw std::runtime_error(
        "Error configuring pre-processing step '" + step_name + "': " + e.what());
    }
  }
  if (processing_steps_.empty()) {
    std::cerr << "Warning: Pre-processing configuration was empty or resulted in no steps."
              << std::endl;
  }
}

void PreProcessing::run(const cv::Mat & input, cv::Mat & output)
{
  if (processing_steps_.empty()) {
    input.copyTo(output);
    return;
  }

  cv::Mat temp_input = input;
  cv::Mat temp_output;

  for (size_t i = 0; i < processing_steps_.size(); ++i) {
    const auto & step = processing_steps_[i];
    step->apply(temp_input, temp_output);

    if (i < processing_steps_.size() - 1) {
      temp_input = temp_output;
    }
  }

  output = temp_output;
}

StandardizeImage::StandardizeImage(const json & params)
{
  try {
    max_value_ = params.at("max_value").get<double>();
    if (max_value_ <= 0) {
      throw std::invalid_argument("'max_value' must be positive.");
    }
  } catch (const json::exception & e) {
    throw std::runtime_error(
      "Error parsing parameters for StandardizeImage: " + std::string(e.what()));
  } catch (const std::invalid_argument & e) {
    throw std::runtime_error(
      "Invalid parameter value for StandardizeImage: " + std::string(e.what()));
  }
}

void StandardizeImage::apply(const cv::Mat & input, cv::Mat & output) const
{
  input.convertTo(output, CV_32F, 1.0 / max_value_);
}

std::string StandardizeImage::name() const { return "StandardizeImage"; }

NormalizeImage::NormalizeImage(const json & params)
{
  try {
    mean_ = params.at("mean").get<std::vector<double>>();
    std_ = params.at("std").get<std::vector<double>>();

    if (mean_.size() != 3 || std_.size() != 3) {
      throw std::runtime_error(
        "NormalizeImage requires 'mean' and 'std' to be arrays of exactly 3 numbers.");
    }
    // Basic check for non-zero std dev
    for (double val : std_) {
      if (std::abs(val) < 1e-9) {
        throw std::invalid_argument("Standard deviation values must be non-zero.");
      }
    }
  } catch (const json::exception & e) {
    throw std::runtime_error(
      "Error parsing parameters for NormalizeImage: " + std::string(e.what()));
  } catch (const std::invalid_argument & e) {
    throw std::runtime_error(
      "Invalid parameter value for NormalizeImage: " + std::string(e.what()));
  }
}

void NormalizeImage::apply(const cv::Mat & input, cv::Mat & output) const
{
  if (input.channels() != 3) {
    throw std::runtime_error("NormalizeImage::apply expects a 3-channel input image.");
  }
  cv::Mat input_float;
  if (input.depth() != CV_32F) {
    input.convertTo(input_float, CV_32F);
  } else {
    input_float = input;
  }

  output = (input_float - cv::Scalar(mean_[0], mean_[1], mean_[2])) /
           cv::Scalar(std_[0], std_[1], std_[2]);
}

std::string NormalizeImage::name() const { return "NormalizeImage"; }

DetectionCenterPadding::DetectionCenterPadding(const json & params)
{
  try {
    pad_value_ = params.at("pad_value").get<int>();
    out_shape_ = parse_cv_size(params.at("output_shape"));

    if (out_shape_.width <= 0 || out_shape_.height <= 0) {
      throw std::invalid_argument("output_shape dimensions must be positive.");
    }

  } catch (const json::exception & e) {
    throw std::runtime_error(
      "Error parsing parameters for DetectionCenterPadding: " + std::string(e.what()));
  } catch (const std::invalid_argument & e) {
    throw std::runtime_error(
      "Invalid parameter value for DetectionCenterPadding: " + std::string(e.what()));
  }
}

void DetectionCenterPadding::apply(const cv::Mat & input, cv::Mat & output) const
{
  if (input.rows > out_shape_.height || input.cols > out_shape_.width) {
    throw std::runtime_error(
      "DetectionCenterPadding::apply: Input image dimensions (" + std::to_string(input.cols) + "x" +
      std::to_string(input.rows) + ") exceed target output_shape (" +
      std::to_string(out_shape_.width) + "x" + std::to_string(out_shape_.height) + ").");
  }

  int pad_height = out_shape_.height - input.rows;
  int pad_width = out_shape_.width - input.cols;

  int pad_top = pad_height / 2;
  int pad_bottom = pad_height - pad_top;
  int pad_left = pad_width / 2;
  int pad_right = pad_width - pad_left;

  cv::copyMakeBorder(
    input, output, pad_top, pad_bottom, pad_left, pad_right, cv::BORDER_CONSTANT,
    cv::Scalar::all(pad_value_));
}

std::string DetectionCenterPadding::name() const { return "DetectionCenterPadding"; }

// DetectionBottomRightPadding implementation
DetectionBottomRightPadding::DetectionBottomRightPadding(const json & params)
{
  try {
    pad_value_ = params.at("pad_value").get<int>();
    out_shape_ = parse_cv_size(params.at("output_shape"));

    if (out_shape_.width <= 0 || out_shape_.height <= 0) {
      throw std::invalid_argument("output_shape dimensions must be positive.");
    }

  } catch (const json::exception & e) {
    throw std::runtime_error(
      "Error parsing parameters for DetectionBottomRightPadding: " + std::string(e.what()));
  } catch (const std::invalid_argument & e) {
    throw std::runtime_error(
      "Invalid parameter value for DetectionBottomRightPadding: " + std::string(e.what()));
  }
}

void DetectionBottomRightPadding::apply(const cv::Mat & input, cv::Mat & output) const
{
  if (input.rows > out_shape_.height || input.cols > out_shape_.width) {
    throw std::runtime_error(
      "DetectionBottomRightPadding::apply: Input image dimensions (" + std::to_string(input.cols) +
      "x" + std::to_string(input.rows) + ") exceed target output_shape (" +
      std::to_string(out_shape_.width) + "x" + std::to_string(out_shape_.height) + ").");
  }
  int pad_height = out_shape_.height - input.rows;
  int pad_width = out_shape_.width - input.cols;

  cv::copyMakeBorder(
    input, output, 0, pad_height, 0, pad_width, cv::BORDER_CONSTANT, cv::Scalar::all(pad_value_));
}

std::string DetectionBottomRightPadding::name() const { return "DetectionBottomRightPadding"; }

// ImagePermute implementation
ImagePermute::ImagePermute(const json & params)
{
  try {
    order_ = params.at("order").get<std::vector<int>>();

    // Check if order is the specifically implemented PyTorch conversion [2,0,1] (HWC->CHW)
    if (order_.size() != 3 || order_[0] != 2 || order_[1] != 0 || order_[2] != 1) {
      throw std::invalid_argument(
        "Only permutation order [2,0,1] (HWC->CHW) is currently implemented for ImagePermute.");
    }
  } catch (const json::exception & e) {
    throw std::runtime_error("Error parsing parameters for ImagePermute: " + std::string(e.what()));
  }
}

void ImagePermute::apply(const cv::Mat & input, cv::Mat & output) const
{
  // Ensure input is 3-channel HWC format expected for [2,0,1] permutation
  if (input.channels() != 3 || input.dims > 2) {
    throw std::runtime_error(
      "ImagePermute::apply with order [2,0,1] expects a 3-channel HWC input cv::Mat.");
  }

  const int H = input.rows;
  const int W = input.cols;
  const int C = input.channels();

  // Create output Mat with dimensions [C, H, W] and same depth as input
  // Note: OpenCV's Mat interpretation is different from pure numpy-style NCHW.
  // cv::dnn::blobFromImage handles this conversion correctly.
  // This manual implementation might be slow and less robust.
  // Consider using cv::dnn::blobFromImage if possible.
  // If manual permutation is strictly needed:
  int sizes[] = {C, H, W};
  output.create(3, sizes, input.depth());  // Create a 3D Mat (Channels, Height, Width)

  // Create views for each channel in the output Mat
  std::vector<cv::Mat> output_channels;
  for (int i = 0; i < C; ++i) {
    // Create a Mat header for the i-th channel slice in the 3D Mat
    // Note: This assumes data is contiguous in the C dimension first.
    // This might require careful handling depending on how Mat allocates 3D data.
    // A safer (but potentially slower) way is manual copying element by element.
    // The original code used element-wise copy, let's stick to that for correctness.
    // The performance might be an issue for large images.
  }

  // Split input into separate channel planes (more reliable)
  std::vector<cv::Mat> input_channels(C);
  cv::split(input, input_channels);

  // Copy the data with the new arrangement [C, H, W]
  // This loop assumes output is CV_MAKETYPE(input.depth(), 1) which might not be right
  // Let's stick to the safer element-wise copy from the original code, ensuring type safety.

  // Re-create output with the correct dimensions and type
  output.create(
    3, sizes,
    input.type());  // Use input.type() to preserve channels in type info for element access

  size_t elem_size =
    input.elemSize1();  // Size of one element in bytes (e.g., 1 for uchar, 4 for float)
  uchar * output_ptr = output.ptr<uchar>();

  for (int c = 0; c < C; ++c) {
    cv::Mat plane = input_channels[c];  // This is HxW single channel
    size_t plane_step = plane.step;     // Bytes per row in the plane
    uchar * plane_ptr = plane.ptr<uchar>();
    for (int h = 0; h < H; ++h) {
      uchar * row_ptr = plane_ptr + h * plane_step;
      for (int w = 0; w < W; ++w) {
        // Calculate the output address: output[c, h, w]
        // For C,H,W layout, index is c * (H*W) + h * W + w
        size_t output_idx = static_cast<size_t>(c) * H * W + static_cast<size_t>(h) * W + w;
        // Copy element bytes
        std::memcpy(output_ptr + output_idx * elem_size, row_ptr + w * elem_size, elem_size);
      }
    }
  }
  // Note: The resulting 'output' Mat has a 3D structure but OpenCV functions
  // might not interpret it as CHW directly. cv::dnn::blobFromImage is preferred
  // if the target is a DNN framework expecting NCHW format.
}

std::string ImagePermute::name() const { return "ImagePermute"; }

DetectionLongestMaxSizeRescale::DetectionLongestMaxSizeRescale(const json & params)
{
  try {
    out_shape_ = parse_cv_size(params.at("output_shape"));

    if (out_shape_.width <= 0 || out_shape_.height <= 0) {
      throw std::invalid_argument("output_shape dimensions must be positive.");
    }
  } catch (const json::exception & e) {
    throw std::runtime_error(
      "Error parsing parameters for DetectionLongestMaxSizeRescale: " + std::string(e.what()));
  } catch (const std::invalid_argument & e) {
    throw std::runtime_error(
      "Invalid parameter value for DetectionLongestMaxSizeRescale: " + std::string(e.what()));
  }
}

void DetectionLongestMaxSizeRescale::apply(const cv::Mat & input, cv::Mat & output) const
{
  if (input.empty()) {
    throw std::runtime_error(
      "DetectionLongestMaxSizeRescale::apply received an empty input image.");
  }
  float scale_factor = 1.0f;
  // Calculate scale factor based on limiting dimension
  if (input.rows > 0 && input.cols > 0) {
    scale_factor = std::min(
      static_cast<float>(out_shape_.height) / input.rows,
      static_cast<float>(out_shape_.width) / input.cols);
  } else {
    // Handle zero dimension case if necessary, maybe default to no resize?
    input.copyTo(output);
    return;
  }

  // Only resize if the scale factor is meaningfully different from 1
  if (std::abs(scale_factor - 1.0f) > 1e-6) {
    // Calculate new dimensions, rounding to nearest integer
    int new_height = static_cast<int>(std::round(input.rows * scale_factor));
    int new_width = static_cast<int>(std::round(input.cols * scale_factor));

    // Ensure dimensions are at least 1x1 if scale_factor is very small
    new_height = std::max(1, new_height);
    new_width = std::max(1, new_width);

    cv::resize(input, output, cv::Size(new_width, new_height), 0, 0, cv::INTER_LINEAR);
  } else {
    input.copyTo(output);
  }
}

std::string DetectionLongestMaxSizeRescale::name() const
{
  return "DetectionLongestMaxSizeRescale";
}

// DetectionRescale implementation
DetectionRescale::DetectionRescale(const json & params)
{
  try {
    out_shape_ = parse_cv_size(params.at("output_shape"));  // Use helper

    if (out_shape_.width <= 0 || out_shape_.height <= 0) {
      throw std::invalid_argument("output_shape dimensions must be positive.");
    }
  } catch (const json::exception & e) {
    throw std::runtime_error(
      "Error parsing parameters for DetectionRescale: " + std::string(e.what()));
  } catch (const std::invalid_argument & e) {
    throw std::runtime_error(
      "Invalid parameter value for DetectionRescale: " + std::string(e.what()));
  }
}

void DetectionRescale::apply(const cv::Mat & input, cv::Mat & output) const
{
  if (input.empty()) {
    throw std::runtime_error("DetectionRescale::apply received an empty input image.");
  }
  cv::resize(input, output, out_shape_, 0, 0, cv::INTER_LINEAR);
}

std::string DetectionRescale::name() const { return "DetectionRescale"; }

}  // namespace yolo_nas_cpp