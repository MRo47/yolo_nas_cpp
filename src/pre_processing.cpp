#include "yolo_nas_cpp/pre_processing.hpp"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <stdexcept>
#include <string>

#include "yolo_nas_cpp/utils.hpp"

namespace yolo_nas_cpp
{

std::pair<std::unique_ptr<PreProcessingStep>, PreProcessingMetadata>
PreProcessingStep::create_from_json(
  const std::string & step_name, const json & params, const cv::Size & input_shape)
{
  std::unique_ptr<PreProcessingStep> step_ptr;

  PreProcessingMetadata metadata;
  metadata.step_name = step_name;
  metadata.output_shape = {-1, -1};
  metadata.input_shape = input_shape;
  metadata.params = params;

  // Create the specific step (constructors now may need input_shape)
  if (step_name == "StandardizeImage") {
    step_ptr = std::make_unique<StandardizeImage>(params);
    metadata.output_shape = step_ptr->calculate_output_shape(input_shape);
  } else if (step_name == "NormalizeImage") {
    step_ptr = std::make_unique<NormalizeImage>(params);
    metadata.output_shape = step_ptr->calculate_output_shape(input_shape);
  } else if (step_name == "DetectionCenterPadding") {
    step_ptr = std::make_unique<DetectionCenterPadding>(params);
    metadata.output_shape = step_ptr->calculate_output_shape(input_shape);
  } else if (step_name == "DetectionBottomRightPadding") {
    step_ptr = std::make_unique<DetectionBottomRightPadding>(params);
    metadata.output_shape = step_ptr->calculate_output_shape(input_shape);
  } else if (step_name == "ImagePermute") {
    step_ptr = std::make_unique<PassthroughStep>(params);
    metadata.output_shape = step_ptr->calculate_output_shape(input_shape);
  } else if (step_name == "DetectionLongestMaxSizeRescale") {
    step_ptr = std::make_unique<DetectionLongestMaxSizeRescale>(params);
    metadata.output_shape = step_ptr->calculate_output_shape(input_shape);
  } else if (step_name == "DetectionRescale") {
    step_ptr = std::make_unique<DetectionRescale>(params);
    metadata.output_shape = step_ptr->calculate_output_shape(input_shape);
  } else {
    throw std::runtime_error("Unknown pre-processing step type requested: " + step_name);
  }

  return {std::move(step_ptr), std::move(metadata)};
}

PreProcessing::PreProcessing(const json & config, const cv::Size input_shape)
{
  if (!config.is_array()) {
    throw std::runtime_error("Expected 'pre_processing' config to be a JSON array.");
  }

  spdlog::info("Initializing PreProcessing pipeline...");

  cv::Size current_shape = input_shape;

  for (const auto & step_config : config) {
    if (!step_config.is_object() || step_config.size() != 1) {
      throw std::runtime_error(
        "Each step in 'pre_processing' must be an object with exactly one key (the step name).");
    }

    auto it = step_config.begin();
    const std::string & step_name = it.key();
    const json & params = it.value();

    try {
      // Create step and get metadata, passing the output shape from the previous step
      auto [step_ptr, metadata] =
        PreProcessingStep::create_from_json(step_name, params, current_shape);

      processing_steps_.push_back(std::move(step_ptr));
      metadata_.push_back(std::move(metadata));

      current_shape = metadata_.back().output_shape;

      spdlog::info(
        "+ Added step: {} (Input: {}x{}, Output: {}x{})", step_name,
        metadata_.back().input_shape.width, metadata_.back().input_shape.height,
        metadata_.back().output_shape.width, metadata_.back().output_shape.height);

    } catch (const std::exception & e) {
      throw std::runtime_error(
        "Error configuring pre-processing step '" + step_name + "': " + e.what());
    }
  }

  if (processing_steps_.empty()) {
    spdlog::warn("Pre-processing configuration was empty or resulted in no steps.");
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

    // shallow copy until last step
    if (i < processing_steps_.size() - 1) {
      temp_input = temp_output;
    }
  }

  output = temp_output;
}

const std::vector<PreProcessingMetadata> & PreProcessing::get_metadata() const { return metadata_; }

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

cv::Size StandardizeImage::calculate_output_shape(const cv::Size & input_shape) const
{
  return input_shape;
}

NormalizeImage::NormalizeImage(const json & params)
{
  try {
    mean_ = params.at("mean").get<std::vector<double>>();
    std_ = params.at("std").get<std::vector<double>>();

    if (mean_.size() != 3 || std_.size() != 3) {
      throw std::runtime_error(
        "NormalizeImage requires 'mean' and 'std' to be arrays of exactly 3 numbers (1 per "
        "channel).");
    }
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
  cv::subtract(
    input_float, cv::Scalar(mean_[0], mean_[1], mean_[2]), output);   // output = input_float - mean
  cv::divide(output, cv::Scalar(std_[0], std_[1], std_[2]), output);  // output = output / std
}

std::string NormalizeImage::name() const { return "NormalizeImage"; }

cv::Size NormalizeImage::calculate_output_shape(const cv::Size & input_shape) const
{
  return input_shape;
}

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
  int content_width = std::min(input.cols, out_shape_.width);
  int content_height = std::min(input.rows, out_shape_.height);

  // Calculate the top-left coordinate (x, y) of the Region of Interest (ROI)
  // within the *input* image. This ROI is the portion of the input image
  // that will be placed into the center of the output image.
  // If input is larger than output, this calculates the start of the centered crop.
  // If input is smaller or equal, this is 0.
  int roi_x = (input.cols - content_width) / 2;
  int roi_y = (input.rows - content_height) / 2;

  // Create the ROI from the input image. This is the part that is NOT cropped.
  cv::Mat content_roi = input(cv::Rect(roi_x, roi_y, content_width, content_height));

  // Now, calculate the padding needed *around* this 'content_roi' to make it
  int pad_left = (out_shape_.width - content_width) / 2;
  int pad_right =
    (out_shape_.width - content_width) - pad_left;  // Ensures total width difference is handled
  int pad_top = (out_shape_.height - content_height) / 2;
  int pad_bottom =
    (out_shape_.height - content_height) - pad_top;  // Ensures total height difference is handled

  cv::copyMakeBorder(
    content_roi, output, pad_top, pad_bottom, pad_left, pad_right, cv::BORDER_CONSTANT,
    cv::Scalar::all(static_cast<double>(pad_value_)));
}

std::string DetectionCenterPadding::name() const { return "DetectionCenterPadding"; }

cv::Size DetectionCenterPadding::calculate_output_shape(const cv::Size & /*input_shape*/) const
{
  return out_shape_;
}

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
  int content_width = std::min(input.cols, out_shape_.width);
  int content_height = std::min(input.rows, out_shape_.height);

  // For bottom-right alignment, the ROI always starts at (0,0) in the input image.
  // Cropping effectively happens from the bottom and right sides of the input.
  int roi_x = 0;
  int roi_y = 0;

  // Create the ROI from the input image. This is the portion that is NOT cropped
  // from the top or left.
  cv::Mat content_roi = input(cv::Rect(roi_x, roi_y, content_width, content_height));

  // Calculate the padding needed *around* this 'content_roi' to make it
  // fill the 'out_shape_'.
  // For bottom-right padding, all padding is on the bottom and right.
  // These padding values will always be non-negative.
  int pad_left = 0;
  int pad_top = 0;
  int pad_right = out_shape_.width - content_width;     // Will be >= 0
  int pad_bottom = out_shape_.height - content_height;  // Will be >= 0

  cv::copyMakeBorder(
    content_roi, output, pad_top, pad_bottom, pad_left, pad_right, cv::BORDER_CONSTANT,
    cv::Scalar::all(static_cast<double>(pad_value_)));  // Cast pad_value to double
}

std::string DetectionBottomRightPadding::name() const { return "DetectionBottomRightPadding"; }

cv::Size DetectionBottomRightPadding::calculate_output_shape(const cv::Size & /*input_shape*/) const
{
  return out_shape_;
}

PassthroughStep::PassthroughStep(const json & /*params*/)
{
  // Constructor body is empty, parameters are ignored.
}

void PassthroughStep::apply(const cv::Mat & input, cv::Mat & output) const
{
  // Perform a shallow copy: output header points to input data buffer.
  output = input;
}

std::string PassthroughStep::name() const { return "PassthroughStep"; }

cv::Size PassthroughStep::calculate_output_shape(const cv::Size & input_shape) const
{
  return input_shape;
}

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

  auto output_shape = calculate_output_shape(input.size());

  if (output_shape.width <= 0 || output_shape.height <= 0) {
    throw std::runtime_error(
      "DetectionLongestMaxSizeRescale::apply could not determine output shape.");
  }

  if (input.size() != output_shape) {
    cv::resize(input, output, output_shape, 0, 0, cv::INTER_LINEAR);
    return;
  }

  input.copyTo(output);
}

std::string DetectionLongestMaxSizeRescale::name() const
{
  return "DetectionLongestMaxSizeRescale";
}

// Calculate the exact output shape if input shape is known, otherwise return unknown
cv::Size DetectionLongestMaxSizeRescale::calculate_output_shape(const cv::Size & input_shape) const
{
  if (input_shape.width <= 0 || input_shape.height <= 0) {
    spdlog::warn("DetectionLongestMaxSizeRescale input shape is unknown.");
    return {-1, -1};
  }

  float scale_factor = std::min(
    static_cast<float>(out_shape_.height) / static_cast<float>(input_shape.height),
    static_cast<float>(out_shape_.width) / static_cast<float>(input_shape.width));

  if (std::abs(scale_factor - 1.0f) < 1e-6) {
    return input_shape;
  }

  int new_height = static_cast<int>(std::round(input_shape.height * scale_factor));
  int new_width = static_cast<int>(std::round(input_shape.width * scale_factor));

  return {new_width, new_height};
}

DetectionRescale::DetectionRescale(const json & params)
{
  try {
    out_shape_ = parse_cv_size(params.at("output_shape"));

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

cv::Size DetectionRescale::calculate_output_shape(const cv::Size & /*input_shape*/) const
{
  return out_shape_;
}

}  // namespace yolo_nas_cpp