#include "yolo_nas_cpp/pre_processing.hpp"

#include <algorithm>
#include <cstring>
#include <iostream>

namespace yolo_nas_cpp
{

// PreProcessing implementation
PreProcessing::PreProcessing(const json & config)
{
  if (!config.is_array()) {
    throw std::runtime_error("Expected 'pre_processing' to be a JSON array.");
  }

  for (const auto & step : config) {
    if (!step.is_object()) {
      throw std::runtime_error("Each step must be an object with one key.");
    }

    for (auto it = step.begin(); it != step.end(); ++it) {
      const std::string & step_name = it.key();
      const json & params = it.value();

      try {
        if (step_name == "StandardizeImage") {
          double max_value = params.at("max_value");
          processing_steps_.push_back(std::make_unique<StandardizeImage>(max_value));

        } else if (step_name == "NormalizeImage") {
          std::vector<double> mean = params.at("mean");
          std::vector<double> std = params.at("std");
          processing_steps_.push_back(
            std::make_unique<NormalizeImage>(std::move(mean), std::move(std)));

        } else if (step_name == "DetectionCenterPadding") {
          int pad_value = params.at("pad_value");
          const auto & out_shape_arr = params.at("output_shape");
          if (out_shape_arr.size() != 2) {
            throw std::runtime_error("out_shape must have exactly 2 elements");
          }
          cv::Size out_shape(out_shape_arr[1], out_shape_arr[0]);
          processing_steps_.push_back(
            std::make_unique<DetectionCenterPadding>(pad_value, out_shape));

        } else if (step_name == "DetectionBottomRightPadding") {
          int pad_value = params.at("pad_value");
          const auto & out_shape_arr = params.at("output_shape");
          if (out_shape_arr.size() != 2) {
            throw std::runtime_error("out_shape must have exactly 2 elements");
          }
          cv::Size out_shape(out_shape_arr[1], out_shape_arr[0]);
          processing_steps_.push_back(
            std::make_unique<DetectionBottomRightPadding>(pad_value, out_shape));

        } else if (step_name == "ImagePermute") {
          std::vector<int> order = params.at("order");
          processing_steps_.push_back(std::make_unique<ImagePermute>(std::move(order)));

        } else if (step_name == "DetectionLongestMaxSizeRescale") {
          const auto & shape_arr = params.at("output_shape");
          if (shape_arr.size() != 2) {
            throw std::runtime_error("output_shape must have exactly 2 elements");
          }
          cv::Size out_shape(shape_arr[1], shape_arr[0]);
          processing_steps_.push_back(std::make_unique<DetectionLongestMaxSizeRescale>(out_shape));

        } else if (step_name == "DetectionRescale") {
          const auto & shape_arr = params.at("output_shape");
          if (shape_arr.size() != 2) {
            throw std::runtime_error("output_shape must have exactly 2 elements");
          }
          cv::Size out_shape(shape_arr[1], shape_arr[0]);
          processing_steps_.push_back(std::make_unique<DetectionRescale>(out_shape));

        } else {
          throw std::runtime_error("pre-processing step not implemented: " + step_name);
        }
      } catch (const json::exception & e) {
        throw std::runtime_error(
          "Error parsing parameters for step '" + step_name + "': " + e.what());
      }

      std::cout << "Adding preprocessing step: " << step_name << std::endl;
    }
  }
}

void PreProcessing::run(const cv::Mat & input, cv::Mat & output)
{
  cv::Mat temp_input = input;
  cv::Mat temp_output;

  for (const auto & step : processing_steps_) {
    step->apply(temp_input, temp_output);
    temp_input = temp_output;
  }

  output = temp_output;
}

// StandardizeImage implementation
StandardizeImage::StandardizeImage(double max_value) : max_value_(max_value) {}

void StandardizeImage::apply(const cv::Mat & input, cv::Mat & output) const
{
  input.convertTo(output, CV_32F, 1.0 / max_value_);
}

std::string StandardizeImage::name() const { return "StandardizeImage"; }

// Normalize implementation
NormalizeImage::NormalizeImage(std::vector<double> mean, std::vector<double> std)
: mean_(std::move(mean)), std_(std::move(std))
{
  if (mean_.size() != 3 || std_.size() != 3) {
    throw std::runtime_error("Normalize requires mean and std of size 3.");
  }
}

void NormalizeImage::apply(const cv::Mat & input, cv::Mat & output) const
{
  output =
    (input - cv::Scalar(mean_[0], mean_[1], mean_[2])) / cv::Scalar(std_[0], std_[1], std_[2]);
}

std::string NormalizeImage::name() const { return "NormalizeImage"; }

// PadCenter implementation
DetectionCenterPadding::DetectionCenterPadding(int pad_value, const cv::Size & out_shape)
: pad_value_(pad_value), out_shape_(out_shape)
{
}

void DetectionCenterPadding::apply(const cv::Mat & input, cv::Mat & output) const
{
  auto pad_height = out_shape_.height - input.rows;
  auto pad_width = out_shape_.width - input.cols;

  auto pad_left = pad_width / 2;
  auto pad_top = pad_height / 2;

  cv::copyMakeBorder(
    input, output, pad_top, pad_height - pad_top, pad_left, pad_width - pad_left,
    cv::BORDER_CONSTANT, cv::Scalar(pad_value_, pad_value_, pad_value_));
}

std::string DetectionCenterPadding::name() const { return "DetectionCenterPadding"; }

// PadBottomRight implementation
DetectionBottomRightPadding::DetectionBottomRightPadding(int pad_value, const cv::Size & out_shape)
: pad_value_(pad_value), out_shape_(out_shape)
{
}

void DetectionBottomRightPadding::apply(const cv::Mat & input, cv::Mat & output) const
{
  auto pad_height = out_shape_.height - input.rows;
  auto pad_width = out_shape_.width - input.cols;

  cv::copyMakeBorder(
    input, output, 0, pad_height, 0, pad_width, cv::BORDER_CONSTANT,
    cv::Scalar(pad_value_, pad_value_, pad_value_));
}

std::string DetectionBottomRightPadding::name() const { return "DetectionBottomRightPadding"; }

// ImagePermute implementation
ImagePermute::ImagePermute(std::vector<int> order = {2, 0, 1}) : order_(std::move(order))
{
  // Check if order is the PyTorch conversion [2,0,1]
  if (order_.size() != 3 || order_[0] != 2 || order_[1] != 0 || order_[2] != 1) {
    throw std::invalid_argument("Only permutation order [2,0,1] (HWC->CHW) is implemented");
  }
}

void ImagePermute::apply(const cv::Mat & input, cv::Mat & output) const
{
  // Get dimensions
  const int H = input.rows;        // Height
  const int W = input.cols;        // Width
  const int C = input.channels();  // Channels

  // Split input into separate channel planes
  std::vector<cv::Mat> channels(C);
  cv::split(input, channels);

  // Create output with dimensions [C, H, W]
  int sizes[] = {C, H, W};
  output.create(3, sizes, CV_MAKETYPE(input.depth(), 1));

  // Copy the data with the new arrangement
  for (int c = 0; c < C; ++c) {
    for (int h = 0; h < H; ++h) {
      for (int w = 0; w < W; ++w) {
        output.at<uchar>(c, h, w) = channels[c].at<uchar>(h, w);
      }
    }
  }
}

std::string ImagePermute::name() const { return "ImagePermute"; }

// DetectionLongestMaxSizeRescale implementation
DetectionLongestMaxSizeRescale::DetectionLongestMaxSizeRescale(const cv::Size & out_shape)
: out_shape_(out_shape)
{
}

void DetectionLongestMaxSizeRescale::apply(const cv::Mat & input, cv::Mat & output) const
{
  auto scale_factor = std::min(
    out_shape_.height / static_cast<float>(input.rows),
    out_shape_.width / static_cast<float>(input.cols));

  if (std::abs(scale_factor - 1.0f) > 1e-6) {
    auto new_height = static_cast<int>(std::round(input.rows * scale_factor));
    auto new_width = static_cast<int>(std::round(input.cols * scale_factor));
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
DetectionRescale::DetectionRescale(const cv::Size & out_shape) : out_shape_(out_shape) {}

void DetectionRescale::apply(const cv::Mat & input, cv::Mat & output) const
{
  cv::resize(input, output, out_shape_, 0, 0, cv::INTER_LINEAR);
}

std::string DetectionRescale::name() const { return "DetectionRescale"; }

}  // namespace yolo_nas_cpp