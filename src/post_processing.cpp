#include "yolo_nas_cpp/post_processing.hpp"

#include <algorithm>  // For std::reverse, std::max/min
#include <iostream>
#include <numeric>          // For std::iota
#include <opencv2/dnn.hpp>  // For NMSBoxes
#include <stdexcept>
#include <vector>

#include "yolo_nas_cpp/utils.hpp"

namespace yolo_nas_cpp
{

PostProcessing::PostProcessing(
  const json & post_processing_config,
  const std::vector<PreProcessingMetadata> & pre_processing_metadata)
{
  std::cout << "Initializing PostProcessing pipeline..." << std::endl;

  // 1. Add NMS Step
  try {
    const auto & nms_conf = post_processing_config.at("NMS");
    float conf_threshold = nms_conf.at("conf").get<float>();
    float iou_threshold = nms_conf.at("iou").get<float>();
    post_processing_steps_.push_back(
      std::make_unique<NonMaximumSuppression>(conf_threshold, iou_threshold));
    std::cout << "  Added step: NonMaximumSuppression (Conf: " << conf_threshold
              << ", IoU: " << iou_threshold << ")" << std::endl;
  } catch (const std::exception & e) {
    throw std::runtime_error("Failed to parse NMS configuration: " + std::string(e.what()));
  }

  // 2. Add Inverse Preprocessing Steps (in reverse order with size tracking)
  std::cout << "  Adding inverse geometric transformations..." << std::endl;

  // Iterate backwards through the preprocessing metadata
  for (auto it = pre_processing_metadata.rbegin(); it != pre_processing_metadata.rend(); ++it) {
    const auto & metadata = *it;

    std::cout << "    Processing PreStep: " << metadata.step_name
              << " (Input: " << metadata.input_shape << ", Output: " << metadata.output_shape << ")"
              << std::endl;

    // Determine which inverse step to add based on the pre-processing step name
    try {
      if (
        metadata.step_name == "DetectionLongestMaxSizeRescale" ||
        metadata.step_name == "DetectionRescale") {
        post_processing_steps_.emplace_back(
          std::make_unique<RescaleBoxes>(metadata.output_shape, metadata.input_shape));
        std::cout << "    Added step: RescaleBoxes (From " << metadata.output_shape << " to "
                  << metadata.input_shape << ")" << std::endl;

      } else if (metadata.step_name == "DetectionCenterPadding") {
        post_processing_steps_.emplace_back(
          std::make_unique<ShiftBoxes>(
            metadata.output_shape, metadata.input_shape, ShiftBoxes::PaddingType::CENTER));
        std::cout << "    Added step: ShiftBoxes (CENTER, From " << metadata.output_shape
                  << " to " << metadata.input_shape << ")" << std::endl;

      } else if (metadata.step_name == "DetectionBottomRightPadding") {
        post_processing_steps_.emplace_back(
          std::make_unique<ShiftBoxes>(
            metadata.output_shape, metadata.input_shape,
            ShiftBoxes::PaddingType::BOTTOM_RIGHT));
        std::cout << "    Added step: ShiftBoxes (BOTTOM_RIGHT, From "
                  << metadata.output_shape << " to " << metadata.input_shape << ")" << std::endl;

      } else {
        std::cout << "    Skipping non-geometric step: " << metadata.step_name << std::endl;
        // If shape tracking mismatched, warning was already printed.
        if (metadata.output_shape != metadata.input_shape) {
          std::cerr << "Warning: Non-geometric step '" << metadata.step_name
                    << "' changed shape from " << metadata.input_shape << " to "
                    << metadata.output_shape << ", but no inverse geometric step added."
                    << std::endl;
        }
      }
    } catch (const std::exception & e) {
      throw std::runtime_error(
        "Failed to create inverse step for '" + metadata.step_name + "': " + e.what());
    }
  }
  std::cout << "PostProcessing pipeline initialization complete. Total steps: "
            << post_processing_steps_.size() << std::endl;
}

void PostProcessing::run(DetectionData & data, const cv::Size & /*original_image_size*/)
{
  if (data.boxes.size() != data.scores.size() || data.boxes.size() != data.class_ids.size()) {
    throw std::runtime_error(
      "PostProcessing::run input data vectors (boxes, scores, class_ids) must have the same size.");
  }

  for (const auto & step : post_processing_steps_) {
    step->apply(data);
  }
}

NonMaximumSuppression::NonMaximumSuppression(float conf_threshold, float iou_threshold)
: conf_threshold_(conf_threshold), iou_threshold_(iou_threshold)
{
  if (conf_threshold < 0.0f || conf_threshold > 1.0f) {
    throw std::invalid_argument("NMS confidence threshold must be between 0.0 and 1.0");
  }
  if (iou_threshold < 0.0f || iou_threshold > 1.0f) {
    throw std::invalid_argument("NMS IoU threshold must be between 0.0 and 1.0");
  }
}

void NonMaximumSuppression::apply(DetectionData & data) const
{
  // Filter out boxes below confidence threshold first - required by NMSBoxes behavior
  std::vector<cv::Rect2d> candidate_boxes;
  std::vector<float> candidate_scores;
  std::vector<int> candidate_original_indices;

  for (int original_idx : data.kept_indices) {
    if (data.scores[original_idx] >= conf_threshold_) {
      candidate_boxes.push_back(data.boxes[original_idx]);
      candidate_scores.push_back(data.scores[original_idx]);
      candidate_original_indices.push_back(original_idx);
    }
  }

  if (candidate_boxes.empty()) {
    data.kept_indices.clear();  // No boxes survived confidence thresholding
    return;
  }

  std::vector<int> nms_result_indices;
  cv::dnn::NMSBoxes(
    candidate_boxes, candidate_scores, conf_threshold_, iou_threshold_, nms_result_indices);

  // Update data.kept_indices with the *original* indices of the boxes that survived NMS
  std::vector<int> final_kept_indices;
  final_kept_indices.reserve(nms_result_indices.size());
  for (int nms_idx : nms_result_indices) {
    if (nms_idx < candidate_original_indices.size()) {  // Bounds check
      final_kept_indices.push_back(candidate_original_indices[nms_idx]);
    }
  }
  data.kept_indices = std::move(final_kept_indices);
}

std::string NonMaximumSuppression::name() const { return "NonMaximumSuppression"; }

RescaleBoxes::RescaleBoxes(
  const cv::Size & rescaled_image_size, const cv::Size & pre_scaling_image_size)
: pre_scaling_image_size_(pre_scaling_image_size)
{
  if (rescaled_image_size.width <= 0 || rescaled_image_size.height <= 0) {
    throw std::invalid_argument("RescaleBoxes: Processed size dimensions must be positive.");
  }
  if (pre_scaling_image_size.width <= 0 || pre_scaling_image_size.height <= 0) {
    throw std::invalid_argument("RescaleBoxes: Pre-scaling size dimensions must be positive.");
  }
  scale_x_ = static_cast<double>(pre_scaling_image_size.width) / rescaled_image_size.width;
  scale_y_ = static_cast<double>(pre_scaling_image_size.height) / rescaled_image_size.height;
}

void RescaleBoxes::apply(DetectionData & data) const
{
  for (int idx : data.kept_indices) {
    cv::Rect2d & box = data.boxes[idx];
    box.x *= scale_x_;
    box.y *= scale_y_;
    box.width *= scale_x_;
    box.height *= scale_y_;

    // Optional: Clamp coordinates to original image boundaries
    box.x = std::max(0.0, box.x);
    box.y = std::max(0.0, box.y);
    // Ensure x+width and y+height don't exceed original dimensions
    box.width = std::min(static_cast<double>(pre_scaling_image_size_.width) - box.x, box.width);
    box.height = std::min(static_cast<double>(pre_scaling_image_size_.height) - box.y, box.height);
    // Ensure width/height are not negative if clamping pushes x/y too far
    box.width = std::max(0.0, box.width);
    box.height = std::max(0.0, box.height);
  }
}

std::string RescaleBoxes::name() const { return "RescaleBoxes"; }

ShiftBoxes::ShiftBoxes(
  const cv::Size & padded_size, const cv::Size & pre_padding_size, PaddingType padding_type)
: padded_size_(padded_size), pre_padding_size_(pre_padding_size), padding_type_(padding_type)
{
  if (padded_size_.width <= 0 || padded_size_.height <= 0) {
    throw std::invalid_argument("ShiftBoxes: Padded size dimensions must be positive.");
  }
  if (pre_padding_size_.width <= 0 || pre_padding_size_.height <= 0) {
    throw std::invalid_argument("ShiftBoxes: Pre-padding size dimensions must be positive.");
  }
}

void ShiftBoxes::apply(DetectionData & data) const
{
  double pad_width = padded_size_.width - pre_padding_size_.width;
  double pad_height = padded_size_.height - pre_padding_size_.height;

  // Clamp negative padding to zero (can happen if pre-padding size was already >= padded size)
  pad_width = std::max(0.0, pad_width);
  pad_height = std::max(0.0, pad_height);

  double pad_left = 0.0;
  double pad_top = 0.0;

  if (padding_type_ == PaddingType::CENTER) {
    pad_left = pad_width / 2.0;
    pad_top = pad_height / 2.0;
  } else if (padding_type_ == PaddingType::BOTTOM_RIGHT) {
    pad_left = 0.0;
    pad_top = 0.0;
  }

  // Adjust coordinates for kept boxes
  for (int idx : data.kept_indices) {
    cv::Rect2d & box = data.boxes[idx];

    // Shift coordinates back by subtracting the top-left padding
    box.x -= pad_left;
    box.y -= pad_top;

    // Clamp coordinates to the boundaries of the PRE-PADDING image size
    box.x = std::max(0.0, box.x);
    box.y = std::max(0.0, box.y);
    // Ensure x+width and y+height don't exceed pre_padding_size dimensions
    box.width = std::min(static_cast<double>(pre_padding_size_.width) - box.x, box.width);
    box.height = std::min(static_cast<double>(pre_padding_size_.height) - box.y, box.height);
    // Ensure width/height are not negative if clamping pushes x/y too far
    box.width = std::max(0.0, box.width);
    box.height = std::max(0.0, box.height);
  }
}

std::string ShiftBoxes::name() const
{
  switch (padding_type_) {
    case PaddingType::CENTER:
      return "ShiftBoxes(Center)";
    case PaddingType::BOTTOM_RIGHT:
      return "ShiftBoxes(BottomRight)";
    default:
      return "ShiftBoxes(Unknown)";
  }
}

}  // namespace yolo_nas_cpp