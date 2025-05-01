#include "yolo_nas_cpp/post_processing.hpp"

#include <algorithm>
#include <iostream>
#include <numeric>
#include <opencv2/dnn.hpp>
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
    post_processing_steps_.push_back(
      std::make_unique<NonMaximumSuppression>(nms_conf));  // Pass the JSON sub-object
    std::cout << "  Added step: " << post_processing_steps_.back()->name() << std::endl;
  } catch (const json::exception & e) {
    throw std::runtime_error(
      "Failed to find or parse 'NMS' configuration: " + std::string(e.what()));
  } catch (const std::exception & e) {
    throw std::runtime_error(
      "Failed to create NonMaximumSuppression step: " + std::string(e.what()));
  }

  // 2. Add Inverse Preprocessing Steps (in reverse order with size tracking)
  std::cout << "  Adding inverse geometric transformations..." << std::endl;

  // Iterate backwards through the preprocessing metadata
  for (auto it = pre_processing_metadata.rbegin(); it != pre_processing_metadata.rend(); ++it) {
    const auto & metadata = *it;

    std::cout << "    Processing PreStep: " << metadata.step_name
              << " (Input: " << metadata.input_shape << ", Output: " << metadata.output_shape << ")"
              << std::endl;

    try {
      // Use the factory method to create the inverse step
      std::unique_ptr<PostProcessingStep> inverse_step =
        PostProcessingStep::create_inverse_from_metadata(metadata);

      if (inverse_step) {
        post_processing_steps_.push_back(std::move(inverse_step));
        std::cout << "    Added step: " << post_processing_steps_.back()->name() << std::endl;
      } else {
        // The factory returns nullptr if the step name is not a known inverse geometric type
        std::cout << "    Skipping non-geometric/non-inverse step: " << metadata.step_name
                  << std::endl;
        if (metadata.output_shape != metadata.input_shape) {
          std::cerr << "Warning: Preprocessing step '" << metadata.step_name
                    << "' changed shape from " << metadata.input_shape << " to "
                    << metadata.output_shape << ", but no inverse geometric step was created."
                    << std::endl;
        }
      }
    } catch (const std::exception & e) {
      // Catch exceptions thrown by the factory or step constructors
      throw std::runtime_error(
        "Failed to create inverse step for metadata '" + metadata.step_name + "': " + e.what());
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

std::unique_ptr<PostProcessingStep> PostProcessingStep::create_inverse_from_metadata(
  const PreProcessingMetadata & metadata)
{
  if (
    metadata.step_name == "DetectionLongestMaxSizeRescale" ||
    metadata.step_name == "DetectionRescale") {
    return std::make_unique<RescaleBoxes>(metadata);
  } else if (metadata.step_name == "DetectionCenterPadding") {
    return std::make_unique<CenterShiftBoxes>(metadata);
  } else if (metadata.step_name == "DetectionBottomRightPadding") {
    return std::make_unique<BottomRightShiftBoxes>(metadata);
  }
  return nullptr;
}

NonMaximumSuppression::NonMaximumSuppression(const json & params)
{
  try {
    conf_threshold_ = params.at("conf").get<float>();
    iou_threshold_ = params.at("iou").get<float>();
  } catch (const json::exception & e) {
    throw std::runtime_error(
      "NonMaximumSuppression constructor failed to parse JSON parameters: " +
      std::string(e.what()));
  }

  if (conf_threshold_ < 0.0f || conf_threshold_ > 1.0f) {
    throw std::invalid_argument(
      "NMS confidence threshold must be between 0.0 and 1.0 (parsed: " +
      std::to_string(conf_threshold_) + ")");
  }
  if (iou_threshold_ < 0.0f || iou_threshold_ > 1.0f) {
    throw std::invalid_argument(
      "NMS IoU threshold must be between 0.0 and 1.0 (parsed: " + std::to_string(iou_threshold_) +
      ")");
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
  // Note: NMSBoxes also applies the confidence threshold internally, but filtering beforehand
  // means we only pass candidates above the threshold, potentially optimizing.
  // The conf_threshold_ passed here to NMSBoxes is the *same* threshold used for pre-filtering.
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

RescaleBoxes::RescaleBoxes(const PreProcessingMetadata & metadata)
: pre_scaling_image_size_(metadata.input_shape)  // The target size after inverse scaling
{
  // The input to this step is the output of the preprocessing step
  const cv::Size & rescaled_image_size = metadata.output_shape;
  // The output of this step should be the input of the preprocessing step
  const cv::Size & pre_scaling_image_size = metadata.input_shape;

  if (rescaled_image_size.width <= 0 || rescaled_image_size.height <= 0) {
    throw std::invalid_argument(
      "RescaleBoxes constructor: Source size (" + std::to_string(rescaled_image_size.width) + "x" +
      std::to_string(rescaled_image_size.height) + ") from metadata (" + metadata.step_name +
      ") dimensions must be positive.");
  }
  if (pre_scaling_image_size.width <= 0 || pre_scaling_image_size.height <= 0) {
    throw std::invalid_argument(
      "RescaleBoxes constructor: Target size (" + std::to_string(pre_scaling_image_size.width) +
      "x" + std::to_string(pre_scaling_image_size.height) + ") from metadata (" +
      metadata.step_name + ") dimensions must be positive.");
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

CenterShiftBoxes::CenterShiftBoxes(const PreProcessingMetadata & metadata)
: padded_size_(metadata.output_shape), pre_padding_size_(metadata.input_shape)
{
  if (padded_size_.width <= 0 || padded_size_.height <= 0) {
    throw std::invalid_argument(
      "CenterShiftBoxes constructor: Source size (" + std::to_string(padded_size_.width) + "x" +
      std::to_string(padded_size_.height) + ") from metadata (" + metadata.step_name +
      ") dimensions must be positive.");
  }
  if (pre_padding_size_.width <= 0 || pre_padding_size_.height <= 0) {
    throw std::invalid_argument(
      "CenterShiftBoxes constructor: Target size (" + std::to_string(pre_padding_size_.width) +
      "x" + std::to_string(pre_padding_size_.height) + ") from metadata (" + metadata.step_name +
      ") dimensions must be positive.");
  }
}

void CenterShiftBoxes::apply(DetectionData & data) const
{
  double pad_width = padded_size_.width - pre_padding_size_.width;
  double pad_height = padded_size_.height - pre_padding_size_.height;

  // Clamp negative padding to zero (shouldn't happen with correct metadata)
  pad_width = std::max(0.0, pad_width);
  pad_height = std::max(0.0, pad_height);

  // Center padding applies half the total padding to the top/left
  double pad_left = pad_width / 2.0;
  double pad_top = pad_height / 2.0;

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

std::string CenterShiftBoxes::name() const { return "CenterShiftBoxes"; }

BottomRightShiftBoxes::BottomRightShiftBoxes(const PreProcessingMetadata & metadata)
: padded_size_(metadata.output_shape), pre_padding_size_(metadata.input_shape)
{
  if (padded_size_.width <= 0 || padded_size_.height <= 0) {
    throw std::invalid_argument(
      "BottomRightShiftBoxes constructor: Source size (" + std::to_string(padded_size_.width) +
      "x" + std::to_string(padded_size_.height) + ") from metadata (" + metadata.step_name +
      ") dimensions must be positive.");
  }
  if (pre_padding_size_.width <= 0 || pre_padding_size_.height <= 0) {
    throw std::invalid_argument(
      "BottomRightShiftBoxes constructor: Target size (" + std::to_string(pre_padding_size_.width) +
      "x" + std::to_string(pre_padding_size_.height) + ") from metadata (" + metadata.step_name +
      ") dimensions must be positive.");
  }
}

void BottomRightShiftBoxes::apply(DetectionData & data) const
{
  double pad_width = padded_size_.width - pre_padding_size_.width;
  double pad_height = padded_size_.height - pre_padding_size_.height;

  // Clamp negative padding to zero (shouldn't happen with correct metadata)
  pad_width = std::max(0.0, pad_width);
  pad_height = std::max(0.0, pad_height);

  double pad_left = 0.0;
  double pad_top = 0.0;

  // Adjust coordinates for kept boxes
  for (int idx : data.kept_indices) {
    cv::Rect2d & box = data.boxes[idx];

    // Shift coordinates back by subtracting the top-left padding (which is 0,0 here)
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

std::string BottomRightShiftBoxes::name() const { return "BottomRightShiftBoxes"; }

}  // namespace yolo_nas_cpp