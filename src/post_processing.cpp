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

// Helper to get cv::Size from JSON (assuming defined elsewhere, e.g., preprocessing.cpp)
// If not, define it here.
extern cv::Size parse_cv_size(const json & shape_arr, const std::string & param_name);

// --- PostProcessing Constructor ---
PostProcessing::PostProcessing(
  const json & post_processing_config, const json & pre_processing_config)
{
  std::cout << "Initializing PostProcessing pipeline..." << std::endl;

  // --- 1. Add NMS Step ---
  try {
    const auto & nms_conf = post_processing_config.at("NMS");
    float conf_threshold = nms_conf.at("conf").get<float>();
    float iou_threshold = nms_conf.at("iou").get<float>();
    post_processing_steps_.push_back(
      std::make_unique<NonMaximumSuppression>(conf_threshold, iou_threshold));
    std::cout << "  + Added step: NonMaximumSuppression (Conf: " << conf_threshold
              << ", IoU: " << iou_threshold << ")" << std::endl;
  } catch (const std::exception & e) {
    throw std::runtime_error("Failed to parse NMS configuration: " + std::string(e.what()));
  }

  // --- 2. Add Inverse Preprocessing Steps (in reverse order) ---
  if (!pre_processing_config.is_array()) {
    throw std::runtime_error("Pre-processing config passed to PostProcessing must be an array.");
  }

  // Iterate through preprocessing steps in REVERSE order
  for (auto it = pre_processing_config.rbegin(); it != pre_processing_config.rend(); ++it) {
    const auto & step_config = *it;

    if (!step_config.is_object() || step_config.size() != 1) {
      std::cerr
        << "Warning: Skipping invalid entry in pre_processing config during post-processing setup."
        << std::endl;
      continue;
    }

    const std::string & step_name = step_config.begin().key();
    const json & params = step_config.begin().value();

    try {
      if (step_name == "DetectionLongestMaxSizeRescale" || step_name == "DetectionRescale") {
        // This step rescaled the image to a certain size.
        cv::Size processed_size = parse_cv_size(params.at("output_shape"));
        post_processing_steps_.push_back(std::make_unique<UndoRescaleBoxes>(processed_size));
        std::cout << "  + Added inverse step: UndoRescaleBoxes (for " << step_name << ")"
                  << std::endl;

      } else if (step_name == "DetectionCenterPadding") {
        // This step padded the image to a certain size using center padding.
        cv::Size padded_size = parse_cv_size(params.at("output_shape"));
        post_processing_steps_.push_back(
          std::make_unique<UndoPaddingBoxes>(padded_size, UndoPaddingBoxes::PaddingType::CENTER));
        std::cout << "  + Added inverse step: UndoPaddingBoxes (Center) (for " << step_name << ")"
                  << std::endl;

      } else if (step_name == "DetectionBottomRightPadding") {
        // This step padded the image TO a certain size using bottom-right padding.
        cv::Size padded_size = parse_cv_size(params.at("output_shape"));
        post_processing_steps_.push_back(
          std::make_unique<UndoPaddingBoxes>(
            padded_size, UndoPaddingBoxes::PaddingType::BOTTOM_RIGHT));
        std::cout << "  + Added inverse step: UndoPaddingBoxes (BottomRight) (for " << step_name
                  << ")" << std::endl;
      }
      // Add other inverse steps here if needed (e.g., undoing normalization if it affects coordinates somehow, though unlikely)
      // Steps like StandardizeImage, ImagePermute, NormalizeImage usually don't require an inverse step for box coordinates.

    } catch (const std::exception & e) {
      // Catch errors during parsing of preprocessing params for inverse steps
      throw std::runtime_error(
        "Error creating inverse step for '" + step_name + "': " + std::string(e.what()));
    }
  }
  std::cout << "PostProcessing pipeline initialized with " << post_processing_steps_.size()
            << " steps." << std::endl;
}

void PostProcessing::run(DetectionData & data, const cv::Size & original_image_size)
{
  if (data.boxes.size() != data.scores.size() || data.boxes.size() != data.class_ids.size()) {
    throw std::runtime_error(
      "PostProcessing::run input data vectors (boxes, scores, class_ids) must have the same size.");
  }

  for (const auto & step : post_processing_steps_) {
    step->apply(data, original_image_size);
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

void NonMaximumSuppression::apply(
  DetectionData & data, const cv::Size & /*original_image_size*/) const
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

UndoRescaleBoxes::UndoRescaleBoxes(const cv::Size & processed_size)
: processed_size_(processed_size)
{
  if (processed_size_.width <= 0 || processed_size_.height <= 0) {
    throw std::invalid_argument("UndoRescaleBoxes: Processed size dimensions must be positive.");
  }
}

void UndoRescaleBoxes::apply(DetectionData & data, const cv::Size & original_image_size) const
{
  if (original_image_size.width <= 0 || original_image_size.height <= 0) {
    throw std::runtime_error("UndoRescaleBoxes::apply received invalid original_image_size.");
  }
  if (processed_size_.width <= 0 || processed_size_.height <= 0) {
    throw std::runtime_error(
      "UndoRescaleBoxes::apply has invalid internal processed_size.");  // Should be caught by constructor
  }

  // Calculate scaling factors
  double scale_x = static_cast<double>(original_image_size.width) / processed_size_.width;
  double scale_y = static_cast<double>(original_image_size.height) / processed_size_.height;

  // Apply scaling to the boxes kept after NMS
  for (int idx : data.kept_indices) {
    cv::Rect2d & box = data.boxes[idx];
    box.x *= scale_x;
    box.y *= scale_y;
    box.width *= scale_x;
    box.height *= scale_y;

    // Optional: Clamp coordinates to original image boundaries
    box.x = std::max(0.0, box.x);
    box.y = std::max(0.0, box.y);
    // Ensure x+width and y+height don't exceed original dimensions
    box.width = std::min(static_cast<double>(original_image_size.width) - box.x, box.width);
    box.height = std::min(static_cast<double>(original_image_size.height) - box.y, box.height);
    // Ensure width/height are not negative if clamping pushes x/y too far
    box.width = std::max(0.0, box.width);
    box.height = std::max(0.0, box.height);
  }
}

std::string UndoRescaleBoxes::name() const { return "UndoRescaleBoxes"; }

UndoPaddingBoxes::UndoPaddingBoxes(const cv::Size & padded_size, PaddingType padding_type)
: padded_size_(padded_size), padding_type_(padding_type)
{
  if (padded_size_.width <= 0 || padded_size_.height <= 0) {
    throw std::invalid_argument("UndoPaddingBoxes: Padded size dimensions must be positive.");
  }
  if (padding_type_ == PaddingType::UNKNOWN) {
    throw std::invalid_argument("UndoPaddingBoxes: Padding type cannot be UNKNOWN.");
  }
}

void UndoPaddingBoxes::apply(DetectionData & data, const cv::Size & original_image_size) const
{
  if (original_image_size.width <= 0 || original_image_size.height <= 0) {
    throw std::runtime_error("UndoPaddingBoxes::apply received invalid original_image_size.");
  }
  if (
    padded_size_.width < original_image_size.width ||
    padded_size_.height < original_image_size.height) {
    // This might be valid if original image was larger and scaling shrunk it before padding
    std::cerr << "Warning: Padded size is smaller than original size in UndoPaddingBoxes."
              << std::endl;
  }

  if (padding_type_ == PaddingType::CENTER) {
    // Calculate padding added
    double pad_width = padded_size_.width - original_image_size.width;
    double pad_height = padded_size_.height - original_image_size.height;
    double pad_left = pad_width / 2.0;
    double pad_top = pad_height / 2.0;

    // Shift coordinates back by subtracting the top-left padding
    for (int idx : data.kept_indices) {
      cv::Rect2d & box = data.boxes[idx];
      box.x -= pad_left;
      box.y -= pad_top;

      // Optional: Clamp coordinates after shifting
      box.x = std::max(0.0, box.x);
      box.y = std::max(0.0, box.y);
      box.width = std::min(static_cast<double>(original_image_size.width) - box.x, box.width);
      box.height = std::min(static_cast<double>(original_image_size.height) - box.y, box.height);
      box.width = std::max(0.0, box.width);
      box.height = std::max(0.0, box.height);
    }
  } else if (padding_type_ == PaddingType::BOTTOM_RIGHT) {
    // No coordinate shift needed for bottom-right padding.
    // However, we might still want to clamp boxes to the original image boundaries
    // in case rescaling made them extend beyond where the padding *would have been*.
    for (int idx : data.kept_indices) {
      cv::Rect2d & box = data.boxes[idx];
      box.x = std::max(0.0, box.x);
      box.y = std::max(0.0, box.y);
      box.width = std::min(static_cast<double>(original_image_size.width) - box.x, box.width);
      box.height = std::min(static_cast<double>(original_image_size.height) - box.y, box.height);
      box.width = std::max(0.0, box.width);
      box.height = std::max(0.0, box.height);
    }
  }
}

std::string UndoPaddingBoxes::name() const
{
  switch (padding_type_) {
    case PaddingType::CENTER:
      return "UndoPaddingBoxes(Center)";
    case PaddingType::BOTTOM_RIGHT:
      return "UndoPaddingBoxes(BottomRight)";
    default:
      return "UndoPaddingBoxes(Unknown)";
  }
}

}  // namespace yolo_nas_cpp