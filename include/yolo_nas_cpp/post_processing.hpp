#pragma once

#include <memory>
#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>
#include <string>
#include <vector>

#include "yolo_nas_cpp/detection_data.hpp"
#include "yolo_nas_cpp/pre_processing.hpp"

namespace yolo_nas_cpp
{

using json = nlohmann::json;

// Forward declaration
class PostProcessingStep;

/**
 * @class PostProcessing
 * @brief Manages and executes a sequence of post-processing steps on detections.
 *
 * Constructs a pipeline typically including NMS and inverse geometric transformations
 * derived from the preprocessing configuration.
 */
class PostProcessing
{
public:
  /**
   * @brief Constructs the post-processing pipeline.
   * @param post_processing_config JSON object for post-processing specific steps (e.g., NMS).
   * @param pre_processing_metadata Vector of metadata objects for each preprocessing step.
   * @throws std::runtime_error if configuration parsing fails.
   */
  PostProcessing(
    const json & post_processing_config,
    const std::vector<PreProcessingMetadata> & pre_processing_metadata);

  /**
   * @brief Runs the entire post-processing pipeline on the provided data.
   * @param data The DetectionData struct containing initial boxes, scores, class_ids,
   *             and initialized kept_indices. This struct will be modified in place.
   * @param original_image_size The dimensions (WxH) of the original input image before any preprocessing.
   */
  void run(DetectionData & data, const cv::Size & original_image_size);

private:
  std::vector<std::unique_ptr<PostProcessingStep>> post_processing_steps_;
};

/**
 * @class PostProcessingStep
 * @brief Base class for all detection post-processing steps.
 */
class PostProcessingStep
{
public:
  virtual ~PostProcessingStep() = default;

  /**
   * @brief Apply the post-processing step. Modifies the DetectionData in place.
   * @param data The detection data (boxes, scores, indices) to be processed. Modified by the function.
   */
  virtual void apply(DetectionData & data) const = 0;

  /**
   * @brief Get the name of the processing step.
   * @return Step name as a string.
   */
  virtual std::string name() const = 0;
};

/**
 * @class NonMaximumSuppression
 * @brief Performs Non-Maximum Suppression on detected boxes.
 *
 * Filters overlapping boxes based on confidence and IoU thresholds,
 * updating the kept_indices in DetectionData.
 */
class NonMaximumSuppression : public PostProcessingStep
{
public:
  /**
   * @brief Constructor.
   * @param conf_threshold Confidence threshold. Boxes below this are discarded.
   * @param iou_threshold Intersection over Union (IoU) threshold for suppression.
   */
  NonMaximumSuppression(float conf_threshold, float iou_threshold);

  void apply(DetectionData & data) const override;
  std::string name() const override;

private:
  float conf_threshold_;
  float iou_threshold_;
};

/**
 * @class RescaleBoxes
 * @brief Rescales bounding box coordinates back to the original image dimensions.
 *
 * This step reverses the effect of a preprocessing rescaling step
 * (like DetectionLongestMaxSizeRescale or DetectionRescale).
 */
class RescaleBoxes : public PostProcessingStep
{
public:
  /**
   * @brief Constructor.
   * @param processed_size The size (WxH) the image was rescaled to during preprocessing
   *                       (i.e., the input size for the padding step, if any, or the network input size).
   */
  explicit RescaleBoxes(
    const cv::Size & rescaled_image_size, const cv::Size & pre_scaling_image_size);

  void apply(DetectionData & data) const override;
  std::string name() const override;

private:
  double scale_x_;
  double scale_y_;
  const cv::Size pre_scaling_image_size_;
};

/**
 * @class ShiftBoxes
 * @brief Adjusts bounding box coordinates to reverse the effect of a padding operation.
 *
 * Maps coordinates from the padded image space back to the space before padding was applied.
 */
class ShiftBoxes : public PostProcessingStep
{
public:
  enum class PaddingType
  {
    CENTER,
    BOTTOM_RIGHT
  };

  /**
   * @brief Constructor.
   * @param padded_size The size (WxH) the image was padded to during preprocessing.
   * @param pre_padding_size The size (WxH) the image had immediately before padding was applied.
   * @param padding_type The type of padding applied (Center or BottomRight).
   */
  ShiftBoxes(
    const cv::Size & padded_size, const cv::Size & pre_padding_size, PaddingType padding_type);

  // Apply method now uses pre_padding_size_ for calculations
  void apply(DetectionData & data) const override;
  std::string name() const override;

private:
  cv::Size padded_size_;
  cv::Size pre_padding_size_;
  PaddingType padding_type_;
};

}  // namespace yolo_nas_cpp