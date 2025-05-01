#pragma once

#include <memory>
#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>
#include <stdexcept>
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
   *
   * The constructor delegates the parsing of specific parameters to the
   * individual step constructors or a factory method for inverse steps.
   *
   * @param post_processing_config JSON object for post-processing specific steps (e.g., NMS).
   * @param pre_processing_metadata Vector of metadata objects for each preprocessing step.
   * @throws std::runtime_error if configuration parsing or step creation fails.
   */
  PostProcessing(
    const json & post_processing_config,
    const std::vector<PreProcessingMetadata> & pre_processing_metadata);

  /**
   * @brief Runs the entire post-processing pipeline on the provided data.
   * @param data The DetectionData struct containing initial boxes, scores, class_ids,
   *             and initialized kept_indices. This struct will be modified in place.
   * @param original_image_size The dimensions (WxH) of the original input image before any preprocessing.
   *                            NOTE: original_image_size is currently not used by the steps themselves,
   *                            but could be passed to steps if needed. The metadata provides the necessary
   *                            sizes for inverse transforms.
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

  /**
   * @brief Factory method to create an inverse geometric post-processing step
   *        from preprocessing metadata.
   * @param metadata The metadata from the corresponding preprocessing step.
   * @return A unique_ptr to the created PostProcessingStep, or nullptr if the metadata
   *         does not correspond to a known inverse step type.
   * @throws std::runtime_error if step creation/parameter extraction fails for a known type.
   */
  static std::unique_ptr<PostProcessingStep> create_inverse_from_metadata(
    const PreProcessingMetadata & metadata);
};

/**
 * @class NonMaximumSuppression
 * @brief Performs Non-Maximum Suppression on detected boxes.
 *
 * Filters overlapping boxes based on confidence and IoU thresholds,
 * updating the kept_indices in DetectionData. Parameters are parsed from JSON.
 */
class NonMaximumSuppression : public PostProcessingStep
{
public:
  /**
   * @brief Constructor. Parses confidence and IoU thresholds from the provided JSON object.
   * @param params JSON object containing "conf" and "iou" fields.
   * @throws std::runtime_error if parsing fails or values are invalid.
   */
  explicit NonMaximumSuppression(const json & params);

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
 * Parameters are extracted from PreProcessingMetadata.
 */
class RescaleBoxes : public PostProcessingStep
{
public:
  /**
   * @brief Constructor. Extracts sizes from the provided PreProcessingMetadata.
   *
   * Assumes the metadata represents a rescaling step, using input_shape as the
   * target size and output_shape as the source size for the inverse transformation.
   *
   * @param metadata Metadata from the corresponding preprocessing step.
   * @throws std::invalid_argument if metadata contains invalid size dimensions.
   */
  explicit RescaleBoxes(const PreProcessingMetadata & metadata);

  void apply(DetectionData & data) const override;
  std::string name() const override;

private:
  double scale_x_;
  double scale_y_;
  const cv::Size
    pre_scaling_image_size_;  // Store the target size (metadata.input_shape) for clamping
};

/**
 * @class CenterShiftBoxes
 * @brief Adjusts bounding box coordinates to reverse the effect of a center padding operation.
 *
 * Maps coordinates from the padded image space back to the space before padding was applied.
 * Parameters are extracted from PreProcessingMetadata.
 */
class CenterShiftBoxes : public PostProcessingStep
{
public:
  /**
   * @brief Constructor. Extracts sizes from the provided PreProcessingMetadata.
   * @param metadata Metadata from the corresponding "DetectionCenterPadding" step.
   * @throws std::invalid_argument if metadata contains invalid size dimensions.
   */
  explicit CenterShiftBoxes(const PreProcessingMetadata & metadata);

  void apply(DetectionData & data) const override;
  std::string name() const override;

private:
  cv::Size padded_size_;
  cv::Size pre_padding_size_;
};

/**
 * @class BottomRightShiftBoxes
 * @brief Adjusts bounding box coordinates to reverse the effect of a bottom-right padding operation.
 *
 * Maps coordinates from the padded image space back to the space before padding was applied.
 * Parameters are extracted from PreProcessingMetadata.
 */
class BottomRightShiftBoxes : public PostProcessingStep
{
public:
  /**
   * @brief Constructor. Extracts sizes from the provided PreProcessingMetadata.
   * @param metadata Metadata from the corresponding "DetectionBottomRightPadding" step.
   * @throws std::invalid_argument if metadata contains invalid size dimensions.
   */
  explicit BottomRightShiftBoxes(const PreProcessingMetadata & metadata);

  void apply(DetectionData & data) const override;
  std::string name() const override;

private:
  cv::Size padded_size_;
  cv::Size pre_padding_size_;
};

}  // namespace yolo_nas_cpp