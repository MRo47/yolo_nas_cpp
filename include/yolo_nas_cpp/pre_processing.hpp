#pragma once

#include <memory>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace yolo_nas_cpp
{

using json = nlohmann::json;

// Forward declaration
class PreProcessingStep;

/**
 * @class PreProcessing
 * @brief Implements a configurable image preprocessing pipeline for object detection
 *
 * The PreProcessing class constructs a sequence of image processing operations
 * from a JSON configuration
 */
class PreProcessing
{
public:
  /**
   * @brief Construct a preprocessing pipeline from JSON configuration
   * @param config JSON array containing processing step configurations.
   *               Each element should be an object with a single key (the step name)
   *               and a value containing the parameters for that step.
   * @throws std::runtime_error if the configuration is invalid or a step fails creation/parsing.
   */
  explicit PreProcessing(const json & config);

  /**
   * @brief Run the preprocessing pipeline on an input image
   * @param input Input image
   * @param output Output preprocessed image
   */
  void run(const cv::Mat & input, cv::Mat & output);

private:
  /** Vector of preprocessing steps to be applied in sequence */
  std::vector<std::unique_ptr<PreProcessingStep>> processing_steps_;
};

/**
 * @class PreProcessingStep
 * @brief Base class for all image preprocessing steps
 */
class PreProcessingStep
{
public:
  /**
   * @brief Virtual destructor
   */
  virtual ~PreProcessingStep() = default;

  /**
   * @brief Apply the processing step to an input image
   * @param input Input image
   * @param output Output processed image
   */
  virtual void apply(const cv::Mat & input, cv::Mat & output) const = 0;

  /**
   * @brief Get the name of the processing step
   * @return Step name
   */
  virtual std::string name() const = 0;

  /**
   * @brief Factory method to create preprocessing steps from JSON config
   * @param step_name Name of the step (e.g., "StandardizeImage")
   * @param params JSON parameters for the step
   * @return A unique pointer to the created step
   * @throws std::runtime_error if the step_name is unknown or parameters are invalid
   *                            (via the specific step's constructor).
   */
  static std::unique_ptr<PreProcessingStep> create_from_json(
    const std::string & step_name, const json & params);
};

/**
 * @class StandardizeImage
 * @brief Scales pixel values to a standard range (typically [0,1])
 */
class StandardizeImage : public PreProcessingStep
{
public:
  explicit StandardizeImage(const json & params);
  void apply(const cv::Mat & input, cv::Mat & output) const override;
  std::string name() const override;

private:
  double max_value_;
};

/**
 * @class NormalizeImage
 * @brief Normalizes an image by subtracting mean and dividing by standard deviation
 */
class NormalizeImage : public PreProcessingStep
{
public:
  explicit NormalizeImage(const json & params);
  void apply(const cv::Mat & input, cv::Mat & output) const override;
  std::string name() const override;

private:
  std::vector<double> mean_;
  std::vector<double> std_;
};

/**
 * @class DetectionCenterPadding
 * @brief Pads an image to a target size, centering the original content
 */
class DetectionCenterPadding : public PreProcessingStep
{
public:
  explicit DetectionCenterPadding(const json & params);
  void apply(const cv::Mat & input, cv::Mat & output) const override;
  std::string name() const override;

private:
  int pad_value_;
  cv::Size out_shape_;
};

/**
 * @class DetectionBottomRightPadding
 * @brief Pads an image to a target size by adding padding at the bottom and right
 */
class DetectionBottomRightPadding : public PreProcessingStep
{
public:
  explicit DetectionBottomRightPadding(const json & params);
  void apply(const cv::Mat & input, cv::Mat & output) const override;
  std::string name() const override;

private:
  int pad_value_;
  cv::Size out_shape_;
};

/**
 * @class PassthroughStep
 * @brief A preprocessing step that performs no operation and passes the input directly to the output.
 *        Used to gracefully handle unsupported steps found in configuration without erroring.
 */
class PassthroughStep : public PreProcessingStep
{
public:
  /**
   * @brief Constructor. Ignores any provided parameters.
   * @param params JSON parameters (ignored).
   */
  explicit PassthroughStep(const json & /*params*/);  // Mark params as unused

  /**
   * @brief Apply the passthrough step. Makes output refer to the same data as input.
   * @param input Input image.
   * @param output Output image (will share data with input).
   */
  void apply(const cv::Mat & input, cv::Mat & output) const override;

  /**
   * @brief Get the name of the processing step.
   * @return Step name ("PassthroughStep").
   */
  std::string name() const override;
};

/**
 * @class DetectionLongestMaxSizeRescale
 * @brief Rescales an image preserving aspect ratio based on the longest dimension
 */
class DetectionLongestMaxSizeRescale : public PreProcessingStep
{
public:
  explicit DetectionLongestMaxSizeRescale(const json & params);
  void apply(const cv::Mat & input, cv::Mat & output) const override;
  std::string name() const override;

private:
  cv::Size out_shape_;  // Note: cv::Size is (width, height)
};

/**
 * @class DetectionRescale
 * @brief Rescales an image to a target size without preserving aspect ratio
 */
class DetectionRescale : public PreProcessingStep
{
public:
  explicit DetectionRescale(const json & params);
  void apply(const cv::Mat & input, cv::Mat & output) const override;
  std::string name() const override;

private:
  cv::Size out_shape_;  // Note: cv::Size is (width, height)
};

}  // namespace yolo_nas_cpp
