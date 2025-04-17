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
 * from a JSON configuration and applies them in order to prepare images for
 * object detection networks.
 */
class PreProcessing
{
public:
  /**
   * @brief Construct a preprocessing pipeline from JSON configuration
   * @param preprocessing_config JSON array containing processing step configurations
   * @throws std::runtime_error if the configuration is invalid
   */
  explicit PreProcessing(const json & preprocessing_config);

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
};

/**
 * @class StandardizeImage
 * @brief Scales pixel values to a standard range (typically [0,1])
 */
class StandardizeImage : public PreProcessingStep
{
public:
  /**
   * @brief Constructor
   * @param max_value The maximum value to scale by (e.g., 255.0 for 8-bit images)
   */
  explicit StandardizeImage(double max_value);

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
  /**
   * @brief Constructor
   * @param mean Vector of mean values for each channel
   * @param std Vector of standard deviation values for each channel
   * @throws std::runtime_error if mean or std vectors don't have exactly 3 elements
   */
  NormalizeImage(std::vector<double> mean, std::vector<double> std);

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
  /**
   * @brief Constructor
   * @param pad_value Value to use for padding pixels
   * @param out_shape Target output shape
   */
  DetectionCenterPadding(int pad_value, const cv::Size & out_shape);

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
  /**
   * @brief Constructor
   * @param pad_value Value to use for padding pixels
   * @param out_shape Target output shape
   */
  DetectionBottomRightPadding(int pad_value, const cv::Size & out_shape);

  void apply(const cv::Mat & input, cv::Mat & output) const override;
  std::string name() const override;

private:
  int pad_value_;
  cv::Size out_shape_;
};

/**
 * @class ImagePermute
 * @brief Permutes the dimensions of a multi-dimensional array
 */
class ImagePermute : public PreProcessingStep
{
public:
  /**
   * @brief Constructor
   * @param order Vector specifying the new order of dimensions
   * @throws std::runtime_error if order is not a valid permutation
   */
  explicit ImagePermute(std::vector<int> order);

  void apply(const cv::Mat & input, cv::Mat & output) const override;
  std::string name() const override;

private:
  std::vector<int> order_;
};

/**
 * @class DetectionLongestMaxSizeRescale
 * @brief Rescales an image preserving aspect ratio based on the longest dimension
 */
class DetectionLongestMaxSizeRescale : public PreProcessingStep
{
public:
  /**
   * @brief Constructor
   * @param out_shape Target output shape
   */
  explicit DetectionLongestMaxSizeRescale(const cv::Size & out_shape);

  void apply(const cv::Mat & input, cv::Mat & output) const override;
  std::string name() const override;

private:
  cv::Size out_shape_;
};

/**
 * @class DetectionRescale
 * @brief Rescales an image to a target size without preserving aspect ratio
 */
class DetectionRescale : public PreProcessingStep
{
public:
  /**
   * @brief Constructor
   * @param out_shape Target output shape
   */
  explicit DetectionRescale(const cv::Size & out_shape);

  void apply(const cv::Mat & input, cv::Mat & output) const override;
  std::string name() const override;

private:
  cv::Size out_shape_;
};

}  // namespace yolo_nas_cpp
