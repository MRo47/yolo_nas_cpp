#pragma once

#include <memory>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace yolo_nas_cpp
{

using json = nlohmann::json;

struct PreProcessingMetadata
{
  std::string step_name;
  cv::Size input_shape = {-1, -1};
  cv::Size output_shape = {-1, -1};
  json params;
};

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
   * @param input_shape The input shape of the image, used to calculate the output shape of some steps
   * @throws std::runtime_error if the configuration is invalid or a step fails creation/parsing.
   */
  explicit PreProcessing(const json & config, const cv::Size input_shape = {-1, -1});

  /**
   * @brief Run the preprocessing pipeline on an input image
   * @param input Input image
   * @param output Output preprocessed image
   */
  void run(const cv::Mat & input, cv::Mat & output);

  /**
   * @brief Get the collected metadata for each preprocessing step.
   * @return A constant reference to the vector of metadata objects.
   */
  const std::vector<PreProcessingMetadata> & get_metadata() const;

private:
  /** Vector of preprocessing steps to be applied in sequence */
  std::vector<std::unique_ptr<PreProcessingStep>> processing_steps_;
  /** Metadata collected for each preprocessing step during configuration */
  std::vector<PreProcessingMetadata> metadata_;
};

/**
 * @class PreProcessingStep
 * @brief Base class for all image preprocessing steps
 */
class PreProcessingStep
{
public:
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
   * @brief Calculate the output shape produced by this step, given an input shape.
   * @param input_shape The shape of the image entering this step.
   * @return The shape of the image after this step is applied.
   *         Returns {-1, -1} if the output shape cannot be determined without the actual image
   *         (e.g., for aspect-ratio-preserving resize with unknown input).
   */
  virtual cv::Size calculate_output_shape(const cv::Size & input_shape) const = 0;

  /**
   * @brief Factory method to create preprocessing steps from JSON config
   * @param step_name Name of the step (e.g., "StandardizeImage")
   * @param params JSON parameters for the step
   * @param input_shape Size of the input image
   * @return A pair containing the created step and its metadata
   * @throws std::runtime_error if the step_name is unknown or parameters are invalid
   *                            (via the specific step's constructor).
   */
  static std::pair<std::unique_ptr<PreProcessingStep>, PreProcessingMetadata> create_from_json(
    const std::string & step_name, const json & params, const cv::Size & input_shape);
};

/**
 * @class StandardizeImage
 * @brief Scales pixel values to a standard range (typically [0,1])
 */
class StandardizeImage : public PreProcessingStep
{
public:
  /**
   * @brief StandardizeImage constructor
   * 
   * @param params json params, required param: max_value(double), image scaled as img/max_value
   */
  explicit StandardizeImage(const json & params);
  void apply(const cv::Mat & input, cv::Mat & output) const override;
  std::string name() const override;
  cv::Size calculate_output_shape(const cv::Size & input_shape) const override;

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
   * @brief NormalizeImage constructor
   * 
   * @param params json params, required params: mean(std::vector<double>), std(std::vector<double>)
   * image normalized as (img - mean) / std
   */
  explicit NormalizeImage(const json & params);
  void apply(const cv::Mat & input, cv::Mat & output) const override;
  std::string name() const override;
  cv::Size calculate_output_shape(const cv::Size & input_shape) const override;

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
   * @brief Construct a new Detection Center Padding object
   * 
   * @param params json params, required param: pad_value(int), output shape(list<int>)
   */
  explicit DetectionCenterPadding(const json & params);
  void apply(const cv::Mat & input, cv::Mat & output) const override;
  std::string name() const override;
  cv::Size calculate_output_shape(const cv::Size & input_shape) const override;

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
   * @brief Construct a new Detection Bottom Right Padding object
   * 
   * @param params json params, required param: pad_value(int), output shape(list<int>)
   */
  explicit DetectionBottomRightPadding(const json & params);
  void apply(const cv::Mat & input, cv::Mat & output) const override;
  std::string name() const override;
  cv::Size calculate_output_shape(const cv::Size & input_shape) const override;

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
   * @brief Construct a new Passthrough Step object
   * 
   * @param params json params, not used
   */
  explicit PassthroughStep(const json & /*params*/);
  void apply(const cv::Mat & input, cv::Mat & output) const override;
  std::string name() const override;
  cv::Size calculate_output_shape(const cv::Size & input_shape) const override;
};

/**
 * @class DetectionLongestMaxSizeRescale
 * @brief Rescales an image preserving aspect ratio based on the longest dimension
 */
class DetectionLongestMaxSizeRescale : public PreProcessingStep
{
public:
  /**
   * @brief Construct a new Detection Longest Max Size Rescale object
   * 
   * @param params json params, required param: out_shape(list<int>)
   */
  explicit DetectionLongestMaxSizeRescale(const json & params);
  void apply(const cv::Mat & input, cv::Mat & output) const override;
  std::string name() const override;
  cv::Size calculate_output_shape(const cv::Size & input_shape) const override;

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
   * @brief Construct a new Detection Rescale object
   * 
   * @param params json params, required param: out_shape(list<int>)
   */
  explicit DetectionRescale(const json & params);
  void apply(const cv::Mat & input, cv::Mat & output) const override;
  std::string name() const override;
  cv::Size calculate_output_shape(const cv::Size & input_shape) const override;

private:
  cv::Size out_shape_;
};

}  // namespace yolo_nas_cpp
