#pragma once

#include <memory>
#include <nlohmann/json.hpp>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <string>
#include <vector>

#include "yolo_nas_cpp/detection_data.hpp"
#include "yolo_nas_cpp/post_processing.hpp"
#include "yolo_nas_cpp/pre_processing.hpp"

namespace yolo_nas_cpp
{

using json = nlohmann::json;

/**
 * @class DetectionNetwork
 * @brief Manages loading, preprocessing, inference, and postprocessing for an object detection network.
 */
class DetectionNetwork
{
public:
  /**
   * @brief Constructs the DetectionNetwork.
   * @param config JSON object containing the full network configuration.
   * @param onnx_model_path Path to the ONNX model file.
   * @param input_image_shape Expected input image size (HxW).
   * @param use_cuda If true, attempt to configure OpenCV DNN to use CUDA backend and target.
   * @throws std::runtime_error if configuration parsing fails or the model cannot be loaded.
   */
  DetectionNetwork(
    const json & config, const std::string & onnx_model_path, const cv::Size & input_image_shape,
    bool use_cuda);

  /**
   * @brief Runs the full detection pipeline on a single input image.
   * @param input_image The input image (expected in BGR format, typically CV_8UC3).
   * @return DetectionData struct containing the final processed detections mapped to original image coordinates.
   * @throws std::runtime_error if any stage of the pipeline fails.
   */
  DetectionData detect(const cv::Mat & input_image);

  /**
    * @brief Gets the class labels loaded from the configuration.
    * @return A constant reference to the vector of class label strings.
    */
  const std::vector<std::string> & get_class_labels() const;

  /**
     * @brief Gets the expected network input size (HxW).
     * @return The cv::Size (width, height) of the network input layer.
     */
  cv::Size get_network_input_size() const;

private:
  /**
   * @brief Parses the configuration JSON and initializes pipelines and members.
   * @param config The JSON configuration object.
   * @param input_image_shape Expected input image size (HxW).
   */
  void parse_config(const json & config, const cv::Size & input_image_shape);

  /**
   * @brief Parses the raw network output tensors into boxes, scores, and class IDs.
   * @param raw_outputs Vector of cv::Mat objects from net_.forward().
   * @param boxes Output vector to be filled with detected bounding boxes (left, top, w, h).
   * @param scores Output vector to be filled with the confidence scores of the detections.
   * @param class_ids Output vector to be filled with the class IDs of the detections.
   * @note This implementation assumes a specific output format as in YOLO-NAS
   *       (e.g., 2 tensors [batch, num_detections, 4_bbox] and [batch, num_detections, num_classes]).
   */
  void parse_network_output(
    const std::vector<cv::Mat> & raw_outputs, std::vector<cv::Rect2d> & boxes,
    std::vector<float> & scores, std::vector<int> & class_ids);

  cv::dnn::Net net_;
  std::unique_ptr<PreProcessing> pre_processing_pipeline_;
  std::unique_ptr<PostProcessing> post_processing_pipeline_;

  std::string network_type_;
  cv::Size network_input_size_;
  cv::Size image_input_shape_;
  std::vector<std::string> class_labels_;
  std::vector<std::string> output_layer_names_;
};

}  // namespace yolo_nas_cpp