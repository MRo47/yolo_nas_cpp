#pragma once

#include <memory>
#include <nlohmann/json.hpp>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <string>
#include <vector>

#include "yolo_nas_cpp/detection_data.hpp"
#include "yolo_nas_cpp/enum_mapping.hpp"
#include "yolo_nas_cpp/post_processing.hpp"
#include "yolo_nas_cpp/pre_processing.hpp"

namespace yolo_nas_cpp
{

using json = nlohmann::json;

static const EnumMapping<cv::dnn::Backend> backend_mapping = {
  {cv::dnn::DNN_BACKEND_DEFAULT, "DEFAULT"},
  {cv::dnn::DNN_BACKEND_HALIDE, "HALIDE"},
  {cv::dnn::DNN_BACKEND_INFERENCE_ENGINE, "INFERENCE_ENGINE"},
  {cv::dnn::DNN_BACKEND_OPENCV, "OPENCV"},
  {cv::dnn::DNN_BACKEND_VKCOM, "VKCOM"},
  {cv::dnn::DNN_BACKEND_CUDA, "CUDA"},
  {cv::dnn::DNN_BACKEND_WEBNN, "WEBNN"},
  {cv::dnn::DNN_BACKEND_TIMVX, "TIMVX"}};

static const EnumMapping<cv::dnn::Target> target_mapping = {
  {cv::dnn::DNN_TARGET_CPU, "CPU"},
  {cv::dnn::DNN_TARGET_OPENCL, "OPENCL"},
  {cv::dnn::DNN_TARGET_OPENCL_FP16, "OPENCL_FP16"},
  {cv::dnn::DNN_TARGET_MYRIAD, "MYRIAD"},
  {cv::dnn::DNN_TARGET_VULKAN, "VULKAN"},
  {cv::dnn::DNN_TARGET_FPGA, "FPGA"},
  {cv::dnn::DNN_TARGET_CUDA, "CUDA"},
  {cv::dnn::DNN_TARGET_CUDA_FP16, "CUDA_FP16"},
  {cv::dnn::DNN_TARGET_HDDL, "HDDL"},
  {cv::dnn::DNN_TARGET_NPU, "NPU"}};

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
   * @param backend The OpenCV DNN backend to use for inference.
   * @param target The OpenCV DNN target device to use for inference.
   * @throws std::runtime_error if configuration parsing fails or the model cannot be loaded.
   */
  DetectionNetwork(
    const json & config, const std::string & onnx_model_path, const cv::Size & input_image_shape,
    const std::string & backend = "OPENCV", const std::string & target = "CPU");

  /**
   * @brief Constructs the DetectionNetwork, using OpenVINO for inference.
   * @param config JSON object containing the full network configuration.
   * @param mo_xml_path Path to the OpenVINO model XML file.
   * @param mo_bin_path Path to the OpenVINO model BIN file.
   * @param input_image_shape Expected input image size (HxW).
   * @param target The OpenVINO target device to use for inference.
   * @throws std::runtime_error if configuration parsing fails or the model cannot be loaded.
   */
  DetectionNetwork(
    const json & config, const std::string & mo_xml_path, const std::string & mo_bin_path,
    const cv::Size & input_image_shape, const std::string & target = "CPU");

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
   * @param boxes Output vector to be filled with detected bounding boxes (left, top, right, bottom).
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
  cv::Size network_input_size_;  // actual network input image size after pre-processing
  cv::Size image_input_shape_;   // expected input image size
  std::vector<std::string> class_labels_;
  std::vector<std::string> output_layer_names_;
};

}  // namespace yolo_nas_cpp