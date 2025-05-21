#include "yolo_nas_cpp/network.hpp"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

#include "yolo_nas_cpp/detection_data.hpp"
#include "yolo_nas_cpp/post_processing.hpp"
#include "yolo_nas_cpp/pre_processing.hpp"
#include "yolo_nas_cpp/utils.hpp"

namespace yolo_nas_cpp
{

DetectionNetwork::DetectionNetwork(
  const json & config, const std::string & onnx_model_path, const cv::Size & input_image_shape,
  const std::string & backend, const std::string & target)
{
  spdlog::info("Initializing Detection Network...");

  cv::dnn::Backend cv_backend =
    backend_mapping.from_string(backend).value_or(cv::dnn::DNN_BACKEND_OPENCV);
  cv::dnn::Target cv_target = target_mapping.from_string(target).value_or(cv::dnn::DNN_TARGET_CPU);

  if (cv_backend == cv::dnn::DNN_BACKEND_INFERENCE_ENGINE) {
    throw std::runtime_error(
      "OpenVINO backend requires OpenVINO model optimizer files (.xml and .bin), use "
      "DetectionNetwork(config, mo_xml_path, mo_bin_path, input_image_shape, target) instead.");
  }

  try {
    parse_config(config, input_image_shape);
    spdlog::info("Configuration parsed and pipelines initialized.");
  } catch (const std::exception & e) {
    throw std::runtime_error("Failed to parse network configuration: " + std::string(e.what()));
  }

  try {
    net_ = cv::dnn::readNetFromONNX(onnx_model_path);
    if (net_.empty()) {
      throw std::runtime_error(
        "Network loaded from ONNX is empty. Check model path: " + onnx_model_path);
    }
    spdlog::info("ONNX model loaded successfully from: {}", onnx_model_path);

    net_.setPreferableBackend(cv_backend);
    spdlog::info("Set inference backend to: {}", backend_mapping.to_string(cv_backend));
    net_.setPreferableTarget(cv_target);
    spdlog::info("Set inference target to: {}", target_mapping.to_string(cv_target));

    output_layer_names_ = net_.getUnconnectedOutLayersNames();
    if (output_layer_names_.empty()) {
      throw std::runtime_error(
        "Could not get output layer names from the loaded network, required to perform "
        "inference.");
    }

  } catch (const cv::Exception & e) {
    throw std::runtime_error(
      "OpenCV Error loading ONNX model '" + onnx_model_path + "': " + std::string(e.what()));
  } catch (const std::exception & e) {
    throw std::runtime_error(
      "Error loading ONNX model '" + onnx_model_path + "': " + std::string(e.what()));
  }
  spdlog::info("Detection Network initialized.");
}

DetectionNetwork::DetectionNetwork(
  const json & config, const std::string & mo_xml_path, const std::string & mo_bin_path,
  const cv::Size & input_image_shape, const std::string & target)
{
  spdlog::info("Initializing Detection Network...");

  cv::dnn::Target cv_target = target_mapping.from_string(target).value_or(cv::dnn::DNN_TARGET_CPU);

  try {
    parse_config(config, input_image_shape);
    spdlog::info("Configuration parsed and pipelines initialized.");
  } catch (const std::exception & e) {
    throw std::runtime_error("Failed to parse network configuration: " + std::string(e.what()));
  }

  try {
    net_ = cv::dnn::readNet(mo_xml_path, mo_bin_path);
    if (net_.empty()) {
      throw std::runtime_error(
        "Network loaded from OpenVINO model optimizer files is empty. Check model paths: " +
        mo_xml_path + " and " + mo_bin_path);
    }
    spdlog::info("OpenVINO model loaded successfully from: {} and {}", mo_xml_path, mo_bin_path);

    if (cv_target != cv::dnn::DNN_TARGET_CPU && cv_target != cv::dnn::DNN_TARGET_OPENCL) {
      throw std::runtime_error("Only CPU and OpenCL targets are supported with OpenVINO backend.");
    }

    net_.setPreferableBackend(cv::dnn::DNN_BACKEND_INFERENCE_ENGINE);
    net_.setPreferableTarget(cv_target);
    spdlog::info(
      "Inference backend set to {}",
      backend_mapping.to_string(cv::dnn::DNN_BACKEND_INFERENCE_ENGINE));
    spdlog::info("Inference target set to {}", target_mapping.to_string(cv_target));

    output_layer_names_ = net_.getUnconnectedOutLayersNames();
    if (output_layer_names_.empty()) {
      throw std::runtime_error(
        "Could not get output layer names from the loaded network, required to perform "
        "inference.");
    }
  } catch (const cv::Exception & e) {
    throw std::runtime_error(
      "OpenCV Error loading OpenVINO model '" + mo_xml_path + "': " + std::string(e.what()));
  } catch (const std::exception & e) {
    throw std::runtime_error(
      "Error loading OpenVINO model '" + mo_xml_path + "': " + std::string(e.what()));
  }
  spdlog::info("Detection Network initialized.");
}

void DetectionNetwork::parse_config(const json & config, const cv::Size & input_image_shape)
{
  try {
    network_type_ = config.at("type").get<std::string>();
    class_labels_ = config.at("labels").get<std::vector<std::string>>();

    const auto & shape_arr = config.at("input_shape");
    if (!shape_arr.is_array() || shape_arr.size() != 4) {
      throw std::runtime_error("'input_shape' must be an array of 4 elements [N, C, H, W].");
    }
    int input_height = shape_arr[2].get<int>();
    int input_width = shape_arr[3].get<int>();
    if (input_height <= 0 || input_width <= 0) {
      throw std::runtime_error("Input shape H and W must be positive.");
    }
    network_input_size_ = cv::Size(input_width, input_height);

    const auto & pre_processing_config = config.at("pre_processing");
    pre_processing_pipeline_ =
      std::make_unique<PreProcessing>(pre_processing_config, input_image_shape);

    auto pre_processing_metadata = pre_processing_pipeline_->get_metadata();

    const auto & post_processing_config = config.at("post_processing");
    post_processing_pipeline_ =
      std::make_unique<PostProcessing>(post_processing_config, pre_processing_metadata);

  } catch (const json::exception & e) {
    throw std::runtime_error("JSON parsing error in network config: " + std::string(e.what()));
  } catch (const std::exception & e) {
    throw std::runtime_error(
      "Error initializing preprocessing/postprocessing pipelines: " + std::string(e.what()));
  }
}

DetectionData DetectionNetwork::detect(const cv::Mat & input_image)
{
  const cv::Size original_image_size = input_image.size();

  cv::Mat processed_image;
  pre_processing_pipeline_->run(input_image, processed_image);

  cv::Mat blob;
  blob = cv::dnn::blobFromImage(
    processed_image, 1, network_input_size_, cv::Scalar(0, 0, 0), true, false, CV_32F);

  std::vector<cv::Mat> raw_outputs;
  net_.setInput(blob);
  net_.forward(raw_outputs, output_layer_names_);

  DetectionData detection_results;
  try {
    parse_network_output(
      raw_outputs, detection_results.boxes, detection_results.scores, detection_results.class_ids);
  } catch (const std::exception & e) {
    throw std::runtime_error("Failed to parse network output: " + std::string(e.what()));
  }

  // If boxes were found, keep all indices before any post-processing (NMS will be applied later)
  if (!detection_results.boxes.empty()) {
    detection_results.kept_indices.resize(detection_results.boxes.size());
    std::iota(detection_results.kept_indices.begin(), detection_results.kept_indices.end(), 0);
  }

  post_processing_pipeline_->run(detection_results, original_image_size);

  return detection_results;
}

void DetectionNetwork::parse_network_output(
  const std::vector<cv::Mat> & raw_outputs, std::vector<cv::Rect2d> & boxes,
  std::vector<float> & scores, std::vector<int> & class_ids)
{
  const cv::Mat * score_tensor_ptr = nullptr;
  const cv::Mat * box_tensor_ptr = nullptr;
  const int expected_num_classes = static_cast<int>(class_labels_.size());

  for (const auto & output : raw_outputs) {
    if (output.empty()) continue;

    // Check for Score Tensor (1 x N x num_classes)
    if (output.dims == 3 && output.size[0] == 1 && output.size[2] == expected_num_classes) {
      if (score_tensor_ptr != nullptr) {
        spdlog::warn(
          "Found multiple potential score tensors. Using the last one found. Size: {}x{}x{}",
          output.size[0], output.size[1], output.size[2]);
      }
      score_tensor_ptr = &output;
    }
    // Check for Box Tensor (1 x N x 4)
    else if (output.dims == 3 && output.size[0] == 1 && output.size[2] == 4) {
      if (box_tensor_ptr != nullptr) {
        spdlog::warn(
          "Found multiple potential box tensors. Using the last one found. Size: {}x{}x{}",
          output.size[0], output.size[1], output.size[2]);
      }
      box_tensor_ptr = &output;
    } else {
      spdlog::debug(
        "Skipping output tensor with unexpected shape: {}x{}x{}", output.size[0], output.size[1],
        output.size[2]);
    }
  }

  if (score_tensor_ptr == nullptr) {
    throw std::runtime_error(
      "Could not find score output tensor (expected shape 1xNx" +
      std::to_string(expected_num_classes) + ")");
  }
  if (box_tensor_ptr == nullptr) {
    throw std::runtime_error("Could not find box output tensor (expected shape 1xNx4)");
  }

  const cv::Mat & score_tensor = *score_tensor_ptr;
  const cv::Mat & box_tensor = *box_tensor_ptr;

  const int num_detections_scores = score_tensor.size[1];
  const int num_detections_boxes = box_tensor.size[1];

  const int num_detections = num_detections_scores;
  if (num_detections == 0) {
    spdlog::warn("Output tensors indicate zero detections.");
    boxes.clear();
    scores.clear();
    class_ids.clear();
    return;
  }

  cv::Mat score_mat = score_tensor.reshape(1, num_detections);
  cv::Mat box_mat = box_tensor.reshape(1, num_detections);

  boxes.clear();
  scores.clear();
  class_ids.clear();
  boxes.reserve(num_detections);
  scores.reserve(num_detections);
  class_ids.reserve(num_detections);

  for (int i = 0; i < num_detections; ++i) {
    const float * box_data = box_mat.ptr<float>(i);      // Pointer to [left, top, right, bottom]
    const float * score_data = score_mat.ptr<float>(i);  // Pointer to [score_cls0, score_cls1, ...]

    // Find the class with the highest score for this detection
    auto max_score_it = std::max_element(score_data, score_data + expected_num_classes);
    float max_score = *max_score_it;  // The highest confidence score
    int max_class_id =
      static_cast<int>(std::distance(score_data, max_score_it));  // The index (class ID)

    // Extract bounding box: [left, top, right, bottom] format
    float x = box_data[0];
    float y = box_data[1];
    float w = box_data[2] - x;
    float h = box_data[3] - y;

    boxes.emplace_back(x, y, w, h);
    scores.push_back(max_score);
    class_ids.push_back(max_class_id);
  }
}

const std::vector<std::string> & DetectionNetwork::get_class_labels() const
{
  return class_labels_;
}

cv::Size DetectionNetwork::get_network_input_size() const { return network_input_size_; }

}  // namespace yolo_nas_cpp