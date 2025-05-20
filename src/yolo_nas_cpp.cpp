#include <spdlog/spdlog.h>

#include <chrono>
#include <fstream>
#include <memory>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <string>

#include "yolo_nas_cpp/network.hpp"
#include "yolo_nas_cpp/utils.hpp"

int main(int argc, char ** argv)
{
  std::vector<std::string> model_paths;
  bool using_openvino = false;
  std::string metadata_path;
  std::string input_path;
  std::string backend;
  std::string target;

  if (argc == 6 && std::string(argv[1]) != "--openvino") {
    model_paths.push_back(argv[1]);
    using_openvino = false;
    metadata_path = argv[2];
    backend = argv[3];
    target = argv[4];
    input_path = argv[5];
  } else if (argc == 7 && std::string(argv[1]) == "--openvino") {
    model_paths.push_back(argv[2]);
    model_paths.push_back(argv[3]);
    using_openvino = true;
    metadata_path = argv[4];
    target = argv[5];
    input_path = argv[6];
    spdlog::info("Using OpenVINO");
  } else {
    spdlog::error(
      "Usage: {} <onnx_model_path> <metadata_path> <backend> <target> <image_or_video_path>\nor: "
      "{} [--openvino] <xml_model_path> <bin_model_path> <metadata_path> "
      "<target> <image_or_video_path>"
      "\n for backend and target options see: "
      "https://docs.opencv.org/4.11.0/d6/d0f/"
      "group__dnn.html#ga186f7d9bfacac8b0ff2e26e2eab02625.html"
      "\n for example: to set backend as \"cv::dnn::DNN_BACKEND_OPENCV\" pass backend arg as "
      "\"OPENCV\" \n",
      argv[0], argv[0]);
    spdlog::error("Unknown arguments: {}", argc - 1);
    spdlog::error("argv1 = \"{}\"", argv[1]);
    return 1;
  }

  std::ifstream file(metadata_path);
  if (!file.is_open()) {
    spdlog::error("Could not open metadata file: {}", metadata_path);
    return 1;
  }
  nlohmann::json config;
  try {
    file >> config;
  } catch (const nlohmann::json::exception & e) {
    spdlog::error("Failed to parse metadata JSON {}: {}", metadata_path, e.what());
    return 1;
  }

  // Try opening as an image first
  cv::Mat image = cv::imread(input_path);

  if (!image.empty()) {  // Successfully read as an image
    spdlog::info("Processing input as an image: {}", input_path);

    std::unique_ptr<yolo_nas_cpp::DetectionNetwork> network;

    if (!using_openvino) {
      network = std::make_unique<yolo_nas_cpp::DetectionNetwork>(
        config, model_paths[0], image.size(), backend, target);
    } else {
      network = std::make_unique<yolo_nas_cpp::DetectionNetwork>(
        config, model_paths[0], model_paths[1], image.size(), target);
    }

    // Warmup
    cv::Mat temp_image(image.size(), CV_8UC3);
    cv::randu(temp_image, cv::Scalar(0), cv::Scalar(255));
    spdlog::info("Starting network warmup (image mode)...");
    for (int i = 0; i < 3; i++) {
      network->detect(temp_image);
    }
    spdlog::info("Warmup finished.");

    std::chrono::high_resolution_clock::time_point start =
      std::chrono::high_resolution_clock::now();
    yolo_nas_cpp::DetectionData detections = network->detect(image);
    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    spdlog::info("Inference time: {} ms", duration.count());
    spdlog::info("Num detections: {}", detections.kept_indices.size());

    cv::Mat output_image =
      yolo_nas_cpp::draw_detections(image, detections, network->get_class_labels());
    cv::namedWindow("Detections", cv::WINDOW_NORMAL);
    cv::imshow("Detections", output_image);
    spdlog::info("Press any key to exit.");
    cv::waitKey(0);

  } else {  // Could not read as an image, try as video or camera
    cv::VideoCapture cap;
    bool is_camera = false;

    // Try opening as a file path
    cap.open(input_path);

    if (!cap.isOpened()) {
      // If opening as file path failed, try converting to integer for camera index
      try {
        int camera_index = std::stoi(input_path);
        cap.open(camera_index);
        if (cap.isOpened()) {
          is_camera = true;
          spdlog::info("Processing input as camera with index: {}", camera_index);
        } else {
          spdlog::error(
            "Input path '{}' is not a valid image file, video file, or camera index.", input_path);
          return 1;
        }
      } catch (const std::invalid_argument &) {
        spdlog::error(
          "Input path '{}' is not a valid image file, video file, or camera index.", input_path);
        return 1;
      } catch (const std::out_of_range &) {
        spdlog::error("Input path '{}' is out of range for a camera index.", input_path);
        return 1;
      }
    } else {
      spdlog::info("Processing input as a video file: {}", input_path);
    }

    // Read the first frame to get dimensions for network initialization
    cv::Mat frame;
    cap >> frame;
    if (frame.empty()) {
      spdlog::error("Could not read first frame from video/camera.");
      cap.release();
      return 1;
    }

    std::unique_ptr<yolo_nas_cpp::DetectionNetwork> network;

    if (!using_openvino) {
      network = std::make_unique<yolo_nas_cpp::DetectionNetwork>(
        config, model_paths[0], frame.size(), backend, target);
    } else {
      network = std::make_unique<yolo_nas_cpp::DetectionNetwork>(
        config, model_paths[0], model_paths[1], frame.size(), target);
    }

    cv::namedWindow("Detections", cv::WINDOW_NORMAL);

    spdlog::info("Starting video processing. Press 'q' to quit.");

    double total_inference_ms = 0;
    int frame_count = 0;
    std::chrono::high_resolution_clock::time_point frame_start_time;

    while (true) {
      frame_start_time = std::chrono::high_resolution_clock::now();

      cap >> frame;
      if (frame.empty()) {
        spdlog::info("End of video stream.");
        break;
      }

      yolo_nas_cpp::DetectionData detections = network->detect(frame);

      cv::Mat output_frame =
        yolo_nas_cpp::draw_detections(frame, detections, network->get_class_labels());

      auto frame_end_time = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double, std::milli> frame_duration = frame_end_time - frame_start_time;
      double fps = 1000.0 / frame_duration.count();
      spdlog::info(
        "Frame {}: FPS = {:.2f}, Detections = {}", frame_count, fps,
        detections.kept_indices.size());

      cv::imshow("Detections", output_frame);

      frame_count++;
      total_inference_ms += frame_duration.count();

      if (cv::waitKey(1) == 'q') {
        spdlog::info("Quitting video processing.");
        break;
      }
    }

    if (frame_count > 0) {
      double average_fps = static_cast<double>(frame_count) / (total_inference_ms / 1000.0);
      spdlog::info(
        "Average inference time over {} frames: {:.2f}ms ~ {:.2f} FPS", frame_count,
        total_inference_ms / frame_count, average_fps);
    } else {
      spdlog::warn("No frames were processed from the video stream.");
    }

    cap.release();
    cv::destroyAllWindows();
  }

  return 0;
}