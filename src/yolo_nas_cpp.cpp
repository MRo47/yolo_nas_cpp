#include <spdlog/spdlog.h>

#include <chrono>
#include <fstream>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <string>

#include "yolo_nas_cpp/network.hpp"
#include "yolo_nas_cpp/utils.hpp"

int main(int argc, char ** argv)
{
  if (argc < 4) {
    spdlog::error("Usage: {} <model_path> <metadata_path> <image_path>", argv[0]);
    return 1;
  }
  std::string model_path = argv[1];
  std::string metadata_path = argv[2];
  std::string image_path = argv[3];

  std::ifstream file(metadata_path);
  nlohmann::json config;
  file >> config;

  cv::Mat image = cv::imread(image_path);
  if (image.empty()) {
    spdlog::error("Could not read image from path: {}", image_path);
    return 1;
  }

  yolo_nas_cpp::DetectionNetwork network(config, model_path, image.size(), false);

  // warmup, to get better benchmark results, usually slow on first few inferences
  cv::Mat temp_image(image.size(), CV_8UC3);
  cv::randu(temp_image, cv::Scalar(0), cv::Scalar(255));
  spdlog::info("Starting network warmup...");
  for (int i = 0; i < 3; i++) {
    network.detect(temp_image);
  }
  spdlog::info("Warmup finished.");

  std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
  yolo_nas_cpp::DetectionData detections = network.detect(image);
  std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> duration = end - start;
  spdlog::info("Inference time: {} ms", duration.count());

  spdlog::info("Num detections: {}", detections.kept_indices.size());

  cv::Mat output_image =
    yolo_nas_cpp::draw_detections(image, detections, network.get_class_labels());

  cv::namedWindow("Detections", cv::WINDOW_NORMAL);
  cv::imshow("Detections", output_image);
  cv::waitKey(0);

  return 0;
}