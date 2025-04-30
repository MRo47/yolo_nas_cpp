#include <fstream>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <string>

#include "yolo_nas_cpp/network.hpp"
#include "yolo_nas_cpp/utils.hpp"

int main(int argc, char ** argv)
{
  if (argc < 4) {
    std::cerr << "Usage: " << argv[0] << " <model_path> <metadata_path> <image_path>" << std::endl;
    return 1;
  }
  std::string model_path = argv[1];
  std::string metadata_path = argv[2];
  std::string image_path = argv[3];

  std::ifstream file(metadata_path);
  nlohmann::json config;
  file >> config;

  cv::Mat image = cv::imread(image_path);

  yolo_nas_cpp::DetectionNetwork network(config, model_path, image.size(), false);

  yolo_nas_cpp::DetectionData detections = network.detect(image);
  std::cout << "Num detections: " << detections.kept_indices.size() << std::endl;

  cv::Mat output_image =
    yolo_nas_cpp::draw_detections(image, detections, network.get_class_labels());

  cv::namedWindow("Detections", cv::WINDOW_NORMAL);
  cv::imshow("Detections", output_image);
  cv::waitKey(0);

  return 0;
}