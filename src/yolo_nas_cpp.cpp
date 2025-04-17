#include <fstream>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <string>

#include "yolo_nas_cpp/pre_processing.hpp"

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

  yolo_nas_cpp::PreProcessing pre_processing(config["pre_processing"]);
  cv::Mat image = cv::imread(image_path);
  cv::Mat preprocessed_image;
  pre_processing.run(image, preprocessed_image);

  std::cout << "Preprocessed Image dims"
            << "\nrows:" << preprocessed_image.rows << "\ncols:" << preprocessed_image.cols
            << "\nchannels:" << preprocessed_image.channels() << std::endl;

  cv::namedWindow("Preprocessed Image", cv::WINDOW_NORMAL);
  cv::imshow("Preprocessed Image", preprocessed_image);
  cv::waitKey(0);

  return 0;
}