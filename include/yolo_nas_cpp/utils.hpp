#pragma once
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>

namespace yolo_nas_cpp
{
using json = nlohmann::json;

inline cv::Size parse_cv_size(
  const json & shape_arr, const std::string & param_name = "output_shape")
{
  if (!shape_arr.is_array() || shape_arr.size() != 2) {
    throw std::runtime_error(
      "'" + param_name + "' must be a JSON array with exactly 2 elements [height, width].");
  }
  // JSON stores [height, width], cv::Size constructor takes (width, height)
  return cv::Size(shape_arr[1].get<int>(), shape_arr[0].get<int>());
}
}  // namespace yolo_nas_cpp