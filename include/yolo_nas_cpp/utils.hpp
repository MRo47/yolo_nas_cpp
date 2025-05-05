#pragma once

#include <spdlog/spdlog.h>

#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "yolo_nas_cpp/detection_data.hpp"

namespace yolo_nas_cpp
{
using json = nlohmann::json;

inline cv::Size parse_cv_size(const json & shape_arr)
{
  if (!shape_arr.is_array() || shape_arr.size() != 2) {
    throw std::runtime_error(
      "shape_arr must be a JSON array with exactly 2 elements [height, width].");
  }
  // JSON stores [height, width], cv::Size constructor takes (width, height)
  return cv::Size(shape_arr[1].get<int>(), shape_arr[0].get<int>());
}

/**
 * @brief Draws bounding boxes, class names, and scores onto an image.
 *
 * @param image The input image on which to draw detections.
 * @param detections A struct containing detection data (boxes, scores, class_ids, kept_indices).
 * @param labels A vector of strings where the index corresponds to the class_id.
 * @param box_color Color for the bounding box. Default is blue.
 * @param alpha Opacity for the bounding box. Default is 0.15.
 * @param text_color Color for the text label. Default is white.
 * @param thickness Thickness for the bounding box lines and text. Default is 2.
 * @param font_scale Font scale for the text label. Default is 0.6.
 * @param font_face Font face for the text label. Default is FONT_HERSHEY_SIMPLEX.
 * @return A new cv::Mat image with the detections drawn.
 */
inline cv::Mat draw_detections(
  const cv::Mat & image, const DetectionData & detections, const std::vector<std::string> & labels,
  const cv::Scalar & box_color = cv::Scalar(255, 178, 50), double alpha = 0.15,
  const cv::Scalar & text_color = cv::Scalar(255, 255, 255), int thickness = 2,
  double font_scale = 0.6, int font_face = cv::FONT_HERSHEY_SIMPLEX)
{
  cv::Mat output_image = image.clone();

  // Ensure labels are available if there are detections to draw
  if (!detections.kept_indices.empty() && labels.empty()) {
    spdlog::warn("Detections exist but no labels provided. Class names will be unavailable.");
  }

  for (int idx : detections.kept_indices) {
    // 1. Data Validation and Retrieval
    // Check if the index is valid for all required vectors
    if (
      idx < 0 || static_cast<size_t>(idx) >= detections.boxes.size() ||
      static_cast<size_t>(idx) >= detections.scores.size() ||
      static_cast<size_t>(idx) >= detections.class_ids.size()) {
      spdlog::warn("Skipping invalid kept_index: {}", idx);
      continue;
    }

    float score = detections.scores[idx];

    const cv::Rect2d & box_double = detections.boxes[idx];
    int class_id = detections.class_ids[idx];
    cv::Rect box(box_double.x, box_double.y, box_double.width, box_double.height);

    // 2. Get Class Name
    std::string class_name = "Unknown";
    if (!labels.empty()) {
      if (class_id >= 0 && static_cast<size_t>(class_id) < labels.size()) {
        class_name = labels[class_id];
      } else {
        spdlog::warn(
          "Invalid class_id {} encountered for index {}. Max label index: {}", class_id, idx,
          labels.size() - 1);
        class_name = "ID:" + std::to_string(class_id);  // Fallback
      }
    }

    // 3. Format Display Text
    std::stringstream ss;
    ss << class_name << ": " << std::fixed << std::setprecision(2) << score;
    std::string display_text = ss.str();

    // 4. Draw Bounding Box
    // Make sure the box coordinates are within image bounds (optional but safer)
    box.x = std::max(0, box.x);
    box.y = std::max(0, box.y);
    box.width = std::min(box.width, output_image.cols - box.x);
    box.height = std::min(box.height, output_image.rows - box.y);

    if (box.width <= 0 || box.height <= 0) {
      spdlog::warn("Skipping box with non-positive dimensions after clamping for index {}", idx);
      continue;
    }

    // draw box with transparency
    cv::rectangle(output_image, box, box_color, thickness);
    cv::Mat roi = output_image(box);
    cv::Mat color(roi.size(), roi.type(), box_color);
    cv::addWeighted(color, alpha, roi, 1.0 - alpha, 0.0, roi);

    // 5. Prepare and Draw Text Label with Background
    int baseline = 0;
    cv::Size text_size = cv::getTextSize(display_text, font_face, font_scale, thickness, &baseline);
    baseline += thickness;

    // Calculate position for the text label
    cv::Point text_origin = cv::Point(box.x, box.y - thickness);
    if (text_origin.y < text_size.height) {
      text_origin.y = box.y + text_size.height + thickness;
    }

    // Ensure text origin is within image bounds
    text_origin.y = std::min(output_image.rows - baseline, text_origin.y);
    text_origin.x = std::min(output_image.cols - text_size.width, text_origin.x);

    // Calculate background rectangle position based on final text_origin
    cv::Point bg_rect_tl(box.x, text_origin.y - text_size.height - thickness);
    cv::Point bg_rect_br(box.x + text_size.width, text_origin.y + baseline - thickness);

    // Clamp background rectangle coordinates to be within image bounds
    bg_rect_tl.x = std::max(0, bg_rect_tl.x);
    bg_rect_tl.y = std::max(0, bg_rect_tl.y);
    bg_rect_br.x = std::min(output_image.cols, bg_rect_br.x);
    bg_rect_br.y = std::min(output_image.rows, bg_rect_br.y);

    cv::rectangle(output_image, bg_rect_tl, bg_rect_br, box_color, cv::FILLED);
    cv::putText(
      output_image, display_text, text_origin, font_face, font_scale, text_color, thickness,
      cv::LINE_AA);
  }

  return output_image;
}

}  // namespace yolo_nas_cpp