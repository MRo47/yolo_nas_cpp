#pragma once

#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

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
 * @param score_threshold Optional minimum score threshold to draw a detection. Default 0.0 (draw all kept).
 * @param box_color Color for the bounding box. Default is blue.
 * @param text_color Color for the text label. Default is white.
 * @param thickness Thickness for the bounding box lines and text. Default is 2.
 * @param font_scale Font scale for the text label. Default is 0.6.
 * @param font_face Font face for the text label. Default is FONT_HERSHEY_SIMPLEX.
 * @return A new cv::Mat image with the detections drawn.
 */
inline cv::Mat draw_detections(
  const cv::Mat & image, const DetectionData & detections, const std::vector<std::string> & labels,
  float score_threshold = 0.0f, const cv::Scalar & box_color = cv::Scalar(255, 178, 50),
  const cv::Scalar & text_color = cv::Scalar(255, 255, 255), int thickness = 2,
  double font_scale = 0.6, int font_face = cv::FONT_HERSHEY_SIMPLEX)
{
  cv::Mat output_image = image.clone();

  // Ensure labels are available if there are detections to draw
  if (!detections.kept_indices.empty() && labels.empty()) {
    std::cerr
      << "Warning: Detections exist but no labels provided. Class names will be unavailable."
      << std::endl;
  }

  for (int idx : detections.kept_indices) {
    // 1. Data Validation and Retrieval
    // Check if the index is valid for all required vectors
    if (
      idx < 0 || static_cast<size_t>(idx) >= detections.boxes.size() ||
      static_cast<size_t>(idx) >= detections.scores.size() ||
      static_cast<size_t>(idx) >= detections.class_ids.size()) {
      std::cerr << "Warning: Skipping invalid kept_index: " << idx << std::endl;
      continue;
    }

    float score = detections.scores[idx];

    // Skip if score is below the threshold
    if (score < score_threshold) {
      continue;
    }

    const cv::Rect2d & box_double = detections.boxes[idx];
    int class_id = detections.class_ids[idx];

    // Convert Rect2d to Rect for drawing functions that prefer integers
    cv::Rect box(box_double.x, box_double.y, box_double.width, box_double.height);

    // 2. Get Class Name
    std::string class_name = "Unknown";
    if (!labels.empty()) {
      if (class_id >= 0 && static_cast<size_t>(class_id) < labels.size()) {
        class_name = labels[class_id];
      } else {
        std::cerr << "Warning: Invalid class_id " << class_id << " encountered for index " << idx
                  << ". Max label index: " << labels.size() - 1 << std::endl;
        class_name = "ID:" + std::to_string(class_id);  // Fallback
      }
    }

    // 3. Format Display Text
    std::stringstream ss;
    ss << class_name << ": " << std::fixed << std::setprecision(2) << score;
    std::string display_text = ss.str();

    // 4. Draw Bounding Box
    // Make sure the box coordinates are within image bounds (optional but safer)
    // Clamp top-left corner
    box.x = std::max(0, box.x);
    box.y = std::max(0, box.y);
    // Clamp bottom-right corner implicitly by adjusting width/height
    box.width = std::min(box.width, output_image.cols - box.x);
    box.height = std::min(box.height, output_image.rows - box.y);

    // Only draw if the box has a valid area after clamping
    if (box.width <= 0 || box.height <= 0) {
      std::cerr << "Warning: Skipping box with non-positive dimensions after clamping for index "
                << idx << std::endl;
      continue;
    }

    cv::rectangle(output_image, box, box_color, thickness);

    // 5. Prepare and Draw Text Label with Background
    int baseline = 0;
    cv::Size text_size = cv::getTextSize(display_text, font_face, font_scale, thickness, &baseline);
    baseline += thickness;

    // Calculate position for the text label
    cv::Point text_origin = cv::Point(box.x, box.y - thickness);
    // If the text would go off the top screen edge, place it inside the box instead
    if (text_origin.y < text_size.height) {
      text_origin.y = box.y + text_size.height + thickness;  // Move inside, below top line
    }

    // Ensure text origin doesn't go beyond image bottom
    text_origin.y = std::min(output_image.rows - baseline, text_origin.y);
    // Ensure text origin doesn't go beyond image right edge
    text_origin.x = std::min(output_image.cols - text_size.width, text_origin.x);

    // Calculate background rectangle position based on final text_origin
    // Top-left corner of the background rectangle
    cv::Point bg_rect_tl(box.x, text_origin.y - text_size.height - thickness);
    // Bottom-right corner of the background rectangle
    cv::Point bg_rect_br(box.x + text_size.width, text_origin.y + baseline - thickness);

    // Clamp background rectangle coordinates to be within image bounds
    bg_rect_tl.x = std::max(0, bg_rect_tl.x);
    bg_rect_tl.y = std::max(0, bg_rect_tl.y);
    bg_rect_br.x = std::min(output_image.cols, bg_rect_br.x);
    bg_rect_br.y = std::min(output_image.rows, bg_rect_br.y);

    // Draw the filled background rectangle
    // Check if the background rectangle has valid dimensions before drawing
    if (bg_rect_br.x > bg_rect_tl.x && bg_rect_br.y > bg_rect_tl.y) {
      cv::rectangle(output_image, bg_rect_tl, bg_rect_br, box_color, cv::FILLED);
    }

    // Draw the text itself
    cv::putText(
      output_image, display_text, text_origin, font_face, font_scale, text_color, thickness,
      cv::LINE_AA);
  }

  return output_image;
}

}  // namespace yolo_nas_cpp