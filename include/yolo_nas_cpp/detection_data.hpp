#pragma once

#include <opencv2/core.hpp>
#include <vector>

namespace yolo_nas_cpp
{

/**
 * @struct DetectionData
 * @brief Holds intermediate and final results during post-processing.
 *
 * Contains the raw detections (boxes, scores, class IDs) and the indices
 * of detections that survive filtering steps like NMS. Subsequent steps
 * typically operate only on the elements referenced by kept_indices.
 */
struct DetectionData
{
  // Raw detections from the network output (or previous steps)
  std::vector<cv::Rect2d> boxes;  // Bounding boxes [x, y, width, height]
  std::vector<float> scores;
  std::vector<int> class_ids;

  // Indices into the above vectors indicating which detections are currently considered valid
  // Initialized with all indices, modified primarily by NMS.
  std::vector<int> kept_indices;
};

/**
 * @brief Overloads the << operator to output DetectionData to a stream.
 *
 * @param os The output stream (e.g., std::cout).
 * @param data The DetectionData object to output.
 * @return The output stream.
 */
inline std::ostream & operator<<(std::ostream & os, const DetectionData & data)
{
  os << "--- DetectionData ---" << std::endl;

  size_t num_raw_detections = data.boxes.size();

  os << "Total Raw Detections: " << num_raw_detections << std::endl;

  // Print Raw Detections
  // os << std::fixed << std::setprecision(4);  // Set precision for scores and box coords
  // if (num_raw_detections > 0) {
  //   os << "Raw Detections Details:" << std::endl;
  //   for (size_t i = 0; i < num_raw_detections; ++i) {
  //     os << "  [" << i << "]: "
  //        << "Box=[" << data.boxes[i].x << ", " << data.boxes[i].y << ", " << data.boxes[i].width
  //        << ", " << data.boxes[i].height << "], "
  //        << "Score=" << data.scores[i] << ", "
  //        << "ClassID=" << data.class_ids[i] << std::endl;
  //   }
  // } else {
  //   os << "  (No raw detections)" << std::endl;
  // }

  // Print Kept Detections
  os << "\nKept Detections: " << data.kept_indices.size() << std::endl;
  if (!data.kept_indices.empty()) {
    os << "Kept Detections Details (Original Index -> Data):" << std::endl;
    for (int kept_idx : data.kept_indices) {
      // Validate index before accessing data
      if (kept_idx >= 0 && static_cast<size_t>(kept_idx) < num_raw_detections) {
        os << "  Kept Idx [" << kept_idx << "]: "
           << "Box=[" << data.boxes[kept_idx].x << ", " << data.boxes[kept_idx].y << ", "
           << data.boxes[kept_idx].width << ", " << data.boxes[kept_idx].height << "], "
           << "Score=" << data.scores[kept_idx] << ", "
           << "ClassID=" << data.class_ids[kept_idx] << std::endl;
      } else {
        os << "  Kept Idx [" << kept_idx << "]: *** Invalid Index ***" << std::endl;
      }
    }
  } else {
    os << "  (No detections kept)" << std::endl;
  }

  os << "--- End DetectionData ---";
  return os;
}

}  // namespace yolo_nas_cpp