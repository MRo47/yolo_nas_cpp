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
  std::vector<float> scores;      // Confidence scores for each box
  std::vector<int> class_ids;     // Class IDs for each box

  // Indices into the above vectors indicating which detections are currently considered valid
  // Initialized with all indices, modified primarily by NMS.
  std::vector<int> kept_indices;

  // Optional: Store the original image size for context if needed directly in steps
  // cv::Size original_image_size;
};

}  // namespace yolo_nas_cpp