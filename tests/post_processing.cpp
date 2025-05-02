#include "yolo_nas_cpp/post_processing.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>
#include <stdexcept>
#include <vector>

#include "yolo_nas_cpp/detection_data.hpp"
#include "yolo_nas_cpp/pre_processing.hpp"
#include "yolo_nas_cpp/utils.hpp"

using namespace yolo_nas_cpp;
using json = nlohmann::json;

// Helper function to create sample DetectionData
DetectionData create_sample_detection_data(
  const std::vector<cv::Rect2d> & boxes, const std::vector<float> & scores,
  const std::vector<int> & class_ids)
{
  DetectionData data;
  data.boxes = boxes;
  data.scores = scores;
  data.class_ids = class_ids;
  // Initially, kept_indices contains all indices
  data.kept_indices.resize(boxes.size());
  std::iota(data.kept_indices.begin(), data.kept_indices.end(), 0);
  return data;
}

// Helper to check if two Rect2d are approximately equal
bool are_rects_approx_equal(const cv::Rect2d & r1, const cv::Rect2d & r2, double epsilon = 1e-9)
{
  return std::abs(r1.x - r2.x) < epsilon && std::abs(r1.y - r2.y) < epsilon &&
         std::abs(r1.width - r2.width) < epsilon && std::abs(r1.height - r2.height) < epsilon;
}

//  Tests for NonMaximumSuppression

TEST(NonMaximumSuppressionTest, ConstructorValidParams)
{
  json params;
  params["conf"] = 0.5;
  params["iou"] = 0.4;
  ASSERT_NO_THROW({ NonMaximumSuppression nms(params); });
}

TEST(NonMaximumSuppressionTest, ConstructorMissingConf)
{
  json params;
  params["iou"] = 0.4;
  ASSERT_THROW(
    { NonMaximumSuppression nms(params); },
    std::runtime_error);  // constructor re-throws json::exception as runtime_error
}

TEST(NonMaximumSuppressionTest, ConstructorMissingIou)
{
  json params;
  params["conf"] = 0.5;
  ASSERT_THROW(
    { NonMaximumSuppression nms(params); },
    std::runtime_error);  // constructor re-throws json::exception as runtime_error
}

TEST(NonMaximumSuppressionTest, ApplyBasicNMS)
{
  json params;
  params["conf"] = 0.5;  // Keep anything above 0.5 initially
  params["iou"] = 0.3;   // Suppress if IoU > 0.3

  NonMaximumSuppression nms_step(params);

  // Create overlapping boxes with different scores
  std::vector<cv::Rect2d> boxes = {
    {10, 10, 50, 50},   // Box 0 (score 0.9) - kept
    {15, 15, 50, 50},   // Box 1 (score 0.8) - overlaps with 0, likely suppressed
    {70, 70, 40, 40},   // Box 2 (score 0.7) - separate, kept
    {75, 75, 40, 40},   // Box 3 (score 0.6) - overlaps with 2, likely suppressed
    {10, 10, 50, 50},   // Box 4 (score 0.4) - duplicate of 0, below conf threshold
    {150, 150, 30, 30}  // Box 5 (score 0.95) - separate, kept
  };
  std::vector<float> scores = {0.9, 0.8, 0.7, 0.6, 0.4, 0.95};
  std::vector<int> class_ids = {0, 0, 1, 1, 0, 2};

  DetectionData data = create_sample_detection_data(boxes, scores, class_ids);

  // Check initial state
  ASSERT_EQ(data.boxes.size(), 6);
  ASSERT_EQ(data.kept_indices.size(), 6);  // Initially all kept

  nms_step.apply(data);

  // Expected results depend on OpenCV's NMS implementation details (esp. stable sort)
  // But for these simple cases, we can predict. Scores 0.9, 0.7, 0.95 should be kept
  // as they are the highest in their overlap groups or are separate.
  // Box 4 (score 0.4) is filtered by the confidence threshold passed to NMSBoxes.
  std::vector<int> expected_kept_indices = {0, 2, 5};  // Original indices
  std::sort(
    data.kept_indices.begin(), data.kept_indices.end());  // NMSBoxes might not return in order
  std::sort(expected_kept_indices.begin(), expected_kept_indices.end());

  ASSERT_EQ(data.kept_indices, expected_kept_indices);

  // Verify data vectors weren't modified except for kept_indices
  ASSERT_EQ(data.boxes.size(), 6);      // Original size
  ASSERT_EQ(data.scores.size(), 6);     // Original size
  ASSERT_EQ(data.class_ids.size(), 6);  // Original size
}

TEST(NonMaximumSuppressionTest, ApplyBelowConfidence)
{
  json params;
  params["conf"] = 0.8;  // Only keep things >= 0.8
  params["iou"] = 0.3;

  NonMaximumSuppression nms_step(params);

  std::vector<cv::Rect2d> boxes = {
    {10, 10, 50, 50},  // Box 0 (score 0.9) - kept
    {15, 15, 50, 50},  // Box 1 (score 0.7) - below conf
    {70, 70, 40, 40},  // Box 2 (score 0.6) - below conf
    {75, 75, 40, 40},  // Box 3 (score 0.85) - kept
  };
  std::vector<float> scores = {0.9, 0.7, 0.6, 0.85};
  std::vector<int> class_ids = {0, 0, 1, 1};

  DetectionData data = create_sample_detection_data(boxes, scores, class_ids);
  nms_step.apply(data);

  std::vector<int> expected_kept_indices = {0, 3};  // Boxes 0 and 3
  std::sort(data.kept_indices.begin(), data.kept_indices.end());
  std::sort(expected_kept_indices.begin(), expected_kept_indices.end());

  ASSERT_EQ(data.kept_indices, expected_kept_indices);
}

TEST(NonMaximumSuppressionTest, ApplyNoOverlap)
{
  json params;
  params["conf"] = 0.5;
  params["iou"] = 0.3;  // IoU threshold high enough that non-overlapping boxes are kept

  NonMaximumSuppression nms_step(params);

  std::vector<cv::Rect2d> boxes = {
    {10, 10, 50, 50},   // Box 0 (score 0.9)
    {100, 10, 50, 50},  // Box 1 (score 0.8)
    {10, 100, 50, 50},  // Box 2 (score 0.7)
    {100, 100, 50, 50}  // Box 3 (score 0.6)
  };
  std::vector<float> scores = {0.9, 0.8, 0.7, 0.6};  // All above conf threshold
  std::vector<int> class_ids = {0, 0, 1, 1};

  DetectionData data = create_sample_detection_data(boxes, scores, class_ids);
  nms_step.apply(data);

  std::vector<int> expected_kept_indices = {0, 1, 2, 3};  // All should be kept
  std::sort(data.kept_indices.begin(), data.kept_indices.end());
  std::sort(expected_kept_indices.begin(), expected_kept_indices.end());

  ASSERT_EQ(data.kept_indices, expected_kept_indices);
}

TEST(NonMaximumSuppressionTest, ApplyEmptyInput)
{
  json params;
  params["conf"] = 0.5;
  params["iou"] = 0.3;

  NonMaximumSuppression nms_step(params);

  std::vector<cv::Rect2d> boxes;
  std::vector<float> scores;
  std::vector<int> class_ids;

  DetectionData data = create_sample_detection_data(boxes, scores, class_ids);
  nms_step.apply(data);

  ASSERT_TRUE(data.kept_indices.empty());
}

//  Tests for RescaleBoxes

TEST(RescaleBoxesTest, ConstructorValidMetadata)
{
  PreProcessingMetadata metadata;
  metadata.step_name = "DetectionRescale";  // Or DetectionLongestMaxSizeRescale
  metadata.input_shape = {640, 480};        // Target size
  metadata.output_shape = {1280, 960};      // Source size (after pre-scaling)

  ASSERT_NO_THROW({ RescaleBoxes rescale_step(metadata); });
}

TEST(RescaleBoxesTest, ConstructorInvalidMetadataZeroDims)
{
  PreProcessingMetadata metadata;
  metadata.step_name = "DetectionRescale";
  metadata.input_shape = {640, 0};
  metadata.output_shape = {1280, 960};
  ASSERT_THROW({ RescaleBoxes rescale_step(metadata); }, std::invalid_argument);

  metadata.input_shape = {640, 480};
  metadata.output_shape = {0, 960};
  ASSERT_THROW({ RescaleBoxes rescale_step(metadata); }, std::invalid_argument);
}

TEST(RescaleBoxesTest, ApplyScaling)
{
  // Scale from 640x640 (network input) back to 1280x960 (original, assuming padding after scale)
  PreProcessingMetadata metadata;
  metadata.step_name = "DetectionLongestMaxSizeRescale";  // Metadata represents the *pre* step
  metadata.input_shape = {1280, 960};                     // Original size (target after inverse)
  metadata.output_shape = {640, 640};                     // Size after scaling (source for inverse)

  RescaleBoxes rescale_step(metadata);

  std::vector<cv::Rect2d> boxes = {
    {160, 240, 320, 160},  // Box at 1/4 W, 1/4 H, spanning 1/2 W, 1/4 H in 640x640 space
    {0, 0, 640, 640},      // Box covering entire 640x640 space
    {600, 600, 50, 50}     // Box near edge
  };
  std::vector<float> scores = {1.0, 1.0, 1.0};
  std::vector<int> class_ids = {0, 0, 0};

  DetectionData data = create_sample_detection_data(boxes, scores, class_ids);

  rescale_step.apply(data);

  // Expected scaled boxes (scales x=1280/640=2, y=960/640=1.5)
  std::vector<cv::Rect2d> expected_boxes = {
    {160.0 * 2.0, 240.0 * 1.5, 320.0 * 2.0, 160.0 * 1.5},  // {320, 360, 640, 240}
    {0.0 * 2.0, 0.0 * 1.5, 640.0 * 2.0, 640.0 * 1.5},      // {0, 0, 1280, 960}
    {600.0 * 2.0, 600.0 * 1.5, 50.0 * 2.0, 50.0 * 1.5}     // {1200, 900, 100, 75}
  };

  ASSERT_EQ(data.kept_indices.size(), boxes.size());  // Rescale doesn't filter

  for (size_t i = 0; i < data.kept_indices.size(); ++i) {
    int original_idx = data.kept_indices[i];
    SCOPED_TRACE("Box index " + std::to_string(original_idx));  // Helps identify failed box
    ASSERT_TRUE(are_rects_approx_equal(data.boxes[original_idx], expected_boxes[original_idx]));
  }
}

//  Tests for CenterShiftBoxes

TEST(CenterShiftBoxesTest, ConstructorValidMetadata)
{
  PreProcessingMetadata metadata;
  metadata.step_name = "DetectionCenterPadding";
  metadata.input_shape = {640, 480};   // Size before padding (target after inverse)
  metadata.output_shape = {640, 640};  // Size after padding (source for inverse)

  ASSERT_NO_THROW({ CenterShiftBoxes shift_step(metadata); });
}

TEST(CenterShiftBoxesTest, ConstructorInvalidMetadataZeroDims)
{
  PreProcessingMetadata metadata;
  metadata.step_name = "DetectionCenterPadding";
  metadata.input_shape = {640, 0};
  metadata.output_shape = {640, 640};
  ASSERT_THROW({ CenterShiftBoxes shift_step(metadata); }, std::invalid_argument);

  metadata.input_shape = {640, 480};
  metadata.output_shape = {0, 640};
  ASSERT_THROW({ CenterShiftBoxes shift_step(metadata); }, std::invalid_argument);
}

TEST(CenterShiftBoxesTest, ApplyShifting)
{
  // Pad from 640x480 to 640x640 (center padding adds (640-480)/2 = 80px top/bottom)
  PreProcessingMetadata metadata;
  metadata.step_name = "DetectionCenterPadding";
  metadata.input_shape = {640, 480};   // Size before padding (target)
  metadata.output_shape = {640, 640};  // Size after padding (source)

  CenterShiftBoxes shift_step(metadata);

  std::vector<cv::Rect2d> boxes = {
    {100, 100, 200, 200},  // Box within padded area
    {0, 80, 640, 480},     // Box covering the original 640x480 area within padding
    {0, 0, 640, 640}       // Box covering the entire padded area
  };
  std::vector<float> scores = {1.0, 1.0, 1.0};
  std::vector<int> class_ids = {0, 0, 0};

  DetectionData data = create_sample_detection_data(boxes, scores, class_ids);

  shift_step.apply(data);

  // Expected shifted boxes (subtract top_pad = 80, left_pad = 0)
  std::vector<cv::Rect2d> expected_boxes = {
    {100.0 - 0.0, 100.0 - 80.0, 200.0, 200.0},  // {100, 20, 200, 200}
    {0.0 - 0.0, 80.0 - 80.0, 640.0, 480.0},     // {0, 0, 640, 480}
    {0.0 - 0.0, 0.0 - 80.0, 640.0, 640.0}       // {0, -80, 640, 640}
  };

  ASSERT_EQ(data.kept_indices.size(), boxes.size());  // Shift doesn't filter

  for (size_t i = 0; i < data.kept_indices.size(); ++i) {
    int original_idx = data.kept_indices[i];
    SCOPED_TRACE("Box index " + std::to_string(original_idx));
    ASSERT_TRUE(are_rects_approx_equal(data.boxes[original_idx], expected_boxes[original_idx]));
  }
}

//  Tests for BottomRightShiftBoxes

TEST(BottomRightShiftBoxesTest, ConstructorValidMetadata)
{
  PreProcessingMetadata metadata;
  metadata.step_name = "DetectionBottomRightPadding";
  metadata.input_shape = {640, 480};   // Size before padding (target after inverse)
  metadata.output_shape = {640, 640};  // Size after padding (source for inverse)

  ASSERT_NO_THROW({ BottomRightShiftBoxes shift_step(metadata); });
}

TEST(BottomRightShiftBoxesTest, ConstructorInvalidMetadataZeroDims)
{
  PreProcessingMetadata metadata;
  metadata.step_name = "DetectionBottomRightPadding";
  metadata.input_shape = {640, 0};
  metadata.output_shape = {640, 640};
  ASSERT_THROW({ BottomRightShiftBoxes shift_step(metadata); }, std::invalid_argument);

  metadata.input_shape = {640, 480};
  metadata.output_shape = {0, 640};
  ASSERT_THROW({ BottomRightShiftBoxes shift_step(metadata); }, std::invalid_argument);
}

TEST(BottomRightShiftBoxesTest, ApplyShifting)
{
  // Pad from 640x480 to 640x640 (bottom-right padding adds 160px bottom)
  PreProcessingMetadata metadata;
  metadata.step_name = "DetectionBottomRightPadding";
  metadata.input_shape = {640, 480};   // Size before padding (target)
  metadata.output_shape = {640, 640};  // Size after padding (source)

  BottomRightShiftBoxes shift_step(metadata);

  std::vector<cv::Rect2d> boxes = {
    {100, 100, 200, 200},  // Box within padded area
    {0, 0, 640, 480},      // Box covering the original 640x480 area (top-left corner of padded)
    {0, 0, 640, 640}       // Box covering the entire padded area
  };
  std::vector<float> scores = {1.0, 1.0, 1.0};
  std::vector<int> class_ids = {0, 0, 0};

  DetectionData data = create_sample_detection_data(boxes, scores, class_ids);

  shift_step.apply(data);

  // Expected shifted boxes (subtract top_pad = 0, left_pad = 0)
  std::vector<cv::Rect2d> expected_boxes = {
    {100.0 - 0.0, 100.0 - 0.0, 200.0, 200.0},  // {100, 100, 200, 200}
    {0.0 - 0.0, 0.0 - 0.0, 640.0, 480.0},      // {0, 0, 640, 480}
    {0.0 - 0.0, 0.0 - 0.0, 640.0, 640.0}       // {0, 0, 640, 640}
  };

  ASSERT_EQ(data.kept_indices.size(), boxes.size());  // Shift doesn't filter

  for (size_t i = 0; i < data.kept_indices.size(); ++i) {
    int original_idx = data.kept_indices[i];
    SCOPED_TRACE("Box index " + std::to_string(original_idx));
    ASSERT_TRUE(are_rects_approx_equal(data.boxes[original_idx], expected_boxes[original_idx]));
  }
}

//  Tests for PostProcessingStep::create_inverse_from_metadata (Factory Method)

TEST(PostProcessingStepFactoryTest, CreateRescale)
{
  PreProcessingMetadata metadata;
  metadata.step_name = "DetectionRescale";
  metadata.input_shape = {100, 100};
  metadata.output_shape = {200, 200};

  std::unique_ptr<PostProcessingStep> step =
    PostProcessingStep::create_inverse_from_metadata(metadata);

  ASSERT_NE(nullptr, step);
  ASSERT_NE(nullptr, dynamic_cast<RescaleBoxes *>(step.get()));
  ASSERT_EQ(step->name(), "RescaleBoxes");
}

TEST(PostProcessingStepFactoryTest, CreateCenterShift)
{
  PreProcessingMetadata metadata;
  metadata.step_name = "DetectionCenterPadding";
  metadata.input_shape = {100, 100};
  metadata.output_shape = {150, 150};

  std::unique_ptr<PostProcessingStep> step =
    PostProcessingStep::create_inverse_from_metadata(metadata);

  ASSERT_NE(nullptr, step);
  ASSERT_NE(nullptr, dynamic_cast<CenterShiftBoxes *>(step.get()));
  ASSERT_EQ(step->name(), "CenterShiftBoxes");
}

TEST(PostProcessingStepFactoryTest, CreateBottomRightShift)
{
  PreProcessingMetadata metadata;
  metadata.step_name = "DetectionBottomRightPadding";
  metadata.input_shape = {100, 100};
  metadata.output_shape = {150, 150};

  std::unique_ptr<PostProcessingStep> step =
    PostProcessingStep::create_inverse_from_metadata(metadata);

  ASSERT_NE(nullptr, step);
  ASSERT_NE(nullptr, dynamic_cast<BottomRightShiftBoxes *>(step.get()));
  ASSERT_EQ(step->name(), "BottomRightShiftBoxes");
}

TEST(PostProcessingStepFactoryTest, CreateUnknownStep)
{
  PreProcessingMetadata metadata;
  metadata.step_name = "DetectionColorConvert";  // Not a geometric step
  metadata.input_shape = {100, 100};
  metadata.output_shape = {100, 100};  // Shape might not change

  std::unique_ptr<PostProcessingStep> step =
    PostProcessingStep::create_inverse_from_metadata(metadata);

  ASSERT_EQ(nullptr, step);  // Should return nullptr
}

TEST(PostProcessingStepFactoryTest, CreateInvalidMetadataThrows)
{
  PreProcessingMetadata metadata;
  metadata.step_name = "DetectionRescale";
  metadata.input_shape = {0, 100};  // Invalid size
  metadata.output_shape = {200, 200};

  ASSERT_THROW(
    { PostProcessingStep::create_inverse_from_metadata(metadata); },
    std::invalid_argument);  // Exception from RescaleBoxes constructor should propagate
}

//  Tests for PostProcessing (Pipeline)

TEST(PostProcessingTest, ConstructorValidConfig)
{
  json post_config;
  post_config["NMS"]["conf"] = 0.5;
  post_config["NMS"]["iou"] = 0.4;

  std::vector<PreProcessingMetadata> pre_metadata = {
    {"DetectionRescale", {128, 128}, {256, 256}},        // Inverse is RescaleBoxes
    {"DetectionCenterPadding", {256, 256}, {256, 300}},  // Inverse is CenterShiftBoxes
    {"DetectionColorConvert", {256, 300}, {256, 300}}    // Skipped
  };

  ASSERT_NO_THROW({
    PostProcessing pipeline(post_config, pre_metadata);
    // Can optionally inspect the pipeline steps if needed, but constructor not throwing is a good start
  });
}

TEST(PostProcessingTest, RunPipelineEndToEnd)
{
  json post_config;
  post_config["NMS"]["conf"] = 0.5;  // Keep score >= 0.5
  post_config["NMS"]["iou"] = 0.3;   // Suppress overlap > 0.3

  // Preprocessing: Rescale from 300x200 to 600x400 -> Pad to 600x600 (center)
  // Postprocessing should be: Shift (center) from 600x600 to 600x400 -> Rescale from 600x400 to 300x200
  std::vector<PreProcessingMetadata> pre_metadata = {
    {"DetectionRescale", {300, 200}, {600, 400}},        // Rescale 300x200 -> 600x400
    {"DetectionCenterPadding", {600, 400}, {600, 600}},  // Pad 600x400 -> 600x600 (center)
    {"DetectionColorConvert", {600, 600}, {600, 600}}    // Skipped
  };
  // Inverse order:
  // 1. CenterShiftBoxes (from 600x600 to 600x400, pad_top = (600-400)/2 = 100)
  // 2. RescaleBoxes (from 600x400 to 300x200, scales x=300/600=0.5, y=200/400=0.5)

  PostProcessing pipeline(post_config, pre_metadata);

  // Raw detections (assume from model output at 600x600 size)
  std::vector<cv::Rect2d> initial_boxes = {
    {100, 100, 100, 100},  // Box 0 (score 0.9, class 0) - Within the original content area
    {110, 110, 100, 100},  // Box 1 (score 0.8, class 0) - Overlaps Box 0, likely suppressed by NMS
    {500, 500, 50,
     50},  // Box 2 (score 0.7, class 1) - In padded area (bottom-right), below original 600x400, but in 600x600
    {10, 10, 50, 50},   // Box 3 (score 0.4, class 0) - Below NMS conf threshold
    {300, 300, 50, 50}  // Box 4 (score 0.85, class 1) - Within original content area
  };
  std::vector<float> initial_scores = {0.9, 0.8, 0.7, 0.4, 0.85};
  std::vector<int> initial_class_ids = {0, 0, 1, 0, 1};

  DetectionData data =
    create_sample_detection_data(initial_boxes, initial_scores, initial_class_ids);
  cv::Size original_image_size(300, 200);  // Original image size

  pipeline.run(data, original_image_size);

  // Expected outcome:
  // 1. NMS (conf=0.5, iou=0.3) on boxes with score >= 0.5:
  //    Candidates: Box 0 (0.9), Box 1 (0.8), Box 2 (0.7), Box 4 (0.85)
  //    Box 0 and 1 overlap. Box 0 (0.9) wins over Box 1 (0.8). Box 1 suppressed.
  //    Box 2 (0.7) is separate. Kept.
  //    Box 4 (0.85) is separate. Kept.
  //    Kept indices after NMS: {0, 2, 4}
  // 2. CenterShiftBoxes (from 600x600 to 600x400, pad_top=100, pad_left=0):
  //    Boxes {0, 2, 4} shifted by (0, -100) and clamped to 600x400
  //    Box 0: {100, 100, 100, 100} -> {100, 0, 100, 100} (clamped y)
  //    Box 2: {500, 500, 50, 50} -> {500, 400, 50, 50} (clamped y, clamped height) -> {500, 400, 50, 0} -> clamped height 0
  //    Box 4: {300, 300, 50, 50} -> {300, 200, 50, 50} (clamped y, clamped height) -> {300, 200, 50, 0} -> clamped height 0
  //    This clamping might result in zero-height boxes if the original box was entirely in the padding region. Let's refine expected boxes assuming valid content is *not* in padding unless intended. Let's adjust Box 2 and 4 to be within the 600x400 region of the 600x600 input.
  //    Let's use detections at 600x400 within the 600x600 input.
  //    Initial boxes (at 600x600)
  initial_boxes = {
    {100, 150, 100, 100},  // Box 0 (score 0.9, class 0) - within 600x400 central region
    {110, 160, 100, 100},  // Box 1 (score 0.8, class 0) - Overlaps Box 0
    {500, 350, 50, 50},    // Box 2 (score 0.7, class 1) - separate, within 600x400 region
    {10, 10, 50, 50},      // Box 3 (score 0.4, class 0) - Below NMS conf
    {300, 200, 50, 50}     // Box 4 (score 0.85, class 1) - separate, within 600x400 region
  };
  initial_scores = {0.9, 0.8, 0.7, 0.4, 0.85};
  initial_class_ids = {0, 0, 1, 0, 1};

  data = create_sample_detection_data(initial_boxes, initial_scores, initial_class_ids);
  pipeline.run(data, original_image_size);

  // Expected kept indices after NMS (on original indices): {0, 2, 4} (as before)
  std::vector<int> expected_kept_indices_orig = {0, 2, 4};
  std::sort(data.kept_indices.begin(), data.kept_indices.end());
  std::sort(expected_kept_indices_orig.begin(), expected_kept_indices_orig.end());
  ASSERT_EQ(data.kept_indices, expected_kept_indices_orig);

  // Now check the transformed boxes for the kept indices {0, 2, 4}
  // Step 1 (CenterShift, 600x600 -> 600x400, pad_top=100):
  // Box 0: {100, 150, 100, 100} -> {100, 150-100, 100, 100} = {100, 50, 100, 100}
  // Box 2: {500, 350, 50, 50} -> {500, 350-100, 50, 50} = {500, 250, 50, 50}
  // Box 4: {300, 200, 50, 50} -> {300, 200-100, 50, 50} = {300, 100, 50, 50}
  // (These are within the 600x400 target, so no clamping applies yet)

  // Step 2 (Rescale, 600x400 -> 300x200, scales 0.5, 0.5):
  // Box 0: {100, 50, 100, 100} -> {100*0.5, 50*0.5, 100*0.5, 100*0.5} = {50, 25, 50, 50}
  // Box 2: {500, 250, 50, 50} -> {500*0.5, 250*0.5, 50*0.5, 50*0.5} = {250, 125, 25, 25}
  // Box 4: {300, 100, 50, 50} -> {300*0.5, 100*0.5, 50*0.5, 50*0.5} = {150, 50, 25, 25}
  // (These are within the 300x200 target, so no clamping applies yet)

  std::vector<cv::Rect2d> expected_final_boxes = {
    {50, 25, 50, 50},    // Original index 0
    {250, 125, 25, 25},  // Original index 2
    {150, 50, 25, 25}    // Original index 4
  };

  // Need to find the boxes corresponding to the kept indices
  std::vector<cv::Rect2d> final_kept_boxes;
  for (int original_idx : data.kept_indices) {
    final_kept_boxes.push_back(data.boxes[original_idx]);
  }

  // We need to sort both the expected boxes and the actual kept boxes based on
  // the sorted original indices to ensure correct comparison.
  // Data.kept_indices is already sorted.
  // We need a way to get expected boxes in the same order as data.kept_indices.
  std::vector<cv::Rect2d> sorted_expected_final_boxes(data.kept_indices.size());
  for (size_t i = 0; i < data.kept_indices.size(); ++i) {
    int original_idx = data.kept_indices[i];
    if (original_idx == 0)
      sorted_expected_final_boxes[i] = expected_final_boxes[0];
    else if (original_idx == 2)
      sorted_expected_final_boxes[i] = expected_final_boxes[1];
    else if (original_idx == 4)
      sorted_expected_final_boxes[i] = expected_final_boxes[2];
    // Add more cases if more boxes were expected to be kept
  }

  ASSERT_EQ(final_kept_boxes.size(), sorted_expected_final_boxes.size());

  for (size_t i = 0; i < final_kept_boxes.size(); ++i) {
    SCOPED_TRACE(
      "Kept box index " + std::to_string(i) + ", Original index " +
      std::to_string(data.kept_indices[i]));
    ASSERT_TRUE(are_rects_approx_equal(final_kept_boxes[i], sorted_expected_final_boxes[i]));
  }

  // Also check that scores and class_ids weren't touched for kept indices
  for (size_t i = 0; i < data.kept_indices.size(); ++i) {
    int original_idx = data.kept_indices[i];
    SCOPED_TRACE("Kept index " + std::to_string(original_idx));
    ASSERT_EQ(data.scores[original_idx], initial_scores[original_idx]);
    ASSERT_EQ(data.class_ids[original_idx], initial_class_ids[original_idx]);
  }
}

TEST(PostProcessingTest, RunPipelineEmptyInput)
{
  json post_config;
  post_config["NMS"]["conf"] = 0.5;
  post_config["NMS"]["iou"] = 0.4;
  std::vector<PreProcessingMetadata> pre_metadata = {{"DetectionRescale", {128, 128}, {256, 256}}};
  PostProcessing pipeline(post_config, pre_metadata);

  std::vector<cv::Rect2d> initial_boxes;
  std::vector<float> initial_scores;
  std::vector<int> initial_class_ids;

  DetectionData data =
    create_sample_detection_data(initial_boxes, initial_scores, initial_class_ids);
  cv::Size original_image_size(128, 128);

  ASSERT_NO_THROW({ pipeline.run(data, original_image_size); });

  ASSERT_TRUE(data.kept_indices.empty());
}

TEST(PostProcessingTest, RunPipelineInputMismatchThrows)
{
  json post_config;
  post_config["NMS"]["conf"] = 0.5;
  post_config["NMS"]["iou"] = 0.4;
  std::vector<PreProcessingMetadata> pre_metadata;
  PostProcessing pipeline(post_config, pre_metadata);

  // Mismatched sizes
  std::vector<cv::Rect2d> initial_boxes = {{10, 10, 20, 20}};
  std::vector<float> initial_scores = {0.9};
  std::vector<int> initial_class_ids = {0, 1};  // Class IDs size is different

  DetectionData data;
  data.boxes = initial_boxes;
  data.scores = initial_scores;
  data.class_ids = initial_class_ids;
  data.kept_indices.resize(initial_boxes.size());
  std::iota(data.kept_indices.begin(), data.kept_indices.end(), 0);

  cv::Size original_image_size(100, 100);

  ASSERT_THROW(
    { pipeline.run(data, original_image_size); },
    std::runtime_error);  // Expect the size check in run() to throw
}

// Add more specific tests for apply methods, including edge cases for clamping