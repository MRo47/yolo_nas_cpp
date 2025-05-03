
#include "yolo_nas_cpp/pre_processing.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <stdexcept>
#include <vector>

#include "yolo_nas_cpp/utils.hpp"

using namespace yolo_nas_cpp;
using json = nlohmann::json;

// Helper to create a dummy image for testing
cv::Mat create_dummy_image(int width, int height, int type, cv::Scalar value)
{
  cv::Mat img(height, width, type, value);
  // Optional: set a few specific pixels for verification
  if (width > 1 && height > 1) {
    if (type == CV_8UC3) {
      img.at<cv::Vec3b>(0, 0) = cv::Vec3b(10, 20, 30);
      img.at<cv::Vec3b>(height - 1, width - 1) = cv::Vec3b(250, 240, 230);
    } else if (type == CV_32FC3) {
      img.at<cv::Vec3f>(0, 0) = cv::Vec3f(0.1f, 0.2f, 0.3f);
      img.at<cv::Vec3f>(height - 1, width - 1) = cv::Vec3f(0.9f, 0.8f, 0.7f);
    } else if (type == CV_8UC1) {
      img.at<uchar>(0, 0) = 50;
      img.at<uchar>(height - 1, width - 1) = 200;
    } else if (type == CV_32FC1) {
      img.at<float>(0, 0) = 0.5f;
      img.at<float>(height - 1, width - 1) = 0.95f;
    }
  }
  return img;
}

// Helper to compare two images approximately (for float types)
// Using cv::norm is robust for aggregate comparison
// Corrected signature for ASSERT_PRED_FORMAT3
::testing::AssertionResult IsImageApproxEqual(
  const char * expected_expr, const char * actual_expr,
  const char * tolerance_expr,                                         // Expression strings
  const cv::Mat & expected, const cv::Mat & actual, double tolerance)  // Evaluated values
{
  // The function body remains largely the same, as it uses the evaluated values
  // You could optionally use tolerance_expr in the failure message if needed,
  // but it's not strictly necessary.

  if (expected.empty() && actual.empty()) {
    return ::testing::AssertionSuccess();
  }
  if (expected.empty() != actual.empty()) {
    return ::testing::AssertionFailure() << "One image (" << expected_expr
                                         << ") is empty, the other (" << actual_expr << ") is not.";
  }
  if (expected.size() != actual.size()) {
    return ::testing::AssertionFailure()
           << "Image sizes mismatch: " << expected_expr << " size " << expected.size() << ", "
           << actual_expr << " size " << actual.size();
  }
  if (expected.type() != actual.type()) {
    return ::testing::AssertionFailure()
           << "Image types mismatch: " << expected_expr << " type " << expected.type() << ", "
           << actual_expr << " type " << actual.type();
  }

  double diff_norm = cv::norm(expected, actual, cv::NORM_INF);  // Max absolute difference
  if (diff_norm <= tolerance) {
    return ::testing::AssertionSuccess();
  } else {
    return ::testing::AssertionFailure()
           << "Image mismatch: max absolute difference (" << diff_norm << ") exceeds tolerance ("
           << tolerance_expr << " = " << tolerance << ")";
  }
}

// The macro definition remains the same
#define ASSERT_IMAGE_APPROX_EQ(expected, actual, tolerance) \
  ASSERT_PRED_FORMAT3(IsImageApproxEqual, expected, actual, tolerance)

// --- Tests for StandardizeImage ---

TEST(StandardizeImageTest, ConstructorValidParams)
{
  json params;
  params["max_value"] = 255.0;
  ASSERT_NO_THROW({ StandardizeImage step(params); });
}

TEST(StandardizeImageTest, ConstructorMissingMaxValue)
{
  json params;
  // No "max_value"
  ASSERT_THROW(
    { StandardizeImage step(params); },
    std::runtime_error);  // Constructor re-throws json::exception as runtime_error
}

TEST(StandardizeImageTest, Apply8UC3)
{
  json params;
  params["max_value"] = 255.0;
  StandardizeImage step(params);

  cv::Mat input = create_dummy_image(10, 10, CV_8UC3, cv::Scalar(128, 64, 32));
  cv::Mat output;

  step.apply(input, output);

  ASSERT_EQ(output.size(), input.size());
  ASSERT_EQ(output.type(), CV_32FC3);

  // Check some known pixels
  cv::Vec3b original_pixel = input.at<cv::Vec3b>(0, 0);  // Should be (10, 20, 30)
  cv::Vec3f output_pixel = output.at<cv::Vec3f>(0, 0);
  ASSERT_NEAR(output_pixel[0], original_pixel[0] / 255.0f, 1e-6);
  ASSERT_NEAR(output_pixel[1], original_pixel[1] / 255.0f, 1e-6);
  ASSERT_NEAR(output_pixel[2], original_pixel[2] / 255.0f, 1e-6);

  original_pixel = input.at<cv::Vec3b>(5, 5);  // Should be (128, 64, 32)
  output_pixel = output.at<cv::Vec3f>(5, 5);
  ASSERT_NEAR(output_pixel[0], 128.0f / 255.0f, 1e-6);
  ASSERT_NEAR(output_pixel[1], 64.0f / 255.0f, 1e-6);
  ASSERT_NEAR(output_pixel[2], 32.0f / 255.0f, 1e-6);
}

TEST(StandardizeImageTest, Apply32FC3)
{
  json params;
  params["max_value"] = 1.0;  // Already float [0, 1], dividing by 1.0 should yield same
  StandardizeImage step(params);

  cv::Mat input = create_dummy_image(10, 10, CV_32FC3, cv::Scalar(0.5f, 0.3f, 0.1f));
  cv::Mat output;

  step.apply(input, output);

  ASSERT_EQ(output.size(), input.size());
  ASSERT_EQ(output.type(), CV_32FC3);

  ASSERT_IMAGE_APPROX_EQ(input, output, 1e-6);  // Should be almost identical

  // Check some known pixels for exact calculation
  cv::Vec3f original_pixel = input.at<cv::Vec3f>(0, 0);  // Should be (0.1, 0.2, 0.3)
  cv::Vec3f output_pixel = output.at<cv::Vec3f>(0, 0);
  ASSERT_NEAR(output_pixel[0], original_pixel[0] / 1.0f, 1e-6);
  ASSERT_NEAR(output_pixel[1], original_pixel[1] / 1.0f, 1e-6);
  ASSERT_NEAR(output_pixel[2], original_pixel[2] / 1.0f, 1e-6);
}

TEST(StandardizeImageTest, CalculateOutputShape)
{
  json params;
  params["max_value"] = 255.0;
  StandardizeImage step(params);

  cv::Size input_shape = {640, 480};
  ASSERT_EQ(step.calculate_output_shape(input_shape), input_shape);

  input_shape = {100, 200};
  ASSERT_EQ(step.calculate_output_shape(input_shape), input_shape);
}

// --- Tests for NormalizeImage ---

TEST(NormalizeImageTest, ConstructorValidParams)
{
  json params;
  params["mean"] = {0.0, 0.0, 0.0};
  params["std"] = {1.0, 1.0, 1.0};
  ASSERT_NO_THROW({ NormalizeImage step(params); });
}

TEST(NormalizeImageTest, ConstructorMissingMean)
{
  json params;
  params["std"] = {1.0, 1.0, 1.0};
  ASSERT_THROW(
    { NormalizeImage step(params); },
    std::runtime_error);  // Missing key, re-throws json::exception
}

TEST(NormalizeImageTest, ConstructorMissingStd)
{
  json params;
  params["mean"] = {0.0, 0.0, 0.0};
  ASSERT_THROW(
    { NormalizeImage step(params); },
    std::runtime_error);  // Missing key, re-throws json::exception
}

TEST(NormalizeImageTest, Apply3ChannelFloat)
{
  json params;
  params["mean"] = {0.1, 0.2, 0.3};
  params["std"] = {0.5, 0.6, 0.7};
  NormalizeImage step(params);

  cv::Size image_size(10, 10);
  // Input image base: 10x10 CV_32FC3 filled with (0.5f, 0.5f, 0.5f)
  cv::Mat input =
    create_dummy_image(image_size.width, image_size.height, CV_32FC3, cv::Scalar(0.5f, 0.5f, 0.5f));
  // create_dummy_image also sets (0,0) to (0.1f, 0.2f, 0.3f)
  // create_dummy_image also sets (h-1, w-1) = (9,9) to (0.9f, 0.8f, 0.7f)

  // Manually set pixel (1,1) as before
  input.at<cv::Vec3f>(1, 1) = cv::Vec3f(0.7f, 0.8f, 0.9f);

  cv::Mat output;
  step.apply(input, output);

  ASSERT_EQ(output.size(), input.size());
  ASSERT_EQ(output.type(), CV_32FC3);

  cv::Mat expected_output(image_size.height, image_size.width, CV_32FC3);

  // Calculate the normalized value for the default fill color (0.5, 0.5, 0.5)
  cv::Vec3f fill_norm_val;
  fill_norm_val[0] = (0.5f - 0.1f) / 0.5f;  // 0.8
  fill_norm_val[1] = (0.5f - 0.2f) / 0.6f;  // 0.5
  fill_norm_val[2] = (0.5f - 0.3f) / 0.7f;  // ~0.285714
  expected_output.setTo(cv::Scalar(fill_norm_val[0], fill_norm_val[1], fill_norm_val[2]));

  // Calculate and set the normalized value for the pixel at (0,0) (original value: 0.1, 0.2, 0.3)
  cv::Vec3f p00_norm_val;
  p00_norm_val[0] = (0.1f - 0.1f) / 0.5f;  // 0.0
  p00_norm_val[1] = (0.2f - 0.2f) / 0.6f;  // 0.0
  p00_norm_val[2] = (0.3f - 0.3f) / 0.7f;  // 0.0
  expected_output.at<cv::Vec3f>(0, 0) = p00_norm_val;

  // Calculate and set the normalized value for the pixel at (1,1) (original value: 0.7, 0.8, 0.9)
  cv::Vec3f p11_norm_val;
  p11_norm_val[0] = (0.7f - 0.1f) / 0.5f;  // 1.2
  p11_norm_val[1] = (0.8f - 0.2f) / 0.6f;  // 1.0
  p11_norm_val[2] = (0.9f - 0.3f) / 0.7f;  // ~0.857143
  expected_output.at<cv::Vec3f>(1, 1) = p11_norm_val;

  // Calculate and set the normalized value for the pixel at (9,9) (original value: 0.9, 0.8, 0.7)
  // This is based on create_dummy_image behavior for CV_32FC3
  cv::Vec3f p99_norm_val;
  p99_norm_val[0] = (0.9f - 0.1f) / 0.5f;  // 1.6
  p99_norm_val[1] = (0.8f - 0.2f) / 0.6f;  // 1.0
  p99_norm_val[2] = (0.7f - 0.3f) / 0.7f;  // ~0.571428
  expected_output.at<cv::Vec3f>(9, 9) = p99_norm_val;

  // --- Compare the actual output with the expected output ---
  // Use the custom assertion macro for the entire image
  ASSERT_IMAGE_APPROX_EQ(expected_output, output, 1e-6);
}

TEST(NormalizeImageTest, Apply8UC3)
{
  json params;
  params["mean"] = {100.0, 50.0, 20.0};
  params["std"] = {50.0, 25.0, 10.0};
  NormalizeImage step(params);

  cv::Size image_size(10, 10);
  // Input image base: 10x10 CV_8UC3 filled with (150, 75, 30)
  // create_dummy_image also sets (0,0) to (10, 20, 30)
  // create_dummy_image also sets (h-1, w-1) = (9,9) to (250, 240, 230)
  cv::Mat input =
    create_dummy_image(image_size.width, image_size.height, CV_8UC3, cv::Scalar(150, 75, 30));

  // Manually set pixel (1,1) as before
  input.at<cv::Vec3b>(1, 1) = cv::Vec3b(200, 100, 40);

  cv::Mat output;
  step.apply(input, output);

  ASSERT_EQ(output.size(), input.size());
  ASSERT_EQ(output.type(), CV_32FC3);

  // --- Calculate Expected Output Image ---
  cv::Mat expected_output(image_size.height, image_size.width, CV_32FC3);

  // Get mean and std as float vectors for easier calculation
  cv::Vec3f mean_f = {
    static_cast<float>(params["mean"][0]), static_cast<float>(params["mean"][1]),
    static_cast<float>(params["mean"][2])};
  cv::Vec3f std_f = {
    static_cast<float>(params["std"][0]), static_cast<float>(params["std"][1]),
    static_cast<float>(params["std"][2])};

  // Iterate through all pixels to calculate the expected output
  for (int r = 0; r < image_size.height; ++r) {
    for (int c = 0; c < image_size.width; ++c) {
      cv::Vec3b input_pixel_8u = input.at<cv::Vec3b>(r, c);
      cv::Vec3f input_pixel_f = {
        static_cast<float>(input_pixel_8u[0]), static_cast<float>(input_pixel_8u[1]),
        static_cast<float>(input_pixel_8u[2])};

      cv::Vec3f expected_pixel_f;
      expected_pixel_f[0] = (input_pixel_f[0] - mean_f[0]) / std_f[0];
      expected_pixel_f[1] = (input_pixel_f[1] - mean_f[1]) / std_f[1];
      expected_pixel_f[2] = (input_pixel_f[2] - mean_f[2]) / std_f[2];

      expected_output.at<cv::Vec3f>(r, c) = expected_pixel_f;
    }
  }

  // --- Compare the actual output with the expected output ---
  // Use the custom assertion macro for the entire image
  ASSERT_IMAGE_APPROX_EQ(expected_output, output, 1e-6);
}

TEST(NormalizeImageTest, CalculateOutputShape)
{
  json params;
  params["mean"] = {0.0, 0.0, 0.0};
  params["std"] = {1.0, 1.0, 1.0};
  NormalizeImage step(params);

  cv::Size input_shape = {640, 480};
  ASSERT_EQ(step.calculate_output_shape(input_shape), input_shape);
}

// --- Tests for DetectionCenterPadding ---

TEST(DetectionCenterPaddingTest, ConstructorValidParams)
{
  json params;
  params["pad_value"] = 114;
  params["output_shape"] = {640, 640};
  ASSERT_NO_THROW({ DetectionCenterPadding step(params); });
}

TEST(DetectionCenterPaddingTest, ConstructorMissingParams)
{
  json params;
  // Missing output_shape
  params["pad_value"] = 114;
  ASSERT_THROW(
    { DetectionCenterPadding step(params); },
    std::runtime_error);  // Constructor re-throws json::exception
}

TEST(DetectionCenterPaddingTest, ApplyPadding)
{
  json params;
  params["pad_value"] = 10;
  params["output_shape"] = {300, 200};  // Target H=300, W=200
  DetectionCenterPadding step(params);

  cv::Mat input =
    create_dummy_image(100, 200, CV_8UC3, cv::Scalar(128, 128, 128));  // Input H=200, W=100
  cv::Mat output;

  step.apply(input, output);

  ASSERT_EQ(output.size().width, 200);
  ASSERT_EQ(output.size().height, 300);
  ASSERT_EQ(output.type(), input.type());  // Padding doesn't change type

  // Expected padding:
  // pad_height = 300 - 200 = 100 -> pad_top = 50, pad_bottom = 50
  // pad_width = 200 - 100 = 100 -> pad_left = 50, pad_right = 50

  // Check padding value pixels
  ASSERT_EQ(output.at<cv::Vec3b>(0, 0), cv::Vec3b(10, 10, 10));      // Top-left padding
  ASSERT_EQ(output.at<cv::Vec3b>(299, 199), cv::Vec3b(10, 10, 10));  // Bottom-right padding
  ASSERT_EQ(output.at<cv::Vec3b>(25, 25), cv::Vec3b(10, 10, 10));  // Inside top-left padding region
  ASSERT_EQ(
    output.at<cv::Vec3b>(275, 175), cv::Vec3b(10, 10, 10));  // Inside bottom-right padding region

  // Check original image content region
  // Original (0,0) is at (50, 50) in the output
  ASSERT_EQ(output.at<cv::Vec3b>(50, 50), input.at<cv::Vec3b>(0, 0));  // Should be (10, 20, 30)
  // Original (height-1, width-1) is at (50+height-1, 50+width-1) = (50+199, 50+99) = (249, 149)
  ASSERT_EQ(
    output.at<cv::Vec3b>(249, 149),
    input.at<cv::Vec3b>(input.rows - 1, input.cols - 1));  // Should be (250, 240, 230)
}

TEST(DetectionCenterPaddingTest, ApplyNoPadding)
{
  json params;
  params["pad_value"] = 10;
  params["output_shape"] = {200, 100};  // Target is same size as input H=200, W=100
  DetectionCenterPadding step(params);

  cv::Mat input = create_dummy_image(100, 200, CV_8UC3, cv::Scalar(128, 128, 128));
  cv::Mat output;

  step.apply(input, output);

  ASSERT_EQ(output.size(), input.size());
  ASSERT_EQ(output.type(), input.type());

  // Check if output is a copy of input
  ASSERT_IMAGE_APPROX_EQ(input, output, 0);
}

TEST(DetectionCenterPaddingTest, ApplyInputLargerThanOutput)
{
  json params;
  params["pad_value"] = 10;
  params["output_shape"] = {100, 50};  // Target smaller than input, H=100, W=50
  DetectionCenterPadding step(params);

  cv::Mat input = create_dummy_image(100, 200, CV_8UC3, cv::Scalar(128, 128, 128));
  cv::Mat output;

  // center padding with negative padding implicitly crops.
  // The warning message indicates this behavior. Test that it runs without throwing.
  ASSERT_NO_THROW({ step.apply(input, output); });

  // Expected output size is the target size
  ASSERT_EQ(output.size().width, 50);
  ASSERT_EQ(output.size().height, 100);
  ASSERT_EQ(output.type(), input.type());

  // Check content (should be top-left part of original)
  // Negative padding: pad_height = 100-200 = -100, pad_width = 50-100 = -50
  // pad_top = -50, pad_bottom = -50
  // pad_left = -25, pad_right = -25
  // center padding will effectively take a ROI starting at (25, 50) of size (50, 100)
  cv::Mat expected_output = input(cv::Rect(25, 50, 50, 100));

  ASSERT_IMAGE_APPROX_EQ(expected_output, output, 0);
}

TEST(DetectionCenterPaddingTest, CalculateOutputShape)
{
  json params;
  params["pad_value"] = 10;
  params["output_shape"] = {300, 200};  // H=300, W=200
  DetectionCenterPadding step(params);

  cv::Size input_shape = {100, 200};
  ASSERT_EQ(step.calculate_output_shape(input_shape), (cv::Size{200, 300}));
}

// --- Tests for DetectionBottomRightPadding ---

TEST(DetectionBottomRightPaddingTest, ConstructorValidParams)
{
  json params;
  params["pad_value"] = 114;
  params["output_shape"] = {640, 640};
  ASSERT_NO_THROW({ DetectionBottomRightPadding step(params); });
}

TEST(DetectionBottomRightPaddingTest, ConstructorMissingParams)
{
  json params;
  // Missing pad_value
  params["output_shape"] = {640, 640};
  ASSERT_THROW(
    { DetectionBottomRightPadding step(params); },
    std::runtime_error);  // Constructor re-throws json::exception
}

TEST(DetectionBottomRightPaddingTest, ApplyPadding)
{
  json params;
  params["pad_value"] = 10;
  params["output_shape"] = {300, 200};  // Target H=300, W=200
  DetectionBottomRightPadding step(params);

  cv::Mat input =
    create_dummy_image(100, 200, CV_8UC3, cv::Scalar(128, 128, 128));  // Input H=200, W=100
  cv::Mat output;

  step.apply(input, output);

  ASSERT_EQ(output.size().width, 200);
  ASSERT_EQ(output.size().height, 300);
  ASSERT_EQ(output.type(), input.type());  // Padding doesn't change type

  // Expected padding:
  // pad_height = 300 - 200 = 100 -> pad_top = 0, pad_bottom = 100
  // pad_width = 200 - 100 = 100 -> pad_left = 0, pad_right = 100

  // Check padding value pixels
  ASSERT_EQ(
    output.at<cv::Vec3b>(250, 150), cv::Vec3b(10, 10, 10));  // Inside bottom-right padding region

  // Check original image content region (should be at top-left 0,0)
  ASSERT_EQ(output.at<cv::Vec3b>(0, 0), input.at<cv::Vec3b>(0, 0));  // Should be (10, 20, 30)
  ASSERT_EQ(
    output.at<cv::Vec3b>(input.rows - 1, input.cols - 1),
    input.at<cv::Vec3b>(input.rows - 1, input.cols - 1));  // Should be (250, 240, 230)
}

TEST(DetectionBottomRightPaddingTest, ApplyNoPadding)
{
  json params;
  params["pad_value"] = 10;
  params["output_shape"] = {200, 100};  // Target matches input, H=200, W=100
  DetectionBottomRightPadding step(params);

  cv::Mat input = create_dummy_image(100, 200, CV_8UC3, cv::Scalar(128, 128, 128));
  cv::Mat output;

  step.apply(input, output);

  ASSERT_EQ(output.size(), input.size());
  ASSERT_EQ(output.type(), input.type());

  // Check if output is a copy of input
  ASSERT_IMAGE_APPROX_EQ(input, output, 0);
}

TEST(DetectionBottomRightPaddingTest, ApplyInputLargerThanOutput)
{
  json params;
  params["pad_value"] = 10;
  params["output_shape"] = {100, 50};  // Target smaller than input, H=100, W=50
  DetectionBottomRightPadding step(params);

  cv::Mat input = create_dummy_image(100, 200, CV_8UC3, cv::Scalar(128, 128, 128));
  cv::Mat output;

  // OpenCV's copyMakeBorder with negative padding implicitly crops.
  ASSERT_NO_THROW({ step.apply(input, output); });

  // Expected output size is the target size
  ASSERT_EQ(output.size().width, 50);
  ASSERT_EQ(output.size().height, 100);
  ASSERT_EQ(output.type(), input.type());

  // Check content (should be top-left part of original)
  // Negative padding: pad_height = 100-200 = -100, pad_width = 50-100 = -50
  // pad_top = 0, pad_bottom = -100
  // pad_left = 0, pad_right = -50
  // copyMakeBorder will effectively take a ROI starting at (0, 0) of size (50, 100)
  cv::Mat expected_output = input(cv::Rect(0, 0, 50, 100));

  ASSERT_IMAGE_APPROX_EQ(expected_output, output, 0);
}

TEST(DetectionBottomRightPaddingTest, CalculateOutputShape)
{
  json params;
  params["pad_value"] = 10;
  params["output_shape"] = {300, 200};  // Target H=300, W=200
  DetectionBottomRightPadding step(params);

  cv::Size input_shape = {100, 200};
  ASSERT_EQ(step.calculate_output_shape(input_shape), (cv::Size{200, 300}));
}

// --- Tests for PassthroughStep (used by ImagePermute) ---

TEST(PassthroughStepTest, ConstructorValidParams)
{
  json params;
  params["arbitrary_param"] = "some_value";  // Passthrough ignores params
  ASSERT_NO_THROW({ PassthroughStep step(params); });
  ASSERT_NO_THROW({
    PassthroughStep step({});  // Empty params
  });
}

TEST(PassthroughStepTest, ApplyShallowCopy)
{
  json params;
  PassthroughStep step(params);

  cv::Mat input = create_dummy_image(50, 50, CV_8UC3, cv::Scalar(100, 150, 200));
  cv::Mat output;

  step.apply(input, output);

  ASSERT_EQ(output.size(), input.size());
  ASSERT_EQ(output.type(), input.type());
  ASSERT_EQ(output.data, input.data);  // Check for shallow copy (same data buffer)

  // Verify modifying input affects output (shallow copy)
  input.at<cv::Vec3b>(0, 0) = cv::Vec3b(1, 2, 3);
  ASSERT_EQ(output.at<cv::Vec3b>(0, 0), cv::Vec3b(1, 2, 3));
}

TEST(PassthroughStepTest, CalculateOutputShape)
{
  json params;
  PassthroughStep step(params);

  cv::Size input_shape = {640, 480};
  ASSERT_EQ(step.calculate_output_shape(input_shape), input_shape);
}

// --- Tests for DetectionLongestMaxSizeRescale ---

TEST(DetectionLongestMaxSizeRescaleTest, ConstructorValidParams)
{
  json params;
  params["output_shape"] = {640, 640};
  ASSERT_NO_THROW({ DetectionLongestMaxSizeRescale step(params); });
}

TEST(DetectionLongestMaxSizeRescaleTest, ConstructorMissingParams)
{
  json params;
  // Missing output_shape
  ASSERT_THROW(
    { DetectionLongestMaxSizeRescale step(params); },
    std::runtime_error);  // Constructor re-throws json::exception
}

TEST(DetectionLongestMaxSizeRescaleTest, Apply)
{
  json params;
  params["output_shape"] = {200, 200};  // Target 200x200
  DetectionLongestMaxSizeRescale step(params);

  // Input {100, 150} (Portrait), target {200, 200}
  // scale = min(200/150, 200/100) = min(1.333, 2.0) = 1.333...
  // new_w = round(100 * 1.333...) = 133
  // new_h = round(150 * 1.333...) = 200
  cv::Mat input = create_dummy_image(100, 150, CV_8UC3, cv::Scalar(128, 128, 128));
  cv::Mat output;

  step.apply(input, output);

  ASSERT_EQ(output.size(), (cv::Size{133, 200}));
  ASSERT_EQ(output.type(), input.type());  // Resize preserves type by default for 8UC

  // Test with input equal to target shape
  cv::Mat input_equal = create_dummy_image(200, 200, CV_8UC3, cv::Scalar(50, 50, 50));
  cv::Mat output_equal;
  step.apply(input_equal, output_equal);
  ASSERT_EQ(output_equal.size(), (cv::Size{200, 200}));
  ASSERT_EQ(output_equal.type(), input_equal.type());
  ASSERT_IMAGE_APPROX_EQ(input_equal, output_equal, 0);  // Should be a copy

  // Test empty input
  cv::Mat input_empty;
  cv::Mat output_empty;
  ASSERT_THROW({ step.apply(input_empty, output_empty); }, std::runtime_error);
}

// --- Tests for DetectionRescale ---

TEST(DetectionRescaleTest, ConstructorValidParams)
{
  json params;
  params["output_shape"] = {300, 300};
  ASSERT_NO_THROW({ DetectionRescale step(params); });
}

TEST(DetectionRescaleTest, ConstructorMissingParams)
{
  json params;
  // Missing output_shape
  ASSERT_THROW(
    { DetectionRescale step(params); },
    std::runtime_error);  // Constructor re-throws json::exception
}

TEST(DetectionRescaleTest, CalculateOutputShape)
{
  json params;
  params["output_shape"] = {300, 300};
  DetectionRescale step(params);

  cv::Size input_shape = {640, 480};  // Arbitrary input shape
  ASSERT_EQ(
    step.calculate_output_shape(input_shape),
    (cv::Size{300, 300}));  // Should always return output_shape
}

TEST(DetectionRescaleTest, Apply)
{
  json params;
  params["output_shape"] = {100, 100};  // Target 100x100
  DetectionRescale step(params);

  cv::Mat input =
    create_dummy_image(200, 300, CV_8UC3, cv::Scalar(128, 128, 128));  // Input 200x300
  cv::Mat output;

  step.apply(input, output);

  ASSERT_EQ(output.size(), (cv::Size{100, 100}));
  ASSERT_EQ(output.type(), input.type());

  // Test empty input
  cv::Mat input_empty;
  cv::Mat output_empty;
  ASSERT_THROW({ step.apply(input_empty, output_empty); }, std::runtime_error);
}

// --- Tests for PreProcessingStep::create_from_json (Factory) ---

TEST(PreProcessingStepFactoryTest, CreateStandardizeImage)
{
  json params;
  params["max_value"] = 255.0;
  cv::Size input_shape = {100, 100};

  auto result = PreProcessingStep::create_from_json("StandardizeImage", params, input_shape);

  ASSERT_NE(nullptr, result.first);  // Check step_ptr is not null
  ASSERT_NE(nullptr, dynamic_cast<StandardizeImage *>(result.first.get()));  // Check type

  // Check metadata
  ASSERT_EQ(result.second.step_name, "StandardizeImage");
  ASSERT_EQ(result.second.input_shape, input_shape);
  ASSERT_EQ(result.second.output_shape, input_shape);  // Standardize keeps shape
  ASSERT_EQ(result.second.params, params);
}

TEST(PreProcessingStepFactoryTest, CreateUnknownStepThrows)
{
  json params;
  cv::Size input_shape = {100, 100};

  ASSERT_THROW(
    { PreProcessingStep::create_from_json("UnknownStepType", params, input_shape); },
    std::runtime_error);  // Factory should throw
}

TEST(PreProcessingStepFactoryTest, InvalidParamsForKnownStepThrows)
{
  json params;
  // Missing max_value for StandardizeImage
  cv::Size input_shape = {100, 100};

  ASSERT_THROW(
    { PreProcessingStep::create_from_json("StandardizeImage", params, input_shape); },
    std::runtime_error);  // Exception from step constructor should propagate
}

// --- Tests for PreProcessing (Pipeline) ---

TEST(PreProcessingTest, ConstructorValidConfig)
{
  json config = R"(
        [
            {"DetectionLongestMaxSizeRescale": {"output_shape": [640, 640]}},
            {"DetectionCenterPadding": {"output_shape": [640, 640], "pad_value": 114}},
            {"StandardizeImage": {"max_value": 255}},
            {"NormalizeImage": {"mean": [0, 0, 0], "std": [255, 255, 255]}},
            {"ImagePermute": {}}
        ]
    )"_json;  // Using raw string literal + _json for readability

  cv::Size input_shape = {800, 600};  // Starting size

  ASSERT_NO_THROW({
    PreProcessing pipeline(config, input_shape);
    ASSERT_EQ(pipeline.get_metadata().size(), 5);  // Check number of steps added
  });
}

TEST(PreProcessingTest, ConstructorEmptyConfig)
{
  json config = json::array();  // Empty array
  cv::Size input_shape = {800, 600};

  // Should not throw, but might print a warning
  ASSERT_NO_THROW({
    PreProcessing pipeline(config, input_shape);
    ASSERT_TRUE(pipeline.get_metadata().empty());
  });
}

TEST(PreProcessingTest, ConstructorInvalidConfigStepNotObject)
{
  json config = R"(
        [
            "DetectionRescale"
        ]
    )"_json;
  cv::Size input_shape = {800, 600};

  ASSERT_THROW(
    { PreProcessing pipeline(config, input_shape); },
    std::runtime_error);  // Expect error because step is not an object
}

TEST(PreProcessingTest, GetMetadataReturnsCorrectData)
{
  json config = R"(
        [
            {"DetectionRescale": {"output_shape": [416, 416]}},
            {"StandardizeImage": {"max_value": 255.0}}
        ]
    )"_json;
  cv::Size input_shape = {800, 600};

  PreProcessing pipeline(config, input_shape);
  const auto & metadata = pipeline.get_metadata();

  ASSERT_EQ(metadata.size(), 2);

  ASSERT_EQ(metadata[0].step_name, "DetectionRescale");
  ASSERT_EQ(metadata[0].input_shape, input_shape);            // Input to step 1
  ASSERT_EQ(metadata[0].output_shape, (cv::Size{416, 416}));  // Output of step 1
  ASSERT_EQ(metadata[0].params, config[0].begin().value());

  ASSERT_EQ(metadata[1].step_name, "StandardizeImage");
  ASSERT_EQ(metadata[1].input_shape, (cv::Size{416, 416}));  // Input to step 2 is output of step 1
  ASSERT_EQ(
    metadata[1].output_shape, (cv::Size{416, 416}));  // Output of step 2 (Standardize keeps shape)
  ASSERT_EQ(metadata[1].params, config[1].begin().value());
}
