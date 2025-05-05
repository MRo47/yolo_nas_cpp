# yolo_nas_cpp
High-Performance YOLO-NAS Inference in C++ using OpenCV DNN.

![Detections on a photo by Brett Sayles from Pexels: https://www.pexels.com/photo/women-walking-on-side-street-1119078/](images/detections.png "Detected objects on a street photo by Brett Sayles")

`yolo_nas_cpp` is a C++ library designed for efficient inference of [YOLO-NAS](https://github.com/Deci-AI/super-gradients/blob/master/YOLONAS.md) (YOLO-Neural Architecture Search) models exported to the ONNX format. It leverages OpenCV's DNN module for executing the network and provides a flexible pipeline architecture for pre- and post-processing steps, configured via a JSON file.

## Features

*   **ONNX Support:** Load YOLO-NAS models exported to the ONNX format.
*   **Configurable Pipelines:** Define pre- and post-processing pipelines using a JSON configuration file, allowing for flexibility in handling different model requirements (resizing, padding, normalization, NMS, etc.).
*   **OpenCV DNN Backend:** Utilizes OpenCV's DNN module, supporting various inference backends including CPU and CUDA.
*   **Test Suite:** Provides tests that can be enabled via CMake.
*   **FetchContent Support:** Designed for easy integration into other CMake projects using `FetchContent`.
*   **Model Export Script:** Includes a Python script to help users export YOLO-NAS models from SuperGradients to ONNX and generate the required metadata JSON.

## Prerequisites

### For the c++ library
*   C++ Compiler (supporting C++17 or later, e.g., g++, clang++, MSVC)
*   CMake (version 3.10 or higher)
*   OpenCV (with DNN module enabled, preferably built with CUDA support if you intend to use GPU acceleration)
*   spdlog library (pulled in cmake via FetchContent)
*   GoogleTest (pulled in cmake via FetchContent, if testing enabled)
*   nlohmann/json library

### For exporting the model from super_gradients/yolo_nas in python.
*   Python 3 (for the export script)
*   `super-gradients` Python library (for the export script)

TODO: steps to use the jupyter notebook.

## Model Export

YOLO-NAS models are typically trained using frameworks like SuperGradients. This library consumes models in the ONNX format along with a metadata JSON file that describes the pre- and post-processing steps required.

A Python script `export_yolo_nas.py` is provided in the `scripts` directory of this repository to facilitate this export process.

1.  **Install Python Dependencies:**
    ```bash
    pip install super-gradients torch nlohmann-json opencv-python
    ```
2.  **Run the Export Script:**
    Navigate to the root of the `yolo_nas_cpp` repository and run the script. You'll need to specify the YOLO-NAS model name and the desired output ONNX and JSON paths. The script will download the pre-trained weights (if not cached), build the model using SuperGradients, export it, and generate the metadata JSON based on common YOLO-NAS export configurations.

    ```bash
    # Example for exporting YOLO-NAS S with default parameters
    python scripts/export_yolo_nas.py --model_name yolo_nas_s --onnx_output_path yolo_nas_s.onnx --metadata_output_path yolo_nas_s_metadata.json
    ```
    Refer to the script's help (`python scripts/export_yolo_nas.py --help`) for available model names and options.

## Building the Library

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/MRo47/yolo_nas_cpp.git
    cd yolo_nas_cpp
    ```
2.  **Create Build Directory and Run CMake:**
    ```bash
    mkdir build
    cd build
    cmake ..
    ```
    *   To enable tests, add `-DENABLE_TESTS=ON` to the CMake command.
3.  **Build the Project:**
    ```bash
    make -j$(nproc) # On Linux/macOS
    # or
    # cmake --build . # On Windows or cross-platform
    ```

## Running Inference

An example executable `yolo_nas_cpp_example` is built as part of the project. You can use it to run inference on a single image using the exported ONNX model and metadata.

1.  **Build the library** as described in the previous section.
2.  **Export your YOLO-NAS model and metadata** using the `export_yolo_nas.py` script.
3.  **Run the example executable:**
    ```bash
    ./yolo_nas_cpp <path/to/your/model.onnx> <path/to/your/metadata.json> <path/to/your/image.jpg>
    ```
    Replace the placeholder paths with the actual paths to your exported files and the image you want to process. The output will display the image with bounding boxes and labels.

## Running Tests

If you enabled tests during the build (`-DENABLE_TESTS=ON`), you can run them using `ctest`.

1.  **Build the library with tests enabled:**
    ```bash
    cd yolo_nas_cpp/build
    cmake -DENABLE_TESTS=ON ..
    make -j$(nproc)
    ```
2.  **Run the tests:**
    ```bash
    ctest
    ```

## Using the Library in Your Project

This library provides CMake targets, allowing easy integration into your own CMake projects, especially using `FetchContent`.

Here's a minimal example of how to integrate `yolo_nas_cpp` into your `CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.10)
project(your_project)

# Find dependencies (adjust paths/methods as needed for your system)
# find_package(OpenCV REQUIRED)

include(FetchContent)

# Declare yolo_nas_cpp as a dependency
FetchContent_Declare(
    yolo_nas_cpp
    GIT_REPOSITORY https://github.com/MRo47/yolo_nas_cpp.git
    GIT_TAG main # Or specify a specific branch, tag, or commit
)

# Make yolo_nas_cpp available (downloads and configures it)
FetchContent_MakeAvailable(yolo_nas_cpp)

# Add your executable
add_executable(your_detection_app main.cpp)

# Link against the yolo_nas_cpp library target
target_link_libraries(your_detection_app PRIVATE yolo_nas_cpp::yolo_nas_cpp)
```

TODO: mention yolo_nas_onnx, license on weights.