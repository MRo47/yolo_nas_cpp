cmake_minimum_required(VERSION 3.14)
project(yolo_nas_cpp VERSION 0.1 LANGUAGES CXX)

# Use C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV 
find_package(OpenCV REQUIRED core imgproc highgui imgcodecs dnn)

include(FetchContent)

set(SPDLOG_INSTALL ON)
# set(SPDLOG_HEADER_ONLY ON) # Uncomment for header-only

FetchContent_Declare(
  spdlog
  GIT_REPOSITORY https://github.com/gabime/spdlog.git
  GIT_TAG        v1.13.0 # Or the latest release tag, or a specific commit hash
)

FetchContent_MakeAvailable(spdlog)

# Define the library
add_library(yolo_nas_cpp
    src/pre_processing.cpp
    src/post_processing.cpp
    src/network.cpp
)

# Include directories for consumers
target_include_directories(yolo_nas_cpp
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# Link OpenCV
target_link_libraries(yolo_nas_cpp
    PUBLIC
        ${OpenCV_LIBS}
)

target_link_libraries(yolo_nas_cpp PRIVATE spdlog::spdlog)

add_executable(yolo_nas_cpp_exec
    src/yolo_nas_cpp.cpp
)

set_target_properties(yolo_nas_cpp_exec PROPERTIES
    OUTPUT_NAME yolo_nas_cpp
)

target_link_libraries(yolo_nas_cpp_exec
    PRIVATE
        yolo_nas_cpp
        spdlog::spdlog
    # No need to explicitly link ${OpenCV_LIBS} here because yolo_nas_cpp links it PUBLICLY
)

# --- Testing ---
enable_testing()

FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/release-1.11.0.zip # Use a specific release tag
)

FetchContent_MakeAvailable(googletest)

add_executable(yolo_nas_cpp_tests
    tests/post_processing.cpp
    tests/pre_processing.cpp
)

target_link_libraries(yolo_nas_cpp_tests
    PRIVATE
        yolo_nas_cpp
        gtest_main
)

add_test(NAME yolo_nas_cpp_all_tests COMMAND yolo_nas_cpp_tests)
# --- End Testing ---


# Install rules (excluding models and config)
include(GNUInstallDirs)

install(TARGETS yolo_nas_cpp
    EXPORT yolo_nas_cppTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT yolo_nas_cppTargets
    FILE yolo_nas_cppTargets.cmake
    NAMESPACE yolo_nas_cpp::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/yolo_nas_cpp
)

include(CMakePackageConfigHelpers)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/yolo_nas_cppConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/yolo_nas_cppConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/yolo_nas_cpp
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/yolo_nas_cppConfig.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/yolo_nas_cpp
)
