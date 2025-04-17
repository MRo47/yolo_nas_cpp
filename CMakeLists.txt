cmake_minimum_required(VERSION 3.14)
project(yolo_nas_cpp VERSION 0.1 LANGUAGES CXX)

# Use C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV (require at least core and imgproc, adjust if needed)
find_package(OpenCV REQUIRED core imgproc highgui imgcodecs)

# Define the library
add_library(yolo_nas_cpp
    src/pre_processing.cpp
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

add_executable(yolo_nas_cpp_exec
    src/yolo_nas_cpp.cpp
)

set_target_properties(yolo_nas_cpp_exec PROPERTIES
    OUTPUT_NAME yolo_nas_cpp
)

target_link_libraries(yolo_nas_cpp_exec
    PRIVATE
        yolo_nas_cpp
        ${OpenCV_LIBS}
)


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

# Optional: CMake config for find_package
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
