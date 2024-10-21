cmake_minimum_required(VERSION 3.28)

project(VideoFrameExtractor)
add_executable(VideoFrameExtractor main.cpp)

set(OpenCV_STATIC ON)
find_package(OpenCV REQUIRED
	core
	imgcodecs # cv::imwrite
	videoio # cv::VideoCapture
	# imgproc # cv::absdiff, cv::threshold
)
target_link_libraries(VideoFrameExtractor ${OpenCV_LIBS})

set_target_properties(VideoFrameExtractor PROPERTIES
	CXX_STANDARD 23
	# CXX_STANDARD_REQUIRED ON
	CXX_EXTENSIONS OFF
	OUTPUT_NAME "frame-extractor"
)

find_package(CUDAToolkit)
if (OpenCV_CUDA_VERSION)
	message(STATUS "CUDA support is enabled in OpenCV")
	target_compile_definitions(VideoFrameExtractor PRIVATE USE_OPENCV_CUDA)
else()
	message(STATUS "CUDA support is not enabled in OpenCV")
endif()
