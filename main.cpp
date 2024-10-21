#include <cxxopts.hpp>
#include <indicators/block_progress_bar.hpp>
#include <opencv2/opencv.hpp>
#ifdef USE_OPENCV_CUDA
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>
#endif // USE_OPENCV_CUDA

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>
// #include <ranges>

#ifdef USE_OPENCV_CUDA
using Frame = cv::cuda::GpuMat;
#else
using Frame = cv::Mat;
#endif // USE_OPENCV_CUDA
using Pixel = cv::Vec3b;
const Pixel WHITE = Pixel::all(255);
const Pixel BLACK = Pixel::all(0);

// template<typename T>
// requires std::is_base_of_v<cv::Matx<typename T::value_type, T::rows, T::cols>, T>
// bool isSimilarColor(const T& pixel1, const T& pixel2, int color_threshold) {
// 	T diff;
// 	cv::absdiff(pixel1, pixel2, diff);
// 	cv::threshold(diff, diff, color_threshold, 255, cv::THRESH_BINARY);
// 	return cv::countNonZero(diff) == 0;
// }

// Function to calculate color difference
bool isSimilarColor(const Pixel& pixel1, const Pixel& pixel2, int color_threshold) {
	return (
		std::abs(pixel1[0] - pixel2[0]) <= color_threshold &&
		std::abs(pixel1[1] - pixel2[1]) <= color_threshold &&
		std::abs(pixel1[2] - pixel2[2]) <= color_threshold
	);
}

// Function to detect margin dynamically
std::pair<int, int> detectHMargins(const Frame& frame, int color_threshold, float margin_threshold) {
	const int width = frame.cols;
	const int height = frame.rows;
	auto isAlmostBlack = [&](Pixel pixel) {
		return isSimilarColor(pixel, BLACK, color_threshold);
	};
	// Assume the margin is same on both sides
	for (int x = 0; x < width / 2; x++) {
		int black_count = 0;
		for (int y = 0; y < height; y++) {
			if (isAlmostBlack(frame.at<Pixel>(y, x))) {
				black_count++;
			}
		}
		if (static_cast<float>(black_count) / height <= margin_threshold) {
			return {x, width - x};
		}
	}
	return {0, width};
}

// Function to calculate Intersection over Union (IoU)
float calculateIoU(const Frame& frame1, const Frame& frame2, const cv::Rect& extracted_region, int color_threshold) {
	int intersection_count = 0;
	int union_count = 0;
	auto isNotAlmostWhite = [&](Pixel pixel) {
		return !isSimilarColor(pixel, WHITE, color_threshold);
	};
	for (int x = extracted_region.x; x < extracted_region.x + extracted_region.width; x++) {
		for (int y = extracted_region.y; y < extracted_region.y + extracted_region.height; y++) {
			const Pixel& pixel1 = frame1.at<Pixel>(y, x);
			const Pixel& pixel2 = frame2.at<Pixel>(y, x);
			bool is_pixel1_colored = isNotAlmostWhite(pixel1);
			bool is_pixel2_colored = isNotAlmostWhite(pixel2);
			if (!is_pixel1_colored && !is_pixel2_colored) continue;
			if (is_pixel1_colored || is_pixel2_colored) {
				union_count++;
			}
			if (is_pixel1_colored && is_pixel2_colored) {
				intersection_count++;
			}
		}
	}
	return static_cast<float>(intersection_count) / union_count;
}

[[maybe_unused]] float calculateNormL2(const Frame& frame1, const Frame& frame2, const cv::Rect& extracted_region) {
	const auto& mat1 = frame1(extracted_region);
	const auto& mat2 = frame2(extracted_region);
	return cv::norm(mat1, mat2, cv::NORM_L2) / (extracted_region.width * extracted_region.height);
}

void setPureWhite(Frame& frame, cv::Rect extracted_region, int color_threshold) {
	for (int x = extracted_region.x; x < extracted_region.x + extracted_region.width; x++) {
		for (int y = extracted_region.y; y < extracted_region.y + extracted_region.height; y++) {
			if (isSimilarColor(frame.at<Pixel>(y, x), WHITE, color_threshold)) {
				frame.at<Pixel>(y, x) = WHITE;
			}
		}
	}
}

void saveFrameToFile(
	const Frame& frame, int frame_number, const cv::Rect& extracted_region,
	const std::string& prefix = "frame_", const std::string& suffix = ".png"
) {
	std::string frame_filename = prefix + std::to_string(frame_number) + suffix;
	cv::imwrite(frame_filename, frame(extracted_region));
}

// bool is_magick_available(const std::string& prog) {
// 	return std::system((prog + " -version").c_str()) == 0;
// }
// void merge_pdf(const std::string& prog, const std::vector<std::string>& img_files, const std::string& pdf_file) {
// 	std::string delim {" "};
// 	std::string quote {"\""};
// 	auto joinView = img_files | std::views::join_with(quote + delim + quote);
// 	std::string img_files_str(joinView.begin(), joinView.end());
// 	std::string command = prog + delim + quote + img_files_str + quote + delim + quote + pdf_file + quote;
// 	std::system(command.c_str());
// }

int main(int argc, char* argv[]) {
	cxxopts::Options options("frame-extract", "Extract key frames from a video file based on IoU threshold");
	options.add_options()
		("i,input", "Input video file",
			cxxopts::value<std::string>())
		("o,output", "Output images directory",
			cxxopts::value<std::string>()->default_value("./"))
		("p,prefix", "Prefix for the extracted frames",
			cxxopts::value<std::string>()->default_value("frame_"))
		("s,suffix", "Suffix for the extracted frames",
			cxxopts::value<std::string>()->default_value(".png"))
		("f,framestep", "frame steps between frames",
			cxxopts::value<int>()->default_value("1"))
		("w,white", "Set pure white color to the extracted region",
			cxxopts::value<bool>()->default_value("true"))
		("W,no-white", "Do not set pure white color to the extracted region",
			cxxopts::value<bool>()->default_value("false"))
		("d,diff", "Difference threshold for L2 norm, range: [0, 1], higher value extracts more frames",
			cxxopts::value<float>()->default_value("0.1"))
		("t,iou", "IoU threshold, range: [0, 1], higher value extracts more frames",
			cxxopts::value<float>()->default_value("0.6"))
		("c,color", "Color threshold, range: [0, 255], higher value extracts more frames",
			cxxopts::value<int>()->default_value("16"))
		("m,margin", "Magin difference rate threshold, range: [0, 1], lower value extracts more frames",
			cxxopts::value<float>()->default_value("0.8"))
		("log", "Log file",
			cxxopts::value<std::string>()->default_value(""))
		("h,help", "Print usage");
	options.parse_positional({"input"});
	options.positional_help("input");
	auto result = options.parse(argc, argv);
	if (result.count("help") || !result.count("input")) {
		std::cout << options.help() << std::endl;
		return EXIT_FAILURE;
	}
	std::string videoFile = result["input"].as<std::string>();
	std::string outputDir = result["output"].as<std::string>();
	std::string prefix = result["prefix"].as<std::string>();
	if (prefix.contains(std::filesystem::path::preferred_separator)) {
		std::cerr << "Prefix cannot contain path separator, use -o option for output directory." << std::endl;
		return EXIT_FAILURE;
	}
	std::string suffix = result["suffix"].as<std::string>();
	int framestep = result["framestep"].as<int>();
	bool set_pure_white = result["white"].as<bool>();
	if (result["no-white"].as<bool>()) {
		set_pure_white = false;
	}
	float diff_threshold = result["diff"].as<float>();
	if (diff_threshold < 0 || diff_threshold > 1) {
		std::cerr << "Difference threshold must be in the range [0, 1]." << std::endl;
		return EXIT_FAILURE;
	}
	float iou_threshold = result["iou"].as<float>();
	if (diff_threshold < 0 || diff_threshold > 1) {
		std::cerr << "Difference threshold must be in the range [0, 1]." << std::endl;
		return EXIT_FAILURE;
	}
	int color_threshold = result["color"].as<int>();
	if (color_threshold < 0 || color_threshold > 255) {
		std::cerr << "Color threshold must be in the range [0, 255]." << std::endl;
		return EXIT_FAILURE;
	}
	float margin_threshold = result["margin"].as<float>();
	if (margin_threshold < 0 || margin_threshold > 1) {
		std::cerr << "Margin threshold must be in the range [0, 1]." << std::endl;
		return EXIT_FAILURE;
	}
	std::string log_file = result["log"].as<std::string>();

	if (!std::filesystem::exists(outputDir)) {
		try {
			std::filesystem::create_directories(outputDir);
			std::cout << "Output directory created: " << outputDir << std::endl;
		} catch (const std::filesystem::filesystem_error& e) {
			std::cerr << "Error creating output directory: " << e.what() << std::endl;
			return EXIT_FAILURE;
		}
	}
	prefix = std::filesystem::path(outputDir) / prefix;
	std::unique_ptr<std::ostream> log_stream;
	if (log_file.empty()) { // If log file is not specified, drop the log
		log_stream = std::make_unique<std::ostream>(nullptr);
	} else {
		log_stream = std::make_unique<std::ofstream>(log_file);
		if (!static_cast<std::ofstream*>(log_stream.get())->is_open()) {
			std::cerr << "Error opening IoU log file." << std::endl;
			return EXIT_FAILURE;
		}
	}

	cv::VideoCapture cap(videoFile);
	if (!cap.isOpened()) {
		std::cerr << "Error opening video file." << std::endl;
		return EXIT_FAILURE;
	}
	int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
	double fps = cap.get(cv::CAP_PROP_FPS);
	std::string total_duration = std::to_string(static_cast<int>(total_frames / fps)) + "s";
	std::cout << "Total frames: " << total_frames << ", FPS: " << fps << std::endl;
	int extract_frames = total_frames / framestep;
	indicators::BlockProgressBar bar{
		indicators::option::BarWidth{20},
		indicators::option::Start{"["},
		indicators::option::End{"]"},
		indicators::option::ForegroundColor{indicators::Color::green},
		indicators::option::ShowElapsedTime{true},
		indicators::option::ShowRemainingTime{true},
		indicators::option::MaxProgress{extract_frames}
	};
	Frame prev_frame, curr_frame;
	int frame_number = 0;
	prev_frame = Frame::zeros(
		cap.get(cv::CAP_PROP_FRAME_HEIGHT),
		cap.get(cv::CAP_PROP_FRAME_WIDTH),
		// cap.get(cv::CAP_PROP_POS_FRAMES); // Return WRONG value
		CV_8UC3
	);
	std::string video_text = "V:0s/100s";
	auto set_video_text = [&]() { // Lambda function to generate video text based the context
		int current_time_ms = cap.get(cv::CAP_PROP_POS_MSEC);
		std::string current_time = std::to_string(current_time_ms / 1000) + "s";
		video_text = "V:" + current_time + "/" + total_duration;
	};
	std::string frame_text = "F:0/100";
	auto set_frame_text = [&]() {
		frame_text = "F:" + std::to_string(frame_number) + "/" + std::to_string(extract_frames);
	};
	auto skip_frames = [&](int framestep) {
		for (int i = 0; i < framestep - 1; i++) {
			cap.grab();
		}
	};
	// cap >> curr_frame;
	while (
		frame_number++,
		cap.read(curr_frame)
	) {
		set_video_text();
		auto [left_margin, right_margin] = detectHMargins(curr_frame, color_threshold, margin_threshold);
		cv::Rect extracted_region(left_margin, 0, right_margin - left_margin, curr_frame.rows);
		float iou = calculateIoU(prev_frame, curr_frame, extracted_region, color_threshold);
		*log_stream << frame_number << "," << iou << std::endl;
		// float l2_norm = calculateNormL2(prev_frame, curr_frame, extracted_region);
		// *log_stream << iou << "," << l2_norm << std::endl;
		if (iou < iou_threshold) {
			if (set_pure_white) {
				setPureWhite(curr_frame, extracted_region, color_threshold);
			}
			saveFrameToFile(curr_frame, frame_number, extracted_region, prefix, suffix);
			set_frame_text();
		}

		bar.set_progress(frame_number);
		bar.set_option(indicators::option::PostfixText{video_text + "|" + frame_text});
		prev_frame = std::move(curr_frame); // Move current frame to previous frame
		skip_frames(framestep);
	}
	// cap.release(); // Automatically called when the VideoCapture object is destroyed
	return EXIT_SUCCESS;
}
