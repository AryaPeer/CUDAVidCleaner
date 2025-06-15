#include "video_denoise.h"
#include <opencv2/opencv.hpp>
#include <iostream>

#ifdef WITH_CUDA
// Include CUDA headers
#include <cuda_runtime.h>
// Include the header
#include "video_denoise.cuh" 
#endif

// Check if CUDA is available
bool isCudaAvailable() {
#ifdef WITH_CUDA
    // Check CUDA availability
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    
    // Return true if CUDA is available
    return (error == cudaSuccess && deviceCount > 0);
#else
    // CUDA support not compiled
    return false;
#endif
}

// Factory function to create denoiser
std::unique_ptr<VideoDenoiser> createVideoDenoiser(float strength, bool forceCPU) {
    if (!forceCPU && isCudaAvailable()) {
        // Include this section only with CUDA support
#ifdef WITH_CUDA
        try {
            return std::make_unique<CUDAVideoDenoiser>(strength);
        } catch (const std::exception& e) {
            std::cerr << "Failed to initialize CUDA denoiser: " << e.what() << std::endl;
            std::cerr << "Falling back to CPU implementation" << std::endl;
            return std::make_unique<CPUVideoDenoiser>(strength);
        }
#endif
    }

    return std::make_unique<CPUVideoDenoiser>(strength);
}

// CPU denoiser implementation
CPUVideoDenoiser::CPUVideoDenoiser(float strength)
    : VideoDenoiser(strength), m_initialized(false) {
}

CPUVideoDenoiser::~CPUVideoDenoiser() {
    // No resources to clean up
}

void CPUVideoDenoiser::initialize(int width, int height) {
    m_width = width;
    m_height = height;
    m_initialized = true;
}

cv::Mat CPUVideoDenoiser::denoise(const cv::Mat& inputFrame) {
    if (!m_initialized) {
        initialize(inputFrame.cols, inputFrame.rows);
    }
    
    // Use OpenCV's CPU-based denoising
    cv::Mat result;
    
    // Apply different strength levels
    if (m_strength < 33.0f) {
        cv::fastNlMeansDenoisingColored(inputFrame, result, 3.0f, 3.0f, 7, 21);
    } else if (m_strength < 66.0f) {
        cv::bilateralFilter(inputFrame, result, 9, 75, 75);
    } else {
        cv::Mat temp;
        cv::bilateralFilter(inputFrame, temp, 9, 100, 100);
        cv::fastNlMeansDenoisingColored(temp, result, 5.0f, 5.0f, 7, 35);
    }
    
    return result;
}