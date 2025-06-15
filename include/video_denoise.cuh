#pragma once

#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include "video_denoise.h"

/**
 * CUDA-accelerated video denoising
 */
class CUDAVideoDenoiser : public VideoDenoiser {
public:
    /**
     * Constructor
     * @param strength Denoising strength
     */
    CUDAVideoDenoiser(float strength);
    
    /**
     * Destructor
     */
    ~CUDAVideoDenoiser() override;

    /**
     * Initializes CUDA resources
     * @param width Frame width
     * @param height Frame height
     */
    void initialize(int width, int height) override;

    /**
     * Denoises a frame using CUDA
     * @param inputFrame Input frame
     * @return Denoised frame
     */
    cv::Mat denoise(const cv::Mat& inputFrame) override;

private:
    float m_strength;
    int m_width;
    int m_height;
    bool m_initialized;
    
    // CUDA memory
    unsigned char* d_inputFrame;
    unsigned char* d_outputFrame;
    float* d_tempBuffer1;
    float* d_tempBuffer2;
    
    // Helper methods
    void uploadFrame(const cv::Mat& frame);
    cv::Mat downloadFrame();
    void cleanup();
};

// CUDA kernel declarations
__global__ void rgb2yuv_kernel(unsigned char* input, float* y, float* u, float* v, int width, int height);
__global__ void bilateral_filter_kernel(float* input, float* output, int width, int height, float sigma_space, float sigma_color);
__global__ void nlm_denoise_kernel(float* input, float* output, int width, int height, float h_param);
__global__ void yuv2rgb_kernel(float* y, float* u, float* v, unsigned char* output, int width, int height);