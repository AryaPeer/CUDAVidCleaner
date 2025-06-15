#include "video_denoise.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <iostream>
#include <cmath> // For fmaxf, fminf

// CUDA error checking
#define CHECK_CUDA_ERROR(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ << ": " << \
                  cudaGetErrorString(err) << std::endl; \
        exit(1); \
    } \
}

// Define block size for CUDA kernels
#define BLOCK_SIZE 16

// CUDA kernel: RGB to YUV
__global__ void rgb2yuv_kernel(unsigned char* input, float* y, float* u, float* v, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y_idx < height) {
        int idx = y_idx * width + x;
        int rgb_idx = idx * 3;
        
        float r = input[rgb_idx];
        float g = input[rgb_idx + 1];
        float b = input[rgb_idx + 2];
        
        // RGB to YUV conversion
        y[idx] = 0.299f * r + 0.587f * g + 0.114f * b;
        u[idx] = -0.169f * r - 0.331f * g + 0.5f * b + 128.0f;
        v[idx] = 0.5f * r - 0.419f * g - 0.081f * b + 128.0f;
    }
}

// CUDA kernel: bilateral filtering
__global__ void bilateral_filter_kernel(float* input, float* output, int width, int height, 
                                        float sigma_space, float sigma_color) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = y * width + x;
        
        float sum = 0.0f;
        float total_weight = 0.0f;
        float center_val = input[idx];
        
        // Bilateral filter radius
        int radius = static_cast<int>(2.0f * sigma_space);
        
        for (int dy = -radius; dy <= radius; dy++) {
            for (int dx = -radius; dx <= radius; dx++) {
                int nx = x + dx;
                int ny = y + dy;
                
                if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                    int neighbor_idx = ny * width + nx;
                    float neighbor_val = input[neighbor_idx];
                    
                    // Calculate spatial and color weights
                    float space_dist = dx*dx + dy*dy;
                    float color_dist = (center_val - neighbor_val) * (center_val - neighbor_val);
                    
                    float weight = expf(-space_dist/(2.0f*sigma_space*sigma_space)) * 
                                   expf(-color_dist/(2.0f*sigma_color*sigma_color));
                    
                    sum += weight * neighbor_val;
                    total_weight += weight;
                }
            }
        }
        
        // Output filtered value
        output[idx] = sum / total_weight;
    }
}

// CUDA kernel: non-local means denoising
__global__ void nlm_denoise_kernel(float* input, float* output, int width, int height, float h_param) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = y * width + x;
        
        // Patch size and search window
        const int patch_radius = 3;
        const int search_radius = 7;
        const int patch_size = (2 * patch_radius + 1) * (2 * patch_radius + 1);
        
        float sum_weights = 0.0f;
        float sum_values = 0.0f;
        
        // For each pixel in search window
        for (int sy = y - search_radius; sy <= y + search_radius; sy++) {
            for (int sx = x - search_radius; sx <= x + search_radius; sx++) {
                if (sx >= 0 && sx < width && sy >= 0 && sy < height) {
                    // Skip the center pixel
                    if (sx == x && sy == y) continue;
                    
                    // Calculate patch distance
                    float distance = 0.0f;
                    float weight = 0.0f;
                    
                    for (int py = -patch_radius; py <= patch_radius; py++) {
                        for (int px = -patch_radius; px <= patch_radius; px++) {
                            int p1x = x + px;
                            int p1y = y + py;
                            int p2x = sx + px;
                            int p2y = sy + py;
                            
                            if (p1x >= 0 && p1x < width && p1y >= 0 && p1y < height &&
                                p2x >= 0 && p2x < width && p2y >= 0 && p2y < height) {
                                float diff = input[p1y * width + p1x] - input[p2y * width + p2x];
                                distance += diff * diff;
                            }
                        }
                    }
                    
                    // Calculate weight
                    distance /= patch_size;
                    weight = expf(-distance / (h_param * h_param));
                    
                    // Accumulate weighted value
                    sum_weights += weight;
                    sum_values += weight * input[sy * width + sx];
                }
            }
        }
        
        // Self weight
        sum_weights += 1.0f;
        sum_values += input[idx];
        
        // Calculate final value
        output[idx] = sum_values / sum_weights;
    }
}

// CUDA kernel: YUV to RGB
__global__ void yuv2rgb_kernel(float* y, float* u, float* v, unsigned char* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y_idx < height) {
        int idx = y_idx * width + x;
        int rgb_idx = idx * 3;
        
        float y_val = y[idx];
        float u_val = u[idx] - 128.0f;
        float v_val = v[idx] - 128.0f;
        
        // YUV to RGB conversion
        float r = y_val + 1.402f * v_val;
        float g = y_val - 0.344f * u_val - 0.714f * v_val;
        float b = y_val + 1.772f * u_val;
        
        // Clamp values
        output[rgb_idx] = static_cast<unsigned char>(fmaxf(0.0f, fminf(255.0f, r)));
        output[rgb_idx + 1] = static_cast<unsigned char>(fmaxf(0.0f, fminf(255.0f, g)));
        output[rgb_idx + 2] = static_cast<unsigned char>(fmaxf(0.0f, fminf(255.0f, b)));
    }
}

// CUDAVideoDenoiser implementation
CUDAVideoDenoiser::CUDAVideoDenoiser(float strength) 
    : VideoDenoiser(strength), m_width(0), m_height(0), m_initialized(false),
      d_inputFrame(nullptr), d_outputFrame(nullptr), d_tempBuffer1(nullptr), d_tempBuffer2(nullptr) {
}

CUDAVideoDenoiser::~CUDAVideoDenoiser() {
    cleanup();
}

void CUDAVideoDenoiser::initialize(int width, int height) {
    // Clean up any previously allocated resources
    cleanup();
    
    m_width = width;
    m_height = height;
    
    // Allocate CUDA memory
    size_t frame_size = width * height * 3 * sizeof(unsigned char); // RGB
    size_t channel_size = width * height * sizeof(float); // Single channel
    
    CHECK_CUDA_ERROR(cudaMalloc(&d_inputFrame, frame_size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_outputFrame, frame_size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_tempBuffer1, channel_size * 3)); // For YUV channels
    CHECK_CUDA_ERROR(cudaMalloc(&d_tempBuffer2, channel_size * 3)); // For denoised YUV
    
    m_initialized = true;
}

void CUDAVideoDenoiser::cleanup() {
    if (d_inputFrame) {
        cudaFree(d_inputFrame);
        d_inputFrame = nullptr;
    }
    
    if (d_outputFrame) {
        cudaFree(d_outputFrame);
        d_outputFrame = nullptr;
    }
    
    if (d_tempBuffer1) {
        cudaFree(d_tempBuffer1);
        d_tempBuffer1 = nullptr;
    }
    
    if (d_tempBuffer2) {
        cudaFree(d_tempBuffer2);
        d_tempBuffer2 = nullptr;
    }
    
    m_initialized = false;
}

void CUDAVideoDenoiser::uploadFrame(const cv::Mat& frame) {
    if (!m_initialized) {
        std::cerr << "CUDAVideoDenoiser not initialized" << std::endl;
        return;
    }
    
    // Check dimensions
    if (frame.cols != m_width || frame.rows != m_height) {
        std::cerr << "Frame dimensions don't match the initialized size" << std::endl;
        return;
    }
    
    // Upload frame data to GPU
    size_t frame_size = m_width * m_height * 3 * sizeof(unsigned char);
    CHECK_CUDA_ERROR(cudaMemcpy(d_inputFrame, frame.data, frame_size, cudaMemcpyHostToDevice));
}

cv::Mat CUDAVideoDenoiser::downloadFrame() {
    if (!m_initialized) {
        std::cerr << "CUDAVideoDenoiser not initialized" << std::endl;
        return cv::Mat();
    }
    
    // Create output Mat
    cv::Mat result(m_height, m_width, CV_8UC3);
    
    // Download frame data from GPU
    size_t frame_size = m_width * m_height * 3 * sizeof(unsigned char);
    CHECK_CUDA_ERROR(cudaMemcpy(result.data, d_outputFrame, frame_size, cudaMemcpyDeviceToHost));
    
    return result;
}

cv::Mat CUDAVideoDenoiser::denoise(const cv::Mat& inputFrame) {
    if (!m_initialized) {
        // Initialize if not already done
        initialize(inputFrame.cols, inputFrame.rows);
    }
    
    // Upload frame to GPU
    uploadFrame(inputFrame);
    
    // Configure CUDA grid and block dimensions
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((m_width + block.x - 1) / block.x, (m_height + block.y - 1) / block.y);
    
    // Extract Y, U, V components
    float* d_y = d_tempBuffer1;
    float* d_u = d_tempBuffer1 + m_width * m_height;
    float* d_v = d_tempBuffer1 + 2 * m_width * m_height;
    
    float* d_y_denoised = d_tempBuffer2;
    float* d_u_denoised = d_tempBuffer2 + m_width * m_height;
    float* d_v_denoised = d_tempBuffer2 + 2 * m_width * m_height;
    
    // Convert RGB to YUV
    rgb2yuv_kernel<<<grid, block>>>(d_inputFrame, d_y, d_u, d_v, m_width, m_height);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    // Calculate denoising parameters
    // Strength is [0, 100]
    float strength_normalized = m_strength / 100.0f;
    float bilateral_space_sigma = 3.0f + 5.0f * strength_normalized;
    float bilateral_color_sigma = 20.0f + 30.0f * strength_normalized;
    float nlm_h = 5.0f + 15.0f * strength_normalized;
    
    // Apply bilateral filter to Y channel
    bilateral_filter_kernel<<<grid, block>>>(d_y, d_y_denoised, m_width, m_height, 
                                            bilateral_space_sigma, bilateral_color_sigma);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    // Apply non-local means to U, V channels
    nlm_denoise_kernel<<<grid, block>>>(d_u, d_u_denoised, m_width, m_height, nlm_h);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    nlm_denoise_kernel<<<grid, block>>>(d_v, d_v_denoised, m_width, m_height, nlm_h);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    // Convert back to RGB
    yuv2rgb_kernel<<<grid, block>>>(d_y_denoised, d_u_denoised, d_v_denoised, d_outputFrame, m_width, m_height);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    // Ensure all operations are complete
    cudaDeviceSynchronize();
    
    // Download result
    return downloadFrame();
}