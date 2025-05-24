#pragma once

#include <opencv2/opencv.hpp>

/**
 * Interface for video denoising
 */
class VideoDenoiser {
public:
    /**
     * Constructor for VideoDenoiser
     * @param strength Denoising strength (0-100)
     */
    VideoDenoiser(float strength) : m_strength(strength) {}
    
    /**
     * Virtual destructor
     */
    virtual ~VideoDenoiser() {}

    /**
     * Initialize denoiser resources
     * @param width Frame width
     * @param height Frame height
     */
    virtual void initialize(int width, int height) = 0;

    /**
     * Denoise a frame
     * @param inputFrame Input frame
     * @return Denoised frame
     */
    virtual cv::Mat denoise(const cv::Mat& inputFrame) = 0;

protected:
    float m_strength;
};

/**
 * CPU implementation of video denoising
 */
class CPUVideoDenoiser : public VideoDenoiser {
public:
    /**
     * Constructor for CPUVideoDenoiser
     * @param strength Denoising strength (0-100)
     */
    CPUVideoDenoiser(float strength);
    
    /**
     * Destructor
     */
    ~CPUVideoDenoiser() override;

    /**
     * Initialize CPU resources
     * @param width Frame width
     * @param height Frame height
     */
    void initialize(int width, int height) override;

    /**
     * Denoise a frame using CPU algorithms
     * @param inputFrame Input frame
     * @return Denoised frame
     */
    cv::Mat denoise(const cv::Mat& inputFrame) override;

private:
    int m_width = 0;
    int m_height = 0;
    bool m_initialized = false;
};

/**
 * Factory function to create appropriate video denoiser based on hardware
 * @param strength Denoising strength (0-100)
 * @param forceCPU Force CPU implementation even if CUDA is available
 * @return A video denoiser instance
 */
std::unique_ptr<VideoDenoiser> createVideoDenoiser(float strength, bool forceCPU = false);

/**
 * Check if CUDA is available on the system
 * @return true if CUDA is available, false otherwise
 */
bool isCudaAvailable(); 