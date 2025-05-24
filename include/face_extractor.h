#pragma once

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>

/**
 * FaceExtractor class for detecting and extracting faces from videos at specific timestamps
 */
class FaceExtractor {
public:
    /**
     * Constructor - initializes face detection models
     */
    FaceExtractor();
    
    /**
     * Extract faces from a video at a specific timestamp
     * @param videoPath Path to the video file
     * @param timeInSeconds Timestamp in seconds to extract faces from
     * @param outputDir Directory where extracted faces will be saved
     * @return True if extraction was successful, false otherwise
     */
    bool extractFaces(const std::string& videoPath, float timeInSeconds, const std::string& outputDir);
    
    /**
     * Extract faces from a video within a time range at regular intervals
     * @param videoPath Path to the video file
     * @param startTime Start time in seconds
     * @param endTime End time in seconds
     * @param interval Time interval in seconds between extractions
     * @param outputDir Directory where extracted faces will be saved
     * @return True if extraction was successful, false otherwise
     */
    bool extractFacesFromRange(const std::string& videoPath, float startTime, float endTime, 
                               float interval, const std::string& outputDir);

    /**
     * Check if the face extractor was initialized correctly (models loaded)
     * @return True if initialized, false otherwise
     */
    bool isInitialized() const;

private:
    /**
     * Detect faces in a frame using the loaded classifier
     * @param frame Input video frame
     * @return Vector of rectangles containing detected faces
     */
    std::vector<cv::Rect> detectFaces(const cv::Mat& frame);

    bool m_initialized = false;
    cv::CascadeClassifier m_faceClassifier;
}; 