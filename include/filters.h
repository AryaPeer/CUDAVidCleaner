#pragma once

#include <vector>
#include <complex>
#include <memory>

/**
 * BandPassFilter class for audio filtering
 */
class BandPassFilter {
public:
    /**
     * Constructor for band pass filter
     * @param sampleRate Audio sample rate in Hz
     * @param lowCutoff Lower cutoff frequency in Hz
     * @param highCutoff Higher cutoff frequency in Hz
     */
    BandPassFilter(int sampleRate, float lowCutoff, float highCutoff);

    /**
     * Apply the filter to audio data
     * @param input Input audio samples
     * @return Filtered audio samples
     */
    std::vector<float> apply(const std::vector<float>& input);

private:
    int m_sampleRate;
    float m_lowCutoff;
    float m_highCutoff;
    std::vector<float> m_coefficients;
    
    void calculateCoefficients();
};

/**
 * SpectralSubtraction class for noise reduction
 */
class SpectralSubtraction {
public:
    /**
     * Constructor for spectral subtraction
     * @param fftSize FFT size to use for processing
     * @param hopSize Hop size between consecutive frames
     * @param reductionFactor Noise reduction factor (0-1)
     */
    SpectralSubtraction(int fftSize, int hopSize, float reductionFactor);

    /**
     * Process audio data to remove noise using spectral subtraction
     * @param input Input audio samples
     * @param noiseProfile Optional pre-recorded noise profile (magnitude spectrum).
     *                     If nullptr, noise is estimated from the initial part of the input.
     * @return Processed audio with reduced noise
     */
    std::vector<float> process(const std::vector<float>& input, const std::vector<float>* noiseProfile = nullptr);

    /**
     * Estimate noise profile from audio
     * @param input Input audio samples
     * @param durationSec Duration in seconds to use for estimation (from beginning)
     * @return Noise profile spectrum
     */
    std::vector<float> estimateNoiseProfile(const std::vector<float>& input, float durationSec = 0.5);

private:
    int m_fftSize;
    int m_hopSize;
    float m_reductionFactor;
    
    void fft_complex_inplace(std::vector<std::complex<float>>& buffer);
    std::vector<std::complex<float>> performFFT(const std::vector<float>& input, int start, int size);
    std::vector<float> performIFFT(const std::vector<std::complex<float>>& spectrum);
    std::vector<float> getWindowFunction(int size);
};

/**
 * AudioProcessor class to combine filters and processing
 */
class AudioProcessor {
public:
    /**
     * Constructor for audio processor
     * @param sampleRate Audio sample rate in Hz
     * @param lowCutoff Lower cutoff frequency in Hz
     * @param highCutoff Higher cutoff frequency in Hz
     * @param noiseReduction Noise reduction factor (0-1)
     */
    AudioProcessor(int sampleRate, float lowCutoff, float highCutoff, float noiseReduction);

    /**
     * Process audio data with both filters
     * @param input Input audio samples
     * @return Processed audio
     */
    std::vector<float> process(const std::vector<float>& input);

private:
    std::unique_ptr<BandPassFilter> m_bandPassFilter;
    std::unique_ptr<SpectralSubtraction> m_spectralSubtraction;
    int m_sampleRate;
}; 