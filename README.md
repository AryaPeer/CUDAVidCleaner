# CUDA Video Cleaner

A video processing tool that cleans video and removes background noise from audio. Supports both CUDA-accelerated processing on NVIDIA GPUs and CPU-only processing on non-NVIDIA systems.

## Features
- Custom band pass filtering for audio
- Spectral subtraction for background noise removal
- Video denoising (CUDA-accelerated if available, CPU fallback otherwise)
- Face extraction from video timestamps

## Requirements
- OpenCV (4.0+)
- FFmpeg libraries (libavcodec, libavformat, libavutil, libswresample)
- CMake (3.10+)
- CUDA Toolkit (11.0+) - *Optional* for GPU acceleration

## Installation

### Install dependencies (Ubuntu/Debian)
```bash
# Install build tools
sudo apt-get update
sudo apt-get install -y build-essential cmake pkg-config

# Install OpenCV
sudo apt-get install -y libopencv-dev

# Install FFmpeg development libraries
sudo apt-get install -y libavcodec-dev libavformat-dev libavutil-dev libswresample-dev

# Optional: Install CUDA (for NVIDIA GPUs)
# Follow NVIDIA's instructions for your specific distribution
```

### Install dependencies (Manjaro/Arch)
```bash
# Install build tools
sudo pacman -Syu
sudo pacman -S --needed base-devel cmake pkg-config

# Install OpenCV
sudo pacman -S opencv

# Install FFmpeg
sudo pacman -S ffmpeg

# Optional: Install CUDA (for NVIDIA GPUs)
sudo pacman -S cuda
source /etc/profile  # Update environment variables
```

## Building
```bash
mkdir build
cd build
cmake ..
make
```

## Usage

### Video Cleaner
```bash
./cuda_video_cleaner input_video.mp4 output_video.mp4
```

#### Options
- `--low-cutoff` (default: 100): Low cutoff frequency for bandpass filter in Hz
- `--high-cutoff` (default: 8000): High cutoff frequency for bandpass filter in Hz
- `--noise-reduction` (default: 0.5): Spectral subtraction noise reduction factor (0-1)
- `--video-denoise-strength` (default: 10): Video denoising strength (0-100)
- `--force-cpu`: Force CPU-based video processing even if CUDA is available

### Face Extractor
Extract faces from a video at specific timestamps:

```bash
# Extract faces from a single timestamp
./face_extractor video.mp4 10.5 faces/

# Extract faces from a time range with specified interval
./face_extractor --range video.mp4 5.0 15.0 1.0 faces/
```

#### Face Extractor Options
- Single timestamp: `./face_extractor video_path timestamp output_directory`
- Time range: `./face_extractor --range video_path start_time end_time interval output_directory`

## Testing Without NVIDIA GPU

If you don't have an NVIDIA GPU, the application will automatically use the CPU implementation for video denoising. You can also force CPU processing even on systems with NVIDIA GPUs by using the `--force-cpu` option:

```bash
./cuda_video_cleaner --force-cpu input_video.mp4 output_video.mp4
```

## Performance Notes

- CUDA-accelerated processing is significantly faster but only available on NVIDIA GPUs
- CPU-based denoising provides similar quality but at reduced speed
- For best performance on non-NVIDIA systems, use a smaller `--video-denoise-strength` value
