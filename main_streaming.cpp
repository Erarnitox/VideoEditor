#include <opencv2/highgui.hpp>
#include <print>
#include <stacktrace>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <filesystem>
#include <random>
#include <thread>
#include <execution>
#include <future>
#include <sndfile.h>
#include <opencv2/opencv.hpp>
#include <numeric>
#include <format>
#include <chrono>
#include <limits>
#include <stdexcept>
#include <sys/resource.h>
#include <unistd.h>
#include <sys/mman.h>
#include <fcntl.h>

//------------------------------------------------------
// Configuration constants - OPTIMIZED FOR LOW MEMORY
//------------------------------------------------------
constexpr int g_sampleLengthMs = 200;
constexpr int g_chunkProcessingInterval = 2;  // Reset VideoCapture more frequently
constexpr double g_maxAudioVideoDurationDiff = 0.5;
constexpr size_t g_maxChunkSize = 200;  // Smaller chunks to reduce memory usage
constexpr size_t g_maxSilentFramesBuffer = 50;  // Limit silent frames buffer
constexpr size_t g_maxAudioChunkSamples = 44100 * 2 * 2;  // Max 2 seconds of audio in memory

//------------------------------------------------------
// Memory monitoring with more aggressive thresholds
//------------------------------------------------------
[[nodiscard]] static inline
size_t getCurrentRSS() {
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS info;
    GetProcessMemoryInfo(GetCurrentProcess(), &info, sizeof(info));
    return (size_t)info.WorkingSetSize;
#elif defined(__APPLE__) && defined(__MACH__)
    struct mach_task_basic_info info;
    mach_msg_type_number_t infoCount = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, 
        (task_info_t)&info, &infoCount) != KERN_SUCCESS)
        return (size_t)0L;
    return (size_t)info.resident_size;
#else    
    long rss = 0L;
    FILE* fp = fopen("/proc/self/statm", "r");
    if (fp == NULL) return (size_t)0L;
    if (fscanf(fp, "%*s%ld", &rss) != 1) {
        fclose(fp);
        return (size_t)0L;
    }
    fclose(fp);
    return (size_t)rss * (size_t)sysconf(_SC_PAGESIZE);
#endif
}

[[nodiscard]] static inline
bool isMemoryUsageCritical() {
    const size_t rss = getCurrentRSS();
    const size_t totalSystemMemory = []{
        #if defined(_SC_PHYS_PAGES) && defined(_SC_PAGESIZE)
        return static_cast<size_t>(sysconf(_SC_PHYS_PAGES)) * 
               static_cast<size_t>(sysconf(_SC_PAGESIZE));
        #else
        return static_cast<size_t>(4ULL * 1024 * 1024 * 1024); // Assume 4GB
        #endif
    }();
    
    // More aggressive: use only 50% of system memory
    return (rss > (totalSystemMemory * 0.50));
}

static inline
void forceMemoryCleanup() {
    // Force garbage collection and memory cleanup
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    // Try to trigger OS memory cleanup (Linux)
#ifdef __linux__
    if (system("echo 1 > /proc/sys/vm/drop_caches 2>/dev/null") != 0) {
        // Fallback: just sleep longer
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
#else
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
#endif
}

//------------------------------------------------------
// CPU throttling class
//------------------------------------------------------
class CpuThrottler {
public:
    explicit CpuThrottler(int cpuLimitPercent) 
        : cpuLimitPercent_(std::clamp(cpuLimitPercent, 10, 100)),
          interval_(std::chrono::milliseconds(100)) {
        sleepTime_ = interval_ * (100 - cpuLimitPercent_) / 100;
    }
    
    void throttle() {
        if (cpuLimitPercent_ < 100) {
            std::this_thread::sleep_for(sleepTime_);
        }
        
        // Additional memory-based throttling
        if (isMemoryUsageCritical()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
    }
    
    void resetTimer() {
        lastReport_ = std::chrono::steady_clock::now();
    }
    
    bool shouldReportProgress(int frameCount, int totalFrames) {
        auto now = std::chrono::steady_clock::now();
        bool timeToReport = (now - lastReport_) >= std::chrono::seconds(5);  // Less frequent reporting
        bool frameProgress = (frameCount % std::max(1, totalFrames/20)) == 0;  // Every 5%
        
        if (timeToReport || frameProgress) {
            lastReport_ = now;
            return true;
        }
        return false;
    }

private:
    int cpuLimitPercent_;
    std::chrono::milliseconds interval_;
    std::chrono::milliseconds sleepTime_;
    std::chrono::time_point<std::chrono::steady_clock> lastReport_ = std::chrono::steady_clock::now();
};

//------------------------------------------------------
// Parameters structure
//------------------------------------------------------
struct Parameters {
    std::string inputFile = "input.mp4";
    std::string outputFile = "output.mp4";
    std::string watermarkString = "Visit: https://www.erarnitox.de/pub/thanks/ to support me!";
    int silentSpeed = 4;
    int silentThreshold = 20;
    float minSilenceDuration = 0.2F;
    float maxSilenceDuration = 2.0F;
    int visualDiffThreshold = 1;
    int minInhaleDurationMs = 60;
    int maxInhaleDurationMs = 600;
    int inhaleLowThreshold = 40;
    int inhaleHighThreshold = 300;
    bool debugMode = false;
    int cpuLimit = 60;  // Lower default CPU limit
    bool useProgressiveProcessing = true;
};

//------------------------------------------------------
// Utility functions
//------------------------------------------------------
[[nodiscard]] static inline
std::string generateTempFilename(const std::string& extension) {
    static const auto tempDir = std::filesystem::current_path();
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 1000000000);
    return (tempDir / (std::to_string(dis(gen)) + extension)).string();
}

[[nodiscard]] static inline
int getMaxVolume(const std::vector<short>& samples) {
    if (samples.empty()) return 0;
    return *std::max_element(samples.begin(), samples.end(),
        [](short a, short b) { return std::abs(a) < std::abs(b); });
}

[[nodiscard]] static inline
double getRMSVolume(const std::vector<short>& samples) {
    if (samples.empty()) return 0;
    double sumSquares = std::accumulate(samples.begin(), samples.end(), 0.0,
        [](double sum, short s) { return sum + static_cast<double>(s) * s; });
    return std::sqrt(sumSquares / samples.size());
}

static inline
void normalizeAudio(std::vector<short>& audio) {
    const int maxVol = getMaxVolume(audio);
    if (maxVol > 0) {
        const double scale = 32767.0 / maxVol;
        for (auto& sample : audio) {
            sample = static_cast<short>(std::clamp(static_cast<int>(std::round(sample * scale)), -32767, 32767));
        }
    }
}

[[nodiscard]] static inline
double computeFrameDifference(const cv::Mat& frame1, const cv::Mat& frame2) {
    // Even smaller ROI to reduce computation
    const int roiWidth = frame1.cols / 8;
    const int roiHeight = frame1.rows / 8;
    const cv::Rect roi(
        (frame1.cols - roiWidth) / 2,
        (frame1.rows - roiHeight) / 2,
        roiWidth,
        roiHeight
    );
    
    cv::Mat diff;
    cv::absdiff(frame1(roi), frame2(roi), diff);
    const cv::Scalar avgDiff = cv::mean(diff);
    return (avgDiff[0] + avgDiff[1] + avgDiff[2]) / 3.0;
}

[[nodiscard]] static inline
std::vector<short> resampleAudio(const std::vector<short>& audio, const size_t targetSamples) {
    if (audio.empty() || targetSamples == 0) return {};
    
    // Limit resampling size to prevent memory explosion
    const size_t maxResampleSize = g_maxAudioChunkSamples;
    const size_t actualTargetSamples = std::min(targetSamples, maxResampleSize);
    
    std::vector<short> result(actualTargetSamples);
    const size_t originalSize = audio.size();
    for (size_t i = 0; i < actualTargetSamples; ++i) {
        const size_t idx = static_cast<size_t>(std::floor(static_cast<double>(i) * originalSize / actualTargetSamples));
        result[i] = audio[std::min(idx, originalSize - 1)];
    }
    return result;
}

[[nodiscard]] static inline
bool isInInhale(const double time, const std::vector<std::pair<double, double>>& inhaleTimes) {
    return std::any_of(inhaleTimes.begin(), inhaleTimes.end(),
        [time](const auto& interval) {
            return time >= interval.first && time <= interval.second;
        });
}

//------------------------------------------------------
// Memory-optimized audio processor
//------------------------------------------------------
class AudioProcessor {
public:
    explicit AudioProcessor(const Parameters& params) : params_(params) {}
    
    void extractAudio(const std::string& videoFile, const std::string& audioFile) const {
        const auto cmd = std::format("ffmpeg -i \"{}\" -ab 160k -ac 2 -ar 44100 -vn \"{}\" > /dev/null 2>&1", videoFile, audioFile);
        const int result = std::system(cmd.c_str());
        if (result != 0) {
             throw std::runtime_error("Failed to extract audio using ffmpeg");
        }
    }
    
    void readAudioInfo(const std::string& audioFile) {
        SNDFILE* sndFile = sf_open(audioFile.c_str(), SFM_READ, &sfinfo_);
        if (not sndFile)
            throw std::runtime_error("Unable to open audio file: " + audioFile);
        sf_close(sndFile);
        
        // Don't load the entire audio into memory - just get the info
        calculateDynamicThresholds(audioFile);
    }
    
    void calculateDynamicThresholds(const std::string& audioFile) {
        SNDFILE* sndFile = sf_open(audioFile.c_str(), SFM_READ, &sfinfo_);
        if (!sndFile) return;
        
        // Sample only 5 seconds to reduce memory usage
        const int sampleDuration = 5 * sfinfo_.samplerate * sfinfo_.channels;
        std::vector<short> sampleData(sampleDuration);
        
        const sf_count_t read_count = sf_read_short(sndFile, sampleData.data(), sampleData.size());
        sf_close(sndFile);
        
        if (read_count > 0) {
            sampleData.resize(static_cast<size_t>(read_count));
            
            std::vector<int> volumes;
            const int windowSize = sfinfo_.samplerate / 10 * sfinfo_.channels; // 100ms window
            for (size_t i = 0; i + windowSize < sampleData.size(); i += windowSize) {
                const std::vector<short> window(sampleData.begin() + i, sampleData.begin() + i + windowSize);
                volumes.push_back(getMaxVolume(window));
            }
            
            if (!volumes.empty()) {
                std::sort(volumes.begin(), volumes.end());
                const size_t silentIndex = static_cast<size_t>(volumes.size() * 0.15);
                dynamicSilentThreshold_ = static_cast<int>(volumes[silentIndex] * 1.5);
                dynamicSilentThreshold_ = std::max(15, std::min(dynamicSilentThreshold_, params_.silentThreshold));
                std::println("Dynamic silence threshold: {}", dynamicSilentThreshold_);
            }
        }
    }
    
    // Process inhale detection in streaming fashion to avoid memory buildup
    std::vector<std::pair<int, int>> detectInhales(const std::string& audioFile) const {
        SNDFILE* sndFile = sf_open(audioFile.c_str(), SFM_READ, const_cast<SF_INFO*>(&sfinfo_));
        if (!sndFile) throw std::runtime_error("Unable to open audio file for inhale detection");
        
        std::vector<std::pair<int, int>> inhales;
        const int windowSizeMs = g_sampleLengthMs;
        const int windowSizeSamples = (sfinfo_.samplerate * windowSizeMs / 1000) * sfinfo_.channels;
        
        // Process in small chunks to avoid memory buildup
        const size_t chunkSize = 44100 * sfinfo_.channels;  // 1 second chunks
        std::vector<short> chunk(chunkSize);
        size_t totalSamplesRead = 0;
        
        while (true) {
            const sf_count_t read_count = sf_read_short(sndFile, chunk.data(), chunkSize);
            if (read_count <= 0) break;
            
            chunk.resize(static_cast<size_t>(read_count));
            
            // Process this chunk for inhales
            for (size_t i = 0; i + windowSizeSamples <= chunk.size(); i += windowSizeSamples / 4) {
                const std::vector<short> window(chunk.begin() + i, chunk.begin() + i + windowSizeSamples);
                const int rms = getMaxVolume(window);
                
                if (rms >= params_.inhaleLowThreshold && rms <= params_.inhaleHighThreshold) {
                    const int globalStart = static_cast<int>(totalSamplesRead + i);
                    const int globalEnd = static_cast<int>(totalSamplesRead + i + windowSizeSamples);
                    
                    if (inhales.empty() || globalStart > inhales.back().second + sfinfo_.channels) {
                        inhales.emplace_back(globalStart, globalEnd);
                    } else {
                        inhales.back().second = globalEnd;
                    }
                }
            }
            
            totalSamplesRead += static_cast<size_t>(read_count);
            chunk.resize(chunkSize);  // Reset for next iteration
        }
        
        sf_close(sndFile);
        return inhales;
    }
    
    void removeInhaleSegmentsToFile(const std::string& inputAudioFile, const std::string& outputAudioFile,
                                   const std::vector<std::pair<int, int>>& inhales) const {
        SNDFILE* inFile = sf_open(inputAudioFile.c_str(), SFM_READ, const_cast<SF_INFO*>(&sfinfo_));
        if (!inFile) throw std::runtime_error("Unable to open input audio file");
        
        SF_INFO outSfinfo = sfinfo_;
        SNDFILE* outFile = sf_open(outputAudioFile.c_str(), SFM_WRITE, &outSfinfo);
        if (!outFile) {
            sf_close(inFile);
            throw std::runtime_error("Unable to create output audio file");
        }
        
        // Process in small chunks, skipping inhale segments
        const size_t chunkSize = 44100 * sfinfo_.channels;  // 1 second chunks
        std::vector<short> chunk(chunkSize);
        size_t currentPos = 0;
        size_t inhaleIndex = 0;
        
        while (true) {
            const sf_count_t read_count = sf_read_short(inFile, chunk.data(), chunkSize);
            if (read_count <= 0) break;
            
            // Process this chunk, skipping inhale segments
            size_t chunkWriteStart = 0;
            const size_t chunkEnd = currentPos + static_cast<size_t>(read_count);
            
            // Check for inhale segments that overlap with this chunk
            while (inhaleIndex < inhales.size() && static_cast<size_t>(inhales[inhaleIndex].first) < chunkEnd) {
                const auto& inhale = inhales[inhaleIndex];
                const size_t inhaleStart = static_cast<size_t>(std::max(0, inhale.first));
                const size_t inhaleEnd = static_cast<size_t>(inhale.second);
                
                // Write data before this inhale
                if (inhaleStart > currentPos && inhaleStart > currentPos + chunkWriteStart) {
                    const size_t writeEnd = std::min(inhaleStart - currentPos, static_cast<size_t>(read_count));
                    const size_t samplesToWrite = writeEnd - chunkWriteStart;
                    
                    if (samplesToWrite > 0) {
                        sf_write_short(outFile, chunk.data() + chunkWriteStart, static_cast<sf_count_t>(samplesToWrite));
                    }
                }
                
                // Skip the inhale segment
                if (inhaleEnd > currentPos) {
                    chunkWriteStart = std::min(inhaleEnd - currentPos, static_cast<size_t>(read_count));
                }
                
                inhaleIndex++;
            }
            
            // Write remaining data in this chunk
            if (chunkWriteStart < static_cast<size_t>(read_count)) {
                const size_t samplesToWrite = static_cast<size_t>(read_count) - chunkWriteStart;
                sf_write_short(outFile, chunk.data() + chunkWriteStart, static_cast<sf_count_t>(samplesToWrite));
            }
            
            currentPos += static_cast<size_t>(read_count);
        }
        
        sf_close(inFile);
        sf_close(outFile);
    }
    
    static std::vector<short> readAudioSegment(const std::string& audioFile, size_t startSample, size_t numSamples) {
        SF_INFO sfinfo = {};
        SNDFILE* sndFile = sf_open(audioFile.c_str(), SFM_READ, &sfinfo);
        if (!sndFile) {
            return {};
        }
        
        const size_t totalSamples = static_cast<size_t>(sfinfo.frames) * static_cast<size_t>(sfinfo.channels);
        
        if (startSample >= totalSamples) {
            sf_close(sndFile);
            return {};
        }
        
        const size_t adjustedNumSamples = std::min(numSamples, totalSamples - startSample);
        const size_t startFrame = startSample / static_cast<size_t>(sfinfo.channels);
        
        if (startFrame >= static_cast<size_t>(sfinfo.frames)) {
            sf_close(sndFile);
            return {};
        }
        
        const sf_count_t seekResult = sf_seek(sndFile, static_cast<sf_count_t>(startFrame), SEEK_SET);
        if (seekResult < 0 || seekResult != static_cast<sf_count_t>(startFrame)) {
            sf_close(sndFile);
            return {};
        }
        
        std::vector<short> segment(adjustedNumSamples);
        const sf_count_t read_count = sf_read_short(sndFile, segment.data(), static_cast<sf_count_t>(adjustedNumSamples));
        sf_close(sndFile);
        
        if (read_count > 0) {
            segment.resize(static_cast<size_t>(read_count));
        } else {
            segment.clear();
        }
        
        return segment;
    }
    
    static void normalize(std::vector<short>& audio) {
        normalizeAudio(audio);
    }
    
    void writeAudio(const std::string& audioFile, const std::vector<short>& audio) const {
        SF_INFO outSfinfo = sfinfo_;
        outSfinfo.frames = static_cast<sf_count_t>(audio.size() / static_cast<size_t>(outSfinfo.channels));
        SNDFILE* outFile = sf_open(audioFile.c_str(), SFM_WRITE, &outSfinfo);
        if (!outFile) throw std::runtime_error("Unable to write audio file: " + audioFile);
        const sf_count_t written_count = sf_write_short(outFile, audio.data(), audio.size());
        if (written_count != static_cast<sf_count_t>(audio.size())) {
             sf_close(outFile);
             throw std::runtime_error("Failed to write all audio data");
        }
        sf_close(outFile);
    }
    
    [[nodiscard]] int getSampleRate() const { return sfinfo_.samplerate; }
    [[nodiscard]] int getChannels() const { return sfinfo_.channels; }
    [[nodiscard]] int getDynamicSilentThreshold() const { return dynamicSilentThreshold_; }

private:
    const Parameters params_;
    SF_INFO sfinfo_{};
    int dynamicSilentThreshold_ = 0;
};

//------------------------------------------------------
// Ultra-memory-optimized video processor
//------------------------------------------------------
class VideoProcessor {
public:
    VideoProcessor(const Parameters& params, const std::string& inputFile, const std::string& videoOutputFile,
                   const std::string& processedAudioFile, const std::vector<std::pair<double, double>>& inhaleTimes,
                   const int sampleRate, const int audioChannels, const int dynamicSilentThreshold)
        : params_(params), inputFile_(inputFile), videoOutputFile_(videoOutputFile),
          processedAudioFile_(processedAudioFile), inhaleTimes_(inhaleTimes), 
          sampleRate_(sampleRate), audioChannels_(audioChannels),
          silentThreshold_(dynamicSilentThreshold > 0 ? dynamicSilentThreshold : params.silentThreshold),
          cpuThrottler_(params.cpuLimit) {
          
        // Get basic video info without keeping capture open
        cv::VideoCapture tempCapture(inputFile_);
        if (!tempCapture.isOpened()) {
            throw std::runtime_error("Cannot open video file: " + inputFile_);
        }
        
        fps_ = tempCapture.get(cv::CAP_PROP_FPS);
        totalFrames_ = static_cast<int>(tempCapture.get(cv::CAP_PROP_FRAME_COUNT));
        videoWidth_ = static_cast<int>(tempCapture.get(cv::CAP_PROP_FRAME_WIDTH));
        videoHeight_ = static_cast<int>(tempCapture.get(cv::CAP_PROP_FRAME_HEIGHT));
        
        if (fps_ <= 0.0) throw std::runtime_error("Invalid FPS detected in video");
        
        numSamplesPerFrame_ = static_cast<int>(std::round(static_cast<double>(sampleRate) / fps_)) * audioChannels;
        chunkSize_ = std::min(g_maxChunkSize, static_cast<size_t>(totalFrames_ / 50));  // Very small chunks
        
        tempCapture.release();
        
        // Get processed audio info
        SF_INFO sfinfo = {};
        SNDFILE* sndFile = sf_open(processedAudioFile_.c_str(), SFM_READ, &sfinfo);
        if (sndFile) {
            totalProcessedAudioSamples_ = static_cast<size_t>(sfinfo.frames) * static_cast<size_t>(sfinfo.channels);
            sf_close(sndFile);
        }
        
        std::println("Video: {} frames, {:.2f} fps, {}x{}", totalFrames_, fps_, videoWidth_, videoHeight_);
        std::println("Processed audio: {} samples", totalProcessedAudioSamples_);
        std::println("Chunk size: {} frames", chunkSize_);
    }
    
    void process() {
        // Process video in ultra-small chunks
        size_t currentAudioPosition = 0;
        int processedFrames = 0;
        
        // Create output video writer once
        const int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
        cv::VideoWriter videoOutput(videoOutputFile_, fourcc, fps_, cv::Size(videoWidth_, videoHeight_));
        if (!videoOutput.isOpened()) {
            throw std::runtime_error("Cannot create video output file");
        }
        
        for (int chunkStart = 0; chunkStart < totalFrames_; chunkStart += static_cast<int>(chunkSize_)) {
            const int chunkEnd = std::min(chunkStart + static_cast<int>(chunkSize_), totalFrames_);
            
            std::println("Processing chunk {}-{}/{} (Memory: {:.1f} MB)", 
                        chunkStart, chunkEnd, totalFrames_, 
                        static_cast<double>(getCurrentRSS()) / (1024.0 * 1024.0));
            
            // Process this chunk with aggressive memory management
            processChunk(chunkStart, chunkEnd, videoOutput, currentAudioPosition);
            
            // Force memory cleanup after each chunk
            if (isMemoryUsageCritical()) {
                std::println("Critical memory usage detected - forcing cleanup");
                forceMemoryCleanup();
            }
            
            processedFrames = chunkEnd;
        }
        
        videoOutput.release();
        
        // Finalize audio
        finalizeAudio();
        
        std::println("Processing complete. Processed {} frames", processedFrames);
    }
    
    const std::string& getFinalAudioFilePath() const { return finalAudioFilePath_; }

private:
    void processChunk(int chunkStart, int chunkEnd, cv::VideoWriter& videoOutput, size_t& currentAudioPosition) {
        // Open video capture just for this chunk
        cv::VideoCapture capture(inputFile_);
        if (!capture.isOpened()) {
            throw std::runtime_error("Cannot open video file for chunk");
        }
        
        capture.set(cv::CAP_PROP_POS_FRAMES, chunkStart);
        
        // Very small buffers to minimize memory usage
        std::vector<cv::Mat> silentFrames;
        silentFrames.reserve(g_maxSilentFramesBuffer);
        std::vector<short> chunkAudio;
        chunkAudio.reserve(g_maxAudioChunkSamples);
        
        std::optional<cv::Mat> silentBaseline;
        int silentStartAudioPos = -1;
        
        for (int frameIndex = chunkStart; frameIndex < chunkEnd; ++frameIndex) {
            // Memory check every few frames
            if ((frameIndex % 10) == 0 && isMemoryUsageCritical()) {
                std::println("Memory critical at frame {}, forcing cleanup", frameIndex);
                
                // Process any accumulated silent frames immediately
                if (!silentFrames.empty()) {
                    processAccumulatedSilentFrames(silentFrames, silentStartAudioPos, 
                                                 currentAudioPosition, videoOutput, chunkAudio);
                    silentFrames.clear();
                    silentBaseline.reset();
                }
                
                // Write accumulated audio immediately
                if (chunkAudio.size() > g_maxAudioChunkSamples / 2) {
                    writeChunkAudio(frameIndex, chunkAudio);
                    chunkAudio.clear();
                    std::vector<short>().swap(chunkAudio);
                    chunkAudio.reserve(g_maxAudioChunkSamples);
                }
                
                forceMemoryCleanup();
            }
            
            cv::Mat frame;
            if (!capture.read(frame)) {
                std::println("Warning: Could not read frame {}", frameIndex);
                break;
            }
            
            const double frameTime = static_cast<double>(frameIndex) / fps_;
            
            // Progress reporting
            if (cpuThrottler_.shouldReportProgress(frameIndex, totalFrames_)) {
                std::println("Frame {}/{} ({:.1f}%) - Audio pos: {} - Mem: {:.1f}MB", 
                            frameIndex, totalFrames_, 
                            (frameIndex * 100.0) / totalFrames_,
                            currentAudioPosition,
                            static_cast<double>(getCurrentRSS()) / (1024.0 * 1024.0));
            }
            
            // Skip inhale frames
            if (isInInhale(frameTime, inhaleTimes_)) {
                cpuThrottler_.throttle();
                continue;
            }
            
            // Get audio for this frame
            const size_t samplesForFrame = static_cast<size_t>(numSamplesPerFrame_);
            const size_t audioEndPos = std::min(currentAudioPosition + samplesForFrame, totalProcessedAudioSamples_);
            const size_t actualSamplesToRead = audioEndPos - currentAudioPosition;
            
            std::vector<short> audioSample;
            if (actualSamplesToRead > 0) {
                audioSample = AudioProcessor::readAudioSegment(processedAudioFile_, currentAudioPosition, actualSamplesToRead);
            }
            
            // Pad with silence if needed
            if (audioSample.size() < samplesForFrame) {
                audioSample.resize(samplesForFrame, 0);
            }
            
            currentAudioPosition += actualSamplesToRead;
            
            const bool isSilent = checkSilence(audioSample);
            
            // Add watermark or debug info
            if (params_.debugMode) {
                drawDebugInfo(frame, isSilent, frameTime);
            } else {
                drawWatermark(frame, frameTime);
            }
            
            if (isSilent) {
                // Accumulate silent frames, but limit buffer size
                if (silentFrames.empty()) {
                    silentStartAudioPos = static_cast<int>(currentAudioPosition - actualSamplesToRead);
                    silentBaseline = frame.clone();
                }
                
                silentFrames.push_back(std::move(frame));
                
                // Process immediately if buffer is full
                if (silentFrames.size() >= g_maxSilentFramesBuffer) {
                    processAccumulatedSilentFrames(silentFrames, silentStartAudioPos, 
                                                 currentAudioPosition, videoOutput, chunkAudio);
                    silentFrames.clear();
                    silentBaseline.reset();
                }
            } else {
                // Process any accumulated silent frames first
                if (!silentFrames.empty()) {
                    processAccumulatedSilentFrames(silentFrames, silentStartAudioPos, 
                                                 currentAudioPosition, videoOutput, chunkAudio);
                    silentFrames.clear();
                    silentBaseline.reset();
                }
                
                // Write regular frame
                videoOutput.write(frame);
                
                // Add audio, but check size limits
                if (chunkAudio.size() + audioSample.size() > g_maxAudioChunkSamples) {
                    writeChunkAudio(frameIndex, chunkAudio);
                    chunkAudio.clear();
                    std::vector<short>().swap(chunkAudio);
                    chunkAudio.reserve(g_maxAudioChunkSamples);
                }
                
                chunkAudio.insert(chunkAudio.end(), audioSample.begin(), audioSample.end());
                totalOutputFrames_++;
            }
            
            cpuThrottler_.throttle();
        }
        
        // Process any remaining silent frames
        if (!silentFrames.empty()) {
            processAccumulatedSilentFrames(silentFrames, silentStartAudioPos, 
                                         currentAudioPosition, videoOutput, chunkAudio);
        }
        
        // Write any remaining audio for this chunk
        if (!chunkAudio.empty()) {
            writeChunkAudio(chunkStart, chunkAudio);
        }
        
        capture.release();
        
        // Force cleanup at end of chunk
        std::vector<cv::Mat>().swap(silentFrames);
        std::vector<short>().swap(chunkAudio);
    }
    
    void processAccumulatedSilentFrames(const std::vector<cv::Mat>& silentFrames, int silentStartAudioPos,
                                      size_t currentAudioPosition, cv::VideoWriter& videoOutput, 
                                      std::vector<short>& chunkAudio) {
        if (silentFrames.empty()) return;
        
        const double silenceDuration = static_cast<double>(silentFrames.size()) / fps_;
        
        // Simple processing to minimize memory usage
        if (silenceDuration >= static_cast<double>(params_.minSilenceDuration) &&
            silenceDuration <= static_cast<double>(params_.maxSilenceDuration)) {
            
            // Speed up: write every Nth frame
            const size_t numOutputFrames = (silentFrames.size() + static_cast<size_t>(params_.silentSpeed) - 1) / 
                                         static_cast<size_t>(params_.silentSpeed);
            
            for (size_t i = 0; i < silentFrames.size(); i += static_cast<size_t>(params_.silentSpeed)) {
                videoOutput.write(silentFrames[i]);
                totalOutputFrames_++;
            }
            
            // Create resampled audio
            const size_t originalAudioLength = silentFrames.size() * static_cast<size_t>(numSamplesPerFrame_);
            const size_t newAudioLength = numOutputFrames * static_cast<size_t>(numSamplesPerFrame_);
            
            // Read original silent audio in small chunks to avoid memory explosion
            std::vector<short> resampledAudio;
            resampledAudio.reserve(newAudioLength);
            
            const size_t readChunkSize = 8192;  // Small read chunks
            for (size_t offset = 0; offset < originalAudioLength; offset += readChunkSize) {
                const size_t toRead = std::min(readChunkSize, originalAudioLength - offset);
                const size_t audioPos = static_cast<size_t>(silentStartAudioPos) + offset;
                
                if (audioPos < totalProcessedAudioSamples_) {
                    auto chunk = AudioProcessor::readAudioSegment(processedAudioFile_, audioPos, toRead);
                    if (!chunk.empty()) {
                        // Simple downsampling
                        const double ratio = static_cast<double>(params_.silentSpeed);
                        for (size_t i = 0; i < chunk.size(); i += static_cast<size_t>(ratio)) {
                            if (resampledAudio.size() < newAudioLength) {
                                resampledAudio.push_back(chunk[i]);
                            }
                        }
                    }
                }
            }
            
            // Pad or truncate to exact size needed
            resampledAudio.resize(newAudioLength, 0);
            
            // Add to chunk audio with size check
            if (chunkAudio.size() + resampledAudio.size() > g_maxAudioChunkSamples) {
                writeChunkAudio(static_cast<int>(totalOutputFrames_), chunkAudio);
                chunkAudio.clear();
                std::vector<short>().swap(chunkAudio);
                chunkAudio.reserve(g_maxAudioChunkSamples);
            }
            
            chunkAudio.insert(chunkAudio.end(), resampledAudio.begin(), resampledAudio.end());
            
        } else {
            // Keep normal speed
            for (const auto& frame : silentFrames) {
                videoOutput.write(frame);
                totalOutputFrames_++;
            }
            
            // Read audio in chunks
            const size_t totalAudioNeeded = silentFrames.size() * static_cast<size_t>(numSamplesPerFrame_);
            const size_t readChunkSize = 8192;
            
            for (size_t offset = 0; offset < totalAudioNeeded; offset += readChunkSize) {
                const size_t toRead = std::min(readChunkSize, totalAudioNeeded - offset);
                const size_t audioPos = static_cast<size_t>(silentStartAudioPos) + offset;
                
                if (audioPos < totalProcessedAudioSamples_) {
                    auto chunk = AudioProcessor::readAudioSegment(processedAudioFile_, audioPos, toRead);
                    if (chunk.empty()) {
                        chunk.resize(toRead, 0);  // Silence
                    }
                    
                    // Add to chunk audio with size check
                    if (chunkAudio.size() + chunk.size() > g_maxAudioChunkSamples) {
                        writeChunkAudio(static_cast<int>(totalOutputFrames_), chunkAudio);
                        chunkAudio.clear();
                        std::vector<short>().swap(chunkAudio);
                        chunkAudio.reserve(g_maxAudioChunkSamples);
                    }
                    
                    chunkAudio.insert(chunkAudio.end(), chunk.begin(), chunk.end());
                }
            }
        }
    }
    
    bool checkSilence(const std::vector<short>& audioSample) const {
        if (audioSample.empty()) return true;
        const int maxVol = getMaxVolume(audioSample);
        const double rmsVol = getRMSVolume(audioSample);
        return maxVol < silentThreshold_ && rmsVol < (static_cast<double>(silentThreshold_) * 0.7);
    }
    
    void drawDebugInfo(cv::Mat& frame, bool isSilent, double time) const {
        if (isSilent) {
            cv::rectangle(frame, cv::Point(0, 0), cv::Point(20, 20), cv::Scalar(0, 0, 255), -1);
        } else {
            cv::rectangle(frame, cv::Point(0, 0), cv::Point(20, 20), cv::Scalar(0, 255, 0), -1);
        }
        if (isInInhale(time, inhaleTimes_)) {
            cv::rectangle(frame, cv::Point(20, 0), cv::Point(40, 20), cv::Scalar(255, 0, 0), -1);
        }
        cv::putText(frame, std::format("{:.1f}s", time), cv::Point(50, 20), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1);
    }
    
    //------------------------------------------------------
    // Draw enhanced watermark with adaptive background and animation
    //------------------------------------------------------
    void drawWatermark(cv::Mat& frame, double time) const {
        // Define multiple watermark strings that will rotate
        const std::vector<std::string> watermarkStrings = {
            "Visit: https://www.erarnitox.de/pub/thanks/ to support me!",
            "Don't forget to Like and Subscribe and Stuff...",
            "Hack the Planet!",
            "Learn Game Hacking at GuidedHacking.com!",
            "Erarnitox.de"
        };

        static int lastIndex = 0;
        static int x = 0;
        static int y = 30;
        static double lastTime = time;
        
        // Calculate which string to use based on time (changes every 8 seconds)
        const int activeStringIndex = static_cast<int>(time / 60.0) % watermarkStrings.size();
        const std::string currentWatermark = watermarkStrings[activeStringIndex];
        
        if(lastIndex != activeStringIndex)
        {
            x = 0;
            y = 30;
            lastIndex = activeStringIndex;
            lastTime = time;
        }

        // Configure font properties
        const int fontFace = cv::FONT_HERSHEY_DUPLEX;
        const double fontScale = 0.65;
        const int thickness = 1;
        
        // Calculate text size with baseline
        int baseline = 0;
        cv::Size textSize = cv::getTextSize(currentWatermark, fontFace, fontScale, thickness, &baseline);
        
        // Add padding around text
        const int padding = 8;
        const int totalHeight = textSize.height + 2 * padding;
        const int totalWidth = textSize.width + 2 * padding;
        
        // Calculate position with animation
        
        // Animation logic - different effects based on time

        if (params_.debugMode) {
            // In debug mode, keep it in a fixed position to avoid interfering with debug info
            x = 50;
            y = frame.rows - 20;
        } else {
            // Main animation logic - scrolling from right to left
            const double animationSpeed = 0.5; // Pixels per second
            const double scrollPeriod = frame.cols + totalWidth * 10; // Total distance for full scroll
            const double positionOffset = fmod((time - lastTime) * animationSpeed, scrollPeriod);
            
            // Start off-screen to the right, scroll left
            x += static_cast<int>(positionOffset);
            
            // Add subtle vertical bounce effect
            // y += static_cast<int>(sin(time));
        }
        
        // Ensure position stays within frame boundaries
        x = std::max(5, std::min(x, frame.cols - totalWidth - 5));
        y = std::max(totalHeight + 5, std::min(y, frame.rows - 5));
        
        // Create semi-transparent overlay for better visibility
        cv::Rect backgroundRect(0, 0, 1920, 50);
        
        // Draw semi-transparent black background
        cv::Mat roi = frame(backgroundRect);
        cv::Mat overlay = roi.clone();
        cv::rectangle(overlay, cv::Point(0, 0), cv::Point(1920, 50), 
                    cv::Scalar(0, 0, 0), cv::FILLED);
        
        // Blend overlay with original (70% overlay, 30% original)
        double alpha = 1;
        cv::addWeighted(overlay, alpha, roi, 1 - alpha, 0, roi);
        
        // Draw the text with subtle glow effect
        cv::Point textPos(x + padding, y - padding);
        
        // Draw shadow for better readability
        cv::putText(frame, currentWatermark, textPos + cv::Point(1, 1), 
                fontFace, fontScale, cv::Scalar(0, 0, 0), thickness + 1);
        
        // Draw main text
        cv::putText(frame, currentWatermark, textPos, 
                fontFace, fontScale, cv::Scalar(255, 255, 255), thickness);
    }
    
    void writeChunkAudio(int chunkId, const std::vector<short>& chunkAudio) {
        if (chunkAudio.empty()) return;
        
        const std::string chunkAudioFile = generateTempFilename("_chunk_" + std::to_string(chunkId) + ".wav");
        chunkAudioFiles_.push_back(chunkAudioFile);
        
        SF_INFO sfinfo = {};
        sfinfo.samplerate = sampleRate_;
        sfinfo.channels = audioChannels_;
        sfinfo.format = SF_FORMAT_WAV | SF_FORMAT_PCM_16;
        sfinfo.frames = static_cast<sf_count_t>(chunkAudio.size() / static_cast<size_t>(sfinfo.channels));
        
        SNDFILE* outFile = sf_open(chunkAudioFile.c_str(), SFM_WRITE, &sfinfo);
        if (!outFile) {
            std::println("Warning: Could not write chunk audio file {}", chunkAudioFile);
            return;
        }
        
        sf_write_short(outFile, chunkAudio.data(), chunkAudio.size());
        sf_close(outFile);
    }
    
    void finalizeAudio() {
        if (chunkAudioFiles_.empty()) {
            std::println("Warning: No audio chunks to finalize");
            return;
        }
        
        finalAudioFilePath_ = generateTempFilename("_final_audio.wav");
        
        // Merge audio files using streaming approach
        SF_INFO outSfinfo = {};
        outSfinfo.samplerate = sampleRate_;
        outSfinfo.channels = audioChannels_;
        outSfinfo.format = SF_FORMAT_WAV | SF_FORMAT_PCM_16;
        
        SNDFILE* outFile = sf_open(finalAudioFilePath_.c_str(), SFM_WRITE, &outSfinfo);
        if (!outFile) {
            throw std::runtime_error("Cannot create final audio file");
        }
        
        // Merge chunks one by one to minimize memory usage
        for (const auto& chunkFile : chunkAudioFiles_) {
            SF_INFO chunkInfo = {};
            SNDFILE* chunkSndFile = sf_open(chunkFile.c_str(), SFM_READ, &chunkInfo);
            if (!chunkSndFile) continue;
            
            // Copy in small buffers
            const size_t bufferSize = 8192;
            std::vector<short> buffer(bufferSize);
            
            while (true) {
                const sf_count_t read_count = sf_read_short(chunkSndFile, buffer.data(), bufferSize);
                if (read_count <= 0) break;
                
                sf_write_short(outFile, buffer.data(), read_count);
            }
            
            sf_close(chunkSndFile);
            
            // Clean up chunk file immediately
            std::filesystem::remove(chunkFile);
        }
        
        sf_close(outFile);
        chunkAudioFiles_.clear();
        
        std::println("Final audio created with {} output frames", totalOutputFrames_);
    }
    
    // Member variables
    const Parameters params_;
    const std::string inputFile_;
    const std::string videoOutputFile_;
    const std::string processedAudioFile_;
    const std::vector<std::pair<double, double>>& inhaleTimes_;
    const int sampleRate_, audioChannels_;
    const int silentThreshold_;
    CpuThrottler cpuThrottler_;
    
    double fps_;
    int totalFrames_;
    int videoWidth_, videoHeight_;
    int numSamplesPerFrame_;
    size_t chunkSize_;
    size_t totalProcessedAudioSamples_ = 0;
    int totalOutputFrames_ = 0;
    
    std::vector<std::string> chunkAudioFiles_;
    std::string finalAudioFilePath_;
};

//------------------------------------------------------
// Convert inhale intervals from samples to seconds
//------------------------------------------------------
static std::vector<std::pair<double, double>> convertInhaleIntervalsToSeconds(
    const std::vector<std::pair<int, int>>& intervals,
    const int sampleRate, const int channels) {
    std::vector<std::pair<double, double>> times;
    times.reserve(intervals.size());
    const double factor = 1.0 / (static_cast<double>(sampleRate) * static_cast<double>(channels));
    for (const auto& [start, end] : intervals) {
        times.emplace_back(static_cast<double>(start) * factor,
                           static_cast<double>(end) * factor);
    }
    return times;
}

//------------------------------------------------------
// Parse command line arguments
//------------------------------------------------------
Parameters parseCommandLine(int argc, char* argv[]) {
    Parameters params;
    if (argc > 1) params.inputFile = argv[1];
    if (argc > 2) params.outputFile = argv[2];
    
    for (int i = 3; i < argc; i += 2) {
        if (i + 1 >= argc) break;
        std::string paramName = argv[i];
        std::string paramValue = argv[i+1];
        
        if (paramName == "--silent-speed") params.silentSpeed = std::stoi(paramValue);
        else if (paramName == "--silent-threshold") params.silentThreshold = std::stoi(paramValue);
        else if (paramName == "--min-silence") params.minSilenceDuration = std::stof(paramValue);
        else if (paramName == "--max-silence") params.maxSilenceDuration = std::stof(paramValue);
        else if (paramName == "--visual-diff") params.visualDiffThreshold = std::stoi(paramValue);
        else if (paramName == "--min-inhale") params.minInhaleDurationMs = std::stoi(paramValue);
        else if (paramName == "--max-inhale") params.maxInhaleDurationMs = std::stoi(paramValue);
        else if (paramName == "--inhale-low") params.inhaleLowThreshold = std::stoi(paramValue);
        else if (paramName == "--inhale-high") params.inhaleHighThreshold = std::stoi(paramValue);
        else if (paramName == "--watermark") params.watermarkString = paramValue;
        else if (paramName == "--debug") params.debugMode = (paramValue == "true" || paramValue == "1");
        else if (paramName == "--cpu-limit") params.cpuLimit = std::clamp(std::stoi(paramValue), 10, 100);
        else if (paramName == "--progressive") params.useProgressiveProcessing = (paramValue == "true" || paramValue == "1");
    }
    return params;
}

//------------------------------------------------------
// MEMORY-OPTIMIZED MAIN FUNCTION
//------------------------------------------------------
int main(int argc, char* argv[]) {
    const Parameters params = (argc > 1) ? parseCommandLine(argc, argv) : Parameters();
    
    // Use unique temp filenames to avoid conflicts
    const std::string originalAudioFile = generateTempFilename("_orig.wav");
    const std::string processedAudioFile = generateTempFilename("_proc.wav");
    const std::string videoFile = generateTempFilename("_video.mp4");
    const std::string finalAudioFile = generateTempFilename("_final.wav");
    
    std::println("=== MEMORY-OPTIMIZED VIDEO PROCESSOR ===");
    std::println("Input: {}", params.inputFile);
    std::println("Output: {}", params.outputFile);
    std::println("CPU limit: {}%", params.cpuLimit);
    std::println("Initial memory: {:.1f} MB", static_cast<double>(getCurrentRSS()) / (1024.0 * 1024.0));
    
    try {
        std::println("\n1. Extracting audio...");
        AudioProcessor audioProcessor(params);
        audioProcessor.extractAudio(params.inputFile, originalAudioFile);
        audioProcessor.readAudioInfo(originalAudioFile);
        
        std::println("Memory after audio extraction: {:.1f} MB", 
                    static_cast<double>(getCurrentRSS()) / (1024.0 * 1024.0));
        
        std::println("\n2. Detecting inhales (streaming mode)...");
        const auto inhaleIntervals = audioProcessor.detectInhales(originalAudioFile);
        const auto inhaleTimes = convertInhaleIntervalsToSeconds(inhaleIntervals, 
                                                              audioProcessor.getSampleRate(),
                                                              audioProcessor.getChannels());
        
        std::println("Found {} inhale intervals", inhaleIntervals.size());
        std::println("Memory after inhale detection: {:.1f} MB", 
                    static_cast<double>(getCurrentRSS()) / (1024.0 * 1024.0));
        
        std::println("\n3. Removing inhales (streaming mode)...");
        audioProcessor.removeInhaleSegmentsToFile(originalAudioFile, processedAudioFile, inhaleIntervals);
        
        // Clean up original audio immediately
        std::filesystem::remove(originalAudioFile);
        
        std::println("Memory after inhale removal: {:.1f} MB", 
                    static_cast<double>(getCurrentRSS()) / (1024.0 * 1024.0));
        
        std::println("\n4. Processing video (ultra-low memory mode)...");
        VideoProcessor videoProcessor(params, params.inputFile, videoFile, processedAudioFile, inhaleTimes,
                                    audioProcessor.getSampleRate(), audioProcessor.getChannels(),
                                    audioProcessor.getDynamicSilentThreshold());
        
        videoProcessor.process();
        
        // Get final audio path
        const std::string& finalAudioPath = videoProcessor.getFinalAudioFilePath();
        
        std::println("Memory after video processing: {:.1f} MB", 
                    static_cast<double>(getCurrentRSS()) / (1024.0 * 1024.0));
        
        std::println("\n5. Normalizing final audio...");
        // Process audio normalization in chunks to avoid loading everything into memory
        {
            SF_INFO sfinfo = {};
            SNDFILE* inFile = sf_open(finalAudioPath.c_str(), SFM_READ, &sfinfo);
            if (!inFile) throw std::runtime_error("Cannot open final audio for normalization");
            
            // Create normalized output
            SNDFILE* outFile = sf_open(finalAudioFile.c_str(), SFM_WRITE, &sfinfo);
            if (!outFile) {
                sf_close(inFile);
                throw std::runtime_error("Cannot create normalized audio file");
            }
            
            // Process in chunks for normalization
            const size_t chunkSize = 44100 * sfinfo.channels * 60;  // 1min chunks
            std::vector<short> chunk(chunkSize);
            
            while (true) {
                const sf_count_t read_count = sf_read_short(inFile, chunk.data(), chunkSize);
                if (read_count <= 0) break;
                
                chunk.resize(static_cast<size_t>(read_count));
                AudioProcessor::normalize(chunk);
                
                sf_write_short(outFile, chunk.data(), read_count);
                chunk.resize(chunkSize);
            }
            
            sf_close(inFile);
            sf_close(outFile);
        }
        
        // Clean up intermediate files
        std::filesystem::remove(processedAudioFile);
        if (!finalAudioPath.empty() && std::filesystem::exists(finalAudioPath)) {
            std::filesystem::remove(finalAudioPath);
        }
        
        std::println("\n6. Merging final video and audio...");
        if (std::filesystem::exists(params.outputFile)) {
            std::filesystem::remove(params.outputFile);
        }
        
        const auto mergeCmd = std::format("ffmpeg -y -i \"{}\" -i \"{}\" -c:v copy -c:a aac \"{}\" > /dev/null 2>&1",
                                        videoFile, finalAudioFile, params.outputFile);
        const int mergeResult = std::system(mergeCmd.c_str());
        if (mergeResult != 0) {
            throw std::runtime_error("Failed to merge video and audio");
        }
        
        std::println("\n7. Cleaning up...");
        std::filesystem::remove(videoFile);
        std::filesystem::remove(finalAudioFile);
        
        std::println("\n=== PROCESSING COMPLETE ===");
        std::println("Output: {}", params.outputFile);
        std::println("Final memory: {:.1f} MB", static_cast<double>(getCurrentRSS()) / (1024.0 * 1024.0));
        
    } catch (const std::bad_alloc& e) {
        std::println(std::cerr, "Memory allocation failed - try with smaller video or more RAM");
        return 2;
    } catch (const std::exception& e) {
        std::println(std::cerr, "Error: {}", e.what());
        return 1;
    }
    
    return 0;
}