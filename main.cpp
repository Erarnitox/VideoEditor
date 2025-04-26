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
#include <future>
#include <sndfile.h>
#include <opencv2/opencv.hpp>
#include <numeric>
#include <format>

//------------------------------------------------------
// Configuration constants
//------------------------------------------------------
constexpr int g_sampleLengthMs = 200;  // Time window to analyze for inhale detection

//------------------------------------------------------
// Parameters structure for configuring video processing
//------------------------------------------------------
struct Parameters {
    std::string inputFile = "input.mp4";
    std::string outputFile = "output.mp4";
    std::string watermarkString = "Visit: https://www.erarnitox.de/pub/thanks/ to support me!";
    int silentSpeed = 4;               // More aggressive speed-up for silent segments
    int silentThreshold = 20;          // Lower threshold to catch more subtle silence
    float minSilenceDuration = 0.2;    // Shorter minimum to catch briefer pauses (seconds)
    float maxSilenceDuration = 2;      // Shorter minimum to catch briefer pauses (seconds)
    int visualDiffThreshold = 1;       // Increased to allow more segments to be sped up
    int minInhaleDurationMs = 60;      // Shorter min inhale duration (ms)
    int maxInhaleDurationMs = 600;     // Longer max inhale duration (ms)
    int inhaleLowThreshold = 40;       // For detecting softer inhales
    int inhaleHighThreshold = 300;     // For avoiding cutting louder noises
    bool debugMode = false;            // Enable visualization of silent/inhale segments
};

//------------------------------------------------------
// Generate a temporary filename
//------------------------------------------------------
[[nodiscard]] static inline 
std::string generateTempFilename(const std::string& extension) {
    static const auto tempDir = std::filesystem::current_path();
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 1000000000);
    return (tempDir / (std::to_string(dis(gen)) + extension)).string();
}

//------------------------------------------------------
// Get the maximum volume from an audio sample
//------------------------------------------------------
[[nodiscard]] static inline
int getMaxVolume(const std::vector<short>& samples) {
    if (samples.empty()) return 0;
    return *std::max_element(samples.begin(), samples.end(), 
        [](short a, short b) { return std::abs(a) < std::abs(b); });
}

//------------------------------------------------------
// Calculate the RMS volume of an audio sample
//------------------------------------------------------
[[nodiscard]] static inline
double getRMSVolume(const std::vector<short>& samples) {
    if (samples.empty()) return 0;
    double sumSquares = std::accumulate(samples.begin(), samples.end(), 0.0,
        [](double sum, short s) { return sum + static_cast<double>(s) * s; });
    return std::sqrt(sumSquares / samples.size());
}

//------------------------------------------------------
// Normalize audio to full dynamic range
//------------------------------------------------------
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

//------------------------------------------------------
// Compute difference between frames
//------------------------------------------------------
[[nodiscard]] static inline
double computeFrameDifference(const cv::Mat& frame1, const cv::Mat& frame2) {
    cv::Mat diff;
    cv::absdiff(frame1, frame2, diff);
    const cv::Scalar avgDiff = cv::mean(diff);
    return (avgDiff[0] + avgDiff[1] + avgDiff[2]) / 3.0;
}

//------------------------------------------------------
// Resample audio to target number of samples
//------------------------------------------------------
[[nodiscard]] static inline 
std::vector<short> resampleAudio(const std::vector<short>& audio, const size_t targetSamples) {
    if (audio.empty() || targetSamples == 0) return {};
    std::vector<short> result(targetSamples);
    const size_t originalSize = audio.size();
    for (size_t i = 0; i < targetSamples; ++i) {
        const size_t idx = static_cast<size_t>(std::floor(static_cast<double>(i) * originalSize / targetSamples));
        result[i] = audio[std::min(idx, originalSize - 1)];
    }
    return result;
}

//------------------------------------------------------
// Check if a time falls within inhale intervals
//------------------------------------------------------
[[nodiscard]] static inline 
bool isInInhale(const double time, const std::vector<std::pair<double, double>>& inhaleTimes) {
    return std::any_of(inhaleTimes.begin(), inhaleTimes.end(),
        [time](const auto& interval) { 
            return time >= interval.first && time <= interval.second; 
        });
}

//------------------------------------------------------
// Check if the audio pattern indicates end of sentence
//------------------------------------------------------
[[nodiscard]] static inline
bool isSentenceEnding(const std::vector<short>& audioWindow) {
    if (audioWindow.size() < 30) return false;
    
    // Calculate RMS energy trend
    const int windowSize = audioWindow.size() / 10;
    std::vector<double> energy;
    for (size_t i = 0; i < audioWindow.size() - windowSize; i += windowSize) {
        double sum = 0;
        for (size_t j = i; j < i + windowSize && j < audioWindow.size(); j++) {
            sum += audioWindow[j] * audioWindow[j];
        }
        energy.push_back(sqrt(sum / windowSize));
    }
    
    // Check for decreasing energy pattern (sentence ending)
    if (energy.size() >= 3) {
        if (energy[energy.size()-1] < energy[energy.size()-2] * 0.7 && 
            energy[energy.size()-2] < energy[energy.size()-3] * 0.7) {
            return true;
        }
    }
    return false;
}

//------------------------------------------------------
// Audio processing class
//------------------------------------------------------
class AudioProcessor {
public:
    //------------------------------------------------------
    // Constructor
    //------------------------------------------------------
    explicit AudioProcessor(const Parameters& params) : params_(params) {}

    //------------------------------------------------------
    // Extract audio from video file
    //------------------------------------------------------
    void extractAudio(const std::string& videoFile, const std::string& audioFile) const {
        const auto cmd = std::format("ffmpeg -i \"{}\" -ab 160k -ac 2 -ar 44100 -vn \"{}\" > /dev/null 2>&1", videoFile, audioFile);
        std::system(cmd.c_str());
    }

    //------------------------------------------------------
    // Read audio data from file
    //------------------------------------------------------
    void readAudio(const std::string& audioFile) {
        SNDFILE* sndFile = sf_open(audioFile.c_str(), SFM_READ, &sfinfo_);

        if (not sndFile) 
            throw std::runtime_error("Unable to open audio file: " + audioFile);

        audioData_.resize(sfinfo_.frames * sfinfo_.channels);
        sf_read_short(sndFile, audioData_.data(), audioData_.size());
        sf_close(sndFile);
        
        // Calculate dynamic threshold
        calculateDynamicThresholds();
    }
    
    //------------------------------------------------------
    // Calculate dynamic thresholds based on audio content
    //------------------------------------------------------
    void calculateDynamicThresholds() {
        if (audioData_.empty()) return;
        
        // Sample 10 seconds of audio or the entire file if shorter
        const int sampleDuration = 10 * sfinfo_.samplerate * sfinfo_.channels;
        const int samplesToAnalyze = std::min(sampleDuration, static_cast<int>(audioData_.size()));
        
        std::vector<int> volumes;
        const int windowSize = sfinfo_.samplerate / 10 * sfinfo_.channels; // 100ms window
        
        for (int i = 0; i < samplesToAnalyze; i += windowSize) {
            const int end = std::min(i + windowSize, samplesToAnalyze);
            const std::vector<short> window(audioData_.begin() + i, audioData_.begin() + end);
            volumes.push_back(getMaxVolume(window));
        }
        
        if (!volumes.empty()) {
            std::sort(volumes.begin(), volumes.end());
            
            // Use the 15th percentile for silent threshold
            const size_t silentIndex = static_cast<size_t>(volumes.size() * 0.15);
            dynamicSilentThreshold_ = volumes[silentIndex] * 1.5; // Add a small buffer
            
            // Clamp threshold to reasonable values
            dynamicSilentThreshold_ = std::max(15, std::min(dynamicSilentThreshold_, params_.silentThreshold));
            
            std::println("Dynamic silence threshold: {}", dynamicSilentThreshold_);
        }
    }

    //------------------------------------------------------
    // Detect inhale segments in audio
    //------------------------------------------------------
    std::vector<std::pair<int, int>> detectInhales() const {
        const int windowSizeMs = g_sampleLengthMs;
        const int windowSizeSamples = (sfinfo_.samplerate * windowSizeMs / 1000) * sfinfo_.channels;
        const int minInhaleSamples = (params_.minInhaleDurationMs * sfinfo_.samplerate / 1000) * sfinfo_.channels;
        const int maxInhaleSamples = (params_.maxInhaleDurationMs * sfinfo_.samplerate / 1000) * sfinfo_.channels;

        // Parallel processing
        const size_t numThreads = std::thread::hardware_concurrency();
        const size_t chunkSize = audioData_.size() / numThreads;
        std::vector<std::future<std::vector<std::pair<int, int>>>> futures;

        for (size_t i{ 0 }; i < numThreads; ++i) {
            const size_t start = i * chunkSize;
            const size_t end = (i == numThreads - 1) ? audioData_.size() : start + chunkSize + windowSizeSamples;
            const size_t boundedEnd = std::min(end, audioData_.size());
            futures.push_back(std::async(std::launch::async, [this, start, boundedEnd, windowSizeSamples, minInhaleSamples, maxInhaleSamples]() {
                std::vector<std::pair<int, int>> localInhales;
                for (size_t j{ start }; j + windowSizeSamples <= boundedEnd; j += sfinfo_.channels) {
                    const std::vector<short> window(audioData_.begin() + j, audioData_.begin() + j + windowSizeSamples);
                    const int rms = getMaxVolume(window);
                    if (rms >= params_.inhaleLowThreshold && rms <= params_.inhaleHighThreshold) {
                        if (localInhales.empty() || j > localInhales.back().second + sfinfo_.channels) {
                            localInhales.emplace_back(j, j + windowSizeSamples);
                        } else {
                            localInhales.back().second = j + windowSizeSamples;
                        }
                    }
                }
                // Filter by duration
                std::vector<std::pair<int, int>> filtered;
                for (const auto& interval : localInhales) {
                    const int duration = interval.second - interval.first;
                    if (duration >= minInhaleSamples && duration <= maxInhaleSamples) {
                        filtered.push_back(interval);
                    }
                }
                return filtered;
            }));
        }

        // Merge results
        std::vector<std::pair<int, int>> inhales;
        for (auto& f : futures) {
            auto chunkInhales = f.get();
            inhales.insert(inhales.end(), chunkInhales.begin(), chunkInhales.end());
        }
        std::sort(inhales.begin(), inhales.end());
        std::vector<std::pair<int, int>> merged;
        for (const auto& interval : inhales) {
            if (merged.empty() || merged.back().second + sfinfo_.channels < interval.first) {
                merged.push_back(interval);
            } else {
                merged.back().second = std::max(merged.back().second, interval.second);
            }
        }
        return merged;
    }
    
    //------------------------------------------------------
    // Check if audio window has characteristics of inhale
    //------------------------------------------------------
    [[nodiscard]] 
    bool hasHighFrequencyContent(const std::vector<short>& window) const {
        // Basic frequency analysis - not as good as FFT but simpler
        if (window.size() < 20) return false;
        
        // Count zero crossings as rough indicator of high frequency content
        int zeroCrossings = 0;
        for (size_t i = 1; i < window.size(); i++) {
            if ((window[i-1] >= 0 && window[i] < 0) || 
                (window[i-1] < 0 && window[i] >= 0)) {
                zeroCrossings++;
            }
        }
        
        // Higher zero crossing rate indicates higher frequencies
        const double zeroCrossingRate = static_cast<double>(zeroCrossings) / window.size();
        return zeroCrossingRate > 0.1; // Threshold determined empirically
    }

    //------------------------------------------------------
    // Remove inhale segments from audio
    //------------------------------------------------------
    std::vector<short> removeInhaleSegments(const std::vector<std::pair<int, int>>& inhales) const {
        std::vector<short> result;
        size_t pos = 0;
        for (const auto& [start, end] : inhales) {
            result.insert(result.end(), audioData_.begin() + pos, audioData_.begin() + start);
            pos = end;
        }
        result.insert(result.end(), audioData_.begin() + pos, audioData_.end());
        return result;
    }

    //------------------------------------------------------
    // Static method to normalize audio
    //------------------------------------------------------
    static void normalize(std::vector<short>& audio) {
        normalizeAudio(audio);
    }

    //------------------------------------------------------
    // Write audio data to file
    //------------------------------------------------------
    void writeAudio(const std::string& audioFile, const std::vector<short>& audio) const {
        SF_INFO outSfinfo = sfinfo_;
        outSfinfo.frames = audio.size() / outSfinfo.channels;
        SNDFILE* outFile = sf_open(audioFile.c_str(), SFM_WRITE, &outSfinfo);
        if (!outFile) throw std::runtime_error("Unable to write audio file: " + audioFile);
        sf_write_short(outFile, audio.data(), audio.size());
        sf_close(outFile);
    }

    //------------------------------------------------------
    // Getters
    //------------------------------------------------------
    [[nodiscard]] int getSampleRate() const { return sfinfo_.samplerate; }
    [[nodiscard]] int getChannels() const { return sfinfo_.channels; }
    [[nodiscard]] int getDynamicSilentThreshold() const { return dynamicSilentThreshold_; }

private:
    const Parameters params_;
    SF_INFO sfinfo_{};
    std::vector<short> audioData_;
    int dynamicSilentThreshold_ = 0;
};

//------------------------------------------------------
// Video processing class
//------------------------------------------------------
class VideoProcessor {
public:
    //------------------------------------------------------
    // Constructor
    //------------------------------------------------------
    VideoProcessor(const Parameters& params, cv::VideoCapture& capture, cv::VideoWriter& videoOutput,
                   const std::vector<short>& processedAudioData, const std::vector<std::pair<double, double>>& inhaleTimes,
                   const int sampleRate, const int audioChannels, const int dynamicSilentThreshold)
        : params_(params), capture_(capture), videoOutput_(videoOutput), processedAudioData_(processedAudioData),
          inhaleTimes_(inhaleTimes), sampleRate_(sampleRate), audioChannels_(audioChannels),
          fps_(capture.get(cv::CAP_PROP_FPS)),
          numSamplesPerFrame_(static_cast<int>(std::round(static_cast<double>(sampleRate) / fps_)) * audioChannels),
          silentThreshold_(dynamicSilentThreshold > 0 ? dynamicSilentThreshold : params.silentThreshold),
          totalOutputFrames_(0) {
              
          // Reserve space for lookahead buffer
          frameBuffer_.reserve(lookAheadFrames_);
          audioBuffer_.reserve(lookAheadFrames_ * numSamplesPerFrame_);
      }

    //------------------------------------------------------
    // Process the video
    //------------------------------------------------------
    void process() {
        int currentFrameIndex = 0;
        int currentAudioPos = 0;
        std::vector<cv::Mat> silentFrames;
        int silentStartPos = -1;
        cv::Mat silentBaseline;
        
        std::println("Starting video processing with silent threshold: {}", silentThreshold_);
        std::println("Frames will be processed at {} fps", fps_);

        while (true) {
            // Fill the lookahead buffer if needed
            fillLookaheadBuffer(currentFrameIndex);
            
            // Break if we've reached the end
            if (frameBuffer_.empty()) break;
            
            // Get current frame
            cv::Mat frame = frameBuffer_.front();
            frameBuffer_.erase(frameBuffer_.begin());
            
            const double origTime = currentFrameIndex / fps_;
            currentFrameIndex++;

            // Skip inhale frames
            if (isInInhale(origTime, inhaleTimes_)) {
                continue;
            }

            if (currentAudioPos >= static_cast<int>(processedAudioData_.size())) {
                break;
            }

            const int audioEndPos = std::min(currentAudioPos + numSamplesPerFrame_, static_cast<int>(processedAudioData_.size()));
            std::vector<short> audioSample(processedAudioData_.begin() + currentAudioPos,
                                          processedAudioData_.begin() + audioEndPos);
            
            // Check if this segment is silent
            bool isSilent = checkSilence(audioSample);
            
            // Add debug visualization if enabled
            if (params_.debugMode) {
                drawDebugInfo(frame, isSilent, origTime);
            } else {
                drawWatermark(frame);
            }

            if (isSilent) {
                // Start tracking silent segment
                if (silentFrames.empty()) {
                    silentStartPos = currentAudioPos;
                    silentBaseline = frame.clone();
                }
                silentFrames.push_back(frame.clone());
                currentAudioPos += numSamplesPerFrame_;
            } else {
                // Process any accumulated silent segment
                if (!silentFrames.empty()) {
                    processSilentSegment(silentFrames, silentStartPos, currentAudioPos, silentBaseline);
                    silentFrames.clear();
                    silentStartPos = -1;
                }
                
                // Write regular frame
                videoOutput_.write(frame);
                modifiedAudio_.insert(modifiedAudio_.end(), audioSample.begin(), audioSample.end());
                currentAudioPos += numSamplesPerFrame_;
                totalOutputFrames_++;
            }
        }

        // Handle any remaining silent frames
        if (!silentFrames.empty()) {
            processSilentSegment(silentFrames, silentStartPos, currentAudioPos, silentBaseline);
        }

        // Print synchronization diagnostics
        const double expectedAudioSamples = totalOutputFrames_ * numSamplesPerFrame_;
        std::println("Total output frames: {}", totalOutputFrames_);
        std::println("Expected audio samples: {}", expectedAudioSamples);
        std::println("Actual audio samples: {}", modifiedAudio_.size());
        
        // Adjust audio length if needed to maintain sync
        if (std::abs(expectedAudioSamples - modifiedAudio_.size()) > numSamplesPerFrame_) {
            std::println("Adjusting audio length for proper synchronization");
            modifiedAudio_ = resampleAudio(modifiedAudio_, static_cast<size_t>(expectedAudioSamples));
        }
    }

    //------------------------------------------------------
    // Fill the lookahead buffer with frames
    //------------------------------------------------------
    void fillLookaheadBuffer(int currentFrameIndex) {
        while (frameBuffer_.size() < lookAheadFrames_) {
            cv::Mat frame;
            if (!capture_.read(frame)) {
                break;
            }
            frameBuffer_.push_back(frame);
        }
    }
    
    //------------------------------------------------------
    // Check if audio sample is silent
    //------------------------------------------------------
    bool checkSilence(const std::vector<short>& audioSample) const {
        if (audioSample.empty()) return true;
        
        // Use both max volume and RMS for better silence detection
        const int maxVol = getMaxVolume(audioSample);
        const double rmsVol = getRMSVolume(audioSample);
        
        return maxVol < silentThreshold_ && rmsVol < (silentThreshold_ * 0.7);
    }
    
    //------------------------------------------------------
    // Check for extended silence over multiple frames
    //------------------------------------------------------
    bool checkForExtendedSilence(int currentPos, int windowSize) const {
        if (currentPos + windowSize * numSamplesPerFrame_ >= processedAudioData_.size()) {
            return false;
        }
        
        int silentFrameCount = 0;
        for (int i = 0; i < windowSize; i++) {
            const int startPos = currentPos + i * numSamplesPerFrame_;
            const int endPos = std::min(startPos + numSamplesPerFrame_, static_cast<int>(processedAudioData_.size()));
            
            if (startPos < endPos) {
                std::vector<short> sample(processedAudioData_.begin() + startPos, processedAudioData_.begin() + endPos);
                if (checkSilence(sample)) {
                    silentFrameCount++;
                }
            }
        }
        
        // Return true if majority of frames in window are silent
        return silentFrameCount > windowSize * 0.7; 
    }
    
    //------------------------------------------------------
    // Draw debug information on frame
    //------------------------------------------------------
    void drawDebugInfo(cv::Mat& frame, bool isSilent, double time) const {
        // Draw colored boxes to indicate frame status
        if (isSilent) {
            // Red box for silent frame
            cv::rectangle(frame, cv::Point(0, 0), cv::Point(20, 20), cv::Scalar(0, 0, 255), -1);
        } else {
            // Green box for speech frame
            cv::rectangle(frame, cv::Point(0, 0), cv::Point(20, 20), cv::Scalar(0, 255, 0), -1);
        }
        
        // Blue box for inhale frame
        if (isInInhale(time, inhaleTimes_)) {
            cv::rectangle(frame, cv::Point(20, 0), cv::Point(40, 20), cv::Scalar(255, 0, 0), -1);
        }
        
        // Add timestamp
        cv::putText(frame, std::format("{:.2f}s", time), 
                   cv::Point(50, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, 
                   cv::Scalar(255, 255, 255), 1);
    }

    //------------------------------------------------------
    // Draw debug information on frame
    //------------------------------------------------------
    void drawWatermark(cv::Mat& frame) const {
        cv::putText(frame, params_.watermarkString, 
                   cv::Point(50, 20), cv::FONT_HERSHEY_DUPLEX, 0.7, 
                   cv::Scalar(255, 255, 255), 1);
    }

    //------------------------------------------------------
    // Get the modified audio
    //------------------------------------------------------
    [[nodiscard]]
    const std::vector<short>& getModifiedAudio() const {
        return modifiedAudio_; 
    }

private:
    //------------------------------------------------------
    // Process a segment of silent frames
    //------------------------------------------------------
    void processSilentSegment(const std::vector<cv::Mat>& silentFrames, const int silentStartPos, const int silentEndPos,
                              const cv::Mat& silentBaseline) {
        if (silentFrames.empty()) return;
        
        const double silenceDuration = static_cast<double>(silentFrames.size()) / fps_;
        
        // Calculate visual difference throughout silent segment
        double sumDiff = 0.0;
        for (const auto& f : silentFrames) {
            sumDiff += computeFrameDifference(silentBaseline, f);
        }
        const double avgVisualDiff = sumDiff / silentFrames.size();

        // Check end of segment to avoid cutting sentences
        bool mightCutSentence = false;

        /*
        if (silentEndPos < static_cast<int>(processedAudioData_.size())) {
            // Look ahead to see if speech resumes strongly
            const int lookAheadSamples = std::min(numSamplesPerFrame_ * 3, 
                                                 static_cast<int>(processedAudioData_.size()) - silentEndPos);
            if (lookAheadSamples > 0) {
                std::vector<short> nextAudio(processedAudioData_.begin() + silentEndPos,
                                           processedAudioData_.begin() + silentEndPos + lookAheadSamples);
                if (getMaxVolume(nextAudio) > silentThreshold_ * 2) {
                    // Strong speech resumes - might be cutting a sentence
                    mightCutSentence = true;
                }
            }
        }*/

        // Decide whether to speed up or keep normal speed
        if (silenceDuration >= params_.minSilenceDuration && !mightCutSentence) {
            if(avgVisualDiff < params_.visualDiffThreshold) {
                std::println("Cutting silent segment (Duration: {:.2f}s, Avg Visual Diff: {:.2f})", silenceDuration, avgVisualDiff);

                // Speed up the segment
                const size_t numOutputFrames = (silentFrames.size() + 50 - 1) / 50;
                const size_t targetSamples = numOutputFrames * numSamplesPerFrame_;
                
                // Get audio for this segment
                const std::vector<short> silentSegment(processedAudioData_.begin() + silentStartPos,
                                                    processedAudioData_.begin() + silentEndPos);
                
                // Resample audio to match sped-up video
                const std::vector<short> processedSilentAudio = resampleAudio(silentSegment, targetSamples);
                modifiedAudio_.insert(modifiedAudio_.end(), processedSilentAudio.begin(), processedSilentAudio.end());
                
                // Add selected frames from the silent segment
                for (size_t i = 0; i < silentFrames.size(); i += 50) {
                    videoOutput_.write(silentFrames[i]);
                    totalOutputFrames_++;
                }
            } else {
                std::println("Speeding up silent segment (Duration: {:.2f}s, Avg Visual Diff: {:.2f})", silenceDuration, avgVisualDiff);

                // Speed up the segment
                const size_t numOutputFrames = (silentFrames.size() + params_.silentSpeed - 1) / params_.silentSpeed;
                const size_t targetSamples = numOutputFrames * numSamplesPerFrame_;
                
                // Get audio for this segment
                const std::vector<short> silentSegment(processedAudioData_.begin() + silentStartPos,
                                                    processedAudioData_.begin() + silentEndPos);
                
                // Resample audio to match sped-up video
                const std::vector<short> processedSilentAudio = resampleAudio(silentSegment, targetSamples);
                modifiedAudio_.insert(modifiedAudio_.end(), processedSilentAudio.begin(), processedSilentAudio.end());
                
                // Add selected frames from the silent segment
                for (size_t i = 0; i < silentFrames.size(); i += params_.silentSpeed) {
                    videoOutput_.write(silentFrames[i]);
                    totalOutputFrames_++;
                }
            }
        } else {
            std::println("Keeping silent segment at normal speed (Duration: {:.2f}s, Avg Visual Diff: {:.2f}, Might cut sentence: {})",
                        silenceDuration, avgVisualDiff, mightCutSentence ? "yes" : "no");
            
            // Keep the segment at normal speed
            for (const auto& frame : silentFrames) {
                videoOutput_.write(frame);
            }
            
            const std::vector<short> silentSegment(processedAudioData_.begin() + silentStartPos,
                                                 processedAudioData_.begin() + silentEndPos);
            modifiedAudio_.insert(modifiedAudio_.end(), silentSegment.begin(), silentSegment.end());
            totalOutputFrames_ += silentFrames.size();
        }
    }

    const Parameters params_;
    cv::VideoCapture& capture_;
    cv::VideoWriter& videoOutput_;
    const std::vector<short>& processedAudioData_;
    const std::vector<std::pair<double, double>>& inhaleTimes_;
    const int sampleRate_, audioChannels_;
    const double fps_;
    const int numSamplesPerFrame_;
    const int silentThreshold_;
    std::vector<short> modifiedAudio_;
    int totalOutputFrames_;
    
    // Lookahead buffer for better processing
    static constexpr int lookAheadFrames_ = 10;
    std::vector<cv::Mat> frameBuffer_;
    std::vector<short> audioBuffer_;
};

//------------------------------------------------------
// Convert inhale intervals from samples to seconds
//------------------------------------------------------
static std::vector<std::pair<double, double>> convertInhaleIntervalsToSeconds(
    const std::vector<std::pair<int, int>>& intervals,
    const int sampleRate, const int channels) {
    std::vector<std::pair<double, double>> times;
    times.reserve(intervals.size());
    
    for (const auto& [start, end] : intervals) {
        times.emplace_back(static_cast<double>(start) / (sampleRate * channels),
                           static_cast<double>(end) / (sampleRate * channels));
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
    
    // Optional parameters
    for (int i = 3; i < argc; i += 2) {
        if (i + 1 >= argc) break;
        
        std::string paramName = argv[i];
        std::string paramValue = argv[i+1];
        
        if (paramName == "--silent-speed") params.silentSpeed = std::stoi(paramValue);
        else if (paramName == "--silent-threshold") params.silentThreshold = std::stoi(paramValue);
        else if (paramName == "--min-silence") params.minSilenceDuration = std::stof(paramValue);
        else if (paramName == "--visual-diff") params.visualDiffThreshold = std::stoi(paramValue);
        else if (paramName == "--min-inhale") params.minInhaleDurationMs = std::stoi(paramValue);
        else if (paramName == "--max-inhale") params.maxInhaleDurationMs = std::stoi(paramValue);
        else if (paramName == "--inhale-low") params.inhaleLowThreshold = std::stoi(paramValue);
        else if (paramName == "--inhale-high") params.inhaleHighThreshold = std::stoi(paramValue);
        else if (paramName == "--watermark") params.watermarkString = paramValue;
        else if (paramName == "--debug") params.debugMode = (paramValue == "true" || paramValue == "1");
    }
    
    return params;
}

//------------------------------------------------------
// Main function
//------------------------------------------------------
int main(int argc, char* argv[]) {
    const Parameters params = (argc > 1) ? parseCommandLine(argc, argv) : Parameters();
    const std::string originalAudioFile = generateTempFilename(".wav");
    const std::string videoFile = generateTempFilename(".mp4");
    const std::string audioFile = generateTempFilename(".wav");

    std::println("Processing video: {}", params.inputFile);
    std::println("Output file: {}", params.outputFile);
    std::println("Parameters:");
    std::println("  Silent speed: {}", params.silentSpeed);
    std::println("  Silent threshold: {}", params.silentThreshold);
    std::println("  Min silence duration: {:.2f}s", params.minSilenceDuration);
    std::println("  Visual difference threshold: {}", params.visualDiffThreshold);
    std::println("  Debug mode: {}", params.debugMode ? "enabled" : "disabled");

    try {
        { // Processing Block
            std::println("Extracting audio...");
            AudioProcessor audioProcessor(params);
            audioProcessor.extractAudio(params.inputFile, originalAudioFile);
            audioProcessor.readAudio(originalAudioFile);

            std::println("Detecting inhales...");
            const auto inhaleIntervals = audioProcessor.detectInhales();
            const auto inhaleTimes = convertInhaleIntervalsToSeconds(inhaleIntervals, audioProcessor.getSampleRate(),
                                                                    audioProcessor.getChannels());
            
            std::println("Removing inhales from audio...");
            const auto processedAudioData = audioProcessor.removeInhaleSegments(inhaleIntervals);

            std::println("Opening video...");
            cv::VideoCapture capture(params.inputFile);

            if (not capture.isOpened()) 
                throw std::runtime_error(std::format("Cannot open video file: {}", params.inputFile));

            const int videoWidth = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_WIDTH));
            const int videoHeight = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_HEIGHT));
            const double fps = capture.get(cv::CAP_PROP_FPS);
            const int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
            cv::VideoWriter videoOutput(videoFile, fourcc, fps, cv::Size(videoWidth, videoHeight));

            if (not videoOutput.isOpened()) 
                throw std::runtime_error(std::format("Cannot open video writer: {}", videoFile));

            std::println("Processing video...");
            VideoProcessor videoProcessor(params, capture, videoOutput, processedAudioData, inhaleTimes,
                                        audioProcessor.getSampleRate(), audioProcessor.getChannels(), 
                                        audioProcessor.getDynamicSilentThreshold());
            videoProcessor.process();
            const auto& modifiedAudio = videoProcessor.getModifiedAudio();

            std::println("Normalizing audio...");
            std::vector<short> mutableAudio = modifiedAudio;
            AudioProcessor::normalize(mutableAudio);
            audioProcessor.writeAudio(audioFile, mutableAudio);
            
            // Close video writer and capture
            videoOutput.release();
            capture.release();
        }

        if(std::filesystem::exists(params.outputFile)) {
            std::filesystem::remove(params.outputFile);
        }

        // Merge the Results
        std::println("Merging audio and video...");
        const auto mergeCmd = std::format("ffmpeg -y -i \"{}\" -i \"{}\" -c:v copy -c:a aac \"{}\" > /dev/null 2>&1", 
                                        videoFile, audioFile, params.outputFile);
        std::system(mergeCmd.c_str());
        
        // Clean up
        std::println("Cleaning up temporary files...");
        std::filesystem::remove(originalAudioFile);
        std::filesystem::remove(videoFile);
        std::filesystem::remove(audioFile);
        
        std::println("Processing complete! Output saved to: {}", params.outputFile);
    } catch (const std::exception& e) {
        std::println(std::cerr, "Error: {}", e.what());
        return 1;
    }

    return 0;
}