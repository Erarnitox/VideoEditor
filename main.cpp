#include <algorithm>
#include <bit>
#include <cassert>
#include <chrono>
#include <cmath>
#include <concepts>
#include <execution>
#include <filesystem>
#include <format>
#include <future>
#include <iterator>
#include <limits>
#include <memory>
#include <numeric>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <optional>
#include <print>
#include <random>
#include <ranges>
#include <sndfile.h>
#include <source_location>
#include <span>
#include <stacktrace>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

// Helper for error handling with source location
#define ERX_THROW_IF(condition, message)                                       \
  do {                                                                         \
    if (condition) {                                                           \
      ::std::println(::std::cerr, "[ERROR] {} at {}:{}", message,              \
                     ::std::source_location::current().file_name(),            \
                     ::std::source_location::current().line());                \
      throw ::std::runtime_error(message);                                     \
    }                                                                          \
  } while (false)

#define ERX_ENSURES(condition)                                                 \
  ERX_THROW_IF(!(condition), "Post-condition violated: " #condition)
#define ERX_EXPECTS(condition)                                                 \
  ERX_THROW_IF(!(condition), "Pre-condition violated: " #condition)

//------------------------------------------------------
// Configuration constants
//------------------------------------------------------
constexpr int g_sampleLengthMs =
    200; // Time window to analyze for inhale detection

//------------------------------------------------------
// Parameters structure for configuring video processing
//------------------------------------------------------
struct Parameters {
  std::string inputFile = "input.mp4";
  std::string outputFile = "output.mp4";
  std::string watermarkString = "Visit: https://www.erarnitox.de/pub/thanks/   to support me!";
  int silentSpeed = 5;      // More aggressive speed-up for silent segments
  int silentThreshold = 20; // Lower threshold to catch more subtle silence
  float minSilenceDuration = 0.2F; // Shorter minimum to catch briefer pauses (seconds)
  float maxSilenceDuration = 2.0F; // Shorter minimum to catch briefer pauses (seconds)
  int visualDiffThreshold = 1; // Increased to allow more segments to be sped up
  int minInhaleDurationMs = 60;  // Shorter min inhale duration (ms)
  int maxInhaleDurationMs = 600; // Longer max inhale duration (ms)
  int inhaleLowThreshold = 40;   // For detecting softer inhales
  int inhaleHighThreshold = 300; // For avoiding cutting louder noises
  bool debugMode = false; // Enable visualization of silent/inhale segments
  int maxThreads = 4;     // Limit parallel threads for CPU usage control
  int cpuThrottle = 3;    // Milliseconds to sleep per frame (0 = disabled)
};

//------------------------------------------------------
// Generate a temporary filename
//------------------------------------------------------
[[nodiscard]] static inline std::string
generateTempFilename(const std::string &extension) {
  static const auto tempDir = std::filesystem::current_path();
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, 1000000000);
  return (tempDir / (std::to_string(dis(gen)) + extension)).string();
}

//------------------------------------------------------
// Concept for numeric types that can be used with volume calculations
//------------------------------------------------------
template <typename T>
concept Numeric = std::integral<T> || std::floating_point<T>;

//------------------------------------------------------
// Get the maximum volume from an audio sample using absolute value
//------------------------------------------------------
template <std::ranges::random_access_range R>
  requires std::same_as<std::ranges::range_value_t<R>, short>
[[nodiscard]] static inline int getMaxVolume(const R &samples) {
  if (std::ranges::empty(samples))
    return 0;
  const auto abs_compare = [](short a, short b) {
    return std::abs(a) < std::abs(b);
  };
  return *std::max_element(std::ranges::begin(samples),
                           std::ranges::end(samples), abs_compare);
}

//------------------------------------------------------
// Calculate the RMS volume of an audio sample
//------------------------------------------------------
template <std::ranges::random_access_range R>
  requires std::same_as<std::ranges::range_value_t<R>, short>
[[nodiscard]] static inline double getRMSVolume(const R &samples) {
  if (std::ranges::empty(samples))
    return 0.0;
  const auto sumSquares = std::transform_reduce(
      std::execution::par_unseq, std::ranges::begin(samples),
      std::ranges::end(samples), 0.0, std::plus<>(),
      [](short s) { return static_cast<double>(s) * s; });
  return std::sqrt(sumSquares /
                   static_cast<double>(std::ranges::size(samples)));
}

//------------------------------------------------------
// Normalize audio to full dynamic range
//------------------------------------------------------
template <std::ranges::random_access_range R>
  requires std::same_as<std::ranges::range_value_t<R>, short>
static inline void normalizeAudio(R &audio) {
  const int maxVol = getMaxVolume(audio);
  if (maxVol > 0) {
    const double scale = 32767.0 / static_cast<double>(maxVol);
    std::for_each(
        std::execution::par_unseq, std::ranges::begin(audio),
        std::ranges::end(audio), [scale](short &sample) {
          const int scaled_val = static_cast<int>(std::round(sample * scale));
          sample = static_cast<short>(std::clamp(scaled_val, -32767, 32767));
        });
  }
}

//------------------------------------------------------
// Compute difference between frames
//------------------------------------------------------
[[nodiscard]] static inline double
computeFrameDifference(const cv::Mat &frame1, const cv::Mat &frame2) {
  if (frame1.empty() || frame2.empty()) {
    return 0.0;
  }
  
  cv::Mat diff;
  cv::absdiff(frame1, frame2, diff);
  const cv::Scalar avgDiff = cv::mean(diff);
  // Assuming 3 channels (BGR), average them. If not, this might need
  // adjustment.
  return (avgDiff[0] + avgDiff[1] + avgDiff[2]) / 3.0;
}

//------------------------------------------------------
// Resample audio to target number of samples
//------------------------------------------------------
[[nodiscard]] static inline std::vector<short>
resampleAudio(std::span<const short> audio, const size_t targetSamples) {
  if (audio.empty() || targetSamples == 0)
    return {};
  std::vector<short> result;
  result.reserve(targetSamples); // Reserve to avoid reallocations
  const size_t originalSize = audio.size();
  for (size_t i = 0; i < targetSamples; ++i) {
    const size_t idx = static_cast<size_t>(
        std::floor(static_cast<double>(i) * originalSize / targetSamples));
    result.push_back(audio[std::min(idx, originalSize - 1)]);
  }
  ERX_ENSURES(result.size() == targetSamples);
  return result;
}

//------------------------------------------------------
// Check if a time falls within inhale intervals
//------------------------------------------------------
[[nodiscard]] static inline bool
isInInhale(const double time,
           std::span<const std::pair<double, double>> inhaleTimes) {
  return std::ranges::any_of(inhaleTimes, [time](const auto &interval) {
    return time >= interval.first && time <= interval.second;
  });
}

//------------------------------------------------------
// Audio processing class
//------------------------------------------------------
class AudioProcessor {
public:
  //------------------------------------------------------
  // Constructor
  //------------------------------------------------------
  explicit AudioProcessor(const Parameters &params) : params_(params) {}

  //------------------------------------------------------
  // Extract audio from video file
  //------------------------------------------------------
  void extractAudio(const std::string &videoFile,
                    const std::string &audioFile) const {
    const auto cmd = std::format(
        "ffmpeg -i \"{}\" -ab 160k -ac 2 -ar 44100 -vn \"{}\" > /dev/null 2>&1",
        videoFile, audioFile);
    const int result = std::system(cmd.c_str());
    ERX_THROW_IF(result != 0, "Failed to extract audio using ffmpeg");
  }

  //------------------------------------------------------
  // Read audio data from file
  //------------------------------------------------------
  void readAudio(const std::string &audioFile) {
    SNDFILE *sndFile = sf_open(audioFile.c_str(), SFM_READ, &sfinfo_);
    ERX_THROW_IF(!sndFile,
                 std::format("Unable to open audio file: {}", audioFile));

    audioData_.resize(sfinfo_.frames * sfinfo_.channels);
    const sf_count_t read_count = sf_read_short(
        sndFile, audioData_.data(), static_cast<sf_count_t>(audioData_.size()));

    ERX_THROW_IF(read_count != static_cast<sf_count_t>(audioData_.size()),
                 "Failed to read all audio data");

    const int close_result = sf_close(sndFile);
    ERX_THROW_IF(close_result != 0, "Failed to close audio file");

    // Calculate dynamic threshold
    calculateDynamicThresholds();
  }

  //------------------------------------------------------
  // Calculate dynamic thresholds based on audio content
  //------------------------------------------------------
  void calculateDynamicThresholds() {
    if (audioData_.empty())
      return;
    // Sample 10 seconds of audio or the entire file if shorter
    const int sampleDuration = 10 * sfinfo_.samplerate * sfinfo_.channels;
    const int samplesToAnalyze =
        std::min(sampleDuration, static_cast<int>(audioData_.size()));

    std::vector<int> volumes;
    const int windowSize =
        sfinfo_.samplerate / 10 * sfinfo_.channels; // 100ms window
    for (int i = 0; i < samplesToAnalyze; i += windowSize) {
      const int end = std::min(i + windowSize, samplesToAnalyze);
      const std::span<const short> window(audioData_.data() + i, end - i);
      volumes.push_back(getMaxVolume(window));
    }

    if (!volumes.empty()) {
      std::ranges::sort(volumes);
      // Use the 15th percentile for silent threshold
      const size_t silentIndex = static_cast<size_t>(volumes.size() * 0.15);
      dynamicSilentThreshold_ =
          static_cast<int>(volumes[silentIndex] * 1.5); // Add a small buffer
      // Clamp threshold to reasonable values
      dynamicSilentThreshold_ = std::max(
          15, std::min(dynamicSilentThreshold_, params_.silentThreshold));
      std::println("Dynamic silence threshold: {}", dynamicSilentThreshold_);
    }
  }

  //------------------------------------------------------
  // Detect inhale segments in audio
  //------------------------------------------------------
  [[nodiscard]] std::vector<std::pair<int, int>> detectInhales() const {
    const int windowSizeMs = g_sampleLengthMs;
    const int windowSizeSamples =
        (sfinfo_.samplerate * windowSizeMs / 1000) * sfinfo_.channels;
    const int minInhaleSamples =
        (params_.minInhaleDurationMs * sfinfo_.samplerate / 1000) *
        sfinfo_.channels;
    const int maxInhaleSamples =
        (params_.maxInhaleDurationMs * sfinfo_.samplerate / 1000) *
        sfinfo_.channels;

    // Limit threads to prevent CPU overload
    const size_t numThreads = std::max(
      1u, 
      std::min(
        static_cast<unsigned>(params_.maxThreads),
        std::thread::hardware_concurrency()
      )
    );
    
    const size_t dataSize = audioData_.size();
    const size_t chunkSize =
        (dataSize + numThreads - 1) / numThreads; // Ceiling division

    std::vector<std::future<std::vector<std::pair<int, int>>>> futures;
    futures.reserve(numThreads);

    for (size_t t = 0; t < numThreads; ++t) {
      const size_t start = t * chunkSize;
      if (start >= dataSize)
        break; // Avoid creating threads for empty chunks
      const size_t end =
          std::min(start + chunkSize + windowSizeSamples, dataSize);

      futures.push_back(std::async(std::launch::async, [this, start, end,
                                                        windowSizeSamples,
                                                        minInhaleSamples,
                                                        maxInhaleSamples]() {
        std::vector<std::pair<int, int>> localInhales;
        for (size_t j = start;
             j + windowSizeSamples <= end && j < audioData_.size();
             j += sfinfo_.channels) {
          const std::span<const short> window(audioData_.data() + j,
                                              windowSizeSamples);
          const int rms = getMaxVolume(window);
          if (rms >= params_.inhaleLowThreshold &&
              rms <= params_.inhaleHighThreshold) {
            if (localInhales.empty() ||
                static_cast<int>(j) >
                    localInhales.back().second + sfinfo_.channels) {
              localInhales.emplace_back(
                  static_cast<int>(j), static_cast<int>(j + windowSizeSamples));
            } else {
              localInhales.back().second =
                  static_cast<int>(j + windowSizeSamples);
            }
          }
        }
        // Filter by duration
        std::vector<std::pair<int, int>> filtered;
        filtered.reserve(localInhales.size());
        for (const auto &interval : localInhales) {
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
    for (auto &f : futures) {
      auto chunkInhales = f.get();
      inhales.insert(inhales.end(),
                     std::make_move_iterator(chunkInhales.begin()),
                     std::make_move_iterator(chunkInhales.end()));
    }
    std::ranges::sort(inhales);

    std::vector<std::pair<int, int>> merged;
    merged.reserve(inhales.size());
    for (const auto &interval : inhales) {
      if (merged.empty() ||
          merged.back().second + sfinfo_.channels < interval.first) {
        merged.push_back(interval);
      } else {
        merged.back().second = std::max(merged.back().second, interval.second);
      }
    }
    return merged;
  }

  //------------------------------------------------------
  // Remove inhale segments from audio
  //------------------------------------------------------
  [[nodiscard]] std::vector<short>
  removeInhaleSegments(std::span<const std::pair<int, int>> inhales) const {
    std::vector<short> result;
    result.reserve(audioData_.size()); // Conservative estimate
    size_t pos = 0;
    for (const auto &[start, end] : inhales) {
      ERX_EXPECTS(start >= 0 && end <= static_cast<int>(audioData_.size()) &&
                  start <= end);
      if (static_cast<size_t>(start) > pos) {
        result.insert(result.end(), audioData_.begin() + pos,
                      audioData_.begin() + start);
      }
      pos = static_cast<size_t>(end);
    }
    if (pos < audioData_.size()) {
      result.insert(result.end(), audioData_.begin() + pos, audioData_.end());
    }
    return result;
  }

  //------------------------------------------------------
  // Static method to normalize audio
  //------------------------------------------------------
  template <typename R>
    requires std::same_as<std::ranges::range_value_t<R>, short>
  static void normalize(R &audio) {
    normalizeAudio(audio);
  }

  //------------------------------------------------------
  // Write audio data to file
  //------------------------------------------------------
  void writeAudio(const std::string &audioFile,
                  std::span<const short> audio) const {
    SF_INFO outSfinfo = sfinfo_;
    outSfinfo.frames = static_cast<sf_count_t>(
        audio.size() / static_cast<size_t>(outSfinfo.channels));
    SNDFILE *outFile = sf_open(audioFile.c_str(), SFM_WRITE, &outSfinfo);
    ERX_THROW_IF(!outFile,
                 std::format("Unable to write audio file: {}", audioFile));

    const sf_count_t written_count = sf_write_short(
        outFile, audio.data(), static_cast<sf_count_t>(audio.size()));
    ERX_THROW_IF(written_count != static_cast<sf_count_t>(audio.size()),
                 "Failed to write all audio data");

    const int close_result = sf_close(outFile);
    ERX_THROW_IF(close_result != 0, "Failed to close output audio file");
  }

  //------------------------------------------------------
  // Getters
  //------------------------------------------------------
  [[nodiscard]] int getSampleRate() const { return sfinfo_.samplerate; }
  [[nodiscard]] int getChannels() const { return sfinfo_.channels; }
  [[nodiscard]] int getDynamicSilentThreshold() const {
    return dynamicSilentThreshold_;
  }

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
  VideoProcessor(const Parameters &params, cv::VideoCapture &capture,
                 cv::VideoWriter &videoOutput,
                 std::span<const short> processedAudioData,
                 std::span<const std::pair<double, double>> inhaleTimes,
                 const int sampleRate, const int audioChannels,
                 const int dynamicSilentThreshold)
      : params_(params), capture_(capture), videoOutput_(videoOutput),
        processedAudioData_(processedAudioData.data(),
                            processedAudioData.data() +
                                processedAudioData.size()),
        inhaleTimes_(inhaleTimes.data(),
                     inhaleTimes.data() + inhaleTimes.size()),
        sampleRate_(sampleRate), audioChannels_(audioChannels),
        fps_(capture.get(cv::CAP_PROP_FPS)),
        numSamplesPerFrame_(static_cast<int>(std::round(
                                static_cast<double>(sampleRate) / fps_)) *
                            audioChannels),
        silentThreshold_(dynamicSilentThreshold > 0 ? dynamicSilentThreshold
                                                    : params.silentThreshold),
        totalOutputFrames_(0) {
    ERX_THROW_IF(fps_ <= 0.0, "Invalid FPS detected in video");
    ERX_THROW_IF(numSamplesPerFrame_ <= 0, "Invalid samples per frame calculated");
    // Reserve space for lookahead buffer - with reasonable limits
    const auto frame_limit{ 50 };
    frameBuffer_.reserve(std::min(lookAheadFrames_, frame_limit));
    audioBuffer_.reserve(lookAheadFrames_ * static_cast<size_t>(numSamplesPerFrame_*frame_limit));
  }

  //------------------------------------------------------
  // Process the video
  //------------------------------------------------------
  void process() {
    int currentFrameIndex = 0;
    int currentAudioPos = 0;
    std::vector<cv::Mat> silentFrames;
    silentFrames.reserve(100); // Estimate reserve
    int silentStartPos = -1;
    std::optional<cv::Mat> silentBaseline; // Use optional for clearer state
    std::println("Starting video processing with silent threshold: {}",
                 silentThreshold_);
    std::println("Frames will be processed at {} fps", fps_);

    auto lastThrottleTime = std::chrono::steady_clock::now();
    
    while (true) {
      // Fill the lookahead buffer if needed
      fillLookaheadBuffer(currentFrameIndex);
      // Break if we've reached the end
      if (frameBuffer_.empty())
        break;
      // Get current frame
      cv::Mat frame = std::move(frameBuffer_.front());
      frameBuffer_.erase(frameBuffer_.begin());
      
      // Safety check for empty frames
      if (frame.empty()) {
        std::println("Skipping empty frame at position {}", currentFrameIndex);
        currentFrameIndex++;
        continue;
      }
      
      const double origTime = static_cast<double>(currentFrameIndex) / fps_;
      currentFrameIndex++;
      // Skip inhale frames
      if (isInInhale(origTime, inhaleTimes_)) {
        continue;
      }
      if (currentAudioPos >= static_cast<int>(processedAudioData_.size())) {
        break;
      }
      const int audioEndPos =
          std::min(currentAudioPos + numSamplesPerFrame_,
                   static_cast<int>(processedAudioData_.size()));
      const std::span<const short> audioSample(processedAudioData_.data() +
                                                   currentAudioPos,
                                               audioEndPos - currentAudioPos);

      // Check if this segment is silent
      const bool isSilent = checkSilence(audioSample);

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
        silentFrames.push_back(std::move(frame)); // Move frame to avoid copy
        currentAudioPos += numSamplesPerFrame_;
      } else {
        // Process any accumulated silent segment
        if (!silentFrames.empty()) {
          processSilentSegment(silentFrames, silentStartPos, currentAudioPos,
                               silentBaseline);
          silentFrames.clear();
          silentBaseline.reset(); // Clear the baseline
          silentStartPos = -1;
        }
        // Write regular frame
        videoOutput_.write(frame);
        modifiedAudio_.insert(modifiedAudio_.end(), audioSample.begin(),
                              audioSample.end());
        currentAudioPos += numSamplesPerFrame_;
        totalOutputFrames_++;
      }
      
      // CPU Throttling - only if enabled
      if (params_.cpuThrottle > 0) {
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - lastThrottleTime);
            
        if (elapsed.count() < params_.cpuThrottle) {
          std::this_thread::sleep_for(
            std::chrono::milliseconds(params_.cpuThrottle) - elapsed
          );
        }
        lastThrottleTime = std::chrono::steady_clock::now();
      }
    }
    // Handle any remaining silent frames
    if (!silentFrames.empty()) {
      processSilentSegment(silentFrames, silentStartPos, currentAudioPos,
                           silentBaseline);
    }
    // Print synchronization diagnostics
    const double expectedAudioSamples =
        static_cast<double>(totalOutputFrames_) *
        static_cast<double>(numSamplesPerFrame_);
    std::println("Total output frames: {}", totalOutputFrames_);
    std::println("Expected audio samples: {}", expectedAudioSamples);
    std::println("Actual audio samples: {}", modifiedAudio_.size());
    // Adjust audio length if needed to maintain sync
    if (std::abs(expectedAudioSamples -
                 static_cast<double>(modifiedAudio_.size())) >
        static_cast<double>(numSamplesPerFrame_)) {
      std::println("Adjusting audio length for proper synchronization");
      modifiedAudio_ = resampleAudio(modifiedAudio_,
                                     static_cast<size_t>(expectedAudioSamples));
    }
  }

  //------------------------------------------------------
  // Fill the lookahead buffer with frames
  //------------------------------------------------------
  void fillLookaheadBuffer(int /*currentFrameIndex*/) {
    while (frameBuffer_.size() < lookAheadFrames_) {
      cv::Mat frame;
      if (!capture_.read(frame)) {
        break;
      }
      frameBuffer_.push_back(std::move(frame));
    }
  }

  //------------------------------------------------------
  // Check if audio sample is silent
  //------------------------------------------------------
  [[nodiscard]] bool checkSilence(std::span<const short> audioSample) const {
    if (audioSample.empty())
      return true;
      
    // Short-circuit: if max volume is above threshold, it's not silent
    const int maxVol = getMaxVolume(audioSample);
    if (maxVol >= silentThreshold_) 
      return false;
      
    // Only calculate RMS if max volume is below threshold
    const double rmsVol = getRMSVolume(audioSample);
    return rmsVol < (static_cast<double>(silentThreshold_) * 0.7);
  }

  //------------------------------------------------------
  // Draw debug information on frame
  //------------------------------------------------------
  void drawDebugInfo(cv::Mat &frame, bool isSilent, double time) const {
    // Draw colored boxes to indicate frame status
    if (isSilent) {
      // Red box for silent frame
      cv::rectangle(frame, cv::Point(0, 0), cv::Point(20, 20),
                    cv::Scalar(0, 0, 255), -1);
    } else {
      // Green box for speech frame
      cv::rectangle(frame, cv::Point(0, 0), cv::Point(20, 20),
                    cv::Scalar(0, 255, 0), -1);
    }
    // Blue box for inhale frame
    if (isInInhale(time, inhaleTimes_)) {
      cv::rectangle(frame, cv::Point(20, 0), cv::Point(40, 20),
                    cv::Scalar(255, 0, 0), -1);
    }
    // Add timestamp
    cv::putText(frame, std::format("{:.2f}s", time), cv::Point(50, 20),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
  }

  //------------------------------------------------------
  // Draw watermark on frame
  //------------------------------------------------------
  void drawWatermark(cv::Mat &frame) const {
    if (frame.empty()) return;
    if (params_.watermarkString.empty()) return;
    
    // Check if frame is too small for watermark
    if (frame.rows < 30 || frame.cols < 100) {
      return;
    }
    cv::rectangle(frame, cv::Point(0, 0), cv::Point(params_.watermarkString.size()*10, 20),
                    cv::Scalar(0, 0, 255), -1);

    cv::putText(frame, params_.watermarkString, cv::Point(50, 20),
                cv::FONT_HERSHEY_DUPLEX, 0.7, cv::Scalar(255, 255, 255), 1);
  }

  //------------------------------------------------------
  // Get the modified audio
  //------------------------------------------------------
  [[nodiscard]]
  const std::vector<short> &getModifiedAudio() const {
    return modifiedAudio_;
  }

private:
  //------------------------------------------------------
  // Process a segment of silent frames
  //------------------------------------------------------
  void processSilentSegment(std::vector<cv::Mat>& silentFrames,
                            const int silentStartPos, const int silentEndPos,
                            std::optional<cv::Mat>& silentBaseline) {
    if (silentFrames.empty() || !silentBaseline.has_value())
      return;

    const double silenceDuration =
        static_cast<double>(silentFrames.size()) / fps_;
    // Calculate visual difference throughout silent segment
    double sumDiff = 0.0;
    const cv::Mat &baseline = silentBaseline.value();
    
    // Skip visual diff calculation if not needed
    if (silenceDuration >= static_cast<double>(params_.minSilenceDuration)) {
      for (const auto &f : silentFrames) {
        if (f.empty() || baseline.empty()) continue;
        sumDiff += computeFrameDifference(baseline, f);
      }
      const double avgVisualDiff =
          sumDiff / static_cast<double>(silentFrames.size());

      // Decide whether to speed up or keep normal speed
      if (avgVisualDiff < static_cast<double>(params_.visualDiffThreshold)) {
        std::println("Cutting silent segment (Duration: {:.2f}s, Avg Visual "
                     "Diff: {:.2f})",
                     silenceDuration, avgVisualDiff);
        // Speed up the segment by cutting (keeping only 1 frame every 50)
        const size_t numOutputFrames =
            (silentFrames.size() + 49) / 50; // Ceiling division
        const size_t targetSamples =
            numOutputFrames * static_cast<size_t>(numSamplesPerFrame_);
        // Get audio for this segment
        const std::span<const short> silentSegmentSpan(
            processedAudioData_.data() + silentStartPos,
            silentEndPos - silentStartPos);
        const std::vector<short> silentSegment(silentSegmentSpan.begin(),
                                               silentSegmentSpan.end());
        // Resample audio to match sped-up video
        const std::vector<short> processedSilentAudio =
            resampleAudio(silentSegment, targetSamples);
        modifiedAudio_.insert(modifiedAudio_.end(),
                              processedSilentAudio.begin(),
                              processedSilentAudio.end());
        // Add selected frames from the silent segment
        for (size_t i = 0; i < silentFrames.size(); i += 50) {
          if (i < silentFrames.size() && !silentFrames[i].empty()) {
            videoOutput_.write(silentFrames[i]);
            totalOutputFrames_++;
          }
        }
      } else {
        std::println("Speeding up silent segment (Duration: {:.2f}s, Avg "
                     "Visual Diff: {:.2f})",
                     silenceDuration, avgVisualDiff);
        // Speed up the segment
        const size_t numOutputFrames =
            (silentFrames.size() + static_cast<size_t>(params_.silentSpeed) -
             1) /
            static_cast<size_t>(params_.silentSpeed); // Ceiling division
        const size_t targetSamples =
            numOutputFrames * static_cast<size_t>(numSamplesPerFrame_);
        // Get audio for this segment
        const std::span<const short> silentSegmentSpan(
            processedAudioData_.data() + silentStartPos,
            silentEndPos - silentStartPos);
        const std::vector<short> silentSegment(silentSegmentSpan.begin(),
                                               silentSegmentSpan.end());
        // Resample audio to match sped-up video
        const std::vector<short> processedSilentAudio =
            resampleAudio(silentSegment, targetSamples);
        modifiedAudio_.insert(modifiedAudio_.end(),
                              processedSilentAudio.begin(),
                              processedSilentAudio.end());
        // Add selected frames from the silent segment
        for (size_t i = 0; i < silentFrames.size();
             i += static_cast<size_t>(params_.silentSpeed)) {
          if (i < silentFrames.size() && !silentFrames[i].empty()) {
            videoOutput_.write(silentFrames[i]);
            totalOutputFrames_++;
          }
        }
      }
    } else {
      std::println("Keeping silent segment at normal speed (Duration: {:.2f}s, "
                   "Avg Visual Diff: {:.2f})",
                   silenceDuration, sumDiff / silentFrames.size());
      // Keep the segment at normal speed
      for (const auto &frame : silentFrames) {
        if (!frame.empty()) {
          videoOutput_.write(frame);
        }
      }
      const std::span<const short> silentSegmentSpan(
          processedAudioData_.data() + silentStartPos,
          silentEndPos - silentStartPos);
      const std::vector<short> silentSegment(silentSegmentSpan.begin(),
                                             silentSegmentSpan.end());
      modifiedAudio_.insert(modifiedAudio_.end(), silentSegment.begin(),
                            silentSegment.end());
      totalOutputFrames_ += static_cast<int>(silentFrames.size());
    }

    std::vector<cv::Mat>(0).swap(silentFrames);  // Force deallocation  
    std::optional<cv::Mat>().swap(silentBaseline);  
  }

  const Parameters params_;
  cv::VideoCapture &capture_;
  cv::VideoWriter &videoOutput_;
  const std::vector<short> processedAudioData_; // Copy of span data for safety
  const std::vector<std::pair<double, double>>
      inhaleTimes_; // Copy of span data for safety
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
static std::vector<std::pair<double, double>>
convertInhaleIntervalsToSeconds(std::span<const std::pair<int, int>> intervals,
                                const int sampleRate, const int channels) {
  std::vector<std::pair<double, double>> times;
  times.reserve(intervals.size());
  const double factor =
      1.0 / (static_cast<double>(sampleRate) * static_cast<double>(channels));
  for (const auto &[start, end] : intervals) {
    times.emplace_back(static_cast<double>(start) * factor,
                       static_cast<double>(end) * factor);
  }
  return times;
}

//------------------------------------------------------
// Parse command line arguments
//------------------------------------------------------
Parameters parseCommandLine(int argc, char *argv[]) {
  Parameters params;
  if (argc > 1)
    params.inputFile = argv[1];
  if (argc > 2)
    params.outputFile = argv[2];
  // Optional parameters
  for (int i = 3; i < argc; i += 2) {
    if (i + 1 >= argc)
      break;
    std::string paramName = argv[i];
    std::string paramValue = argv[i + 1];
    if (paramName == "--silent-speed")
      params.silentSpeed = std::stoi(paramValue);
    else if (paramName == "--silent-threshold")
      params.silentThreshold = std::stoi(paramValue);
    else if (paramName == "--min-silence")
      params.minSilenceDuration = std::stof(paramValue);
    else if (paramName == "--visual-diff")
      params.visualDiffThreshold = std::stoi(paramValue);
    else if (paramName == "--min-inhale")
      params.minInhaleDurationMs = std::stoi(paramValue);
    else if (paramName == "--max-inhale")
      params.maxInhaleDurationMs = std::stoi(paramValue);
    else if (paramName == "--inhale-low")
      params.inhaleLowThreshold = std::stoi(paramValue);
    else if (paramName == "--inhale-high")
      params.inhaleHighThreshold = std::stoi(paramValue);
    else if (paramName == "--watermark")
      params.watermarkString = paramValue;
    else if (paramName == "--debug")
      params.debugMode = (paramValue == "true" || paramValue == "1");
    else if (paramName == "--max-threads")
      params.maxThreads = std::stoi(paramValue);
    else if (paramName == "--cpu-throttle")
      params.cpuThrottle = std::stoi(paramValue);
  }
  return params;
}

//------------------------------------------------------
// Main function
//------------------------------------------------------
int main(int argc, char *argv[]) {
  const Parameters params =
      (argc > 1) ? parseCommandLine(argc, argv) : Parameters();
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
  std::println("  Max threads: {}", params.maxThreads);
  std::println("  CPU throttle: {}ms", params.cpuThrottle);
  std::println("  Debug mode: {}", params.debugMode ? "enabled" : "disabled");
  try {
    { // Processing Block
      std::println("Extracting audio...");
      AudioProcessor audioProcessor(params);
      audioProcessor.extractAudio(params.inputFile, originalAudioFile);
      audioProcessor.readAudio(originalAudioFile);
      std::println("Detecting inhales...");
      const auto inhaleIntervals = audioProcessor.detectInhales();
      const auto inhaleTimes = convertInhaleIntervalsToSeconds(
          inhaleIntervals, audioProcessor.getSampleRate(),
          audioProcessor.getChannels());
      std::println("Removing inhales from audio...");
      const auto processedAudioData =
          audioProcessor.removeInhaleSegments(inhaleIntervals);
      std::println("Opening video...");
      cv::VideoCapture capture(params.inputFile);
      ERX_THROW_IF(!capture.isOpened(),
                   std::format("Cannot open video file: {}", params.inputFile));
      const int videoWidth =
          static_cast<int>(capture.get(cv::CAP_PROP_FRAME_WIDTH));
      const int videoHeight =
          static_cast<int>(capture.get(cv::CAP_PROP_FRAME_HEIGHT));
      const double fps = capture.get(cv::CAP_PROP_FPS);
      const int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
      cv::VideoWriter videoOutput(videoFile, fourcc, fps,
                                  cv::Size(videoWidth, videoHeight));
      ERX_THROW_IF(!videoOutput.isOpened(),
                   std::format("Cannot open video writer: {}", videoFile));
      std::println("Processing video...");
      VideoProcessor videoProcessor(
          params, capture, videoOutput, processedAudioData, inhaleTimes,
          audioProcessor.getSampleRate(), audioProcessor.getChannels(),
          audioProcessor.getDynamicSilentThreshold());
      videoProcessor.process();
      const auto &modifiedAudio = videoProcessor.getModifiedAudio();
      std::println("Normalizing audio...");
      std::vector<short> mutableAudio = modifiedAudio; // Copy for normalization
      AudioProcessor::normalize(mutableAudio);
      audioProcessor.writeAudio(audioFile, mutableAudio);
      // Close video writer and capture
      videoOutput.release();
      capture.release();
    }
    if (std::filesystem::exists(params.outputFile)) {
      std::filesystem::remove(params.outputFile);
    }
    // Merge the Results
    std::println("Merging audio and video...");
    const auto mergeCmd = std::format("ffmpeg -y -i \"{}\" -i \"{}\" -c:v copy "
                                      "-c:a aac \"{}\" > /dev/null 2>&1",
                                      videoFile, audioFile, params.outputFile);
    const int mergeResult = std::system(mergeCmd.c_str());
    ERX_THROW_IF(mergeResult != 0,
                 "Failed to merge audio and video using ffmpeg");
    // Clean up
    std::println("Cleaning up temporary files...");
    std::filesystem::remove(originalAudioFile);
    std::filesystem::remove(videoFile);
    std::filesystem::remove(audioFile);
    std::println("Processing complete! Output saved to: {}", params.outputFile);
  } catch (const std::exception &e) {
    std::println(std::cerr, "Error: {}", e.what());
    return 1;
  } catch (...) {
    std::println(std::cerr, "Unknown error occurred.");
    return 1;
  }
  return 0;
}