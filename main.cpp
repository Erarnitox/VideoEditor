#include <print>
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

//------------------------------------------------------
//
//------------------------------------------------------
constexpr int g_sampleLengthMs  = 200;  // Time window to analyze for inhale detection

//------------------------------------------------------
//
//------------------------------------------------------
struct Parameters {
    std::string inputFile       = "input.mp4";
    std::string outputFile      = "output2.mp4";
    int silentSpeed             = 3;    // Gentler speed-up for silent segments
    int silentThreshold         = 30;   // Higher threshold to preserve soft speech
    float minSilenceDuration    = 1;    // Shorter duration to capture brief pauses (seconds)
    int visualDiffThreshold     = 1;    // Higher threshold to retain visual changes
    int minInhaleDurationMs     = 80;   // Slightly shorter min inhale duration (ms)
    int maxInhaleDurationMs     = 500;  // Slightly longer max inhale duration (ms)
    int inhaleLowThreshold      = 50;   // Lowered for softer inhales
    int inhaleHighThreshold     = 300;  // Raised to avoid cutting louder noises
};

//------------------------------------------------------
//
//------------------------------------------------------
[[nodiscard]] static inline 
std::string generateTempFilename(const std::string& extension) {
    static const auto tempDir = std::filesystem::current_path(); //std::filesystem::temp_directory_path();
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 1000000000);
    return (tempDir / (std::to_string(dis(gen)) + extension)).string();
}

//------------------------------------------------------
//
//------------------------------------------------------
[[nodiscard]] static inline
int getMaxVolume(const std::vector<short>& samples) {
    return std::accumulate(samples.begin(), samples.end(), 0,
        [](int max, short s) { return std::max(max, std::abs(static_cast<int>(s))); });
}

//------------------------------------------------------
//
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
//
//------------------------------------------------------
[[nodiscard]] static inline
double computeFrameDifference(const cv::Mat& frame1, const cv::Mat& frame2) {
    cv::Mat diff;
    cv::absdiff(frame1, frame2, diff);
    const cv::Scalar avgDiff = cv::mean(diff);
    return (avgDiff[0] + avgDiff[1] + avgDiff[2]) / 3.0;
}

//------------------------------------------------------
//
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
//
//------------------------------------------------------
[[nodiscard]] static inline 
bool isInInhale(const double time, const std::vector<std::pair<double, double>>& inhaleTimes) {
    return std::any_of(inhaleTimes.begin(), inhaleTimes.end(),
        [time](const auto& interval) { 
            return time >= interval.first && time <= interval.second; 
        });
}

//------------------------------------------------------
//
//------------------------------------------------------
class AudioProcessor {
public:
    //------------------------------------------------------
    //
    //------------------------------------------------------
    explicit AudioProcessor(const Parameters& params) : params_(params) {}

    //------------------------------------------------------
    //
    //------------------------------------------------------
    void extractAudio(const std::string& videoFile, const std::string& audioFile) const {
        const auto cmd = std::format("ffmpeg -i {} -ab 160k -ac 2 -ar 44100 -vn {} > /dev/null 2>&1", videoFile, audioFile);
        std::system(cmd.c_str());
    }

    //------------------------------------------------------
    //
    //------------------------------------------------------
    void readAudio(const std::string& audioFile) {
        SNDFILE* sndFile = sf_open(audioFile.c_str(), SFM_READ, &sfinfo_);

        if (not sndFile) 
            throw std::runtime_error("Unable to open audio file: " + audioFile);

        audioData_.resize(sfinfo_.frames * sfinfo_.channels);
        sf_read_short(sndFile, audioData_.data(), audioData_.size());
        sf_close(sndFile);
    }

    //------------------------------------------------------
    //
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
    //
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
    //
    //------------------------------------------------------
    static void normalize(std::vector<short>& audio) {
        normalizeAudio(audio);
    }

    //------------------------------------------------------
    //
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
    //
    //------------------------------------------------------
    [[nodiscard]]
    int getSampleRate() const { 
        return sfinfo_.samplerate; 
    }

    //------------------------------------------------------
    //
    //------------------------------------------------------
    [[nodiscard]]
    int getChannels() const { 
        return sfinfo_.channels; 
    }

private:
    const Parameters params_;
    SF_INFO sfinfo_{};
    std::vector<short> audioData_;
};

//------------------------------------------------------
//
//------------------------------------------------------
class VideoProcessor {
public:
    //------------------------------------------------------
    //
    //------------------------------------------------------
    VideoProcessor(const Parameters& params, cv::VideoCapture& capture, cv::VideoWriter& videoOutput,
                   const std::vector<short>& processedAudioData, const std::vector<std::pair<double, double>>& inhaleTimes,
                   const int sampleRate, const int audioChannels)
        : params_(params), capture_(capture), videoOutput_(videoOutput), processedAudioData_(processedAudioData),
          inhaleTimes_(inhaleTimes), sampleRate_(sampleRate), audioChannels_(audioChannels),
          fps_(capture.get(cv::CAP_PROP_FPS)),
          numSamplesPerFrame_(static_cast<int>(std::round(static_cast<double>(sampleRate) / fps_)) * audioChannels),
          totalOutputFrames_(0) {}

    //------------------------------------------------------
    //
    //------------------------------------------------------
    void process() {
        int currentFrameIndex = 0;
        int currentAudioPos = 0;
        std::vector<cv::Mat> silentFrames;
        int silentStartPos = -1;
        cv::Mat silentBaseline;

        while (capture_.isOpened()) {
            cv::Mat frame;

            if (not capture_.read(frame)) 
                break;

            const double origTime = currentFrameIndex / fps_;
            currentFrameIndex++;

            if (isInInhale(origTime, inhaleTimes_))
                continue;

            if (currentAudioPos >= static_cast<int>(processedAudioData_.size()))
                break;

            const int audioSampleEnd = std::min(currentAudioPos + numSamplesPerFrame_, static_cast<int>(processedAudioData_.size()));
            const std::vector<short> audioSample(processedAudioData_.begin() + currentAudioPos,
                                                processedAudioData_.begin() + audioSampleEnd);

            if (getMaxVolume(audioSample) < params_.silentThreshold) {
                if (silentFrames.empty()) {
                    silentStartPos = currentAudioPos;
                    silentBaseline = frame.clone();
                }
                silentFrames.push_back(frame.clone());
                currentAudioPos += numSamplesPerFrame_;
            } else {
                if (not silentFrames.empty()) {
                    processSilentSegment(silentFrames, silentStartPos, currentAudioPos, silentBaseline);
                    silentFrames.clear();
                    silentStartPos = -1;
                }
                videoOutput_.write(frame);
                modifiedAudio_.insert(modifiedAudio_.end(), audioSample.begin(), audioSample.end());
                currentAudioPos += numSamplesPerFrame_;
                totalOutputFrames_++;
            }
        }

        if (not silentFrames.empty()) {
            processSilentSegment(silentFrames, silentStartPos, currentAudioPos, silentBaseline);
        }

        // Synchronization diagnostics
        const double expectedAudioSamples = totalOutputFrames_ * numSamplesPerFrame_;
        std::println("Total output frames: {}", totalOutputFrames_);
        std::println("Expected audio samples: {}", expectedAudioSamples);
        std::println("Actual audio samples: {}", modifiedAudio_.size());
    }

    //------------------------------------------------------
    //
    //------------------------------------------------------
    [[nodiscard]]
    const std::vector<short>& getModifiedAudio() const {
        return modifiedAudio_; 
    }

private:
    //------------------------------------------------------
    //
    //------------------------------------------------------
    void processSilentSegment(const std::vector<cv::Mat>& silentFrames, const int silentStartPos, const int silentEndPos,
                              const cv::Mat& silentBaseline) {
        const double silenceDuration = static_cast<double>(silentFrames.size()) / fps_;
        double sumDiff = 0.0;
        for (const auto& f : silentFrames) {
            sumDiff += computeFrameDifference(silentBaseline, f);
        }
        const double avgVisualDiff = sumDiff / silentFrames.size();

        if (silenceDuration >= params_.minSilenceDuration && avgVisualDiff < params_.visualDiffThreshold) {
            std::println("Speeding up silent segment (Duration: {}s, Avg Visual Diff: {})", silenceDuration, avgVisualDiff);
            const size_t numOutputFrames = (silentFrames.size() + params_.silentSpeed - 1) / params_.silentSpeed;
            const size_t targetSamples = numOutputFrames * numSamplesPerFrame_;
            const std::vector<short> silentSegment(processedAudioData_.begin() + silentStartPos,
                                                   processedAudioData_.begin() + silentEndPos);
            const std::vector<short> processedSilentAudio = resampleAudio(silentSegment, targetSamples);
            modifiedAudio_.insert(modifiedAudio_.end(), processedSilentAudio.begin(), processedSilentAudio.end());
            for (size_t i{ 0 }; i < silentFrames.size(); i += params_.silentSpeed) {
                videoOutput_.write(silentFrames[i]);
                totalOutputFrames_++;
            }
        } else {
            std::println("Keeping silent segment at normal speed (Duration: {}s, Avg Visual Diff: {})\n", silenceDuration, avgVisualDiff);
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
    std::vector<short> modifiedAudio_;
    int totalOutputFrames_;
};

//------------------------------------------------------
//
//------------------------------------------------------
static std::vector<std::pair<double, double>> convertInhaleIntervalsToSeconds(const std::vector<std::pair<int, int>>& intervals,
                                                                              const int sampleRate, const int channels) {
    std::vector<std::pair<double, double>> times;
    for (const auto& [start, end] : intervals) {
        times.emplace_back(static_cast<double>(start) / (sampleRate * channels),
                           static_cast<double>(end) / (sampleRate * channels));
    }
    return times;
}

//------------------------------------------------------
//
//------------------------------------------------------
int main() {
    const Parameters params;
    const std::string originalAudioFile = generateTempFilename(".wav");
    const std::string videoFile = generateTempFilename(".mp4");
    const std::string audioFile = generateTempFilename(".wav");

    try {
        { // Processing Block
            AudioProcessor audioProcessor(params);
            audioProcessor.extractAudio(params.inputFile, originalAudioFile);
            audioProcessor.readAudio(originalAudioFile);

            const auto inhaleIntervals = audioProcessor.detectInhales();
            const auto processedAudioData = audioProcessor.removeInhaleSegments(inhaleIntervals);
            const auto inhaleTimes = convertInhaleIntervalsToSeconds(inhaleIntervals, audioProcessor.getSampleRate(),
                                                                    audioProcessor.getChannels());

            cv::VideoCapture capture(params.inputFile);

            if (not capture.isOpened()) 
                throw std::runtime_error("Cannot open video file: " + params.inputFile);

            const int videoWidth = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_WIDTH));
            const int videoHeight = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_HEIGHT));
            const double fps = capture.get(cv::CAP_PROP_FPS);
            const int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
            cv::VideoWriter videoOutput(videoFile, fourcc, fps, cv::Size(videoWidth, videoHeight));

            if (not videoOutput.isOpened()) 
                throw std::runtime_error("Cannot open video writer: " + videoFile);

            VideoProcessor videoProcessor(params, capture, videoOutput, processedAudioData, inhaleTimes,
                                        audioProcessor.getSampleRate(), audioProcessor.getChannels());
            videoProcessor.process();
            const auto& modifiedAudio = videoProcessor.getModifiedAudio();

            std::vector<short> mutableAudio = modifiedAudio;
            AudioProcessor::normalize(mutableAudio);
            audioProcessor.writeAudio(audioFile, mutableAudio);
        }

        // Merge the Results
        const auto mergeCmd = std::format("ffmpeg -i {} -i {} -c:v copy -c:a aac {}", videoFile, audioFile, params.outputFile);
        std::system(mergeCmd.c_str());
        
        // Clean up
        std::filesystem::remove(originalAudioFile);
        std::filesystem::remove(videoFile);
        std::filesystem::remove(audioFile);
    } catch (const std::exception& e) {
        std::println(std::cerr, "Error: {}", e.what());
        return 1;
    }

    return 0;
}