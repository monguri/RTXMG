#pragma once

// clang-format off

#include <cstdint>
#include <memory>
#include <limits>
#include <filesystem>
#include <vector>

// clang-format on

namespace audio {

struct WaveForm {
    std::vector<float> mins;
    std::vector<float> maxs;
};

class WaveFile {

public:

    struct Fmt {
        uint16_t audioFormat;    // Audio format 1=PCM,6=mulaw,7=alaw, 257=IBM Mu-Law, 258=IBM A-Law, 259=ADPCM
        uint16_t numChannels;    // Number of channels 1=Mono 2=Sterio
        uint32_t samplesPerSec;  // Sampling Frequency in Hz
        uint32_t bytesPerSec;    // bytes per second
        uint16_t blockAlign;     // 2=16-bit mono, 4=16-bit stereo
        uint16_t bitsPerSample;  // Number of bits per sample
    };

    static std::unique_ptr<WaveFile> read(std::filesystem::path const& m_filepath);

    static std::unique_ptr<WaveFile> create(std::vector<uint8_t>&& data);

    std::filesystem::path m_filepath;

    Fmt const& getFormat() const;

    void const * getSamplesData() const;   
    uint32_t getSamplesDataSize() const;
    uint32_t getNumSamplesTotal() const;

    float duration() const;

    std::unique_ptr<WaveForm> computeWaveform(
        uint32_t size, float start_time = 0.f, float end_time = -1.f, int channel = -1) const;

private:

    WaveFile() = default;

    struct Chunk;
    struct Header;

    static Chunk const* getSubChunk(std::vector<uint8_t> const& data, char const* name);

    mutable Fmt const* _fmt = nullptr;
    mutable Chunk const* _dataChunk = nullptr;

    std::vector<uint8_t> _data;
};

}; // end namespace audio
