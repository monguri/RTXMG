// clang-format off

#include "./waveFile.h"

#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>

namespace fs = std::filesystem;

static std::vector<uint8_t> readFile(fs::path const& m_filepath) {
    
    std::ifstream file(m_filepath.generic_string().c_str(), std::ios::binary);

    if (!file.is_open())
        return {};

    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<uint8_t> data(size);

    file.read((char*)data.data(), size);

    if (!file.good())
        return {};
    
    return data;
}

// clang-format on
namespace audio {

struct WaveFile::Chunk{
    char      id[4]; // "data"  string
    uint32_t  size;  // Sampled data length
    
    inline uint8_t const* data() const {
        return reinterpret_cast<uint8_t const*>(this) + sizeof(Chunk);
    }
};

struct WaveFile::Header {
    char      id[4];   // "data"  string
    uint32_t  size;    // Sampled data length
    char      wave[4]; // WAVE signature
    inline bool validate() const {
        if (std::memcmp(id, "RIFF", 4) != 0)
            return false;
        if (std::memcmp(wave, "WAVE", 4) != 0)
            return false;
        return true;
    }
};

WaveFile::Chunk const* WaveFile::getSubChunk(std::vector<uint8_t> const& data, char const* name) {

    // some WAV files have additional undocumented sub-chunks within the header ;
    // this searches for a specific sub-chunk by name.

    for (uint8_t const* ptr = data.data() + sizeof(Header); ptr < (data.data() + data.size()); ) {

        Chunk const* chunk = reinterpret_cast<Chunk const*>(ptr);

        if (std::memcmp(chunk->id, name, 4) == 0)
            return chunk;

        ptr += sizeof(Chunk) + chunk->size;
    }
    return nullptr;
}

std::unique_ptr<WaveFile> WaveFile::read(fs::path const& m_filepath) {
    auto wave = create(std::move(readFile(m_filepath)));   
    if (wave)
        wave->m_filepath = m_filepath;
    return wave;
}

std::unique_ptr<WaveFile> WaveFile::create(std::vector<uint8_t>&& data) {

    // see: http://tiny.systems/software/soundProgrammer/WavFormatDocs.pdf

    if (data.empty())
        return nullptr;

    // header chunk
    Header const* header = reinterpret_cast<Header const*>(data.data());
    if (!header->validate())
        return nullptr;

    // format sub-chunk
    Fmt const* fmt = nullptr;
    if (Chunk const* chunk = getSubChunk(data, "fmt "))
        fmt = reinterpret_cast<Fmt const*>(chunk->data());
    else
        return nullptr;
    if (fmt->audioFormat != 1)
        return nullptr;

    // data sub-chunk
    Chunk const* dataChunk = getSubChunk(data, "data");
    if (!dataChunk)
        return nullptr;

    auto wave = new WaveFile;
    wave->_fmt = fmt;
    wave->_dataChunk = dataChunk;
    wave->_data = std::move(data);
    return std::unique_ptr<WaveFile>(wave);
}

WaveFile::Fmt const& WaveFile::getFormat() const {
    return *_fmt;
}

uint32_t WaveFile::getSamplesDataSize() const {
    return _dataChunk->size;
}

void const* WaveFile::getSamplesData() const {
    return _dataChunk->data();
}

uint32_t WaveFile::getNumSamplesTotal() const {
    return _dataChunk->size / (_fmt->numChannels * _fmt->bitsPerSample / 8);
}

float WaveFile::duration() const {
    return float(getNumSamplesTotal() / _fmt->samplesPerSec);
}

std::unique_ptr<WaveForm> WaveFile::computeWaveform(
    uint32_t size, float start_time, float end_time, int channel) const {

    uint32_t start_index = static_cast<uint32_t>( std::max( 0.f, start_time * static_cast<float>( _fmt->samplesPerSec ) ) );
    uint32_t end_index = end_time < 0.f ? getNumSamplesTotal() :
                                          static_cast<uint32_t>( end_time * static_cast<float>( _fmt->samplesPerSec ) );

    start_index = std::min(start_index, getNumSamplesTotal());
    end_index = std::min(end_index, getNumSamplesTotal());

    if (start_index >= end_index)
        return nullptr;

    auto process = [this, &size]<typename T>(uint32_t start, uint32_t end, int channel) {
        assert(end > start);

        T const* ptr = reinterpret_cast<T const*>(_data.data());
        T const* data_end = reinterpret_cast<T const*>(_data.data() + _data.size());

        uint32_t nsamples = end - start;
        uint16_t nchannels = _fmt->numChannels;

        uint32_t block_count = size;
        uint32_t block_size =
            static_cast<uint32_t>( std::ceil( float( nsamples ) / float( 2 * static_cast<float>( block_count ) ) ) );

        auto wform = std::make_unique<WaveForm>();
        wform->maxs.resize(block_count);
        wform->mins.resize(block_count);

        for (uint32_t block = 0; (block < block_count) && (ptr < data_end); ++block) {

            float min = std::numeric_limits<T>::max();
            float max = std::numeric_limits<T>::min();

            for (uint32_t sample = 0; (sample < block_size) && (ptr < data_end); ++sample) {
                if (channel == -1) {
                    for (uint16_t channel = 0; channel < nchannels; ++channel) {
                        min = std::min(min, float(*ptr++) / std::numeric_limits<T>::max());
                        max = std::max(max, float(*ptr++) / std::numeric_limits<T>::max());
                    }
                } else {
                    T const* channelPtr = ptr + (channel * 2);
                    min = std::min(min, float(*channelPtr++) / std::numeric_limits<T>::max());
                    max = std::max(max, float(*channelPtr) / std::numeric_limits<T>::max());
                    ptr += nchannels * 2;
                }
            }
            wform->mins[block] = min;
            wform->maxs[block] = max;
        }
        return wform;
    };

    switch (_fmt->bitsPerSample) {
        case 8: return process.template operator()<int8_t>(start_index, end_index, channel);
        case 16: return process.template operator()<int16_t>(start_index, end_index, channel);
    }
    return nullptr;
}

} // end namespace audio
