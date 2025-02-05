#pragma once

// clang-format off

#if  defined(AUDIO_ENGINE_WITH_XAUDIO)
    #include <xaudio2.h>
    #include <XAudio2fx.h>
    #define GRAB_VOICE_OBJ(P) (P)
#elif defined(AUDIO_ENGINE_XAUDIO_ON_FAUDIO)
    #include <audio/xaudiofaudio.h>
#endif

#include <memory>
#include <string>
#include <vector>
#include <filesystem>

// clang-format on

namespace donut::vfs
{
    class IFileSystem;
}

namespace audio {

class Engine;
class WaveFile;

//
// Voice
//

class Voice {

public:

    static std::unique_ptr<Voice> create(Engine& engine,
        std::filesystem::path const& wavefile, bool loop=true);

    static std::unique_ptr<Voice> create(Engine& engine,
        std::shared_ptr<WaveFile const> wav, bool loop=true);

    ~Voice();

    void start() { _voice->Start(0); }
    void stop() { _voice->Stop(0); }
    void playOnce() { _voice->Start(0); _voice->ExitLoop(); }
    bool rewind();

    // sets the audio playback starting point (offset in seconds)
    bool setStart(Engine& engine, float offset);

    // returns the number of buffers queued for this voice
    int getBuffersQueued();
    
    // total number of samples played by the voice since creation
    int getSamplesPlayed();

    void setVolume(float volume) { _voice->SetVolume(volume); }
    float getVolume() { float v; _voice->GetVolume(&v); return v; }

    void setPitch(float pitch) { _voice->SetFrequencyRatio(pitch); }
    float getPitch() { float p; _voice->GetFrequencyRatio(&p); return p; }

private:

    bool submitBuffer(Engine& engine, float offset = 0.f);

    Voice() = default;

    IXAudio2SourceVoice * _voice = nullptr;

    XAUDIO2_BUFFER _buffer;

    std::shared_ptr<WaveFile const> _wave;
};

//
// Engine
//

class Engine {

public:

    static std::shared_ptr<Engine> create();

    ~Engine();

    void mute(bool mute);

    static std::weak_ptr<Engine> get();

    bool isReady() const { return _xaudio2 && _masteringVoice; }

    void error(char const * msg);

    static std::vector<std::string> const & getErrors();

private:

    friend class Voice;

    float _masterVolume = 1.f;

    IXAudio2 * _xaudio2 = nullptr;
    IXAudio2MasteringVoice * _masteringVoice = nullptr;
};

}; // end namespace audio

