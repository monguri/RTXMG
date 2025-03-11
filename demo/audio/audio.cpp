// clang-format off
#include "./audio.h"
#include "./waveFile.h"

#include <cstring>

// clang-format on

#ifndef SAFE_RELEASE
    #define SAFE_RELEASE(x) \
       if(x != NULL)        \
       {                    \
          x->Release();     \
          x = NULL;         \
       }
#endif

namespace audio {

static std::vector<std::string> _errors;

std::unique_ptr<Voice> Voice::create(Engine& engine, std::filesystem::path const& m_filepath, bool loop) {

    if (!engine.isReady())
        return nullptr;

    if (auto wave = WaveFile::read(m_filepath))
        return Voice::create(engine, std::move(wave), loop);
    else
        engine.error((std::string("cannpt read: ")+ m_filepath.generic_string()).c_str());
    return nullptr;
}

std::unique_ptr<Voice> Voice::create(Engine& engine, std::shared_ptr<WaveFile const> wave, bool loop) 
{
    if (!engine._xaudio2)
        return nullptr;

    WaveFile::Fmt const& fmt = wave->getFormat();

    IXAudio2SourceVoice * voice = nullptr;

    WAVEFORMATEX wfmtx;
    wfmtx.wFormatTag = WAVE_FORMAT_PCM;
    wfmtx.nChannels = fmt.numChannels;
    wfmtx.nSamplesPerSec = fmt.samplesPerSec;
    wfmtx.nAvgBytesPerSec = fmt.bytesPerSec;
    wfmtx.nBlockAlign = fmt.blockAlign;
    wfmtx.wBitsPerSample = fmt.bitsPerSample;
    wfmtx.cbSize = 0;

    XAUDIO2_SEND_DESCRIPTOR sendDescriptors[1];
    sendDescriptors[0].Flags = XAUDIO2_SEND_USEFILTER;
    sendDescriptors[0].pOutputVoice = GRAB_VOICE_OBJ(engine._masteringVoice);
    const XAUDIO2_VOICE_SENDS sendList = { 1, sendDescriptors };

    HRESULT hr;

    hr = engine._xaudio2->CreateSourceVoice(&voice, &wfmtx, 0, 2.0f, NULL, &sendList);
    if (FAILED(hr)) {
        engine.error((std::string("error CreateSourceVoice : ")+wave->m_filepath.generic_string()).c_str());
        return nullptr;
    }

    std::unique_ptr<Voice> result(new Voice);
    result->_voice = voice;
    result->_wave = std::move(wave);

    std::memset(&result->_buffer, 0, sizeof(XAUDIO2_BUFFER));
    result->_buffer.pAudioData = (BYTE const *)result->_wave->getSamplesData();
    result->_buffer.Flags = XAUDIO2_END_OF_STREAM;
    result->_buffer.AudioBytes = result->_wave->getSamplesDataSize();
    result->_buffer.LoopCount = loop ? XAUDIO2_LOOP_INFINITE : XAUDIO2_NO_LOOP_REGION;

    if (result->submitBuffer(engine))
        return result;
    return nullptr;
}

Voice::~Voice() {
    if (_voice)
        _voice->DestroyVoice();
}

bool Voice::submitBuffer(Engine& engine, float offset) {
    _voice->Stop();
    WaveFile::Fmt const& fmt = _wave->getFormat();
    if (offset > 0.f)
        _buffer.PlayBegin = _buffer.LoopBegin = uint32_t(offset * float(fmt.samplesPerSec));
    if (FAILED(_voice->SubmitSourceBuffer(&_buffer))) {
        engine.error("SubmitSourceBuffer failed");
        return false;
    }
    return true;
}

bool Voice::setStart(Engine& engine, float offset)
{
    if (offset < 0.f)
        return false;
     _voice->FlushSourceBuffers();
    return submitBuffer(engine, offset);
}

bool Voice::rewind() {
    _voice->Stop();
    _voice->FlushSourceBuffers();
    return _voice->SubmitSourceBuffer(&_buffer)==S_OK;
}

int Voice::getBuffersQueued() {
    XAUDIO2_VOICE_STATE xstate;
    _voice->GetState(&xstate, 0);
    return xstate.BuffersQueued;
}

int Voice::getSamplesPlayed() {
    XAUDIO2_VOICE_STATE xstate;
    _voice->GetState(&xstate, 0);
    return (int)xstate.SamplesPlayed;
}

//
//
//

Engine::~Engine() {
    if (_masteringVoice)
        _masteringVoice->DestroyVoice();
    if (_xaudio2)
        _xaudio2->Release();
}

std::shared_ptr<Engine> Engine::create() {

    static std::shared_ptr<Engine> _engine;

    if (_engine && _engine->isReady()) {
        return _engine;
    }

    if (!_engine) {

        _engine = std::make_shared<Engine>(*new Engine);

        HRESULT hr;
        if (FAILED(hr = CoInitializeEx(NULL, COINIT_MULTITHREADED))) {
            _errors.push_back("Error initializing multi-threaded mode");
            return _engine;
        }

        if (FAILED(hr = XAudio2Create(&_engine->_xaudio2, 0))) {
            _errors.push_back("XAudio2Create failed");
            return _engine;
        }

        if (FAILED(hr = _engine->_xaudio2->CreateMasteringVoice(&_engine->_masteringVoice))) {
            SAFE_RELEASE(_engine->_xaudio2);
            _errors.push_back("CreateMasteringVoice failed");
        }
    }
    return _engine;
}

void Engine::mute(bool mute) {
    
    if (!isReady())
        return;
    
    if (mute) {
        _masteringVoice->GetVolume(&_masterVolume);
        _masteringVoice->SetVolume(0.f);
    } else {
        _masteringVoice->SetVolume(_masterVolume);
        _masterVolume = -1.f;
    }
}

std::vector<std::string> const & Engine::getErrors() {
    return _errors;
}

void Engine::error(char const * msg) {
    _errors.push_back(msg);
}

};
