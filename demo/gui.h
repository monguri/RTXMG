/*
 * Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#pragma once

#include <filesystem>
#include <map>

#include <donut/app/imgui_renderer.h>
#include <donut/app/StreamlineInterface.h>
#include <donut/core/math/math.h>
#include <donut/engine/TextureCache.h>

#include "implot.h"
#include "rtxmg_demo.h"

#include "rtxmg/profiler/gui.h"

using namespace donut::math;

namespace fs = std::filesystem;

class RTXMGDemoApp;
struct GLFWindow;

#ifdef AUDIO_ENGINE_ENABLED
#include <audio/audio.h>
#include <audio/waveFile.h>
#else
namespace audio
{
    class Engine {};
    class Voice {};
}
#endif

struct MediaAsset
{
    enum class Type : uint8_t
    {
        OBJ_FILE = 0,
        OBJ_SEQUENCE
    } type = Type::OBJ_FILE;

    char const* name = nullptr;

    std::string sequenceName; // decorated sequence name to display in GUI
    std::string
        sequenceFormat; // format string to generate paths to individual files
    int padding =
        0; // number of digits in sequence numbers (or 0 if no padding detected)

    std::string wavePath;
    float waveStartTime = 0.f; // offset on audio start time

    int2 frameRange = { std::numeric_limits<int>::max(),
                       std::numeric_limits<int>::min() };
    float frameRate = 24.f;

    bool IsSequence() const { return type == Type::OBJ_SEQUENCE; }

    char const* GetName() const
    {
        return IsSequence() ? sequenceName.c_str() : name;
    }

    void GrowFrameRange(int frame)
    {
        frameRange.x = std::min(frame, frameRange.x);
        frameRange.y = std::max(frame, frameRange.y);
    };
};

typedef std::map<std::string, MediaAsset> MediaAssetsMap;

struct TimeLineEditorState
{
    template <typename T> constexpr T clamp(T value, T lower, T upper)
    {
        return std::min(std::max(value, lower), upper);
    }

    enum class Playback : uint8_t { Pause = 0, Play } mode = Playback::Pause;
    bool loop = true;
    int2 frameRange = { 0, 0 };
    float frameRate = 30.f;
    float startTime = 0.f;
    float endTime = 0.f;
    float currentTime = 0.f;

    std::function<void(TimeLineEditorState const&)> playCallback;
    std::function<void(TimeLineEditorState const&)> pauseCallback;
    std::function<void(TimeLineEditorState const&)> setTimeCallback;

    void Update(float elapsedTime);
    float AnimationTime() const { return currentTime - startTime; }

    // programmatic manipulation
    inline bool IsPlaying() const { return mode == Playback::Play; }
    inline bool IsPaused() const { return mode == Playback::Pause; }

    inline void SetFrame(float time)
    {
        currentTime = clamp(time / frameRate, startTime, endTime);
    }
    inline void StepForward()
    {
        currentTime = clamp(currentTime + 1.f / frameRate, startTime, endTime);
        if (setTimeCallback)
            setTimeCallback(*this);
    }
    inline void StepBackward()
    {
        currentTime = clamp(currentTime - 1.f / frameRate, startTime, endTime);
        if (setTimeCallback)
            setTimeCallback(*this);
    }
    inline void Rewind()
    {
        currentTime = startTime;
        if (setTimeCallback)
            setTimeCallback(*this);
    }
    inline void FastForward()
    {
        currentTime = endTime;
        if (setTimeCallback)
            setTimeCallback(*this);
    }

    inline void PlayClicked()
    {
        bool paused = IsPaused();

        if (paused && playCallback)
            playCallback(*this);
        else if (pauseCallback)
            pauseCallback(*this);

        mode = paused ? TimeLineEditorState::Playback::Play
            : TimeLineEditorState::Playback::Pause;
    }
};

struct UIData
{
    bool showUI = true;

    static const char* WindowTitle;

    // path to imgui.ini settings file (auto-saved by imgui)
    std::string iniFilepath;

    MediaAssetsMap mediaAssets;

    MediaAsset const* currentAsset = nullptr;

    void SelectCurrentAsset(const std::string& name)
    {
        currentAsset = nullptr;
        if (name.empty())
            return;
        if (auto it = mediaAssets.find(name); it != mediaAssets.end())
            currentAsset = &it->second;
    }

    bool audioMuted = false;
    float audioStartTime = 0.f;

    // various filters for the scene selector
    bool includeJsonAssets = true;
    bool includeObjAssets = true;

    std::array<char const*, 4> formatFilters() const;

    std::array<char const*, 5> folderFilters() const;

    bool showTimeLineEditor = true;
    bool showRecommendationWindow = false;
    bool showHelpWindow = true;
    bool showLodInspector = false;
    bool forceRebuildAccelStruct = true;
    bool enableMonolithicClusterBuild = false;
    
    TimeLineEditorState timeLineEditorState;

    std::shared_ptr<donut::engine::LoadedTexture> envmap = nullptr;
    std::string envmapFilepath = "";

    // DLSS
#if DONUT_WITH_STREAMLINE
    using StreamlineInterface = donut::app::StreamlineInterface;
    StreamlineInterface::DLSSMode dlssMode = StreamlineInterface::DLSSMode::eMaxQuality;
    bool dlssUseLodBiasOverride = false;
    float dlssLodBiasOverride = 0.f;
    StreamlineInterface::DLSSPreset dlssPreset = StreamlineInterface::DLSSPreset::eDefault;
    StreamlineInterface::DLSSRRPreset dlssRRPreset = StreamlineInterface::DLSSRRPreset::eDefault;
#endif
};

class UserInterface : public donut::app::ImGui_Renderer
{
public:
    UserInterface(RTXMGDemoApp& app);
    void BackBufferResized(const uint32_t width,
        const uint32_t height,
        const uint32_t sampleCount) override;
    void buildUI() override;

    void SetAnimationRange(int2 frameRange, float frameRate);
    void Animate(float elapsedTimeSeconds) override;

    RTXMGDemoApp& GetApp() { return m_app; }

    ImFont* GetIconicFont() { return m_iconicFont; }

    ImGuiContext* GetImGuiContext() const { return m_imgui; }
    ImPlotContext* GetImPlotContext() const { return m_implot; }

    ProfilerGUI& GetProfilerGUI() { return m_profiler; }

    bool CustomInit(std::shared_ptr<donut::engine::ShaderFactory> shaderFactory);

private:
    void BuildMemoryWarning(int2 windowSize);
    void BuildUIMain(int2 windowSize);
    void BuildUITimeline(int2 windowSize, float timeline_width);
    void BuildUIEnvmap(ImVec2 itemSize);

    void BuildMissColorUI();

    bool BuildTimeLineEditor(TimeLineEditorState& state, float2 size);

    void SetupAudioVoice(const std::string& wavepath, float startTime = 0.f);
    void SetupAudioEngine();
    void MuteAudio(bool mute);

    void LoadAsset(MediaAsset const& asset, std::string const& name,
        int2 frameRange);

    static MediaAssetsMap FindMediaAssets(fs::path const& mediapath,
        char const* const* folder_filters,
        char const* const* format_filters);
    // 'iconic' open-source TTF font (lots of standard icons to make buttons
    // with)

    static constexpr float const iconicFontSize = 18.f;
    static uint16_t const* GetOpenIconicFontGlyphRange();
    static char const* GetNVSansFontRgCompressedBase85TTF();
    static char const* GetNVSansFontBoldCompressedBase85TTF();
    static char const* GetOpenIconicFontCompressedBase85TTF();

    ImFont* AddFontFromMemoryCompressedBase85TTF(const char* data, float fontSize,
        const uint16_t* range);
    bool FolderDialog(std::string& m_filepath);
    bool FileDialog(bool bOpen, char const* filters, std::string& m_filepath);

    void SetupIniHandler();

private:
    RTXMGDemoApp& m_app;

    ImFont* m_iconicFont = nullptr;
    ImFont* m_nvidiaRgFont = nullptr;
    ImFont* m_nvidiaBldFont = nullptr;

    ImGuiContext* m_imgui = nullptr;
    ImPlotContext* m_implot = nullptr;
    
    ProfilerGUI m_profiler;

    std::shared_ptr<audio::Engine> m_audioEngine;
    std::unique_ptr<audio::Voice> m_voice;
};