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

// グローバルな構造体定義などがあるので
#include "gui.h"

class MonguriDemoApp;

class MonguriUserInterface : public donut::app::ImGui_Renderer
{
public:
    MonguriUserInterface(MonguriDemoApp& app);
    void BackBufferResized(const uint32_t width,
        const uint32_t height,
        const uint32_t sampleCount) override;
    void buildUI() override;

    void SetAnimationRange(int2 frameRange, float frameRate);
    void Animate(float elapsedTimeSeconds) override;

    MonguriDemoApp& GetApp() { return m_app; }

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
    MonguriDemoApp& m_app;

    ImFont* m_iconicFont = nullptr;
    ImFont* m_nvidiaRgFont = nullptr;
    ImFont* m_nvidiaBldFont = nullptr;

    ImGuiContext* m_imgui = nullptr;
    ImPlotContext* m_implot = nullptr;
    
    ProfilerGUI m_profiler;

    std::shared_ptr<audio::Engine> m_audioEngine;
    std::unique_ptr<audio::Voice> m_voice;
};