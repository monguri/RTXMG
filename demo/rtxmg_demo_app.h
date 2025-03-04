#pragma once

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

#include <array>
#include <donut/app/ApplicationBase.h>
#include <donut/app/Camera.h>
#include <donut/app/DeviceManager.h>
#include <donut/core/math/quat.h>
#include <donut/core/vfs/VFS.h>
#include <donut/engine/BindingCache.h>
#include <donut/engine/DescriptorTableManager.h>
#include <donut/engine/Scene.h>
#include <donut/engine/ShaderFactory.h>
#include <donut/engine/View.h>

#include "args.h"
#include "rtxmg_demo.h"
#include "rtxmg_renderer.h"
#include "gui.h"
#include "zrenderer.h"
#include "trackball.h"

using namespace donut::math;

#include "render_params.h"

#include "rtxmg/scene/camera.h"
#include "rtxmg/scene/scene.h"

class UserInterface;

class RTXMGDemoApp : public donut::app::ApplicationBase
{
public:
    struct WindowState
    {
        donut::math::int2 windowSize = { 0,0 };
        bool isMaximized = false;
        bool isFullscreen = false;
    };

private:
    std::shared_ptr<RTXMGScene> m_scene;
    Camera m_camera;
    Camera m_prevCamera;
    bool m_cameraReset = false;
    Camera m_tesselationCamera;
    Trackball m_trackBall;
    
    std::shared_ptr<donut::engine::DirectionalLight> m_sunLight;
    nvrhi::CommandListHandle m_commandList;

    // path to executable (because argv0 is not reliable)
    std::filesystem::path m_binaryPath;

    // path to local folder w/ media assets
    std::filesystem::path m_mediaPath;

    UIData m_ui;
    Args m_args;
    RenderParams m_renderParams;
    int2 m_displaySize;
    int2 m_renderSize;
    float m_lodBias = 1.0f;
    DenoiserMode m_denoiserMode = DenoiserMode::None; // Desired denoiser mode, but m_renderParams contains effective denoiser mode

    int2 m_loadFrameRange = { std::numeric_limits<int>::max(), std::numeric_limits<int>::min() };
    std::chrono::steady_clock::time_point m_currFrameStart = {};
    std::chrono::steady_clock::time_point m_prevFrameStart = {};
    float m_animationTime = 0.0f;
    bool m_animationLooped = false;
    bool m_animationUpdated = false;

    std::unique_ptr<RTXMGRenderer> m_renderer;
    std::unique_ptr<ZRenderer> m_zRenderer;

    nvrhi::BufferHandle m_lerpKeyFramesParamsBuffer;
    nvrhi::BindingLayoutHandle m_lerpVerticesBL;
    nvrhi::ComputePipelineHandle m_lerpVerticesPSO;

    bool m_accelBuilderNeedsUpdate = true;
    bool m_dumpFineTess = false;
    bool m_screenshot = false;
    bool m_dumpDebugBuffer = false;
    bool m_dumpPixelDebug = false;

    // DLSS State
#if DONUT_WITH_STREAMLINE
    using StreamlineInterface = donut::app::StreamlineInterface;
    StreamlineInterface::DLSSPreset m_dlssLastPreset = StreamlineInterface::DLSSPreset::eDefault;
    StreamlineInterface::DLSSMode m_dlssLastMode = StreamlineInterface::DLSSMode::eOff;
    int2 m_dlssLastDisplaySize;

    StreamlineInterface::DLSSSettings m_RecommendedDLSSSettings;
    StreamlineInterface::DLSSRRSettings m_RecommendedDLSSRRSettings;
#endif

    int m_argc;
    const char** m_argv;

    UserInterface* m_gui;
    WindowState m_windowState;

    struct MessageCallback : public nvrhi::IMessageCallback
    {
        explicit MessageCallback(donut::app::DeviceManager* deviceManager)
            : m_deviceManager(deviceManager)
        {}
        const donut::app::DeviceManager* m_deviceManager;
        void message(nvrhi::MessageSeverity severity, const char* messageText) override;
    };
    MessageCallback m_messageCallback;

    void UpdateParams();
    void UpdateDLSSSettings();

    void DoSaveScreenshot(nvrhi::ITexture* framebufferTexture, std::string const& filename = "");
    void DoDumpFineTess(std::string const& filename = "");
    void DoDumpDebugBuffer(std::string const& filename = "");
    
    void LerpVertices(nvrhi::IBuffer* outBuffer,
        nvrhi::IBuffer* keyFrame0Buffer,
        nvrhi::IBuffer* keyFrame1Buffer,
        unsigned int numVertices, float animTime);
    void DispatchGPUAnimation();

public:
    // AppBase Overrides
    bool LoadScene(std::shared_ptr<donut::vfs::IFileSystem> fs,
        const std::filesystem::path& sceneFileName) override;

    bool KeyboardUpdate(int key, int scancode, int action, int mods) override;
    bool MousePosUpdate(double xpos, double ypos) override;
    bool MouseButtonUpdate(int button, int action, int mods) override;
    bool MouseScrollUpdate(double xoffset, double yoffset) override;

    void BackBufferResizing() override;
    void BackBufferResized(const uint32_t width, const uint32_t height,
        const uint32_t sampleCount) override;
    void Animate(float fElapsedTimeSeconds) override;
    void Render(nvrhi::IFramebuffer* framebuffer) override;

public:
    RTXMGDemoApp(donut::app::DeviceManager* deviceManager, int argc,
        const char** argv);
    virtual ~RTXMGDemoApp();

    bool Init();
    void ResetCamera();
    bool SetEnvmapTex(const std::string& filePath);

    void HandleSceneLoad(std::string const& m_filepath,
        std::string const& mediapathm, int2 frameRange = { std::numeric_limits<int>::max(), std::numeric_limits<int>::min() });
    const RTXMGScene& GetScene() const { return *m_scene; }
    RTXMGRenderer& GetRenderer();

    float GetCPUFrameTime() const;

    ///////////////////////////////////////////////////////
    // GUI access
    ///////////////////////////////////////////////////////
    void SetGui(UserInterface* gui) { m_gui = gui; }
    const std::filesystem::path& GetBinaryPath() const { return m_binaryPath; }
    const std::filesystem::path& GetMediaPath() const { return m_mediaPath; }
    void SetMediaPath(const std::filesystem::path& path) { m_mediaPath = path; }
    UIData& GetUIData() { return m_ui; }

    WindowState GetWindowState() const { return m_windowState; }
    void SetWindowState(const WindowState& state);

    void DumpFineTess();
    void SaveScreenshot();
    void DumpDebugBuffer();

    TessellatorConfig::MemorySettings GetTessMemSettings() const { return m_args.tessMemorySettings; }
    void SetTessMemSettings(const TessellatorConfig::MemorySettings& settings);

    bool GetUpdateTessellationCamera() const { return m_args.updateTessCamera; }
    void SetUpdateTessellationCamera(bool update);

    float GetFineTessellationRate() const { return m_args.fineTessellationRate; }
    void  SetFineTessellationRate(float rate);

    float GetCoarseTessellationRate() const { return m_args.coarseTessellationRate; }
    void  SetCoarseTessellationRate(float rate);

    int GetIsolationLevelSharp() const { return m_args.isoLevelSharp; }
    int GetIsolationLevelSmooth() const { return m_args.isoLevelSmooth; }

    TessellatorConfig::VisibilityMode GetTessellatorVisibilityMode() const { return m_args.visMode; }
    void                              SetTessellatorVisibilityMode(TessellatorConfig::VisibilityMode visMode);

    bool GetBackfaceVisibilityEnabled() const { return m_args.enableBackfaceVisibility; }
    void SetBackfaceVisibilityEnabled(bool enabled);

    int GetQuantizationBits() const { return m_args.quantNBits; }
    void SetQuantizationBits(int quantNBits);

    void SetDisplacementScale(float scale);
    float GetDisplacementScale() const { return m_renderParams.globalDisplacementScale; }

    ClusterPattern GetClusterTessellationPattern() const
    {
        return (ClusterPattern)m_renderParams.clusterPattern;
    }
    void SetClusterTessellationPattern(ClusterPattern clusterPattern);

    TessellatorConfig::AdaptiveTessellationMode GetAdaptiveTessellationMode() const { return m_args.tessMode; }
    void SetAdaptiveTessellationMode(TessellatorConfig::AdaptiveTessellationMode mode);

    bool GetFrustumVisibilityEnabled() const { return m_args.enableFrustumVisibility; }
    void SetFrustumVisibilityEnabled(bool enabled);

    bool GetHiZVisibilityEnabled() const { return m_args.enableHiZVisibility; }
    void SetHiZVisibilityEnabled(bool enabled);

    bool GetAccelBuildLoggingEnabled() const { return m_args.enableAccelBuildLogging; }
    void SetAccelBuildLoggingEnabled(bool enabled) { m_args.enableAccelBuildLogging = enabled; }

    // Desired denoiser mode
    void SetDenoiserMode(DenoiserMode denoiserMode);
    DenoiserMode GetDenoiserMode() const { return m_denoiserMode; }
    DenoiserMode GetEffectiveDenoiserMode() const { return m_renderParams.denoiserMode; }

    void NextShadingMode()
    {
        auto& renderer = GetRenderer();
        if (renderer.GetShowMicroTriangles())
            return;

        ShadingMode shadingMode = ShadingMode((int(renderer.GetShadingMode()) + 1) % ShadingMode::SHADING_MODE_COUNT);
        renderer.SetShadingMode(shadingMode);
    }

    void NextTonemapper()
    {
        auto& renderer = GetRenderer();
        if (renderer.GetShowMicroTriangles())
            return;

        TonemapOperator op = TonemapOperator((int(renderer.GetTonemapOperator()) + 1) % TonemapOperator::Count);
        renderer.SetTonemapOperator(op);
    }

    void IncrementMaxBounces(int delta)
    {
        auto& renderer = GetRenderer();
        if (renderer.GetEffectiveShadingMode() == ShadingMode::PT)
        {
            int maxBounces = std::min(renderer.GetPTMaxBounces() + delta, 10);
            renderer.SetPTMaxBounces(maxBounces);
        }
    }

    void ToggleWireframe()
    {
        auto& renderer = GetRenderer();
        if (renderer.GetShowMicroTriangles())
            return;

        bool wireframe = !renderer.GetWireframe();
        renderer.SetWireframe(wireframe);
    }

    void IncrementColorMode(int delta)
    {
        auto& renderer = GetRenderer();
        if (renderer.GetShowMicroTriangles())
            return;

        ColorMode colorMode = ColorMode(((int)renderer.GetColorMode() + ColorMode::COLOR_MODE_COUNT + delta) % ColorMode::COLOR_MODE_COUNT);
        renderer.SetColorMode(colorMode);
    }

    void ToggleTimeView()
    {
        auto& renderer = GetRenderer();
        if (renderer.GetEffectiveShadingMode() == ShadingMode::PT)
        {
            bool timeView = !renderer.GetTimeView();
            renderer.SetTimeView(timeView);
        }
    }

    void ToggleUpdateTessellationCamera()
    {
        SetUpdateTessellationCamera(!GetUpdateTessellationCamera());
    }

    ClusterStatistics m_BuildStats;
};