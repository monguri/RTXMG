//
// Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#pragma once

#include <donut/core/math/math.h>
#include <donut/engine/BindingCache.h>
#include <donut/engine/DescriptorTableManager.h>
#include <donut/engine/SceneGraph.h>
#include <donut/engine/ShaderFactory.h>
#include <donut/engine/TextureCache.h>
#include <donut/engine/View.h>
#include <nvrhi/nvrhi.h>

#include "render_params.h"
#include "blit_params.h"
#include "motion_vectors_params.h"
#include "pixel_debug.h"
#include "render_targets.h"

#include "rtxmg/cluster_builder/cluster_accel_builder.h"
#include "envmap/preprocess_envmap.h"
#include "envmap/scan_system.h"
#include "rtxmg/hiz/zbuffer.h"
#include "rtxmg/scene/camera.h"
#include "rtxmg/utils/buffer.h"


using namespace donut;
using namespace donut::math;

typedef vector<unsigned short, 2> uint16_t2;
typedef vector<unsigned short, 4> uint16_t4;

class RTXMGScene;
class Camera;

#define cStablePlaneCount (3u)

class RTXMGRenderer
{
public:
    struct Options
    {
        RenderParams& params;
        nvrhi::IDevice* device;
    };

    enum class OutputTexture : uint32_t
    {
        DlssOutputColor,
        Accumulation,
        Depth,
        Normals,
        Albedo,
        Specular,
        SpecularHitT,
        Roughness,
        MotionVectors,
#if ENABLE_DUMP_FLOAT
        Debug1,
        Debug2,
        Debug3,
        Debug4,
#endif
        Count
    };

    enum class Output : uint32_t
    {
        DlssOutputColor,
        Accumulation,
        Depth,
        Normals,
        Albedo,
        Specular,
        SpecularHitT,
        Roughness,
        MotionVectors,
#if ENABLE_DUMP_FLOAT
        Debug1,
        Debug2,
        Debug3,
        Debug4,
#endif
        OutputTextureCount,
        InstanceId = OutputTextureCount,
        SurfaceIndex,
        SurfaceUv,
        Texcoord,
        HiZ,
        Count
    };

    // Output is a superset of OutputTexture since we need to be able to visualize both textures/buffers
    static_assert(uint32_t(Output::OutputTextureCount) == uint32_t(OutputTexture::Count));

    RTXMGRenderer(Options const& opts);
    ~RTXMGRenderer();

    void UpdateAccelerationStructures(const TessellatorConfig& tessConfig,
        ClusterStatistics& buildStats,
        uint32_t frameIndex,
        nvrhi::ICommandList* commandList);
    void BuildOrUpdatePipelines();

    void Launch(nvrhi::ICommandList* commandList, uint32_t frameIndex,
        std::shared_ptr<engine::Light> light);
    void BlitFramebuffer(nvrhi::ICommandList* commandList, nvrhi::IFramebuffer* framebuffer);
    void DlssUpscale(nvrhi::ICommandList* commandList, uint32_t frameIndex);

    void ResetSubframes();
    void ResetDenoiser() { m_resetDenoiser = true; }
    void SetRenderCamera(Camera& camera, bool isCameraCut);

    void SetShadingMode(ShadingMode shadingMode)
    {
        m_shadingMode = shadingMode;
        ResetSubframes();
        ResetDenoiser();
    }
    ShadingMode GetShadingMode() const { return m_shadingMode; }
    ShadingMode GetEffectiveShadingMode() const { return m_showMicroTriangles ? ShadingMode::PRIMARY_RAYS : m_shadingMode; }

    void SetColorMode(ColorMode colorMode)
    {
        m_colorMode = colorMode;
        ResetSubframes();
        ResetDenoiser();
    }
    ColorMode GetColorMode() const { return m_colorMode; }

    void SetOutputIndex(Output output)
    {
        m_outputIndex = output;
    }
    Output GetOutputIndex() const { return m_outputIndex; }

    const char* GetOutputLabel(Output output) const
    {
        uint32_t outputIndex = uint32_t(output);
        if (outputIndex < m_outputTextures.size())
        {
            auto& textureHandle = m_outputTextures[outputIndex];
            return textureHandle.Get() ? textureHandle->getDesc().debugName.c_str() : "";
        }
        else
        {
            const char* label = "";
            switch (output)
            {
            case Output::InstanceId:
                label = "Instance Id";
                break;
            case Output::SurfaceIndex:
                label = "Surface Index";
                break;
            case Output::SurfaceUv:
                label = "Surface UV";
                break;
            case Output::Texcoord:
                label = "Texcoord UV";
                break;
            case Output::HiZ:
                label = "HiZ Buffer";
                break;
            default:
                assert(false);
                label = "Unknown";
                break;
            }
            return label;
        }
    }

    nvrhi::TextureHandle GetOutputTexture(Output output) const
    {
        return m_outputTextures[uint32_t(output)];
    }

    void SetSPP(int spp)
    {
        m_params.spp = spp;
        ResetSubframes();
    }
    int GetSPP() const { return m_params.spp; }

    void SetMissColor(float3 missColor)
    {
        m_params.missColor = missColor;
        ResetSubframes();
        ResetDenoiser();
    }
    float3 GetMissColor() const { return m_params.missColor; }

    void SetWireframe(bool wireframe)
    {
        m_params.enableWireframe = wireframe;
        ResetSubframes();
        ResetDenoiser();
    }
    bool GetWireframe() const { return m_params.enableWireframe; }

    void SetShowMicroTriangles(bool showMicroTriangles) { m_showMicroTriangles = showMicroTriangles; }
    bool GetShowMicroTriangles() const { return m_showMicroTriangles; }

    void SetWireframeThickness(float thickness)
    {
        m_params.wireframeThickness = thickness;
        ResetSubframes();
        ResetDenoiser();
    }
    float GetWireframeThickness() const { return m_params.wireframeThickness; }

    int GetPTMaxBounces() const { return m_params.ptMaxBounces; }
    void SetPTMaxBounces(int maxBounces)
    {
        m_params.ptMaxBounces = maxBounces;
        ResetSubframes();
        ResetDenoiser();
    }

    float GetFireflyMaxIntensity() const { return m_params.fireflyMaxIntensity; }
    void SetFireflyMaxIntensity(float fireflyFilterMaxIntensity)
    {
        m_params.fireflyMaxIntensity = fireflyFilterMaxIntensity;
        ResetSubframes();
        ResetDenoiser();
    }

    float GetRoughnessOverride() const { return m_params.roughnessOverride; }
    void  SetRoughnessOverride(float roughness)
    {
        m_params.roughnessOverride = roughness;
        ResetSubframes();
        ResetDenoiser();
    }

    float GetExposure() const
    {
        return m_exposure;
    }
    void SetExposure(float exposure)
    {
        m_exposure = exposure;
        ResetSubframes();
    }

    TonemapOperator GetTonemapOperator() const { return m_tonemapOperator; }
    void SetTonemapOperator(TonemapOperator tonemapOperator)
    {
        m_tonemapOperator = tonemapOperator;
    }

    MvecDisplacement GetMVecDisplacement() const { return m_mvecDisplacement; }
    void SetMvecDisplacement(MvecDisplacement mvecDisplacement)
    {
        m_mvecDisplacement = mvecDisplacement;
    }

    float GetEnvMapAzimuth() const { return m_environmentMapAzimuth; }
    void  SetEnvMapAzimuth(float azimuth)
    {
        m_environmentMapAzimuth = azimuth;
        UpdateEnvMapTransform();
        ResetSubframes();
    }
    float GetEnvMapElevation() const { return m_environmentMapElevation; }
    void  SetEnvMapElevation(float elevation)
    {
        m_environmentMapElevation = elevation;
        UpdateEnvMapTransform();
        ResetSubframes();
    }
    float GetEnvMapIntensity() const { return m_params.envmapIntensity; }
    void  SetEnvMapIntensity(float intensity)
    {
        m_params.envmapIntensity = intensity;
        ResetSubframes();
    }

    float GetDenoiserSeparator() const { return m_denoiserSeparator; }
    void  SetDenoiserSeparator(float separator)
    {
        m_denoiserSeparator = separator;
    }

    bool GetEnableEnvmapHeatmap() const { return m_params.enableEnvmapHeatmap; }
    void  SetEnableEnvmapHeatmap(bool enableEnvmapHeatmap)
    {
        m_params.enableEnvmapHeatmap = enableEnvmapHeatmap;
        ResetSubframes();
        ResetDenoiser();
    }

    void SetTimeView(bool timeView);
    bool GetTimeView() const { return m_params.enableTimeView; }

    void SetDisplayZBuffer(bool displayZBuffer)
    {
        m_displayZBuffer = displayZBuffer;
        m_outputIndex = displayZBuffer ? Output::HiZ : Output::Accumulation;
    }
    bool GetDisplayZBuffer() const { return m_displayZBuffer; }

    int2& GetDebugPixel() { return m_params.debugPixel; }

    void SceneFinishedLoading(std::shared_ptr<RTXMGScene> scene);

    std::shared_ptr<engine::TextureCache> GetTextureCache() const
    {
        return m_textureCache;
    }

    std::shared_ptr<engine::DescriptorTableManager> GetDescriptorTable() const
    {
        return m_descriptorTable;
    }

    std::shared_ptr<engine::ShaderFactory> GetShaderFactory() const
    {
        return m_shaderFactory;
    }

    std::shared_ptr<engine::CommonRenderPasses> GetCommonPasses() const
    {
        return m_commonPasses;
    }

    void DumpPixelDebugBuffers(nvrhi::ICommandList* commandList);

    void SetRenderSize(int2 renderSize, int2 displaySize);
    ZBuffer* GetZBuffer() { return m_zbuffer.get(); }
    const ZBuffer* GetZBuffer() const { return m_zbuffer.get(); }

    nvrhi::rt::AccelStructHandle GetTopLevelAS() const { return m_topLevelAS; }
    std::unique_ptr<ClusterAccels>& GetSceneAccels() { return m_sceneAccels; }
    std::unique_ptr<ClusterAccelBuilder>& GetAccelBuilder() { return m_clusterAccelBuilder; }

    void SetEnvMap(const std::string& filePath, nvrhi::ICommandList* commandList);
    std::shared_ptr<donut::engine::LoadedTexture> GetEnvMap() const { return m_envMap; }
    void ClearEnvMap() { m_envMap = nullptr; }
private:

    void CreateOutputs(nvrhi::ICommandList* commandList);

    nvrhi::IDevice* GetDevice() const { return m_options.device; }

    bool CreateRayTracingPipeline(engine::ShaderFactory& shaderFactory);

    void FillInstanceDescs(nvrhi::ICommandList* commandList, nvrhi::IBuffer* outInstanceDescs, nvrhi::IBuffer* blasAddresses, uint32_t numInstances);

    void CreateAccelStructs();

    void UpdateEnvMapTransform();
    void UpdateEnvMapSampling(nvrhi::ICommandList* commandList);

    void ComputeMotionVectors(nvrhi::ICommandList* commandList);
private:
    

    Options m_options;
    RenderParams& m_params;
    ShadingMode m_shadingMode = ShadingMode::PT;
    ColorMode m_colorMode = ColorMode::BASE_COLOR;
    TonemapOperator m_tonemapOperator = TonemapOperator::Aces;
    float m_exposure = 1.0f;

    PreprocessEnvMapShaders m_preprocessEnvMapShaders;
    PreprocessEnvMapResources m_preprocessEnvMapResources;
    std::shared_ptr<donut::engine::LoadedTexture> m_envMap;
    ScanSystem m_scanSystem;
    float m_environmentMapAzimuth = 0.f;
    float m_environmentMapElevation = 0.f;

    Camera m_camera;
    Camera m_previousCamera;

    int2 m_renderSize = int2(0, 0);
    int2 m_displaySize = int2(0, 0);

    nvrhi::rt::PipelineHandle m_rayPipeline;
    nvrhi::rt::ShaderTableHandle m_shaderTable;
    nvrhi::BindingLayoutHandle m_bindingLayout;
    nvrhi::BindingSetHandle m_bindingSet;
    nvrhi::BindingLayoutHandle m_bindlessLayout;

    nvrhi::BufferHandle m_dummyBuffer;

    nvrhi::ComputePipelineHandle m_blitPipeline;
    nvrhi::BindingLayoutHandle m_blitBL;
    nvrhi::BufferHandle m_blitParamsBuffer;

    std::unique_ptr<ClusterAccelBuilder> m_clusterAccelBuilder;
    RTXMGBuffer<nvrhi::rt::InstanceDesc> m_instanceDescs;
    nvrhi::rt::AccelStructHandle m_topLevelAS;
    std::unique_ptr<ClusterAccels> m_sceneAccels;

    nvrhi::BufferHandle m_fillInstanceDescsParams;
    nvrhi::BindingLayoutHandle m_fillInstanceDescsBL;
    nvrhi::ComputePipelineHandle m_fillInstanceDescsPSO;

    nvrhi::BufferHandle m_lightingConstantsBuffer;
    nvrhi::BufferHandle m_renderParamsBuffer;

    std::shared_ptr<engine::DescriptorTableManager> m_descriptorTable;

    // Render Textures (RenderSize)
    std::array<nvrhi::TextureHandle, size_t(OutputTexture::Count)> m_outputTextures;
    Output m_outputIndex = Output::Accumulation;
    nvrhi::BufferHandle m_hitResultBuffer;

    // Display Textures (DisplayRes)
    nvrhi::TextureHandle m_dlssOutputColorTexture; // Upscaled color output.
    nvrhi::TextureHandle m_displayTexture; // Can be eliminated when blit is converted to PS

    float m_denoiserSeparator = 0.0f;
    bool m_resetDenoiser = false;
    bool m_showMicroTriangles = false;

    // Motion Vectors
    MvecDisplacement m_mvecDisplacement = MvecDisplacement::FromSubdEval;
    RTXMGBuffer<SubdInstance> m_subdInstancesBuffer;
    nvrhi::BindingLayoutHandle m_motionVectorsBL;
    nvrhi::ComputePipelineHandle m_motionVectorsPSO[size_t(MvecDisplacement::Count)];

    // Debug Buffers
#if ENABLE_PIXEL_DEBUG
    RTXMGBuffer<PixelDebugElement> m_pixelDebugBuffer;
    RTXMGBuffer<PixelDebugElement> m_motionVectorsPixelDebugBuffer;
#endif
    nvrhi::BufferHandle m_timeViewBuffer;

    std::unique_ptr<engine::BindingCache> m_bindingCache;
    std::shared_ptr<engine::TextureCache> m_textureCache;
    std::shared_ptr<RTXMGScene> m_scene;

    nvrhi::ShaderLibraryHandle m_shaderLibrary;

    std::shared_ptr<engine::ShaderFactory> m_shaderFactory;
    std::shared_ptr<engine::CommonRenderPasses> m_commonPasses;

    engine::PlanarView m_view;
    engine::PlanarView m_viewPrevious;

    bool m_pipelinesNeedsUpdate = true;
    bool m_needsRebind = true;
    bool m_displayZBuffer = false;
    std::unique_ptr<ZBuffer> m_zbuffer;
};
