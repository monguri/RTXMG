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

#ifndef RENDER_PARAMS_H // using instead of "#pragma once" due to https://github.com/microsoft/DirectXShaderCompiler/issues/3943
#define RENDER_PARAMS_H

#include "rtxmg_demo.h"

#ifdef __cplusplus
#include <donut/core/math/math.h>
using namespace donut::math;
#endif

#define ENABLE_DUMP_FLOAT 0

#define RTXMG_NVAPI_SHADER_EXT_SLOT 1000

static const uint32_t kInvalidBindlessIndex = ~0u;

// Camera struct for constant bfuffer
struct CameraConstants
{
    float4x4 view;
    float4x4 viewInv;
    float4x4 proj;
    float4x4 projInv;
    float2   dims;
    float2   dimsInv;

#ifndef __cplusplus
    float3 unprojectPixelToWorld_lineardepth(float2 pixel, float z)
    {
        float4 ndcPos = float4(
            pixel.x * dimsInv.x * 2.f - 1.f,
            (1.0f - pixel.y * dimsInv.y) * 2.f - 1.f,
            0.f, 1.f);

        float4 clipPos = ndcPos * z;
        float4 viewPos = mul(projInv, clipPos);
        float4 worldPos = mul(viewInv, float4(viewPos.xyz, 1.0f));

        return worldPos.xyz;
    }

    float3 unprojectPixelToWorld_hwdepth(float2 pixel, float zNDC)
    {
        // zNDC --> zCam
        float  A = proj[2][2]; //[10];
        float  B = proj[2][3]; //[11];
        float  C = proj[3][2]; //[14];
        float  zLinear = -B / (C * zNDC - A);

        return unprojectPixelToWorld_lineardepth(pixel, zLinear);
    }

    float3 unprojectPixelToWorldDirection(float2 pixel)
    {
        float4 ndcPos = float4(
            pixel.x * dimsInv.x * 2.f - 1.f,
            (1.0f - pixel.y * dimsInv.y) * 2.f - 1.f,
            0.f,
            1.f);

        // Position on near plane
        float4 clipPos = ndcPos;
        float4 viewPos = mul(projInv, clipPos);
        float4 worldDir = mul(viewInv, float4(viewPos.xyz, 0.0f));

        // Unnormalized world direction
        return worldDir.xyz;
    }

    float2 projectWorldToPixel(float3 p)
    {
        const float4 viewPos = mul(view, float4(p, 1.0f));
        const float4 clipPos = mul(proj, viewPos);
        const float4 ndcPos = clipPos / clipPos.w;
        float2 screenPos = 0.5f * (ndcPos.xy + 1.0f);
        screenPos.y = 1.0f - screenPos.y;
        const float2 pixel = screenPos * dims;

        return pixel;
    }

    float3 projectWorldToClip(float3 p)
    {
        const float4 viewPos = mul(view, float4(p, 1.0f));
        const float4 clipPos = mul(proj, viewPos);
        const float4 ndcPos = clipPos / clipPos.w;
        return ndcPos.xyz;
    }

    float2 projectWorldDirectionToPixel(float3 v)
    {
        const float4 viewDir = mul(view, float4(v, 0.0f));
        const float4 clipDir = mul(proj, viewDir);
        const float4 ndcPos = clipDir / clipDir.w;
        float2 screenPos = 0.5f * (ndcPos.xy + 1.f);
        screenPos.y = 1.0f - screenPos.y;
        const float2 pixel = screenPos * dims;

        return pixel;
    }
#endif
};

#ifdef __cplusplus
static_assert((sizeof(CameraConstants) % 16) == 0);
#endif

struct RenderParams
{
    ColorMode colorMode;
    ShadingMode shadingMode;
    uint32_t spp;
    int subFrameIndex;

    int enableWireframe;
    float wireframeThickness;
    float fireflyMaxIntensity;
    float roughnessOverride;

    float3 missColor;
    uint32_t ptMaxBounces;

    float3 eye;
    float zFar;

    float3 U;
    int enableTimeView;

    float3 V;
    uint32_t clusterPattern;

    float3 W;
    float globalDisplacementScale;

    float2 jitter;
    int2 debugPixel;

    CameraConstants camera;
    CameraConstants prevCamera;

    // for wireframe thickness
    float4x4 viewProjectionMatrix;

    int hasEnvironmentMap;
    float envmapIntensity;
    int enableEnvmapHeatmap;
    DenoiserMode denoiserMode;

    float4x4 envmapRotation;
    float4x4 envmapRotationInv;
};

struct SubdInstance
{
    // Bindless buffer indices
    uint32_t plansBindlessIndex;
    uint32_t stencilMatrixBindlessIndex;
    uint32_t subpatchTreesBindlessIndex;
    uint32_t patchPointIndicesBindlessIndex;

    uint32_t vertexSurfaceDescriptorBindlessIndex;
    uint32_t vertexControlPointIndicesBindlessIndex;
    uint32_t positionsBindlessIndex;
    uint32_t positionsPrevBindlessIndex;

    uint32_t surfaceToGeometryIndexBindlessIndex;
    uint32_t topologyQualityBindlessIndex;
    uint32_t isolationLevel;

    float3x4 prevLocalToWorld;
    float3x4 worldToLocal;

#ifdef __cplusplus
    SubdInstance()
        : plansBindlessIndex(kInvalidBindlessIndex)
        , stencilMatrixBindlessIndex(kInvalidBindlessIndex)
        , subpatchTreesBindlessIndex(kInvalidBindlessIndex)
        , patchPointIndicesBindlessIndex(kInvalidBindlessIndex)
        , vertexSurfaceDescriptorBindlessIndex(kInvalidBindlessIndex)
        , vertexControlPointIndicesBindlessIndex(kInvalidBindlessIndex)
        , positionsBindlessIndex(kInvalidBindlessIndex)
        , positionsPrevBindlessIndex(kInvalidBindlessIndex)
        , surfaceToGeometryIndexBindlessIndex(kInvalidBindlessIndex)
        , topologyQualityBindlessIndex(kInvalidBindlessIndex)
        , isolationLevel(0)
    {}

    bool operator==(const SubdInstance& other) const
    {
        return memcmp(this, &other, sizeof(*this)) == 0;
    }
#endif
};

#endif // RENDER_PARAMS_H
