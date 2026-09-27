/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
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
    uint32_t subFrameIndex;

    int enableWireframe;
    float wireframeThickness;
    float fireflyMaxIntensity;
    float roughnessOverride;

    uint32_t isolationLevel;
    uint32_t clusterPattern;
    float globalDisplacementScale;
    int debugSurfaceIndex;

    float3 missColor;
    uint32_t ptMaxBounces;

    float3 eye;
    float zFar;

    float3 U;
    int enableTimeView;

    float3 V;
    // Inspector-selected cluster-LOD geometryID; the hit shader draws its
    // wireframe in red. -1 = none.
    int selectedClusterLodGeometry;

    float3 W;
    // Viewport-picked instance of the selected geometry: limits the red
    // wireframe to that one instance. -1 = highlight every instance (row
    // clicks in the Inspector carry no instance).
    int selectedClusterLodInstance;

    float2 jitter;
    int2 debugPixel;

    // Selected-LOD narrowing for the Inspector highlight: only clusters at
    // this LOD level draw the red wireframe (-1 = all levels). Set by
    // viewport picks (the hit cluster's level) and LOD-row clicks.
    int selectedClusterLodLevel;
    // Inspector-selected SubD mesh index; -1 = none.
    int selectedSubdMesh;
    int _padSel1, _padSel2;

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

    uint32_t surfaceToMaterialIndexBindlessIndex;
    uint32_t topologyQualityBindlessIndex;
    uint32_t meshID;
    uint32_t _meshPad;

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
        , surfaceToMaterialIndexBindlessIndex(kInvalidBindlessIndex)
        , topologyQualityBindlessIndex(kInvalidBindlessIndex)
        , meshID(~0u)
        , _meshPad(0)
    {}

    bool operator==(const SubdInstance& other) const
    {
        return memcmp(this, &other, sizeof(*this)) == 0;
    }
#endif
};

#endif // RENDER_PARAMS_H
