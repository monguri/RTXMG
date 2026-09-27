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

#include "envmap/preprocess_envmap_params.h"
#include "rtxmg/utils/constants.h"

ConstantBuffer<PreprocessEnvMapParams> gPreprocessEnvMapParams : register(b0);

Texture2D<float4> gTextureColorIn : register(t0);
RWBuffer<float> g_ConditionalFunc : register(u0);

SamplerState gTextureColorInSampler : register(s0);

// clang-format off
[numthreads(16, 16, 1)]
[shader("compute")]
void main(uint2 dispatchThreadId : SV_DispatchThreadID)
// clang-format on
{
    if (dispatchThreadId.y >= gPreprocessEnvMapParams.envMapHeight || dispatchThreadId.x >= gPreprocessEnvMapParams.envMapWidth)
    {
        return;
    }
    float2 uv = (float2(dispatchThreadId) + 0.5f) / float2(gPreprocessEnvMapParams.envMapWidth, gPreprocessEnvMapParams.envMapHeight);
    float3 color = gTextureColorIn.SampleLevel(gTextureColorInSampler, uv, 0).xyz;
    //    float3 color = gTextureColorIn[dispatchThreadId].xyz;

    const float3 lumConverter = float3(0.299f, 0.587f, 0.114f);
    float lum = dot(lumConverter, color);
    float sinTheta = sin(uv[1] * M_PIf); // prefer values away from the poles to compensate for distortion in latlong mapping (PBRT V3 section 14.2.4)

    g_ConditionalFunc[dispatchThreadId.y * gPreprocessEnvMapParams.envMapWidth + dispatchThreadId.x] = lum * sinTheta;
}

