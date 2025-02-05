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

