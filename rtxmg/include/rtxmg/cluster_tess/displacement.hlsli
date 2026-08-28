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

#ifndef DISPLACEMENT_H
#define DISPLACEMENT_H

#include "rtxmg/subdivision/subdivision_eval.hlsli"
#include "rtxmg/scene/material_constants.h"

void GetDisplacement(RTXMGMaterialConstants material,
                     float globalScale,
                     out int displacementTexIndex,
                     out float scale)
{
    scale = 0.0f; // Default value indicating no displacement
    displacementTexIndex = -1;
    if (material.normalOrDisplacementTextureIndex != -1)
    {
        displacementTexIndex = material.normalOrDisplacementTextureIndex;
        scale = material.normalOrDisplacementTextureScale * globalScale;
    }
}


float DisplacementMipLevel(float2 dx, float2 dy) 
{
    float d = max(dot(dx, dx), dot(dy, dy));
    return max(0.5f * log2(d), 0.f);
}

LimitFrame DoDisplacement(TexcoordEvaluatorHLSL texcoordEval,
    LimitFrame limit,
    uint32_t iSurface,
    float2 uv,
    float du,
    float dv,
    Texture2D<float> displacementTex,
    SamplerState dispSampler,
    float scale)
{
    if (scale == 0)
    {
        return limit;
    }
        
    // compute subd limit and normal
    const float3 normal = normalize(cross(limit.deriv1, limit.deriv2));
    TexCoordLimitFrame texcoord = texcoordEval.EvaluateLinearSubd(uv, iSurface);
    
    // Sample 1 texel 
    float2 gradDu = du * texcoord.deriv1;
    float2 gradDv = dv * texcoord.deriv2;

    float mipLevel = DisplacementMipLevel(gradDu, gradDv);
    float displacement = scale * displacementTex.SampleLevel(dispSampler, texcoord.uv, mipLevel);
    
    // compute derivatives of displacement map, (dD/du) and (dD/dv) from finite differences:
    const float2 delta = float2(max(du, 0.01f), max(dv, 0.01f));
    float2 texcoordDu = texcoord.uv + delta.x * texcoord.deriv1;
    float2 texcoordDv = texcoord.uv + delta.y * texcoord.deriv2;
    
    float displacement1 = scale * displacementTex.SampleLevel(dispSampler, texcoordDu, mipLevel);
    float displacement2 = scale * displacementTex.SampleLevel(dispSampler, texcoordDv, mipLevel);
    float  dDdu = ( displacement1 - displacement ) / delta.x;
    float  dDdv = ( displacement2 - displacement ) / delta.y;
    
    // compute displaced partial derivates
    const float3 dpdu = limit.deriv1 + dDdu * normal;
    const float3 dpdv = limit.deriv2 + dDdv * normal;

    LimitFrame ret;
    
    ret.p = limit.p + displacement * normal;
    ret.deriv1 = dpdu;
    ret.deriv2 = dpdv;
    
    return ret;
}

#endif // DISPLACEMENT_H