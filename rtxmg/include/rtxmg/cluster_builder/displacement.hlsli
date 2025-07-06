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

#ifndef DISPLACEMENT_H
#define DISPLACEMENT_H

#include "rtxmg/subdivision/subdivision_eval.hlsli"
#include <donut/shaders/material_cb.h>

void GetDisplacement(MaterialConstants material,
                     float globalScale,
                     out int displacementTexIndex, 
                     out float scale)
{
    scale = 0.0f; // Default value indicating no displacement
    displacementTexIndex = -1;
    if (material.normalTextureIndex != -1)
    {
        displacementTexIndex = material.normalTextureIndex;
        scale = material.normalTextureScale * globalScale;
    }
}

LimitFrame DoDisplacement(TexcoordEvaluatorHLSL texcoordEval,
    LimitFrame limit,
    uint32_t iSurface,
    float2 uv,
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
    float2 dimensions;
    displacementTex.GetDimensions(dimensions.x, dimensions.y);
    float2 dsdt = 1.0f / dimensions;
    float displacement = scale * displacementTex.SampleLevel(dispSampler, texcoord.uv, 0);
    
    // compute derivatives of displacement map, (dD/du) and (dD/dv) from finite differences:
    float2 texcoordDu         = texcoord.uv + dsdt.x * texcoord.deriv1;
    float2 texcoordDv         = texcoord.uv + dsdt.y * texcoord.deriv2;
    
    float displacement1 = scale * displacementTex.SampleLevel(dispSampler, texcoordDu, 0);
    float displacement2 = scale * displacementTex.SampleLevel(dispSampler, texcoordDv, 0);
    float  displacementDeriv1 = ( displacement1 - displacement ) / dsdt.x;
    float  displacementDeriv2 = ( displacement2 - displacement ) / dsdt.y;
    // compute displaced parital derivates
    const float3 dpdu = limit.deriv1 + displacementDeriv1 * normal;
    const float3 dpdv = limit.deriv2 + displacementDeriv2 * normal;

    LimitFrame ret;
    
    ret.p = limit.p + displacement * normal;
    ret.deriv1 = dpdu;
    ret.deriv2 = dpdv;
    
    return ret;
}

#endif // DISPLACEMENT_H