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


#ifndef ENVMAP_HLSLI  // using instead of "#pragma once" due to https://github.com/microsoft/DirectXShaderCompiler/issues/3943
#define ENVMAP_HLSLI

float sphericalPhi(float3 v)
{
    const float p = atan2(v.z, v.x);
    return (p < 0) ? (p + TWO_PI) : p;
}

float sphericalTheta(float3 v)
{
    return acos(clamp(v.y, -1.f, 1.f));
}

float2 sphericalProjection(float3 d)
{
    d = normalize(d);
    const float u = fmod(sphericalPhi(d), TWO_PI) * INV_2PI;
    const float v = sphericalTheta(d) * M_1_PIf;
    return float2(u, v);
}

float3
sphericalDirection(const float2 u, out float sinTheta)
{
    const float phi = u.x * TWO_PI;
    const float theta = u.y * M_PIf;
    const float cosTheta = cos(theta);
    sinTheta = sin(theta);
    return normalize(float3(sinTheta * cos(phi), cosTheta, sinTheta * sin(phi)));
}

float2 convertDirToTexCoords(float3 d, float4x4 rotation)
{
    float4 transformed = mul(rotation, float4(d, 1.0f));
    return sphericalProjection(transformed.xyz);
}

float3 convertTexCoordsToDir(float2 uv, float4x4 rotationInv, out float sinTheta)
{
    float3 dir = sphericalDirection(uv, sinTheta);
    float4 transformed = mul(rotationInv, float4(dir, 1.f));
    return transformed.xyz;
}

float3 envMapEvaluate(float2 u, Texture2D<float4> tex, float intensity, SamplerState sampler)
{
    return intensity * (tex.SampleLevel(sampler, u, 0).rgb);
}

float3 envMapEvaluate(float3 dir, Texture2D<float4> tex, float intensity, SamplerState sampler, float4x4 rotation)
{
    const float2 u = convertDirToTexCoords(dir, rotation);
    return envMapEvaluate(u, tex, intensity, sampler);
}
float envMapPdf(float2 u, Texture2D<float4> envMap, StructuredBuffer<float> conditional, StructuredBuffer<float> marginalCDF, SamplerState sampler)
{
    uint width, height;
    
    envMap.GetDimensions(width, height);
    uint iu = clamp(int(u[0] * width), 0, width - 1);
    uint iv = clamp(int(u[1] * height), 0, height - 1); 

    float conditionalValue = conditional[iv * width + iu];
    float marginalIntegral = marginalCDF[height+1];

    float sinTheta = sin(u[1] * M_PIf);

    return conditionalValue / (marginalIntegral * TWO_PI * M_PI * sinTheta); // Jacobian term for latlong conversion; see PBRT V3 section 14.2.4.);
}

float envMapPdf(float3 dir, float4x4 rotation, Texture2D<float4> envMap, StructuredBuffer<float> conditional, StructuredBuffer<float> marginalCDF, SamplerState sampler)
{
    const float2 u = convertDirToTexCoords(dir, rotation);
    return envMapPdf(u, envMap, conditional, marginalCDF, sampler);
}

uint FindInterval(StructuredBuffer<float> cdf, float u, uint row, uint width)
{
    // cdf buffer is padded on both sides (firstpad, first, ... , last, lastpad)
    uint cdfPaddedWidth = width + 2;
    uint size = width;
    uint first = 1;
        
    while (size > 0)
    {
        uint halfSize = size >> 1;
        uint middle = first + halfSize;
        uint middleOffset = row * cdfPaddedWidth + middle;
        
        bool predResult = cdf[middleOffset] <= u;
        first = predResult ? middle + 1 : first;
        size = predResult ? size - halfSize - 1 : halfSize;
    }
    
    return clamp(first - 1, 0, width - 1);
}

float Sample1D(float u, StructuredBuffer<float> func, StructuredBuffer<float> cdf, uint row, uint width, out float pdf, out uint offset)
{
    offset = FindInterval(cdf, u, row, width);
    uint cdfPaddedWidth = width + 2;
    uint cdfOffset = row * cdfPaddedWidth + offset;
    uint funcOffset = row * width + offset;
    
    float du = u - cdf[cdfOffset];
    if (cdf[cdfOffset + 1] - cdf[cdfOffset] > 0)
    {
        du /= (cdf[cdfOffset + 1] - cdf[cdfOffset]);
    }
    float funcInt = cdf[row * (width + 2) + width + 1];
    pdf = func[funcOffset] / funcInt;
    return (offset + du) / width;
}

float3 envMapImportanceSample(float2 rndSample, float4x4 rotationInv, out float pdf, 
    out float3 envMapColor, Texture2D<float4> envMap, float intensity, StructuredBuffer<float>conditional, 
    StructuredBuffer<float>marginal, StructuredBuffer<float> conditionalCDF, StructuredBuffer<float> marginalCDF, SamplerState sampler)
{   
    uint width, height;
    float pdfs[2];
    uint indices[2];
    float2 latLongUv;

    envMap.GetDimensions(width, height);

    latLongUv[1] = Sample1D(rndSample[1], marginal, marginalCDF, 0, height, pdfs[1], indices[1]);
    latLongUv[0] = Sample1D(rndSample[0], conditional, conditionalCDF, indices[1], width, pdfs[0], indices[0]);
    envMapColor = envMapEvaluate(latLongUv, envMap, intensity, sampler);
    
    float        sinTheta = 0.f;
    const float3 dir = convertTexCoordsToDir(latLongUv, rotationInv, sinTheta);
    pdf = (pdfs[0] * pdfs[1]) / (TWO_PI * M_PI * sinTheta); // Jacobian term for latlong conversion; see PBRT V3 section 14.2.4.

    return dir;
}

#endif // ENVMAP_HLSLI