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

#include "blit_params.h"

#include "gbuffer.h"
#include "utils.hlsli"

ConstantBuffer<BlitParams> g_Params : register(b0);
RWTexture2D<float4> g_Output : register(u0);
Texture2D<float4> g_Input : register(t0);
Texture2D<float4> g_InputSplitScreen : register(t1);

StructuredBuffer<HitResult> g_HitResult : register(t3);

SamplerState g_Sampler : register(s0);

inline float3 expose(float3 input)
{
    return input * g_Params.m_exposure;
}

inline float3 computeSRGB(float3 c)
{
    // reference: https://www.color.org/chardata/rgb/srgb.xalter
    float  invGamma = 1.0f / 2.4f;
    float3 powed = float3(pow(c.x, invGamma), pow(c.y, invGamma), pow(c.z, invGamma));
    return float3(
        c.x < 0.0031308f ? 12.92f * c.x : 1.055f * powed.x - 0.055f,
        c.y < 0.0031308f ? 12.92f * c.y : 1.055f * powed.y - 0.055f,
        c.z < 0.0031308f ? 12.92f * c.z : 1.055f * powed.z - 0.055f);
}

inline float3 computeACES(float3 c)
{
    // reference : https://github.com/TheRealMJP/BakingLab/blob/master/BakingLab/ACES.hlsl
    // sRGB => XYZ => D65_2_D60 => AP1 => RRT_SAT
    //float3x3 m1 = { 0.59719f, 0.35458f, 0.04823f,
    //    0.07600f, 0.90834f, 0.01566f,
    //    0.02840f, 0.13383f, 0.83777f };

    float3x3 m1 = { 0.59719f, 0.07600f, 0.02840f,
        0.35458f, 0.90834f, 0.13383f,
        0.04823f, 0.01566f, 0.83777f };


    // ODT_SAT => XYZ => D60_2_D65 => sRGB
    //float3x3 m2 = { 1.60475f, -0.53108f, -0.07367f,
    //    -0.10208f, 1.10813f, -0.00605f,
    //    -0.00327f, -0.07276f, 1.07602f };

    float3x3 m2 = { 1.60475f, -0.10208f, -0.00327f,
        -0.53108f, 1.10813f, -0.07276f,
        -0.07367f, -0.00605f, 1.07602f };

    float3 v = mul(c, m1);
    float3 a = v * (v + 0.0245786f) - 0.000090537f;
    float3 b = v * (0.983729f * v + 0.4329510f) + 0.238081f;

    c = clamp(mul(a / b, m2), 0.0f, 1.0f);
    return c;
}

inline float3 uncharted2_partial(float3 x)
{
    float A = 0.15f; float B = 0.50f; float C = 0.10f;
    float D = 0.20f; float E = 0.02f; float F = 0.30f;
    return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F;
}

inline float3 computeUncharted2(float3 c)
{
    // reference : https://64.github.io/tonemapping/
    const float exposureBias = 2.0f;
    float3 curr = uncharted2_partial(c * exposureBias);

    const float3 W = { 11.2f, 11.2f, 11.2f };
    float3 whiteScale = float3(1.f, 1.f, 1.f) / uncharted2_partial(W);

    c = clamp(curr * whiteScale, 0.f, 1.f);

    return c;
}

[numthreads(32, 32, 1)]
void main(uint3 threadIdx : SV_DispatchThreadID)
{
    uint2 pixelPos = uint2(threadIdx.xy);
    uint outputWidth, outputHeight;
    uint inputWidth, inputHeight;

    g_Input.GetDimensions(inputWidth, inputHeight);
    g_Output.GetDimensions(outputWidth, outputHeight);

    if (pixelPos.x >= outputWidth || pixelPos.y >= outputHeight)
        return;

    float2 uv = (float2(pixelPos) + 0.5) / float2(outputWidth, outputHeight);

    uint2 inputPos = uint2(uv * float2(inputWidth, inputHeight));
    uint hitResultIndex = inputPos.x + inputPos.y * inputWidth;

    float3 input;

    if (uv.x < g_Params.m_separator)
    {
        switch (g_Params.m_blitDecodeMode)
        {
        case BlitDecodeMode::SingleChannel:
            input = g_Input.Sample(g_Sampler, uv).rrr;
            break;
        case BlitDecodeMode::Depth:
            input = (g_Input[inputPos].rrr - g_Params.m_zNear) / (g_Params.m_zFar - g_Params.m_zNear);
            break;
        case BlitDecodeMode::MotionVectors:
            // input = float3(abs(g_Input[inputPos].xy / float2(inputWidth, inputHeight)), 0.0f);
            input = float3(abs(g_Input[inputPos].xy), 0.0f);
            break;
        case BlitDecodeMode::Normals:
            input = float3(g_Input[inputPos].xyz * 0.5 + 0.5);
            break;
        case BlitDecodeMode::InstanceId:
        {
            uint instanceId = g_HitResult[hitResultIndex].instanceId;
            input = instanceId != kInvalidInstanceId ? UintToColor(instanceId) : float3(0, 0, 0);
            break;
        }
        case BlitDecodeMode::SurfaceIndex:
        {
            uint surfaceIndex = g_HitResult[hitResultIndex].surfaceIndex;
            input = surfaceIndex != kInvalidSurfaceIndex ? UintToColor(surfaceIndex) : float3(0, 0, 0);
            break;
        }
        case BlitDecodeMode::SurfaceUv:
            input = float3(g_HitResult[hitResultIndex].surfaceUV, 0);
            break;
        case BlitDecodeMode::Texcoord:
            input = float3(frac(g_HitResult[hitResultIndex].texcoord), 0);
            break;
        case BlitDecodeMode::None:
            input = g_Input.Sample(g_Sampler, uv).xyz;
            break;
        }
    }
    else
    {
        input = g_InputSplitScreen.Sample(g_Sampler, uv).xyz;
    }

    input = expose(input);
    
    float3 output;
    switch (g_Params.m_tonemapOperator)
    {
    case TonemapOperator::Srgb:
        output = computeSRGB(input);
        break;
    case TonemapOperator::Aces:
        output = computeSRGB(computeACES(input));
        break;
    case TonemapOperator::Hable:
        output = computeSRGB(computeUncharted2(input));
        break;
    default:
        output = input;
        break;
    }

    float verticalLine = saturate(1.0 - abs(uv.x - g_Params.m_separator) * outputWidth / 3.5);
    verticalLine = saturate(verticalLine / 0.5);
    verticalLine *= float(g_Params.m_separator != 0.0);

    const float3 nvColor = float3(118.0, 185.0, 0.0) / 255.0;
    output = lerp(output, nvColor * verticalLine, verticalLine);

    g_Output[pixelPos] = float4(output, 1.f);
}