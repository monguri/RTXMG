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

#ifndef RTXMG_UTILS_HLSLI // using instead of "#pragma once" due to https://github.com/microsoft/DirectXShaderCompiler/issues/3943
#define RTXMG_UTILS_HLSLI

#include "rtxmg/utils/constants.h"

#if ENABLE_DUMP_FLOAT
#define DUMP_FLOAT(idx, x)  { u_DebugTex##idx[DispatchRaysIndex().xy] = float4(x, 0, 0, 1); }
#define DUMP_FLOAT2(idx, x) { u_DebugTex##idx[DispatchRaysIndex().xy] = float4(x, 0, 1); }
#define DUMP_FLOAT3(idx, x) { u_DebugTex##idx[DispatchRaysIndex().xy] = float4(x, 1); }
#define DUMP_FLOAT4(idx, x) { u_DebugTex##idx[DispatchRaysIndex().xy] = x; }
#else
#define DUMP_FLOAT(idx, x)  
#define DUMP_FLOAT2(idx, x)
#define DUMP_FLOAT3(idx, x)
#define DUMP_FLOAT4(idx, x)
#endif

#define DEBUG_ARGS \
    RWBuffer<float4> u_Debug1, RWTexture2D<float4> u_DebugTex1, \
    RWBuffer<float4> u_Debug2, RWTexture2D<float4> u_DebugTex2, \
    RWBuffer<float4> u_Debug3, RWTexture2D<float4> u_DebugTex3, \
    RWBuffer<float4> u_Debug4, RWTexture2D<float4> u_DebugTex4
#define DEBUG_PARAMS \
    u_Debug1, u_DebugTex1, \
    u_Debug2, u_DebugTex2, \
    u_Debug3, u_DebugTex3, \
    u_Debug4, u_DebugTex4

// Ported from Omniverse Kit Renderer.
// Clamps the contribution of ray in order to prevent fireflies
inline
float3 FireflyFiltering(float3 contribution, float fireflyMaxIntensity)
{
    if (fireflyMaxIntensity > 0.0f)
    {
        // !important! don't use CIE luminance functions here
        // reason: Our fake spectral dispersion creates negative color components and the spectral sampling is tweaked
        // such as that the expected value is (1, 1, 1) in sRGB. If now the firefly filter clamps those components
        // unevenly (in particular it doesn't clamp values where the negative red minimizes the CIE luminance) it
        // results in color shifts. So we use the average of the absolute value here, that seems to behave better in
        // expectation.
        float magicNumber = 1e-5f;
        float3 absValue = float3(abs(contribution.x), abs(contribution.y), abs(contribution.z));
        float averageRGB = (absValue.x + absValue.y + absValue.z) * (1.0f / 3.0f);
        float clampedLuminance = min(averageRGB, fireflyMaxIntensity);
        contribution = (averageRGB > magicNumber) ? (contribution * (clampedLuminance / averageRGB)) : contribution;
    }
    contribution.x = max(0.f, contribution.x);
    contribution.y = max(0.f, contribution.y);
    contribution.z = max(0.f, contribution.z);
    return contribution;
}

inline float ModPositive(float dividend, float divisor)
{
    float result = fmod(dividend, divisor);
    if (result < 0)
        result += divisor;
    return result;
}

inline float3 HSVToRGB(float3 c)
{
    if (c.y == 0.0f)
        return float3(c.zzz);

    float h = ModPositive(c.x, 360.0f) / 60.0f;
    int i = int(floor(h));
    
    float f = h - i;
    float p = c.z * (1.0f - c.y);
    float q = c.z * (1.0f - c.y * f);
    float t = c.z * (1.0f - c.y * (1.0f - f));

    switch (i)
    {
        case 0:
            return float3(c.z, t, p);
        case 1:
            return float3(q, c.z, p);
        case 2:
            return float3(p, c.z, t);
        case 3:
            return float3(p, q, c.z);
        case 4:
            return float3(t, p, c.z);
        case 5:
            return float3(c.z, p, q);
        default:
            return float3(0.0, 0.0f, 0.f);
    }
}

inline uint JenkinsHash(uint a)
{
    // http://burtleburtle.net/bob/hash/integer.html
    a = (a + 0x7ed55d16) + (a << 12);
    a = (a ^ 0xc761c23c) ^ (a >> 19);
    a = (a + 0x165667b1) + (a << 5);
    a = (a + 0xd3a2646c) ^ (a << 9);
    a = (a + 0xfd7046c5) + (a << 3);
    a = (a ^ 0xb55a4f09) ^ (a >> 16);
    return a;
}

inline uint32_t MurmurAdd(uint32_t hash, uint32_t element)
{
    element *= 0xcc9e2d51;
    element = (element << 15) | (element >> (32 - 15));
    element *= 0x1b873593;
    hash ^= element;
    hash = (hash << 13) | (hash >> (32 - 13));
    hash = hash * 5 + 0xe6546b64;
    return hash;
}

inline uint32_t MurmurMix(uint32_t hash)
{
    hash ^= hash >> 16;
    hash *= 0x85ebca6b;
    hash ^= hash >> 13;
    hash *= 0xc2b2ae35;
    hash ^= hash >> 16;
    return hash;
}

inline float3 UintToColor(uint x)
{
    uint xHashed = JenkinsHash(x);
    float hue = (float(xHashed & 0xffff) / float(0xffffu)) * 360.0f;
    float sat = lerp(0.7f, 1.0f, saturate(float((xHashed >> 16) & 0xff) / float(0xffu)));
    float value = lerp(0.5f, 1.0f, saturate(float((xHashed >> 24) & 0xff) / float(0xffu)));
    
    return HSVToRGB(float3(hue, sat, value));
}

float Luminance(float3 rgb)
{
    return dot(rgb, float3(0.2126f, 0.7152f, 0.0722f));
}

inline float SafeLength(float3 v)
{
    const float3 n = float3(abs(v.x), abs(v.y), abs(v.z));
    // avoid overflow by dividing by the max component
    const float m = max(n.x, max(n.y, n.z));
    const float x = n.x / m;
    const float y = n.y / m;
    const float z = n.z / m;
    // scale back by the max component
    const float len = m * (sqrt(x * x + y * y + z * z));
    return len;
}

inline float3 SafeNormalize(float3 n)
{
    // avoid division by 0 by adding numeric_limits::min
    const float len = SafeLength(n) + FLT_MIN;
    return n * (1.f / len);
}

inline float2 ConcentricSampleDisk(float2 u)
{
    // Map uniform random numbers to $[-1,1]^2$
    const float2 uOffset = 2.f * u - float2(1.f, 1.f);

    // Handle degeneracy at the origin
    if (uOffset.x == 0.f && uOffset.y == 0.f)
        return float2(0.f, 0.f);

    // Apply concentric mapping to point
    float theta, r;
    if (abs(uOffset.x) > abs(uOffset.y))
    {
        r = uOffset.x;
        theta = PI_OVER_4 * (uOffset.y / uOffset.x);
    }
    else
    {
        r = uOffset.y;
        theta = PI_OVER_2 - PI_OVER_4 * (uOffset.x / uOffset.y);
    }
    return r * float2(cos(theta), sin(theta));
}

inline float3 CosineSampleHemisphere(float2 u)
{
    const float2 d = ConcentricSampleDisk(u);
    const float  z = sqrt(max(0.0f, 1.f - d.x * d.x - d.y * d.y));
    return float3(d.x, d.y, z);
}

inline float CosineHemispherePdf(float cosTheta)
{
    return cosTheta * M_1_PIf;
}

inline float3 UniformSampleHemisphere(float2 u)
{
    const float z = u.x;
    const float r = sqrt(max(0.f, 1.f - z * z));
    const float phi = TWO_PI * u.y;
    return float3(r * cos(phi), r * sin(phi), z);
}

inline float UniformHemiSpherePdf()
{
    return INV_2PI;
}

inline float PowerHeuristic(int nf, float fPdf, int ng, float gPdf)
{
    float f = nf * fPdf, g = ng * gPdf;
    return (f * f) / (f * f + g * g);
}

inline uint32_t FloatToUInt(
float _V, float _Scale)
{
    return (uint32_t)floor(_V * _Scale + 0.5f);
}

// this is a generalized lerp; map a value in [a, b] to the range [c, d] without clamping
inline float Remap(float x, float a, float b, float c, float d)
{
    return c + (d - c) * (x - a) / (b - a);
}


/**
* Octahedral normal vector encoding.
* @param N must be a unit vector.
* @return An octahedral vector on the [-1, +1] square.
*/

inline float2 UnitVectorToOctahedron(float3 N)
{
    const float d = dot(float3(1.f, 1.f, 1.f), float3(abs(N.x), abs(N.y), abs(N.z)));
    N.x /= d;
    N.y /= d;
    if (N.z <= 0.f)
    {
        float2 signs;
        signs.x = N.x >= 0.f ? 1.f : -1.f;
        signs.y = N.y >= 0.f ? 1.f : -1.f;

        const float2 k = (float2(1.f, 1.f) - float2(abs(N.y), abs(N.x))) * signs;

        N.x = k.x;
        N.y = k.y;
    }
    return float2(N.x, N.y);
}
/**
* Octahedral normal vector decoding.
* @param Oct An octahedral vector as returned from UnitVectorToOctahedron, on the [-1, +1] square.
* @return Returns a unit vector.
*/
inline float3 OctahedronToUnitVector(float2 Oct)
{
    float3 N = float3(Oct.x, Oct.y, 1.f - dot(float2(1.f, 1.f), float2(abs(Oct.x), abs(Oct.y))));
    if (N.z < 0)
    {
        float2 signs;
        signs.x = N.x >= 0.f ? 1.f : -1.f;
        signs.y = N.y >= 0.f ? 1.f : -1.f;

        const float2 k = (float2(1.f, 1.f) - float2(abs(N.y), abs(N.x))) * signs;

        N.x = k.x;
        N.y = k.y;
    }
    return normalize(N);
}

inline uint32_t PackNormalizedVector(float3 x)
{
    float2 XY = UnitVectorToOctahedron(x);

    XY = XY * float2( .5f, .5f ) + float2( .5f, .5f );

    uint32_t X = FloatToUInt( saturate( XY.x ), ( 1 << 16 ) - 1 );
    uint32_t Y = FloatToUInt( saturate( XY.y ), ( 1 << 16 ) - 1 );

    uint32_t PackedOutput = X;
    PackedOutput |= Y << 16;
    return PackedOutput;
}

inline float3 UnpackNormalizedVector(uint32_t PackedInput)
{
    uint32_t  X = PackedInput & ((1 << 16) - 1);
    uint32_t  Y = PackedInput >> 16;
    float2 XY = float2(0.f, 0.f);
    XY.x = (float) X / ((1 << 16) - 1);
    XY.y = (float) Y / ((1 << 16) - 1);
    XY = XY * float2(2.f, 2.f) + float2(-1.f, -1.f);
    return OctahedronToUnitVector(XY);
}


inline float3 Temperature(const float t)
{
    const float b = t < 0.25f ? smoothstep(-0.25f, 0.25f, t) : 1.0f - smoothstep(0.25f, 0.5f, t);
    const float g = t < 0.5f ? smoothstep(0.0f, 0.5f, t) : (t < 0.75f ? 1.0f : 1.0f - smoothstep(0.75f, 1.0f, t));
    const float r = smoothstep(0.5f, 0.75f, t);
    return float3(r, g, b);
}


inline int GetImageIndex()
{
    return DispatchRaysIndex().y * DispatchRaysDimensions().x + DispatchRaysIndex().x;
}

unsigned int TEA(unsigned int N, unsigned int val0, unsigned int val1)
{
    unsigned int v0 = val0;
    unsigned int v1 = val1;
    unsigned int s0 = 0;

    for (unsigned int n = 0; n < N; n++)
    {
        s0 += 0x9e3779b9;
        v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4);
        v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7e95761e);
    }

    return v0;
}
// Generate random unsigned int in [0, 2^24)
unsigned int LCG(inout unsigned int prev)
{
    const unsigned int LCG_A = 1664525u;
    const unsigned int LCG_C = 1013904223u;
    prev = (LCG_A * prev + LCG_C);
    return prev & 0x00FFFFFF;
}

// Generate random float in [0, 1)
float Rnd(inout unsigned int prev)
{
    return ((float) LCG(prev) / (float) 0x01000000);
}

float2 RandomStrat(const int sampleOffset, const int strataCount,
    inout unsigned int prev)
{
    const int sy = sampleOffset / strataCount;
    const int sx = sampleOffset - sy * strataCount;
    const float invStrata = 1.0f / strataCount;
    return float2((sx + Rnd(prev)) * invStrata, (sy + Rnd(prev)) * invStrata);
}

struct Onb
{

    // transform from the coordinate system represented by ONB
    float3 ToWorld(float3 v) { return (v.x * m_tangent + v.y * m_binormal + v.z * m_normal); }

    // transform to the coordinate system represented by ONB
    float3 ToLocal(float3 v)
    {
        const float x = dot(v, m_tangent);
        const float y = dot(v, m_binormal);
        const float z = dot(v, m_normal);
        return float3(x, y, z);
    }

    float3 m_tangent;
    float3 m_binormal;
    float3 m_normal;
};

Onb MakeOnb(float3 normal)
{
    Onb ret;
    ret.m_normal = normal;

    if (abs(normal.x) > abs(normal.z))
    {
        ret.m_binormal.x = -normal.y;
        ret.m_binormal.y = normal.x;
        ret.m_binormal.z = 0.f;
    }
    else
    {
        ret.m_binormal.x = 0.f;
        ret.m_binormal.y = -normal.z;
        ret.m_binormal.z = normal.y;
    }

    ret.m_binormal = SafeNormalize(ret.m_binormal);
    ret.m_tangent = cross(ret.m_binormal, normal);

    return ret;
}

#endif // RTXMG_UTILS_HLSLI