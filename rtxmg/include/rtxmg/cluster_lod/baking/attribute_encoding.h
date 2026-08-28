/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

// Oct-encoding for normals and tangents.  ATTRENC_FLOOR / ATTRENC_CLAMP are
// local helpers because donut::math has no component-wise floor/clamp(vec,
// scalar, scalar).

#pragma once

#include <cmath>
#include <cstdint>
#include <donut/core/math/math.h>

#define ATTRENC_PI     float(3.14159265358979323846264338327950288)
#define ATTRENC_NORMAL_BITS   22
#define ATTRENC_TANGENT_BITS  10

namespace shaderio {

using namespace donut::math;

// ---- helper replacements for glm functions not in donut math ----

inline float2 AttrEncFloorF2(float2 v)
{
    return float2(std::floor(v.x), std::floor(v.y));
}

inline float2 AttrEncClampF2(float2 v, float lo, float hi)
{
    return float2(std::fmax(lo, std::fmin(hi, v.x)),
                  std::fmax(lo, std::fmin(hi, v.y)));
}

#define ATTRENC_INLINE   inline
#define ATTRENC_OUT(a)   a&
#define ATTRENC_ATAN2F   atan2f
#define ATTRENC_FLOOR    AttrEncFloorF2
#define ATTRENC_CLAMP    AttrEncClampF2
#define ATTRENC_ABS      std::abs

static_assert(ATTRENC_NORMAL_BITS % 2 == 0);

// --------------------------------------------------------------------------
// Oct-encoding helpers (from http://jcgt.org/published/0003/02/01/paper.pdf)
// --------------------------------------------------------------------------

ATTRENC_INLINE float2 OctSignNotZero(float2 v)
{
    return float2((v.x >= 0.0f) ? +1.0f : -1.0f,
                  (v.y >= 0.0f) ? +1.0f : -1.0f);
}

ATTRENC_INLINE float3 OctToVec(float2 e)
{
    float3 v = float3(e.x, e.y, 1.0f - ATTRENC_ABS(e.x) - ATTRENC_ABS(e.y));
    if (v.z < 0.0f)
    {
        float2 os = OctSignNotZero(e);
        v.x = (1.0f - ATTRENC_ABS(e.y)) * os.x;
        v.y = (1.0f - ATTRENC_ABS(e.x)) * os.y;
    }
    return normalize(v);
}

ATTRENC_INLINE float2 VecToOct(float3 v)
{
    // Project onto octahedron, then onto xy plane.
    float2 p = float2(v.x, v.y) * (1.0f / (ATTRENC_ABS(v.x) + ATTRENC_ABS(v.y) + ATTRENC_ABS(v.z)));
    // Reflect lower hemisphere folds over the diagonals.
    return (v.z <= 0.0f)
        ? (float2(1.0f - ATTRENC_ABS(p.y), 1.0f - ATTRENC_ABS(p.x)) * OctSignNotZero(p))
        : p;
}

ATTRENC_INLINE float2 VecToOctPrecise(float3 v, int bits)
{
    float2 s = VecToOct(v);
    // Each snorm's max value interpreted as integer, e.g. 127.0 for snorm8.
    float  M = float(1 << (bits - 1)) - 1.0f;
    // Remap to snorm(n/2) precision with floor instead of round (see Eq. 1).
    s                        = ATTRENC_FLOOR(ATTRENC_CLAMP(s, -1.0f, +1.0f) * M) * (1.0f / M);
    float2 bestRepresentation = s;
    float  highestCosine     = dot(OctToVec(s), v);
    // Test all combinations of floor and ceil; keep the best.
    for (int i = 0; i <= 1; ++i)
    {
        for (int j = 0; j <= 1; ++j)
        {
            if ((i != 0) || (j != 0))
            {
                float2 candidate = float2(float(i), float(j)) * (1.0f / M) + s;
                float  cosine    = dot(OctToVec(candidate), v);
                if (cosine > highestCosine)
                {
                    bestRepresentation = candidate;
                    highestCosine      = cosine;
                }
            }
        }
    }
    return bestRepresentation;
}

// --------------------------------------------------------------------------
// Normal pack / unpack
// --------------------------------------------------------------------------

ATTRENC_INLINE float3 NormalUnpack(uint32_t packed)
{
    const uint32_t mask = (1u << (ATTRENC_NORMAL_BITS / 2)) - 1u;
    uint2  pv = uint2(packed, (packed >> 11u)) & uint2(mask);
    float2 v  = (float2(float(pv.x), float(pv.y)) / float(mask)) * 2.0f - 1.0f;
    return OctToVec(v);
}

ATTRENC_INLINE uint32_t NormalPack(float3 normal)
{
    float2         v    = VecToOctPrecise(normal, ATTRENC_NORMAL_BITS / 2);
    const uint32_t mask = (1u << (ATTRENC_NORMAL_BITS / 2)) - 1u;
    v = (v + 1.0f) * 0.5f * float(mask) + 0.5f;
    uint32_t packed  = uint32_t(v.x) & mask;
    packed          |= (uint32_t(v.y) & mask) << 11u;
    return packed;
}

// --------------------------------------------------------------------------
// Tangent pack / unpack
// Based on "3 BYTE TANGENT FRAMES" from RenderingDoomEternal (SIGGRAPH 2020).
// --------------------------------------------------------------------------

// Builds an orthonormal basis from a normal vector (Nelson Max technique).
ATTRENC_INLINE void TangentOrthonormalBasis(float3 normal, ATTRENC_OUT(float3) tangent, ATTRENC_OUT(float3) bitangent)
{
    if (normal.z < -0.99998796F)
    {
        tangent   = float3( 0.0F, -1.0F,  0.0F);
        bitangent = float3(-1.0F,  0.0F,  0.0F);
        return;
    }
    float a   = 1.0F / (1.0F + normal.z);
    float b   = -normal.x * normal.y * a;
    tangent   = float3(1.0F - normal.x * normal.x * a, b, -normal.x);
    bitangent = float3(b, 1.0f - normal.y * normal.y * a, -normal.y);
}

ATTRENC_INLINE uint32_t TangentPack(float3 normal, float4 tangent)
{
    const uint32_t mask = (1u << (ATTRENC_TANGENT_BITS - 1)) - 1u;

    float3 autoTangent;
    float3 autoBitangent;
    TangentOrthonormalBasis(normal, autoTangent, autoBitangent);

    float3 t3 = float3(tangent);
    float  angle = ATTRENC_ATAN2F(dot(autoTangent, t3), dot(autoBitangent, t3)) / ATTRENC_PI;

    float    angleUnorm = std::fmin(std::fmax((angle + 1.0f) * 0.5f, 0.0f), 1.0f);
    uint32_t angleBits  = uint32_t(angleUnorm * float(mask) + 0.5f);
    uint32_t encoded    = (angleBits << 1u) | (tangent.w > 0.0f ? 1u : 0u);
    return encoded;
}

ATTRENC_INLINE float4 TangentUnpack(float3 normal, uint32_t encoded)
{
    const uint32_t mask = (1u << (ATTRENC_TANGENT_BITS - 1)) - 1u;

    uint32_t signBit   = encoded & 1u;
    uint32_t angleBits = (encoded >> 1u) & mask;

    float angleUnorm = float(angleBits) / float(mask);
    float angle      = ((angleUnorm * 2.0f) - 1.0f) * ATTRENC_PI;

    float3 autoTangent;
    float3 autoBitangent;
    TangentOrthonormalBasis(normal, autoTangent, autoBitangent);

    float3 tangent = std::cos(angle) * autoBitangent + std::sin(angle) * autoTangent;
    float  w       = (signBit == 1u) ? 1.0f : -1.0f;
    return float4(tangent, w);
}

#undef ATTRENC_ABS
#undef ATTRENC_FLOOR
#undef ATTRENC_CLAMP
#undef ATTRENC_INLINE
#undef ATTRENC_OUT
#undef ATTRENC_ATAN2F

} // namespace shaderio
