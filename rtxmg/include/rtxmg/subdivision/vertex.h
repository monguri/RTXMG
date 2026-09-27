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

#pragma once

#ifdef __cplusplus
#include <donut/core/math/math.h>
#include <cstdint>

using namespace donut::math;
#endif

static const uint32_t kPatchSize = 16;

struct LimitFrame
{
    float3 p;
    float3 deriv1;
    float3 deriv2;

    void Clear()
    {
        p = float3(0, 0, 0);
        deriv1 = float3(0, 0, 0);
        deriv2 = float3(0, 0, 0);
    }

    void AddWithWeight(float3 src,
        float weight, float d1Weight, float d2Weight)
    {
        p += weight * src;
        deriv1 += d1Weight * src;
        deriv2 += d2Weight * src;
    }
};

// Texture coordinate with partial derivs w.r.t the parametric U and V directions of the surface
struct TexCoordLimitFrame
{
    float2 uv;
    float2 deriv1;  // (dST/du)
    float2 deriv2;  // (dST/du)

    void Clear()
    {
        uv = float2(0,0);
        deriv1 = float2(0,0);
        deriv2 = float2(0,0);
    }

    void AddWithWeight(float2 src, float weight, float du_weight, float dv_weight)
    {
        uv += weight * src;
        deriv1 += du_weight * src;
        deriv2 += dv_weight * src;
    }
};