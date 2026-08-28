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

#include "rtxmg/hiz/hiz_buffer_display_params.h"
#include "rtxmg/hiz/hiz_buffer_constants.h"

ConstantBuffer<HiZDisplayParams> g_params: register(b0);
Texture2D<float> u_hiz[HIZ_MAX_LODS]: register(t0);
RWTexture2D<float4> output: register(u0);

[numthreads(32, 32, 1)]
void main(uint2 threadIdx : SV_GroupThreadID, uint2 dispatchThreadId : SV_DispatchThreadID)
{
    uint32_t width, height;
    u_hiz[g_params.level].GetDimensions(width, height);

    uint32_t outWidth, outHeight;
    output.GetDimensions(outWidth, outHeight);

    uint32_t x = dispatchThreadId.x;
    uint32_t y = dispatchThreadId.y;

    if ((x >= width) || (y >= height))
    {
        return;
    }

    float depth = u_hiz[g_params.level][dispatchThreadId];

    uint32_t2 outputIdx = uint32_t2(x + g_params.offsetX, outHeight + y - height - g_params.offsetY);

    if (isinf(depth))
    {
        output[outputIdx] = float4(1, 0, 0, 0);
    }
    else
    {
        depth = (depth <= 0.f) ? 1.f : 1.f / depth;
        output[outputIdx] = float4(depth, depth, depth, 1);
    }
}