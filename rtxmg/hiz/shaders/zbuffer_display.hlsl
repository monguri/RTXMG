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

Texture2D<float> zbuffer: register(t0);
StructuredBuffer<float> minmax: register(t1);

RWTexture2D<float4> output: register(u0);

[numthreads(16, 16, 1)]
void main(uint2 threadIdx : SV_GroupThreadID, uint2 dispatchThreadId : SV_DispatchThreadID)
{
    uint32_t width, height;
    zbuffer.GetDimensions(width, height);

    uint32_t x = dispatchThreadId.x;
    uint32_t y = dispatchThreadId.y;

    if ((x >= width) || (y >= height))
    {
        return;
    }

    float depth = zbuffer[dispatchThreadId];
    if (!isinf(depth))
    {
        float minz = minmax[0];
        float maxz = minmax[1];
        float deltaz = maxz - minz;
        depth = deltaz > 1e-6 ? (depth - minz) / deltaz : .5f;
        output[dispatchThreadId] = float4(depth, depth, depth, 1);
    }
}