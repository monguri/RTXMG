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

#include "rtxmg/hiz/hiz_buffer_reduce_params.h"
#include "rtxmg/hiz/hiz_buffer_constants.h"

Texture2D<float> t_zbuffer: register(t0);
StructuredBuffer<HiZReducePass1Params> g_params: register(t1);
RWTexture2D<float> u_output[HIZ_MAX_LODS]: register(u0);

groupshared float s_reductionData[HIZ_GROUP_SIZE][HIZ_GROUP_SIZE];
SamplerState s : register(s0);

inline float Reduce(float a, float b)
{
    return max(a, b);
}

inline float Reduce(float4 a)
{
    return Reduce(Reduce(a.x, a.y), Reduce(a.z, a.w));
}

// perform 8x reduction using fixed-function pixel gather-4
inline float GetZFarFromTile(uint2 tile, int tilesize)
{
    float2 uv = (float2(tile) * float(tilesize) + 1.f);

    float zfar = 0;

    uint2 offset;

#pragma unroll 4
    for (offset.y = 0; offset.y < tilesize; offset.y += 2)
    {
#pragma unroll 4
        for (offset.x = 0; offset.x < tilesize; offset.x += 2)
        {
            float x = (uv.x + float(offset.x)) * g_params[0].zBufferInvSize.x;
            float y = (uv.y + float(offset.y)) * g_params[0].zBufferInvSize.y;

            float4 values = t_zbuffer.Gather(s, float2(x, y), 0);
            zfar = Reduce(zfar, Reduce(values));
        }
    }
    return zfar;
}

[numthreads(HIZ_GROUP_SIZE, HIZ_GROUP_SIZE, 1)]
void main(uint2 threadIdx : SV_GroupThreadID, uint2 dispatchThreadId : SV_DispatchThreadID, uint2 groupIdx : SV_GroupID)
{
    uint32_t x = dispatchThreadId.x;
    uint32_t y = dispatchThreadId.y;


    float value = GetZFarFromTile(uint2(x, y), HIZ_LOD0_TILE_SIZE);
    u_output[0][dispatchThreadId.xy] = value;

    // level 0 dimensions are always a multiple of 16
    // so this reduction will never miss any pixels

    // reduce the next 4 LODs using shared memory across each block
#pragma unroll
    for (uint16_t level = 1; level < 5; ++level)
    {
        uint16_t outGroupSize = ((uint16_t)HIZ_GROUP_SIZE) >> level;
        uint16_t inGroupSize = outGroupSize << 1;

        if (threadIdx.x < inGroupSize && threadIdx.y < inGroupSize)
        {
            s_reductionData[threadIdx.y][threadIdx.x] = value;
        }

        GroupMemoryBarrierWithGroupSync();

        if (threadIdx.x < outGroupSize && threadIdx.y < outGroupSize)
        {
            float a = s_reductionData[threadIdx.y * 2 + 0][threadIdx.x * 2 + 0];
            float b = s_reductionData[threadIdx.y * 2 + 0][threadIdx.x * 2 + 1];
            float c = s_reductionData[threadIdx.y * 2 + 1][threadIdx.x * 2 + 0];
            float d = s_reductionData[threadIdx.y * 2 + 1][threadIdx.x * 2 + 1];

            value = Reduce(float4(a, b, c, d));

            uint32_t x = groupIdx.x * outGroupSize + threadIdx.x;
            uint32_t y = groupIdx.y * outGroupSize + threadIdx.y;

            u_output[level][uint2(x, y)] = value;
        }
        GroupMemoryBarrierWithGroupSync();
    }
}