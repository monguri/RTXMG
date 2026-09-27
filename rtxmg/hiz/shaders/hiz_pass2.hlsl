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

[numthreads(HIZ_GROUP_SIZE, HIZ_GROUP_SIZE, 1)]
void main(uint2 threadIdx : SV_GroupThreadID, uint2 dispatchThreadId : SV_DispatchThreadID, uint2 groupIdx : SV_GroupID)
{
    uint32_t x = dispatchThreadId.x;
    uint32_t y = dispatchThreadId.y;

    uint32_t hizwidth, hizheight;
    u_output[0].GetDimensions(hizwidth, hizheight);

    // current LOD may be smaller than a single tile
    uint2 size = uint2(hizwidth >> 4, hizheight >> 4);
    if (x >= size.x || y >= size.y)
        return;


    float value = u_output[4][dispatchThreadId];

#pragma unroll
    for (uint16_t level = 5; level < HIZ_MAX_LODS; level++)
    {
        uint16_t outGroupSize = ((uint16_t)HIZ_GROUP_SIZE) >> (level - 4);
        uint16_t inGroupSize = outGroupSize << 1;

        if (threadIdx.x < inGroupSize && threadIdx.y < inGroupSize)
        {
            s_reductionData[threadIdx.y][threadIdx.x] = value;
        }

        x = groupIdx.x * outGroupSize + threadIdx.x;
        y = groupIdx.y * outGroupSize + threadIdx.y;

        // the base level is guaranteed to be a multiple of 32, so we won't have an 
        // odd sized parent level until we go from L5 --> L6.

        bool extraRow = (size.x & 1) != 0;
        bool extraCol = (size.y & 1) != 0;

        size = uint2(size.x >> 1, size.y >> 1);
        if (x >= size.x || y >= size.y)
            return;

        GroupMemoryBarrierWithGroupSync();

        if (threadIdx.x < outGroupSize && threadIdx.y < outGroupSize)
        {
            float a = s_reductionData[threadIdx.y * 2 + 0][threadIdx.x * 2 + 0];
            float b = s_reductionData[threadIdx.y * 2 + 0][threadIdx.x * 2 + 1];
            float c = s_reductionData[threadIdx.y * 2 + 1][threadIdx.x * 2 + 0];
            float d = s_reductionData[threadIdx.y * 2 + 1][threadIdx.x * 2 + 1];

            value = Reduce(float4(a, b, c, d));
#if 1
            if (extraCol)
            {
                // Get the two values to the right
                a = s_reductionData[threadIdx.y * 2 + 0][threadIdx.x * 2 + 2];
                b = s_reductionData[threadIdx.y * 2 + 1][threadIdx.x * 2 + 2];

                if (extraRow)
                {
                    // Get the corner value
                    c = s_reductionData[threadIdx.y * 2 + 2][threadIdx.x * 2 + 2];
                }

                // okay to re-use d here because reduce is a maximum
                value = Reduce(float4(a, b, c, d));
            }
            if (extraRow)
            {
                // Get the two values below
                a = s_reductionData[threadIdx.y * 2 + 2][threadIdx.x * 2 + 0];
                b = s_reductionData[threadIdx.y * 2 + 2][threadIdx.x * 2 + 1];

                // okay to re-use c and d here because reduce is a maximum
                value = Reduce(float4(a, b, c, d));
            }
#endif
            u_output[level][uint2(x, y)] = value;
        }

        GroupMemoryBarrierWithGroupSync();
    }
}