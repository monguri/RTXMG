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

#include "lerp_keyframes_params.h"

StructuredBuffer<float3> kf0 : register(t0);
StructuredBuffer<float3> kf1 : register(t1);

RWStructuredBuffer<float3> dst : register(u0);

ConstantBuffer<LerpKeyFramesParams> g_lerpParams: register(b0);

[numthreads(32, 1, 1)]
void main(uint3 threadIdx : SV_DispatchThreadID)
{
    const uint32_t vertexIndex = threadIdx.x;
    if (vertexIndex >= g_lerpParams.numVertices)
        return;
    const float3 v0 = kf0[vertexIndex];
    const float3 v1 = kf1[vertexIndex];

    dst[vertexIndex] = lerp(v0, v1, g_lerpParams.animTime);
}
