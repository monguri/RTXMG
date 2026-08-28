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

#pragma pack_matrix(row_major)

#include "zrender_params.h"
#include <donut/shaders/bindless.h>
#include <donut/shaders/packing.hlsli>
#include <donut/shaders/surface.hlsli>
#include <donut/shaders/utils.hlsli>
#include <donut/shaders/binding_helpers.hlsli>

ConstantBuffer<ZRenderParams> g_RenderParams : register(b0);

RWTexture2D<float> u_Output    : register(u0);

RaytracingAccelerationStructure SceneBVH : register(t0);

void GetRay(uint2 pixelPosition, float2 subpixelJitter, uint2 dims, out float3 rayOrigin, out float3 rayDirection)
{
    float2 d = ((float2(pixelPosition) + subpixelJitter) / float2(dims)) * 2.f - 1.f;

    d *= float2(1, -1);

    rayOrigin = g_RenderParams.eye;
    rayDirection = normalize(d.x * g_RenderParams.U + d.y * g_RenderParams.V + g_RenderParams.W);
}

[shader("miss")] void Miss(inout ZRayPayload payload
    : SV_RayPayload)
{
    payload.hitT = 1.#INF;
}

[shader("closesthit")]void ClosestHit(inout ZRayPayload payload
    : SV_RayPayload, in BuiltInTriangleIntersectionAttributes attrib
    : SV_IntersectionAttributes)
{
    payload.hitT = min(payload.hitT, RayTCurrent());
}

[shader("raygeneration")]void RayGen()
{
    uint2 pixelPosition = DispatchRaysIndex().xy;
    uint2 dims = DispatchRaysDimensions().xy;

    float2 subpixelJitter = float2(0.5f, 0.5f);

    float3 rayOrigin, rayDirection;
    GetRay(pixelPosition, subpixelJitter, dims, rayOrigin, rayDirection);

    RayDesc ray = { rayOrigin, 0.f, rayDirection, 1e38 };

    ZRayPayload payload = { 1.#INF };

    // Only instances with the second InstanceMask bit set will be intersected.
    // This prevents animated objects from being written to the depth buffer.
    TraceRay(SceneBVH, RAY_FLAG_NONE, 2, 0, 0, 0, ray, payload);

    float hitT = payload.hitT;
    float3 hitP = rayOrigin + rayDirection * hitT;
    float depth = (!isinf(hitT)) ? dot(normalize(g_RenderParams.W), hitP - g_RenderParams.eye) : 1.#INF;

    u_Output[pixelPosition] = depth;
}
