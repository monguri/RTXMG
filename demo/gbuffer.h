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
//

#ifndef GBUFFER_H  // using instead of "#pragma once" due to https://github.com/microsoft/DirectXShaderCompiler/issues/3943
#define GBUFFER_H

typedef float DepthFormat;
typedef float4 NormalFormat;
typedef float4 AlbedoFormat;
typedef float4 SpecularFormat;
typedef float SpecularHitTFormat;
typedef float RoughnessFormat;

static const uint32_t kInvalidInstanceId = ~0u;
static const uint32_t kInvalidSurfaceIndex = ~0u;

struct HitResult
{
    uint32_t instanceId;
    uint32_t surfaceIndex;
    float2 surfaceUV;
    float2 texcoord; // For displacement texture
};

#ifndef __cplusplus
HitResult DefaultHitResult()
{
    HitResult result;
    result.instanceId = kInvalidInstanceId;
    result.surfaceIndex = kInvalidSurfaceIndex;
    result.surfaceUV = float2(0.0f, 0.f);
    result.texcoord = float2(0.0f, 0.f);

    return result;
}
#endif

#endif // GBUFFER_H