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

#pragma once

#include "rtxmg/utils/box3.h"
#include "nvrhi/nvrhiHLSL.h"

static const uint32_t kComputeClusterTilingWaves = 4;

struct ComputeClusterTilingParams
{
    uint32_t surfaceStart; //inclusive
    uint32_t surfaceEnd; //exclusive
    uint32_t debugSurfaceIndex;
    uint32_t debugLaneIndex;
    
    float4x4 matWorldToClip;
    float3x4 localToWorld;

    float3 cameraPos;
    float pad0;

    Box3 aabb;

    uint4 edgeSegments;

    uint isolationLevel;
    float fineTessellationRate;
    float coarseTessellationRate;
    uint pad1;

    float2 viewportSize;
    float2 invHiZSize;

    int enableFrustumVisibility;
    int enableBackfaceVisibility;
    int enableHiZVisibility;
    int numHiZLODs;
    
    float globalDisplacementScale;
    uint maxClusters;
    uint maxVertices;
    uint maxClasBlocks;

    nvrhi::GpuVirtualAddress clasDataBaseAddress;
    nvrhi::GpuVirtualAddress clusterVertexPositionsBaseAddress;
};

#if defined(__cplusplus)
static_assert(sizeof(ComputeClusterTilingParams) % 16 == 0);
#elif defined(TARGET_D3D12)
_Static_assert(sizeof(ComputeClusterTilingParams) % 16 == 0, "Must be 16 byte aligned");
#endif
