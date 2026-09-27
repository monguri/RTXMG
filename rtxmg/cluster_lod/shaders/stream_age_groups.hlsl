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

/*

  Shader Description
  ==================

  This compute shader writes the streaming request for
  groups to be unloaded. We determine this based on an
  age since the group has been used last.

  A thread represents one resident group.

  Binding notes:
  --------------

  * Buffers are discretely bound at the fixed slots of the unified
    m_streamingBindingLayout (streaming.cpp), shared with the allocator family
    + stream_update_scene.
  * All streaming state lives in one RWStructuredBuffer<SceneStreaming>.
*/

#pragma pack_matrix(row_major)

#include "rtxmg/cluster_lod/shaderio.h"

////////////////////////////////////////////

// Unified streaming binding layout — see streaming.cpp's
// InitShadersAndPipelines comment for the full slot map.  u_ActiveGroups,
// u_GroupIDs and u_Geometries are read through UAV slots because a buffer
// can't be bound at both an SRV and a UAV slot inside one BindingSet (D3D12
// state conflict) and HLSL allows reading a RWStructuredBuffer.
RWStructuredBuffer<shaderio::Geometry>        u_Geometries            : register(u15);
RWStructuredBuffer<shaderio::StreamingGroup>  u_ResidentGroups        : register(u0);
RWStructuredBuffer<shaderio::SceneStreaming>  streamingRW             : register(u7);
RWStructuredBuffer<uint>                      u_ActiveGroups          : register(u8);
RWStructuredBuffer<uint>                      u_GroupIDs              : register(u9);
RWStructuredBuffer<uint2>                     u_UnloadGeometryGroups  : register(u12);

////////////////////////////////////////////

uint GetCachedBlasLodLevel(uint geometryID)
{
    return u_Geometries[geometryID].cachedBlasLodLevel;
}

#include "stream_age_groups.hlsli"

////////////////////////////////////////////

[numthreads(shaderio::kStreamAgeFilterGroupsThreads, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID)
{
    uint threadID = gid.x;

    // Load before the count guard: u_ActiveGroups is over-allocated by one
    // thread group past m_maxGroups (StreamingResident::Init) so padding lanes stay
    // in bounds.  It is bound WHOLE (Vulkan storage-buffer descriptor offsets
    // must be 16B-aligned), so the low-detail prefix skip is applied in-shader.
    uint residentID = u_ActiveGroups[streamingRW[0].resident.persistentGroupsCount + threadID];
    if (threadID < streamingRW[0].resident.activeGroupsCount)
    {
        uint geometryID = u_ResidentGroups[residentID].geometryID;

        StreamingAgeFilter(residentID, geometryID, streamingRW[0].useBlasCaching != 0u);
    }
}
