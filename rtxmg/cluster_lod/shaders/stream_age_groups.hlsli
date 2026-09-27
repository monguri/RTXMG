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

// stream_age_groups.hlsli
// The streaming age filter, shared by stream_age_groups.hlsl and by
// traversal_blas_merging.hlsl, which runs it in place of that dispatch.
//
// Include AFTER the including file's own u_ResidentGroups / streamingRW /
// u_GroupIDs / u_UnloadGeometryGroups declarations, and after its definition of
// GetCachedBlasLodLevel(): the two callers sit on different binding layouts and
// reach the geometry table through different views.

#pragma once

void StreamingAgeFilter(uint residentID, uint geometryID, bool useBlasCaching)
{
    // increase the age of a resident group
    uint lodLevel = uint(u_ResidentGroups[residentID].lodLevel);
    uint age      = uint(u_ResidentGroups[residentID].age);

    if (useBlasCaching)
    {
        uint cachedLevel = GetCachedBlasLodLevel(geometryID);

        // keep cached levels alive
        if (lodLevel >= cachedLevel)
        {
            age = 0u;
        }
    }

    if (age < 0xFFFFu)
    {
        age++;
        u_ResidentGroups[residentID].age = uint16_t(age);
    }

    // detect if we are over the age limit and request the group to be unloaded
    if (age > uint(streamingRW[0].ageThreshold))
    {
        uint unloadOffset;
        InterlockedAdd(streamingRW[0].request.unloadCounter, 1u, unloadOffset);
        if (unloadOffset < streamingRW[0].request.maxUnloads)
        {
            const uint slotBase = streamingRW[0].request.taskIndex
                                * streamingRW[0].request.taskSlotStride;
            u_UnloadGeometryGroups[slotBase
                                   + streamingRW[0].request.unloadGroupsOffsetElems
                                   + unloadOffset] = uint2(geometryID, u_GroupIDs[residentID]);
        }
    }
}
