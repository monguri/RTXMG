/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
  Only used with USE_BLAS_CACHING && streaming.

  Sets up the cluster-AS MOVE_OBJECTS op that copies each freshly-built cached
  BLAS out of the per-frame BLAS scratch storage into its persistent
  cached-BLAS pool allocation.  One thread per cached-BLAS patch.

  Implementation notes
  --------------------
  * Buffers are discretely bound (slots match m_cachingCopyLayout in
    blas_pass.cpp); the copy counter lives in SceneBuildingCounters (u0).
  * The per-build BLAS address (blasBuildAddresses[buildIndex]) is the move
    source; sgpatch.cachedBlasAddress (the pool allocation handed out by
    HandleBlasCaching) is the destination.
*/

#pragma pack_matrix(row_major)

#include "rtxmg/cluster_lod/shaderio.h"
#include "rtxmg/cluster_lod/shader_registers.h"
#include "rtxmg/cluster_lod/blas_build_params.h"

////////////////////////////////////////////

ConstantBuffer<BlasBuildParams>                      g_Params                 : register(b0);

StructuredBuffer<uint64_t>                           t_BlasBuildAddresses     : CLOD_SRV(CLOD_T_BLAS_ADDRESSES);
StructuredBuffer<shaderio::StreamingGeometryPatch>   t_GeometryPatches        : CLOD_SRV(CLOD_T_GEOMETRY_PATCHES);
StructuredBuffer<shaderio::GeometryBuildInfo>        t_GeometryBuildInfos     : CLOD_SRV(CLOD_T_GEOMETRY_BUILD_INFOS);

RWStructuredBuffer<shaderio::SceneBuildingCounters>  u_Counters               : CLOD_UAV(CLOD_U_COUNTERS);
RWStructuredBuffer<uint64_t>                         u_CachedBlasAddressesSrc : CLOD_UAV(CLOD_U_CACHED_BLAS_SRC);
RWStructuredBuffer<uint64_t>                         u_CachedBlasAddressesDst : CLOD_UAV(CLOD_U_CACHED_BLAS_DST);

////////////////////////////////////////////

[numthreads(shaderio::kBlasCachingSetupCopyThreads, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID)
{
    uint threadID = dtid.x;

    // Load inside the count guard.  The dispatch rounds up to a thread group, and
    // unlike stream_age_groups's over-allocated u_ActiveGroups there is no
    // padding past patchCachedBlasCount here, so padding lanes would read out
    // of bounds.
    if (threadID >= g_Params.patchCachedBlasCount)
        return;

    shaderio::StreamingGeometryPatch sgpatch = t_GeometryPatches[threadID];
    uint64_t dstAddress                      = sgpatch.cachedBlasAddress;

    // Some patches carry a null destination: a cached BLAS fully removed this
    // frame.
    if (dstAddress != uint64_t(0))
    {
        uint buildIndex = t_GeometryBuildInfos[sgpatch.geometryID].cachedBuildIndex;

        uint copyOffset;
        InterlockedAdd(u_Counters[0].cachedBlasCopyCounter, 1u, copyOffset);

        u_CachedBlasAddressesSrc[copyOffset] = t_BlasBuildAddresses[buildIndex];
        u_CachedBlasAddressesDst[copyOffset] = dstAddress;
    }
}
