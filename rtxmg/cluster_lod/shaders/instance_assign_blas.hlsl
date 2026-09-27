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


  This compute shader assigns the blasReference address to each
  tlas instance description prior updating the tlas and after
  the per-frame blas were built.

  A single thread represents one instance.

  Binding notes:
  --------------

  * Buffers are discretely bound at fixed register slots.
    Slots match m_updateBlasLayout in [blas_pass.cpp]:
      b0 = BlasBuildParams
      t0 = InstanceBuildInfos (SRV)
      t1 = BlasAddresses      (SRV)  — per-buildIndex BLAS VA pool
      u0 = InstanceBlasAddrs  (UAV)  — output VA[numRenderInstances]
  * shaderio::BlasBuildIndex lives in shaderio.h.
    LowDetail is seeded by traversal_init and overwritten by
    blas_reserve_clusters for slow-path instances; ShareBit / CacheBit are
    set by traversal_init_blas_sharing.  USE_BLAS_SHARING and USE_BLAS_CACHING
    are per-PSO permutations (shaders.cfg compiles both values).
  * BLAS VAs are nvrhi::GpuVirtualAddress (uint64_t) end-to-end — both host
    buffers (m_blasAddresses @ t1, m_instanceBlasAddrs @ u0) are
    RTXMGBuffer<nvrhi::GpuVirtualAddress>, so no uint2 lo/hi packing is needed.
*/

#pragma pack_matrix(row_major)

#include "rtxmg/cluster_lod/shaderio.h"
#include "rtxmg/cluster_lod/shader_registers.h"
#include "feature_gates.hlsli"
#include "rtxmg/cluster_lod/blas_build_params.h"

// kInstancesAssignBlasThreads and shaderio::BlasBuildIndex::LowDetail come from shaderio.h.

////////////////////////////////////////////

ConstantBuffer<BlasBuildParams>                       g_Params              : register(b0);
StructuredBuffer<shaderio::InstanceBuildInfo>         t_InstanceBuildInfos  : CLOD_SRV(CLOD_T_INSTANCE_BUILD_INFOS);
StructuredBuffer<nvrhi::GpuVirtualAddress>            t_BlasAddresses       : CLOD_SRV(CLOD_T_BLAS_ADDRESSES);
RWStructuredBuffer<nvrhi::GpuVirtualAddress>          u_InstanceBlasAddrs   : CLOD_UAV(CLOD_U_INSTANCE_BLAS_ADDRS);

// Geometries — read for the cached-BLAS VA (USE_BLAS_CACHING) and, under
// TRACK_RENDER_STATS, for the per-geometry low-detail counts.  Declared
// unconditionally because the binding layout is fixed.
StructuredBuffer<shaderio::Geometry>                  t_Geometries          : CLOD_SRV(CLOD_T_GEOMETRIES);

// Render-stats (TRACK_RENDER_STATS): always bound, touched only under the gate.
// t_PerInstanceTriangles holds the traversal-side per-instance tally, which this
// shader resolves through each instance's owner; u_PerGeomSeen dedups the
// per-geometry low-detail / cached contribution to the unique count.
RWStructuredBuffer<shaderio::SceneBuildingCounters>   u_Counters             : CLOD_UAV(CLOD_U_COUNTERS);
RWStructuredBuffer<uint>                              u_PerGeomSeen          : CLOD_UAV(CLOD_U_PER_GEOM_SEEN);
StructuredBuffer<uint>                                t_PerInstanceTriangles : CLOD_SRV(CLOD_T_PER_INSTANCE_TRIANGLES);
StructuredBuffer<shaderio::RenderInstance>            t_Instances            : CLOD_SRV(CLOD_T_RENDER_INSTANCES);

// BLAS build stats are not accumulated here: numBlasBuilds is already
// SceneBuildingCounters.blasBuildCounter, and the host sums the built-BLAS
// bytes from m_blasSizes[0..blasBuildCounter).

////////////////////////////////////////////

[numthreads(shaderio::kInstancesAssignBlasThreads, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID)
{
    uint instanceID = gid.x;

    if (instanceID >= g_Params.numRenderInstances)
        return;

    shaderio::InstanceBuildInfo buildInfo = t_InstanceBuildInfos[instanceID];
    uint buildIndex = buildInfo.blasBuildIndex;

    nvrhi::GpuVirtualAddress cachedBlasAddr = 0;

    // Render-stats: which instance's traversal-side triangle tally this instance
    // inherits (self for a per-instance/merged-proxy build; the provider/merged
    // proxy for a sharing/merging consumer), and whether it uses a cached BLAS.
    uint triOwnerInstance  = instanceID;
    bool isCachedInstance  = false;

#if USE_BLAS_SHARING
    // An instance whose blasBuildIndex carries ShareBit (consumer) or CacheBit
    // references another instance's / the geometry's cached BLAS rather than
    // owning one.  traversal_init_blas_sharing is the only producer of those
    // bits; plain build slots are small integers.
    if (buildIndex != shaderio::BlasBuildIndex::LowDetail &&
        (buildIndex & shaderio::BlasBuildIndex::IndirectMask) != 0u)
    {
        uint lookupIndex = buildIndex & ~shaderio::BlasBuildIndex::IndirectMask;
    #if USE_BLAS_CACHING
        if ((buildIndex & shaderio::BlasBuildIndex::CacheBit) != 0u)
        {
            cachedBlasAddr   = t_Geometries[lookupIndex].cachedBlasAddress;
            isCachedInstance = true;
        }
        else
    #endif
        {
            buildIndex        = t_InstanceBuildInfos[lookupIndex].blasBuildIndex;
            triOwnerInstance  = lookupIndex;  // provider / merged proxy traversed
        }
    }
#endif // USE_BLAS_SHARING

    // By default tlasInstances are set to low detail blas (seeded by
    // traversal_init); override when blas_reserve_clusters promoted this
    // instance to a real build slot.
    if (buildIndex != shaderio::BlasBuildIndex::LowDetail)
    {
        u_InstanceBlasAddrs[instanceID] =
        #if USE_BLAS_CACHING
            (cachedBlasAddr != 0) ? cachedBlasAddr :
        #endif
            t_BlasAddresses[buildIndex];
    }

#if TRACK_RENDER_STATS
    // Render-stats: fold this instance's BLAS triangles + clusters into the
    // instanced totals, and the per-geometry low-detail BLAS into the unique
    // counts (once).  The traversed (built / shared / merged) clusters were
    // already counted toward unique in the traversal pass; here we add the
    // instanced totals for every instance plus the low-detail unique addition
    // (low-detail instances skip traversal, so their geometry isn't in the
    // per-clusterID seen-set).
    {
        uint geometryID   = t_Instances[instanceID].geometryID;
        uint instTris     = 0u;
        uint instClusters = 0u;
        bool lowDetail    = false;
        bool cached       = false;

        if (isCachedInstance)
        {
            // Cached instances reference the geometry's cached BLAS instead of
            // traversing, so their geometry isn't in the traversal seen-set.
            // Like low-detail: add the cached level to the instanced totals
            // (per instance) and to unique once per geometry.
            instTris     = t_Geometries[geometryID].cachedBlasTriangles;
            instClusters = t_Geometries[geometryID].cachedBlasClusters;
            cached       = true;
        }
        else if (buildIndex != shaderio::BlasBuildIndex::LowDetail)
        {
            // Built / shared-provider / merged-proxy: the owner instance
            // traversed and accumulated both tallies.
            instTris     = t_PerInstanceTriangles[triOwnerInstance];
            instClusters = t_InstanceBuildInfos[triOwnerInstance].clusterReferencesCount;
        }
        else
        {
            instTris     = t_Geometries[geometryID].lowDetailTriangles;
            instClusters = t_Geometries[geometryID].lowDetailClusters;
            lowDetail    = true;
        }

        // Instanced totals (per instance — mirrors the ray tracer's view).
        if (instTris != 0u)
        {
            uint64_t prevTT;
            InterlockedAdd(u_Counters[0].totalTriangles, (uint64_t)instTris, prevTT);
        }
        if (instClusters != 0u)
        {
            uint prevTC;
            InterlockedAdd(u_Counters[0].totalClusters, instClusters, prevTC);
        }

        // Instanced cached-BLAS contribution: the slice of the instanced totals
        // served by cached BLASes (a subset of total*).
        if (cached)
        {
            if (instTris != 0u)
            {
                uint64_t prevCT;
                InterlockedAdd(u_Counters[0].cachedTriangles, (uint64_t)instTris, prevCT);
            }
            if (instClusters != 0u)
            {
                uint prevCC;
                InterlockedAdd(u_Counters[0].cachedClusters, instClusters, prevCC);
            }
        }

        // Low-detail unique: its geometry skips traversal, so add it once per
        // geometry (bit 0 of the per-geom seen mask).
        if (lowDetail && (instTris != 0u || instClusters != 0u))
        {
            uint prevSeen;
            InterlockedOr(u_PerGeomSeen[geometryID], 1u, prevSeen);
            if ((prevSeen & 1u) == 0u)
            {
                uint64_t prevUT;
                InterlockedAdd(u_Counters[0].uniqueTriangles, (uint64_t)instTris, prevUT);
                uint prevUC;
                InterlockedAdd(u_Counters[0].uniqueClusters, instClusters, prevUC);
            }
        }

        // Cached unique: the cached BLAS is shared by all its instances and its
        // level isn't traversed, so add it once per geometry (bit 1).  Distinct
        // from low-detail's bit 0 so the two coexist on the same geometry.
        if (cached && (instTris != 0u || instClusters != 0u))
        {
            uint prevSeen;
            InterlockedOr(u_PerGeomSeen[geometryID], 2u, prevSeen);
            if ((prevSeen & 2u) == 0u)
            {
                uint64_t prevUT;
                InterlockedAdd(u_Counters[0].uniqueTriangles, (uint64_t)instTris, prevUT);
                uint prevUC;
                InterlockedAdd(u_Counters[0].uniqueClusters, instClusters, prevUC);
                // Also track the cached slice of the unique totals separately
                // (deduped, once per geometry) for the Cached (unique) stat.
                uint64_t prevCUT;
                InterlockedAdd(u_Counters[0].cachedUniqueTriangles, (uint64_t)instTris, prevCUT);
                uint prevCUC;
                InterlockedAdd(u_Counters[0].cachedUniqueClusters, instClusters, prevCUC);
            }
        }
    }
#endif
}
