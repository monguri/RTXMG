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


  This compute shader inserts the CLAS clusters that should be rendered
  into the cluster references list for each instance's BLAS.

  A single thread represents one CLAS

  Binding notes:
  --------------

  * Buffers are discretely bound; slots match m_insertLayout in [blas_pass.cpp].
  * Preloaded and streaming share one CLAS-address path: both funnel through the
    scene-global `m_residentClasAddrs` SRV (t4), indexed by `clusterID`.
  * Per-BLAS cluster references live in a single `u_BlasClasAddrs` pool: each
    instance writes into the sub-range `[blasClasOffset, blasClasOffset+N)` that
    blas_reserve_clusters reserved for it, and `u_BlasArgs[buildIndex]` holds the
    matching `clusterAddresses` VA + `clusterCount` for the BLAS builder.
  * Per-frame counters come from SceneBuildingCounters + the ClusterLod
    traversal async readback in blas_pass.cpp.
*/

#pragma pack_matrix(row_major)

#include "rtxmg/cluster_lod/shaderio.h"
#include "rtxmg/cluster_lod/shader_registers.h"
#include "rtxmg/cluster_lod/blas_build_params.h"
#include <nvrhi/nvrhiHLSL.h>

// traversal_common.hlsli is deliberately not included here: this kernel has its
// own layout (m_insertLayout) and needs none of the traversal-only bindings.

////////////////////////////////////////////

ConstantBuffer<BlasBuildParams>                          g_Params              : register(b0);

RWStructuredBuffer<nvrhi::rt::cluster::IndirectArgs>     u_BlasArgs            : CLOD_UAV(CLOD_U_BLAS_ARGS);
RWStructuredBuffer<uint64_t>                             u_BlasClasAddrs       : CLOD_UAV(CLOD_U_BLAS_CLAS_ADDRS);

StructuredBuffer<shaderio::Geometry>                     t_Geometries          : CLOD_SRV(CLOD_T_GEOMETRIES);
StructuredBuffer<shaderio::RenderInstance>               t_Instances           : CLOD_SRV(CLOD_T_RENDER_INSTANCES);
StructuredBuffer<shaderio::SceneBuildingCounters>        t_Counters            : CLOD_SRV(CLOD_T_COUNTERS);
StructuredBuffer<shaderio::ClusterInfo>                  t_RenderClusters      : CLOD_SRV(CLOD_T_RENDER_CLUSTERS);
StructuredBuffer<uint64_t>                               t_ResidentClasAddrs   : CLOD_SRV(CLOD_T_RESIDENT_CLAS_ADDRS);
StructuredBuffer<shaderio::InstanceBuildInfo>            t_InstanceBuildInfos  : CLOD_SRV(CLOD_T_INSTANCE_BUILD_INFOS);

////////////////////////////////////////////

[numthreads(shaderio::kBlasInsertThreads, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID)
{
    uint renderClusterIndex = dtid.x;

    if (renderClusterIndex < t_Counters[0].renderClusterCounter)
    {
        shaderio::ClusterInfo cluster = t_RenderClusters[renderClusterIndex];
        uint     instanceID    = cluster.instanceID;
        uint     clusterID     = cluster.clusterID;

        uint64_t clusterAddress = t_ResidentClasAddrs[clusterID];

        uint buildIndex     = t_InstanceBuildInfos[instanceID].blasBuildIndex;
        uint blasClasOffset = t_InstanceBuildInfos[instanceID].blasClasOffset;

        // LowDetail / Share / Cache instances reuse another BLAS: blasBuildIndex
        // is a sentinel, not a build slot, and traversal emits no clusters for them.
        if (buildIndex == shaderio::BlasBuildIndex::LowDetail
            || (buildIndex & shaderio::BlasBuildIndex::IndirectMask) != 0u)
            return;

        uint idx;
        InterlockedAdd(u_BlasArgs[buildIndex].clusterCount, 1u, idx);

        // Write the CLAS VA into this instance's sub-range of the global pool.
        // u_BlasArgs[buildIndex].clusterAddresses already points at
        // (blasClasAddressesBaseVA + blasClasOffset * sizeof(GpuVA)), set by
        // blas_reserve_clusters.
        u_BlasClasAddrs[blasClasOffset + idx] = clusterAddress;
    }
}
