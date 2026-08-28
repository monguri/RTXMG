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

#include "rtxmg/cluster_tess/copy_cluster_offset_params.h"
#include "rtxmg/cluster_tess/tessellation_counters.h"
#include "rtxmg/cluster_tess/fill_clusters_params.h"

StructuredBuffer<TessellationCounters> t_TessellationCounters : register(t0);
RWStructuredBuffer<uint2> u_ClusterOffsetCounts : register(u0);
RWStructuredBuffer<uint3> u_FillClustersIndirectArgs : register(u1);
ConstantBuffer<CopyClusterOffsetParams> g_Params : register(b0);

[numthreads(1, 1, 1)]
void main(uint3 threadIdx : SV_GroupThreadID, uint3 groupIdx : SV_GroupID)
{
    uint totalClusterCount = t_TessellationCounters[0].clusters;

    // Offsets goes by the order of ClusterTessDispatchType
    // PureBSpline Clusters
    // RegularBSpline Clusters
    // Limit Clusters
    // All Clusters
    if (g_Params.dispatchTypeIndex <= ClusterTessDispatchType::Limit)
    {
        uint dispatchIndex = g_Params.instanceIndex * ClusterTessDispatchType::NumTypes + g_Params.dispatchTypeIndex;
        uint dispatchClusterCount = 0;
        if (dispatchIndex == 0)
        {
            dispatchClusterCount = totalClusterCount;
            u_ClusterOffsetCounts[0] = uint2(0, dispatchClusterCount);
        }
        else
        {
            uint2 previousOffsetCount = u_ClusterOffsetCounts[dispatchIndex - 1];
            uint instanceOffset = previousOffsetCount.x + previousOffsetCount.y;
            dispatchClusterCount = totalClusterCount - instanceOffset;
            u_ClusterOffsetCounts[dispatchIndex] = uint2(instanceOffset, dispatchClusterCount);
        }

        // Write the number of clusters for the surface type
        const uint32_t vertThreadGroupsX = (dispatchClusterCount + kFillClustersVerticesWaves - 1) / kFillClustersVerticesWaves;
        u_FillClustersIndirectArgs[dispatchIndex] = uint3(vertThreadGroupsX, 1, 1);
    }

    // Write the total number of clusters for the instance
    if (g_Params.dispatchTypeIndex == ClusterTessDispatchType::Limit || g_Params.dispatchTypeIndex == ClusterTessDispatchType::All)
    {
        uint32_t instanceTotalIndex = g_Params.instanceIndex * ClusterTessDispatchType::NumTypes + ClusterTessDispatchType::All;
        uint dispatchClusterCount = 0;
        if (g_Params.instanceIndex == 0)
        {
            dispatchClusterCount = totalClusterCount;
            u_ClusterOffsetCounts[instanceTotalIndex] = uint2(0, dispatchClusterCount);
        }
        else
        {
            uint2 previousOffsetCount = u_ClusterOffsetCounts[(g_Params.instanceIndex - 1) * ClusterTessDispatchType::NumTypes + ClusterTessDispatchType::All];
            uint instanceOffset = previousOffsetCount.x + previousOffsetCount.y;
            dispatchClusterCount = totalClusterCount - instanceOffset;
            u_ClusterOffsetCounts[instanceTotalIndex] = uint2(instanceOffset, dispatchClusterCount);
        }

        if (g_Params.dispatchTypeIndex == ClusterTessDispatchType::All)
        {
            // Write the number of clusters for the surface type
            const uint32_t vertThreadGroupsX = (dispatchClusterCount + kFillClustersVerticesWaves - 1) / kFillClustersVerticesWaves;
            u_FillClustersIndirectArgs[g_Params.instanceIndex * ClusterTessDispatchType::NumTypes + ClusterTessDispatchType::Limit] = uint3(vertThreadGroupsX, 1, 1);
        }

        const uint32_t texcoordsThreadGroupsX = (dispatchClusterCount + kFillClustersTexcoordsThreadsX - 1) / kFillClustersTexcoordsThreadsX;
        u_FillClustersIndirectArgs[g_Params.instanceIndex * ClusterTessDispatchType::NumTypes + ClusterTessDispatchType::All] = uint3(texcoordsThreadGroupsX, 1, 1);
    }
}