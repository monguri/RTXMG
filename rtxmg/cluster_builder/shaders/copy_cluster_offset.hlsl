/*
* Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a
* copy of this software and associated documentation files (the "Software"),
* to deal in the Software without restriction, including without limitation
* the rights to use, copy, modify, merge, publish, distribute, sublicense,
* and/or sell copies of the Software, and to permit persons to whom the
* Software is furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
* THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
* FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
* DEALINGS IN THE SOFTWARE.
*/
#pragma pack_matrix(row_major)

#include "rtxmg/cluster_builder/copy_cluster_offset_params.h"
#include "rtxmg/cluster_builder/tessellation_counters.h"
#include "rtxmg/cluster_builder/fill_clusters_params.h"

StructuredBuffer<TessellationCounters> t_TessellationCounters : register(t0);
RWStructuredBuffer<uint2> u_ClusterOffsetCounts : register(u0);
RWStructuredBuffer<uint3> u_FillClustersIndirectArgs : register(u1);
ConstantBuffer<CopyClusterOffsetParams> g_Params : register(b0);

[numthreads(1, 1, 1)]
void main(uint3 threadIdx : SV_GroupThreadID, uint3 groupIdx : SV_GroupID)
{
    uint currentClusterCount = t_TessellationCounters[0].clusters;
    
    uint instanceClusterCount = 0;
    if (g_Params.instanceIndex == 0)
    {
        instanceClusterCount = currentClusterCount;
        u_ClusterOffsetCounts[0] = uint2(0, instanceClusterCount);
    }
    else
    {
        uint2 previousOffsetCount = u_ClusterOffsetCounts[g_Params.instanceIndex - 1];
        uint instanceOffset = previousOffsetCount.x + previousOffsetCount.y;
        instanceClusterCount = currentClusterCount - instanceOffset;
        u_ClusterOffsetCounts[g_Params.instanceIndex] = uint2(instanceOffset, instanceClusterCount);
    }
    
    // Alternate between vertices and texcoord dispatch args
    const uint32_t vertThreadGroupsX = (instanceClusterCount + kFillClustersVerticesWaves - 1) / kFillClustersVerticesWaves;
    const uint32_t texcoordsThreadGroupsX = (instanceClusterCount + kFillClustersTexcoordsThreadsX - 1) / kFillClustersTexcoordsThreadsX;

    u_FillClustersIndirectArgs[g_Params.instanceIndex * kFillClustersPerInstanceIndirectArgCount + 0] = uint3(vertThreadGroupsX, 1, 1);
    u_FillClustersIndirectArgs[g_Params.instanceIndex * kFillClustersPerInstanceIndirectArgCount + 1] = uint3(texcoordsThreadGroupsX, 1, 1);
}