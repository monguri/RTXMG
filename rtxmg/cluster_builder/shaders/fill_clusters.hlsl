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

#include <nvrhi/nvrhiHLSL.h>

#include <donut/shaders/bindless.h>
#include <donut/shaders/binding_helpers.hlsli>

#include "rtxmg/cluster_builder/fill_clusters_params.h"
#include "rtxmg/subdivision/subdivision_eval.hlsli"
#include "rtxmg/cluster_builder/tessellator_constants.h"
#include "rtxmg/subdivision/subdivision_plan_hlsl.h"
#include "rtxmg/cluster_builder/cluster.h"
#include "rtxmg/subdivision/vertex.h"
#include "rtxmg/cluster_builder/fill_instantiate_template_args_params.h"
#include "rtxmg/cluster_builder/displacement.hlsli"

#include "rtxmg/subdivision/osd_ports/tmr/surfaceDescriptor.h"


StructuredBuffer<GridSampler> t_GridSamplers : register(t0);
StructuredBuffer<uint2> t_ClusterOffsetCounts : register(t1);
StructuredBuffer<Cluster> t_Clusters : register(t2);

// Buffers for the subd surface 
StructuredBuffer<float3> t_VertexControlPoints : register(t3);
StructuredBuffer<SurfaceDescriptor> t_VertexSurfaceDescriptors : register(t4);
StructuredBuffer<Index> t_VertexControlPointIndices : register(t5);
StructuredBuffer<uint32_t> t_VertexPatchPointsOffsets : register(t6);
StructuredBuffer<SubdivisionPlanHLSL> t_Plans : register(t7);
StructuredBuffer<uint32_t> t_SubpatchTrees : register(t8);
StructuredBuffer<Index> t_PatchPointIndices : register(t9);
StructuredBuffer<float> t_StencilMatrix : register(t10);
StructuredBuffer<float3> t_VertexPatchPoints : register(t11);

// Displacement/Materials
StructuredBuffer<GeometryData> t_GeometryData : register(t12);
StructuredBuffer<MaterialConstants> t_MaterialConstants : register(t13);
StructuredBuffer<uint16_t> t_SurfaceToGeometryIndex : register(t14);

// Texcoord evaluation
StructuredBuffer<LinearSurfaceDescriptor> t_TexCoordSurfaceDescriptors : register(t15);
StructuredBuffer<Index> t_TexCoordControlPointIndices : register(t16);
StructuredBuffer<uint32_t> t_TexCoordPatchPointsOffsets : register(t17);
StructuredBuffer<float2> t_TexCoordPatchPoints : register(t18);
StructuredBuffer<float2> t_TexCoords : register(t19);

// Buffers for the gatherer
RWStructuredBuffer<float3> u_ClusterVertexPositions : register(u0);
RWStructuredBuffer<ClusterShadingData> u_ClusterShadingData : register(u1);
RWStructuredBuffer<float4> u_Debug : register(u2);

SamplerState s_DisplacementSampler : register(s0);

VK_BINDING(1, 1) Texture2D t_BindlessTextures[] : register(t0, space2);

ConstantBuffer<FillClustersParams> g_TessParams : register(b0);

void GathererWriteLimit(LimitFrame vertexLimit, Cluster cluster, uint32_t vertexIndex)
{
    u_ClusterVertexPositions[cluster.nVertexOffset + vertexIndex] = quantize(vertexLimit.p, g_TessParams.quantNBits);
}

void GathererWriteTexcoord(TexCoordLimitFrame texcoord, uint32_t clusterIndex, uint32_t cornerIndex)
{
    u_ClusterShadingData[clusterIndex].m_texcoords[cornerIndex] = texcoord.uv;
}

[numthreads(32, kFillClustersVerticesWaves, 1)]
void FillClustersMain(uint3 threadIdx : SV_GroupThreadID, uint3 groupIdx : SV_GroupID)
{
    const uint32_t groupClusterIndex = groupIdx.x * kFillClustersVerticesWaves + threadIdx.y;
    uint2 offsetCount = t_ClusterOffsetCounts[g_TessParams.instanceIndex];
    if (groupClusterIndex >= offsetCount.y)
        return; // early out waves beyond cluster array end

    uint32_t clusterIndex = groupClusterIndex + offsetCount.x;
    const Cluster rCluster = t_Clusters[clusterIndex];
    const uint32_t iSurface = rCluster.iSurface;
    const GridSampler rSampler = t_GridSamplers[iSurface];

    SubdivisionEvaluatorHLSL subd;
    subd.m_surfaceIndex = iSurface;
    subd.m_isolationLevel = (uint16_t)g_TessParams.isolationLevel;
    subd.m_surfaceDescriptors = t_VertexSurfaceDescriptors;
    subd.m_plans = t_Plans;
    subd.m_subpatchTrees = t_SubpatchTrees;
    subd.m_vertexPatchPointIndices = t_PatchPointIndices;
    subd.m_stencilMatrix = t_StencilMatrix;

    subd.m_vertexControlPointIndices = t_VertexControlPointIndices;
    subd.m_vertexControlPoints = t_VertexControlPoints;
    subd.m_vertexPatchPointsOffsets = t_VertexPatchPointsOffsets;
    subd.m_vertexPatchPoints = t_VertexPatchPoints;

#if DISPLACEMENT_MAPS
    TexcoordEvaluatorHLSL texcoordEval;
    texcoordEval.m_surfaceDescriptors = t_TexCoordSurfaceDescriptors;
    texcoordEval.m_texcoordControlPointIndices = t_TexCoordControlPointIndices;
    texcoordEval.m_texcoordPatchPointsOffsets = t_TexCoordPatchPointsOffsets;
    texcoordEval.m_texcoordPatchPoints = t_TexCoordPatchPoints;
    texcoordEval.m_texcoordControlPoints = t_TexCoords;

    float displacementScale;
    int displacementTexIndex;

    uint32_t geometryIndex = t_SurfaceToGeometryIndex[iSurface] + g_TessParams.firstGeometryIndex;
    GeometryData geometry = t_GeometryData[geometryIndex];
    MaterialConstants material = t_MaterialConstants[geometry.materialIndex];

    GetDisplacement(material, g_TessParams.globalDisplacementScale, displacementTexIndex, displacementScale);
    Texture2D displacementTex = t_BindlessTextures[displacementTexIndex];
#endif

    {
        // wave wide loop
        for (uint16_t pointIndex = (uint16_t)threadIdx.x; pointIndex < rCluster.VerticesPerCluster(); pointIndex += 32)
        {
            float2 uv = rSampler.UV(rCluster.Linear2Idx2D(pointIndex) + rCluster.offset, (ClusterPattern)g_TessParams.clusterPattern);

            // always do the non-displaced evaluation first.  Displacement maps will perturb this calculation below
            LimitFrame limit = subd.Evaluate(uv);

#if DISPLACEMENT_MAPS
            limit = DoDisplacement(texcoordEval,
                        limit, iSurface, uv,
                        displacementTex,
                        s_DisplacementSampler, displacementScale);
#endif

            GathererWriteLimit(limit, rCluster, pointIndex);
        }
    }
}

[numthreads(kFillClustersTexcoordsThreadsX, 4, 1)]
void FillClustersTexcoordsMain(uint3 threadIdx : SV_GroupThreadID, uint3 groupIdx : SV_GroupID)
{
    const uint32_t groupClusterIndex = groupIdx.x * kFillClustersTexcoordsThreadsX + threadIdx.x;
    uint2 offsetCount = t_ClusterOffsetCounts[g_TessParams.instanceIndex];
    if (groupClusterIndex >= offsetCount.y)
        return; // early out waves beyond cluster array end

    uint32_t clusterIndex = groupClusterIndex + offsetCount.x;
    const Cluster rCluster = t_Clusters[clusterIndex];
    const uint32_t iSurface = rCluster.iSurface;
    const GridSampler rSampler = t_GridSamplers[iSurface];

    // Extra shading data: texcoords on corners of surfaces (patches). This might not exactly match texcoords used in displacement
    // above if the subd evaluator is cubic.
    
    // TODO: This can be removed here and the uvs in the surface corners should be written into a dedicated array in the coarse
    // rasterizer. If we ever go to higher-order texture coord eval per surface, the per cluster corner uvs or even per vertex
    // uvs can be added back.
    const float2 kSurfaceUVs[4] = { { 0, 0 }, { 1, 0 }, { 1, 1 }, { 0, 1 } };

    TexcoordEvaluatorHLSL texcoordEval;
    texcoordEval.m_surfaceDescriptors = t_TexCoordSurfaceDescriptors;
    texcoordEval.m_texcoordControlPointIndices = t_TexCoordControlPointIndices;
    texcoordEval.m_texcoordPatchPointsOffsets = t_TexCoordPatchPointsOffsets;
    texcoordEval.m_texcoordPatchPoints = t_TexCoordPatchPoints;
    texcoordEval.m_texcoordControlPoints = t_TexCoords;

    TexCoordLimitFrame texcoord = texcoordEval.EvaluateLinearSubd(kSurfaceUVs[threadIdx.y], iSurface);

    GathererWriteTexcoord(texcoord, clusterIndex, threadIdx.y);
}