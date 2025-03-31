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

#include <donut/shaders/bindless.h>
#include <donut/shaders/binding_helpers.hlsli>

#include "render_params.h"
#include "motion_vectors_params.h"
#include "gbuffer.h"

#include "rtxmg/cluster_builder/displacement.hlsli"
#include "rtxmg/subdivision/subdivision_eval.hlsli"

#include "pixel_debug.h"

// MVEC_DISPLACEMENT
#define MVEC_DISPLACEMENT_FROM_SUBD_EVAL 0
#define MVEC_DISPLACEMENT_FROM_MATERIAL 1

#ifndef MVEC_DISPLACEMENT
#error "Must define MVEC_DISPLACEMENT"
#endif

ConstantBuffer<RenderParams>        g_RenderParams          : register(b0);

Texture2D<DepthFormat>              t_Depth                 : register(t0);
StructuredBuffer<HitResult>         t_HitResult             : register(t1);
StructuredBuffer<SubdInstance>      t_SubdInstances         : register(t2); // indexed via instancID, but values will be null.
StructuredBuffer<InstanceData>      t_InstanceData          : register(t3);
StructuredBuffer<GeometryData>      t_GeometryData          : register(t4);
StructuredBuffer<MaterialConstants> t_MaterialConstants     : register(t5);


RWTexture2D<float2>                 u_MotionVectors         : register(u0);

#if ENABLE_PIXEL_DEBUG
RWStructuredBuffer<PixelDebugElement> u_PixelDebug          : register(u1);
#endif


VK_BINDING(0, 1) ByteAddressBuffer t_BindlessBuffers[]  : register(t0, space1);
VK_BINDING(1, 1) Texture2D t_BindlessTextures[]         : register(t0, space2);

SamplerState                        s_DisplacementSampler : register(s0);

static DynamicSubdivisionEvaluatorHLSL MakeDynamicSubdivisionEvaluator(SubdInstance subdInstance, uint32_t surfaceIndex)
{
    DynamicSubdivisionEvaluatorHLSL result;
    result.m_plans = ResourceDescriptorHeap[NonUniformResourceIndex(subdInstance.plansBindlessIndex)];
    result.m_stencilMatrix = ResourceDescriptorHeap[NonUniformResourceIndex(subdInstance.stencilMatrixBindlessIndex)];
    result.m_subpatchTrees = ResourceDescriptorHeap[NonUniformResourceIndex(subdInstance.subpatchTreesBindlessIndex)];
    result.m_vertexPatchPointIndices = ResourceDescriptorHeap[NonUniformResourceIndex(subdInstance.patchPointIndicesBindlessIndex)];
    result.m_surfaceDescriptors = ResourceDescriptorHeap[NonUniformResourceIndex(subdInstance.vertexSurfaceDescriptorBindlessIndex)];
    result.m_vertexControlPointIndices = ResourceDescriptorHeap[NonUniformResourceIndex(subdInstance.vertexControlPointIndicesBindlessIndex)];
    result.m_vertexControlPoints = ResourceDescriptorHeap[NonUniformResourceIndex(subdInstance.positionsBindlessIndex)];
    result.m_vertexControlPointsPrev = ResourceDescriptorHeap[NonUniformResourceIndex(subdInstance.positionsPrevBindlessIndex)];

    result.m_surfaceIndex = surfaceIndex;
    result.m_isolationLevel = uint16_t(subdInstance.isolationLevel);
    return result;
}

float3 TransformPoint(float3 p, const float3x4 mat)
{
    return mul(mat, float4(p, 1.0f)).xyz;
}

[numthreads(kMotionVectorsNumThreadsX, kMotionVectorsNumThreadsY, 1)]
void main(uint3 threadIdx : SV_DispatchThreadID)
{
    uint2 idx = threadIdx.xy;
    if (any(idx >= uint2(g_RenderParams.camera.dims)))
        return;

    PIXEL_DEBUG_INIT(u_PixelDebug, g_RenderParams.debugPixel, idx, true);

    const HitResult hit = t_HitResult[idx.y * g_RenderParams.camera.dims.x + idx.x];

    const float2 curPixel = g_RenderParams.jitter + float2(idx) + 0.5f;

    // Check for miss
    if (hit.instanceId == ~uint32_t(0))
    {
        // Re-project env map direction
        const float3 vw = g_RenderParams.camera.unprojectPixelToWorldDirection(curPixel);
        const float2 prevPixel = g_RenderParams.prevCamera.projectWorldDirectionToPixel(vw);
        u_MotionVectors[idx] = prevPixel - curPixel;
        return;
    }

    const float depth = t_Depth[idx];
    const float3 Pw = g_RenderParams.camera.unprojectPixelToWorld_lineardepth(curPixel, depth);
    float3 PdispW;

    // Check for non-subd geometry
    if (hit.surfaceIndex == ~uint32_t(0))
    {
        //  No deformation, only camera motion
        float2 prevPixel = g_RenderParams.prevCamera.projectWorldToPixel(Pw);
        u_MotionVectors[idx] = prevPixel - curPixel;
        return;
    }

    InstanceData instanceData = t_InstanceData[hit.instanceId];

    float2 prevPixel = 0.0f;
    SubdInstance subdInstance = t_SubdInstances[hit.instanceId];
    if (subdInstance.positionsPrevBindlessIndex != kInvalidBindlessIndex)
    {
        DynamicSubdivisionEvaluatorHLSL subd = MakeDynamicSubdivisionEvaluator(subdInstance, hit.surfaceIndex);

        if (MVEC_DISPLACEMENT == MVEC_DISPLACEMENT_FROM_MATERIAL)
        {
            // Resample displacement from texture and apply to prev frame limit surface
            // If tess rates vary then there can be a mismatch with the current frame hit point.
            LimitFrame limitPrev;
            subd.EvaluatePrev(hit.surfaceUV, limitPrev);

            float3 displacementVec = 0.f;

            StructuredBuffer<uint16_t> surfaceToGeometryIndex = ResourceDescriptorHeap[NonUniformResourceIndex(subdInstance.surfaceToGeometryIndexBindlessIndex)];
            uint32_t geometryIndex = surfaceToGeometryIndex[hit.surfaceIndex] + instanceData.firstGeometryIndex;
            GeometryData geometry = t_GeometryData[geometryIndex];
            MaterialConstants material = t_MaterialConstants[geometry.materialIndex];

            float displacementScale = 0.f;
            int displacementTexIndex = -1;
            GetDisplacement(material, g_RenderParams.globalDisplacementScale, displacementTexIndex, displacementScale);
            if (displacementTexIndex >= 0)
            {
                Texture2D displacementTex = t_BindlessTextures[NonUniformResourceIndex(displacementTexIndex)];

                float displacement = displacementTex.SampleLevel(s_DisplacementSampler, hit.texcoord, 0).r * displacementScale;
                float3 normal = normalize(cross(limitPrev.deriv1, limitPrev.deriv2));
                displacementVec = displacement * normal;
            }

            PdispW = TransformPoint(limitPrev.p + displacementVec, subdInstance.prevLocalToWorld);
            prevPixel = g_RenderParams.prevCamera.projectWorldToPixel(PdispW);
        }
        else
        {
            // Compute displacement using the delta between gbuffer hit point and subd limit point
            // Expensive since it re-evalutes limit surface again, but compensates for tess rates
            LimitFrame limit, limitPrev;
            subd.Evaluate(hit.surfaceUV, limit, limitPrev);

            float3 displacementVec = TransformPoint(Pw, subdInstance.worldToLocal) - limit.p;

            PdispW = TransformPoint(limitPrev.p + displacementVec, subdInstance.prevLocalToWorld);
            prevPixel = g_RenderParams.prevCamera.projectWorldToPixel(PdispW);
        }
    }
    else
    {
        // No deformation, only camera motion
        PdispW = Pw;
        prevPixel = g_RenderParams.prevCamera.projectWorldToPixel(Pw);
    }

    u_MotionVectors[idx] = prevPixel - curPixel;
}