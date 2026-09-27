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

#include "rtxmg/utils/shader_debug.h"

#include <donut/shaders/binding_helpers.hlsli>

#include "render_params.h"
#include "motion_vectors_params.h"
#include "gbuffer.h"

#include "rtxmg/cluster_tess/displacement.hlsli"
#include "rtxmg/subdivision/subdivision_eval.hlsli"
#include "rtxmg/scene/instance_data.h"

// MVEC_DISPLACEMENT
#define MVEC_DISPLACEMENT_FROM_SUBD_EVAL 0
#define MVEC_DISPLACEMENT_FROM_MATERIAL 1

#ifndef MVEC_DISPLACEMENT
#error "Must define MVEC_DISPLACEMENT"
#endif

ConstantBuffer<RenderParams>        g_RenderParams          : register(b0);

Texture2D<DepthFormat>              t_Depth                 : register(t0);
StructuredBuffer<HitResult>         t_HitResult             : register(t1);
StructuredBuffer<SubdInstance>      t_SubdInstances         : register(t2); // indexed via instanceID, but values will be null.
StructuredBuffer<RTXMGMaterialConstants> t_MaterialConstants : register(t3);


VK_IMAGE_FORMAT_UNKNOWN RWTexture2D<float2>                 u_MotionVectors         : register(u0);

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
    result.m_isolationLevel = uint16_t(g_RenderParams.isolationLevel);
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

    // Do not delete: DXC eliminates this entirely, but its presence pins how it
    // associates the limit-surface delta below, which the subd goldens encode.
    SHADER_DEBUG_INIT(g_RenderParams.debugPixel, idx);

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

    float2 prevPixel = 0.0f;
    SubdInstance subdInstance = t_SubdInstances[hit.instanceId];
    if (subdInstance.positionsPrevBindlessIndex != kInvalidBindlessIndex)
    {
        DynamicSubdivisionEvaluatorHLSL subd = MakeDynamicSubdivisionEvaluator(subdInstance, hit.surfaceIndex);

        if (MVEC_DISPLACEMENT == MVEC_DISPLACEMENT_FROM_MATERIAL)
        {
            // Resample displacement from texture and apply to prev frame limit surface
            // If tess rates vary then there can be a mismatch with the current frame hit point.
            LimitFrame limitPrev = subd.EvaluatePrev(hit.surfaceUV);

            float3 displacementVec = 0.f;

            StructuredBuffer<uint32_t> surfaceToMaterialIndex = ResourceDescriptorHeap[NonUniformResourceIndex(subdInstance.surfaceToMaterialIndexBindlessIndex)];
            RTXMGMaterialConstants material = t_MaterialConstants[surfaceToMaterialIndex[hit.surfaceIndex]];

            float displacementScale = 0.f;
            int displacementTexIndex = -1;
            GetDisplacement(material, g_RenderParams.globalDisplacementScale, displacementTexIndex, displacementScale);
            if (displacementTexIndex >= 0)
            {
                Texture2D<float> displacementTex = ResourceDescriptorHeap[NonUniformResourceIndex(displacementTexIndex)];

                float displacement = displacementTex.SampleLevel(s_DisplacementSampler, hit.texcoord, 0) * displacementScale;
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
            subd.Evaluate(limit, limitPrev, hit.surfaceUV);

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