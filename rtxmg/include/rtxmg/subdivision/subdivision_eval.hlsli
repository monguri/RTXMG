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

#ifndef SUBDIVISION_EVAL_HLSLI // using instead of "#pragma once" due to https://github.com/microsoft/DirectXShaderCompiler/issues/3943
#define SUBDIVISION_EVAL_HLSLI

#include "rtxmg/subdivision/osd_ports/tmr/surfaceDescriptor.h"
#include "rtxmg/subdivision/subdivision_plan_hlsl.h"
#include "rtxmg/subdivision/vertex.h"

// SURFACE_TYPE
#define SURFACE_TYPE_PUREBSPLINE 0
#define SURFACE_TYPE_REGULARBSPLINE 1
#define SURFACE_TYPE_LIMIT 2
#define SURFACE_TYPE_ALL 3

#ifndef SURFACE_TYPE
#define SURFACE_TYPE SURFACE_TYPE_ALL
#endif

#ifdef PATCH_POINTS_WRITEABLE // defined in the shader
#define VERTEX_PATCH_POINTS_TYPE RWStructuredBuffer<float3>
#define TEXCOORD_PATCH_POINTS_TYPE RWStructuredBuffer<float2>
#else
#define VERTEX_PATCH_POINTS_TYPE StructuredBuffer<float3>
#define TEXCOORD_PATCH_POINTS_TYPE StructuredBuffer<float2>
#endif

static const uint32_t kPatchSize = 16;
static const Index kPureBSplinePatchPointIndices[kPatchSize] = { 6, 7, 8, 9, 5, 0, 1, 10, 4, 3, 2, 11, 15, 14, 13, 12 };
    
const static uint32_t kNumWaveSurfaceUVSamples = 8;
const static float2 kWaveSurfaceUVSamples[kNumWaveSurfaceUVSamples] =
{
    { 0, 0 },
    { 0.5, 0 },
    { 1, 0 },
    { 1, 0.5 },
    { 1, 1 },
    { 0.5, 1 },
    { 0, 1 },
    { 0, 0.5 }
};

struct SubdivisionEvaluatorHLSL
{
    uint32_t m_surfaceIndex;
    uint16_t m_isolationLevel;
    
    StructuredBuffer<SurfaceDescriptor> m_surfaceDescriptors;
    
    // SubdivisionPlanContext
    StructuredBuffer<SubdivisionPlanHLSL> m_plans;
    StructuredBuffer<uint32_t> m_subpatchTrees;
    StructuredBuffer<Index> m_vertexPatchPointIndices;
    StructuredBuffer<float> m_stencilMatrix;
    
    // Subdivision Surface data
    StructuredBuffer<Index> m_vertexControlPointIndices;
    StructuredBuffer<float3> m_vertexControlPoints; // position data for control points
    StructuredBuffer<uint32_t> m_vertexPatchPointsOffsets; // offsets into m_vertexPatchPoints
    VERTEX_PATCH_POINTS_TYPE m_vertexPatchPoints; // position data for patch level, computed from stencil matrix and control points
    
    SurfaceDescriptor GetSurfaceDesc()
    {
        return m_surfaceDescriptors[m_surfaceIndex];
    }
    
    SubdivisionPlanContext GetPlan()
    {
        SubdivisionPlanContext context = (SubdivisionPlanContext)0;
        context.m_data = m_plans[GetSurfaceDesc().GetSubdivisionPlanIndex()];
        context.m_subpatchTrees = m_subpatchTrees;
        context.m_patchPoints = m_vertexPatchPointIndices;
        context.m_stencilMatrix = m_stencilMatrix;
        return context;
    }
    
    bool IsPureBSplinePatch()
    {
        return GetSurfaceDesc().GetSubdivisionPlanIndex() == 0;
    }

    bool IsBSplinePatch()
    {
        return GetPlan().IsBSplinePatch(m_isolationLevel);
    }
    
    LimitFrame EvaluatePureBsplinePatch(float2 uv)
    {
        float aPtWeights[kPatchSize];
        float aDuWeights[kPatchSize];
        float aDvWeights[kPatchSize];

        EvalBasisBSpline(uv, aPtWeights, aDuWeights, aDvWeights,
                                0, // boundary mask
                                0.0f, // sharpness
                                true // pure bspline
        );

        SurfaceDescriptor desc = GetSurfaceDesc();
        
        LimitFrame limit;
        limit.Clear();
        
        for (int iWeight = 0; iWeight < kPatchSize; ++iWeight)
        {
            Index patchPointIndex = kPureBSplinePatchPointIndices[iWeight];
            Index cpi = m_vertexControlPointIndices[desc.firstControlPoint + patchPointIndex];
            limit.AddWithWeight(m_vertexControlPoints[cpi], aPtWeights[iWeight], aDuWeights[iWeight], aDvWeights[iWeight]);
        }
        
        return limit;
    }

    LimitFrame EvaluateBsplinePatch(float2 uv)
    {
        float aPtWeights[kPatchSize];
        float aDuWeights[kPatchSize];
        float aDvWeights[kPatchSize];

        uint16_t quadrant = 0;
    
        SubdivisionPlanContext plan = GetPlan();
        
        SubdivisionNode node = plan.EvaluateBasis(uv, aPtWeights, aDuWeights, aDvWeights, quadrant, m_isolationLevel);

        SurfaceDescriptor desc = GetSurfaceDesc();

        LimitFrame limit;
        limit.Clear();

        for (int iWeight = 0; iWeight < kPatchSize; ++iWeight)
        {
            Index patchPointIndex = node.GetPatchPoint(iWeight, quadrant, m_isolationLevel);
            Index cpi = m_vertexControlPointIndices[desc.firstControlPoint + patchPointIndex];
            limit.AddWithWeight(m_vertexControlPoints[cpi], aPtWeights[iWeight], aDuWeights[iWeight], aDvWeights[iWeight]);
        }
        
        return limit;
    }
    
    LimitFrame EvaluateLimitSurface(float2 uv)
    {
        uint16_t quadrant = 0;
        float aPtWeights[kPatchSize];
        float aDuWeights[kPatchSize];
        float aDvWeights[kPatchSize];

        SubdivisionPlanContext plan = GetPlan();
        SubdivisionNode node = plan.EvaluateBasis(uv, aPtWeights, aDuWeights, aDvWeights, quadrant, m_isolationLevel);

        const uint16_t numControlPoints = plan.m_data.numControlPoints;
        
        LimitFrame limit;
        limit.Clear();

        for (int iWeight = 0; iWeight < kPatchSize; ++iWeight)
        {
            Index patchPointIndex = node.GetPatchPoint(iWeight, quadrant, m_isolationLevel);
            int index = patchPointIndex - numControlPoints;
            uint32_t supportOffset = m_vertexPatchPointsOffsets[m_surfaceIndex];
            limit.AddWithWeight(m_vertexPatchPoints[supportOffset + index], aPtWeights[iWeight], aDuWeights[iWeight], aDvWeights[iWeight]);
        }
        
        return limit;
    }
    
    LimitFrame Evaluate(float2 uv)
    {
#if SURFACE_TYPE == SURFACE_TYPE_ALL
        if (IsPureBSplinePatch())
            return EvaluatePureBsplinePatch(uv);
        else if (IsBSplinePatch())
            return EvaluateBsplinePatch(uv);
        else
            return EvaluateLimitSurface(uv);
#elif SURFACE_TYPE == SURFACE_TYPE_PUREBSPLINE
        return EvaluatePureBsplinePatch(uv);
#elif SURFACE_TYPE == SURFACE_TYPE_REGULARBSPLINE
        return EvaluateBsplinePatch(uv);
#elif SURFACE_TYPE == SURFACE_TYPE_LIMIT
        return EvaluateLimitSurface(uv);
#endif
    }
    
    float3 GetPureBsplinePatchPoint(uint32_t i, uint32_t j)
    {
        int iWeight = 4 * j + i;
        Index patchPointIndex = kPureBSplinePatchPointIndices[iWeight];

        SurfaceDescriptor desc = GetSurfaceDesc();
        Index cpi = m_vertexControlPointIndices[desc.firstControlPoint + patchPointIndex];
        return m_vertexControlPoints[cpi];
    }
    
    // Wave-parallel pure b-spline patch evaluator for up to 8 sample locations.
    //
    // Inputs:
    //    uvs - a span of up to 8 sample locations
    //    subd - subd object
    //
    // Returns:
    //    limits - a span of LimitFrames with enough space to store the resulting limits (same a uvs.m_size()).
    //
    LimitFrame WaveEvaluatePureBsplinePatch8(uint32_t iLane)
    {
        float2 uv = kWaveSurfaceUVSamples[iLane % kNumWaveSurfaceUVSamples];
        float sWeights = CubicBSplineWeight(uv.x, iLane / kNumWaveSurfaceUVSamples); // weights for s-direction in wave w_s_0, ..., w_s_0, w_s_1, ..., w_s_1, ... w_s_3, ... w_s_3
        float dsWeights = CubicBSplineDerivativeWeight(uv.x, iLane / kNumWaveSurfaceUVSamples); // weights for s_direction derivative, wave layout as for sWeights

        LimitFrame limit;
        limit.Clear();
        for (uint32_t j = 0; j < 4; ++j)
        {
            float tWeights = CubicBSplineWeight(uv.y, j);
            float dtWeights = CubicBSplineDerivativeWeight(uv.y, j);
            float3 vtx = GetPureBsplinePatchPoint(iLane / kNumWaveSurfaceUVSamples, j);

            limit.p += sWeights * tWeights * vtx; // point
            limit.deriv1 += dsWeights * tWeights * vtx; // deriv1
            limit.deriv2 += sWeights * dtWeights * vtx; // deriv2
        }

        limit.p += WaveReadLaneAt(limit.p, WaveGetLaneIndex() + 8);
        limit.p += WaveReadLaneAt(limit.p, WaveGetLaneIndex() + 16);
        limit.deriv1 += WaveReadLaneAt(limit.deriv1, WaveGetLaneIndex() + 8);
        limit.deriv1 += WaveReadLaneAt(limit.deriv1, WaveGetLaneIndex() + 16);
        limit.deriv2 += WaveReadLaneAt(limit.deriv2, WaveGetLaneIndex() + 8);
        limit.deriv2 += WaveReadLaneAt(limit.deriv2, WaveGetLaneIndex() + 16);
        
        return limit;
    }

    LimitFrame WaveEvaluateBsplinePatch(uint32_t iLane)
    {
        const float2 st = kWaveSurfaceUVSamples[iLane % kNumWaveSurfaceUVSamples];

        LimitFrame limit;
        limit.Clear();
        SubdivisionPlanContext plan = GetPlan();
        SubdivisionNode rootNode = plan.GetRootNode();

        NodeDescriptor nodeDesc = rootNode.GetDesc();
        float sharpness = nodeDesc.HasSharpness() ? rootNode.GetSharpness() : 0.f;

        PatchParam param;
        param.Set(INDEX_INVALID, (uint16_t)nodeDesc.GetU(), (uint16_t)nodeDesc.GetV(), 0, false, (uint16_t)nodeDesc.GetBoundaryMask(), 0, true);
        const int boundaryMask = param.GetBoundary();

        float sWeights[4], tWeights[4], dsWeights[4], dtWeights[4], unused[4];
        EvalBSplineCurve(st.x, sWeights, dsWeights, unused, true, false);
        EvalBSplineCurve(st.y, tWeights, dtWeights, unused, true, false);

        if (boundaryMask)
        {
            if (sharpness > 0.0f)
                AdjustCreases(st, sWeights, tWeights, dsWeights, dtWeights, boundaryMask, sharpness);
            else
                AdjustBoundaries(st, sWeights, tWeights, dsWeights, dtWeights, boundaryMask);
        }

        SurfaceDescriptor desc = GetSurfaceDesc();

        const int iPatchPtBase = rootNode.GetPatchPointBase();

        const uint32_t i = iLane / 8;
        float w_t_i = tWeights[i];
        float w_dt_i = dtWeights[i];

        for (int j = 0; j < 4; ++j)
        {
            const int iWeight = 4 * i + j;
            Index patchPointIndex = m_vertexPatchPointIndices[plan.m_data.patchPointsOffset + iPatchPtBase + iWeight];
            Index cpi = m_vertexControlPointIndices[desc.firstControlPoint + patchPointIndex];
            float3 vtx = m_vertexControlPoints[cpi]; // regular face

            limit.p += (w_t_i * sWeights[j]) * vtx;
            limit.deriv1 += (w_t_i * dsWeights[j]) * vtx;
            limit.deriv2 += (w_dt_i * sWeights[j]) * vtx;
        }
        limit.p += WaveReadLaneAt(limit.p, WaveGetLaneIndex() + 8);
        limit.p += WaveReadLaneAt(limit.p, WaveGetLaneIndex() + 16);
        limit.deriv1 += WaveReadLaneAt(limit.deriv1, WaveGetLaneIndex() + 8);
        limit.deriv1 += WaveReadLaneAt(limit.deriv1, WaveGetLaneIndex() + 16);
        limit.deriv2 += WaveReadLaneAt(limit.deriv2, WaveGetLaneIndex() + 8);
        limit.deriv2 += WaveReadLaneAt(limit.deriv2, WaveGetLaneIndex() + 16);

        return limit;
        
    }

#ifdef PATCH_POINTS_WRITEABLE
    void WaveEvaluatePatchPoints(uint32_t iLane)
    {
        SurfaceDescriptor desc = GetSurfaceDesc();
        // desc.firstControlPoint is the offset to the first control point for this surface in vertexControlPointIndices

        SubdivisionPlanContext plan = GetPlan();
        const uint32_t numPatchPoints = plan.GetTreeDescriptor().GetNumPatchPoints(m_isolationLevel);

        uint32_t globalPatchPointOffset = m_vertexPatchPointsOffsets[m_surfaceIndex];
        for (int iPatchPoint = iLane; iPatchPoint < numPatchPoints; iPatchPoint += 32)  // advance wave
        {
            float3 patchPoint = float3(0, 0, 0);
            for (int i = 0; i < plan.m_data.numControlPoints; ++i)
            {
                float3 controlPoint = m_vertexControlPoints[m_vertexControlPointIndices[desc.firstControlPoint + i]];
                patchPoint += m_stencilMatrix[plan.m_data.stencilMatrixOffset + iPatchPoint * plan.m_data.numControlPoints + i] * controlPoint;
            }
            m_vertexPatchPoints[globalPatchPointOffset + iPatchPoint] = patchPoint;
        }
    }
#endif
};

struct DynamicSubdivisionEvaluatorHLSL : SubdivisionEvaluatorHLSL
{
    StructuredBuffer<float3> m_vertexControlPointsPrev;

    void EvaluatePureBsplinePatch(out LimitFrame limit, out LimitFrame limitPrev, float2 uv)
    {
        float aPtWeights[kPatchSize];
        float aDuWeights[kPatchSize];
        float aDvWeights[kPatchSize];

        EvalBasisBSpline(uv, aPtWeights, aDuWeights, aDvWeights,
                                    0, // boundary mask
                                    0.0f, // sharpness
                                    true // pure bspline
        );

        SurfaceDescriptor desc = GetSurfaceDesc();

        limit.Clear();
        limitPrev.Clear();

        for (int iWeight = 0; iWeight < kPatchSize; ++iWeight)
        {
            Index patchPointIndex = kPureBSplinePatchPointIndices[iWeight];
            Index cpi = m_vertexControlPointIndices[desc.firstControlPoint + patchPointIndex];
            limit.AddWithWeight(m_vertexControlPoints[cpi], aPtWeights[iWeight], aDuWeights[iWeight], aDvWeights[iWeight]);
            limitPrev.AddWithWeight(m_vertexControlPointsPrev[cpi], aPtWeights[iWeight], aDuWeights[iWeight], aDvWeights[iWeight]);
        }
    }


    void EvaluateBsplinePatch(out LimitFrame limit, out LimitFrame limitPrev, float2 uv)
    {
        float aPtWeights[kPatchSize];
        float aDuWeights[kPatchSize];
        float aDvWeights[kPatchSize];

        uint16_t quadrant = 0;

        SubdivisionPlanContext plan = GetPlan();
        SubdivisionNode node = plan.EvaluateBasis(uv, aPtWeights, aDuWeights, aDvWeights, quadrant, m_isolationLevel);

        SurfaceDescriptor desc = GetSurfaceDesc();

        limit.Clear();
        limitPrev.Clear();

        for (int iWeight = 0; iWeight < kPatchSize; ++iWeight)
        {
            Index patchPointIndex = node.GetPatchPoint(iWeight, quadrant, m_isolationLevel);
            Index cpi = m_vertexControlPointIndices[desc.firstControlPoint + patchPointIndex];
            limit.AddWithWeight(m_vertexControlPoints[cpi], aPtWeights[iWeight], aDuWeights[iWeight], aDvWeights[iWeight]);
            limitPrev.AddWithWeight(m_vertexControlPointsPrev[cpi], aPtWeights[iWeight], aDuWeights[iWeight], aDvWeights[iWeight]);
        }
    }

    void EvaluateLimitSurface(out LimitFrame limit, out LimitFrame limitPrev, float2 uv)
    {
        uint16_t quadrant = 0;
        const uint32_t numPatchPoints = 16;
        float   aPtWeights[numPatchPoints];
        float   aDuWeights[numPatchPoints];
        float   aDvWeights[numPatchPoints];

        SubdivisionPlanContext plan = GetPlan();
        SubdivisionNode node = plan.EvaluateBasis(uv, aPtWeights, aDuWeights, aDvWeights, quadrant, m_isolationLevel);

        limit.Clear();
        limitPrev.Clear();

        const int numControlPoints = plan.m_data.numControlPoints;
        SurfaceDescriptor desc = GetSurfaceDesc();

        for (int iPoint = 0; iPoint < numPatchPoints; ++iPoint)
        {
            Index patchPointIndex = node.GetPatchPoint(iPoint, quadrant, m_isolationLevel);
            //assert(patchPointIndex >= numControlPoints);  // not a regular bspline surface

            int stencilMatrixRow = plan.m_data.stencilMatrixOffset + (patchPointIndex - numControlPoints) * numControlPoints;
            // Build patch iPoint and accumulate it
            float3 patchPoint0 = 0.0f;
            for (int i = 0; i < numControlPoints; ++i)
            {
                patchPoint0 +=
                    m_vertexControlPoints[m_vertexControlPointIndices[desc.firstControlPoint + i]] *
                    m_stencilMatrix[stencilMatrixRow + i];
            }
            limit.AddWithWeight(patchPoint0, aPtWeights[iPoint], aDuWeights[iPoint], aDvWeights[iPoint]);


            float3 patchPoint1 = 0.0f;
            for (int i = 0; i < numControlPoints; ++i)
            {
                patchPoint1 +=
                    m_vertexControlPointsPrev[m_vertexControlPointIndices[desc.firstControlPoint + i]] *
                    m_stencilMatrix[stencilMatrixRow + i];
            }
            limitPrev.AddWithWeight(patchPoint1, aPtWeights[iPoint], aDuWeights[iPoint], aDvWeights[iPoint]);
        }
    }

    void Evaluate(float2 uv, out LimitFrame limit, out LimitFrame limitPrev)
    {
        if (IsPureBSplinePatch())
        {
            EvaluatePureBsplinePatch(limit, limitPrev, uv);
        }
        else if (IsBSplinePatch())
        {
            EvaluateBsplinePatch(limit, limitPrev, uv);
        }
        else
        {
            EvaluateLimitSurface(limit, limitPrev, uv);
        }
    }

    void EvaluatePrev(float2 uv, out LimitFrame limit)
    {
        LimitFrame dummy;
        Evaluate(uv, dummy, limit);
    }
};

struct TexcoordEvaluatorHLSL
{
    StructuredBuffer<LinearSurfaceDescriptor> m_surfaceDescriptors;
    StructuredBuffer<Index> m_texcoordControlPointIndices;
    StructuredBuffer<uint32_t> m_texcoordPatchPointsOffsets;
    TEXCOORD_PATCH_POINTS_TYPE m_texcoordPatchPoints;
    StructuredBuffer<float2> m_texcoordControlPoints;
    
    void EvalLinearBasis(float u, float v, out float weights[4], out float duWeights[4], out float dvWeights[4])
    {
        weights[0] = (1.0f - u) * (1.0f - v);
        weights[1] = u * (1.0f - v);
        weights[2] = u * v;
        weights[3] = (1.0f - u) * v;

        duWeights[0] = (-1.0f + v);
        duWeights[1] = (1.0f - v);
        duWeights[2] = v;
        duWeights[3] = -v;

        dvWeights[0] = (-1.0f + u);
        dvWeights[1] = (-u);
        dvWeights[2] = u;
        dvWeights[3] = (1.0f - u);
    }

    Index GetPatchPoint(int pointIndex, uint16_t faceSize, LocalIndex subfaceIndex)
    {
        if (subfaceIndex == LOCAL_INDEX_INVALID)
        {
            assert(pointIndex < faceSize);
            return pointIndex;
        }
        else
        {
            assert(pointIndex < 4);
            // patch point indices layout (N = faceSize) :
            // [ N control points ]
            // [ 1 face-point ]
            // [ N edge-points ]
            int N = faceSize;
            switch (pointIndex)
            {
                case 0:
                    return subfaceIndex;
                case 1:
                    return N + 1 + subfaceIndex; // edge-point after
                case 2:
                    return N; // central face-point
                case 3:
                    return N + (subfaceIndex > 0 ? subfaceIndex : N);
            }
        }
        return INDEX_INVALID;
    }
    
#ifdef PATCH_POINTS_WRITEABLE
    void WaveEvaluateTexCoordPatchPoints(uint32_t iLane, uint32_t iSurface)
    {
        LinearSurfaceDescriptor desc = m_surfaceDescriptors[iSurface];
        LocalIndex subface = desc.GetQuadSubfaceIndex();

        if (subface == LOCAL_INDEX_INVALID)
            return;

        uint16_t faceSize = desc.GetFaceSize();
        float2 center = (iLane < faceSize) ?
            m_texcoordControlPoints[m_texcoordControlPointIndices[desc.firstControlPoint + iLane]] : 
            float2(0, 0);

        for (int offset = 16; offset > 0; offset /= 2)
        {
            center.x += WaveReadLaneAt(center.x, WaveGetLaneIndex() + offset);
            center.y += WaveReadLaneAt(center.y, WaveGetLaneIndex() + offset);
        }

        if (iLane == 0)
        {
            m_texcoordPatchPoints[m_texcoordPatchPointsOffsets[iSurface] + 0] = center / faceSize;
        }

        if (iLane < faceSize)
        {
            float2 a = m_texcoordControlPoints[m_texcoordControlPointIndices[desc.firstControlPoint + iLane]];
            float2 b = m_texcoordControlPoints[m_texcoordControlPointIndices[desc.firstControlPoint + (iLane + 1) % faceSize]];
            m_texcoordPatchPoints[m_texcoordPatchPointsOffsets[iSurface] + iLane + 1] = 0.5f * (a + b);
        }
    }
#endif
    
    TexCoordLimitFrame EvaluateLinearSubd(float2 uv, uint32_t iSurface)
    {
        TexCoordLimitFrame limit;
        limit.Clear();
    
        LinearSurfaceDescriptor desc = m_surfaceDescriptors[iSurface];

        const uint16_t faceSize = desc.GetFaceSize();
        LocalIndex subface = desc.GetQuadSubfaceIndex();

        float pointWeights[4], duWeights[4], dvWeights[4];
        EvalLinearBasis(uv.x, uv.y, pointWeights, duWeights, dvWeights);
        
        for (int k = 0; k < 4; ++k)
        {
            Index patchPointIndex = GetPatchPoint(k, faceSize, subface);
                
            if (patchPointIndex < faceSize)
            {
                limit.AddWithWeight(m_texcoordControlPoints[m_texcoordControlPointIndices[desc.firstControlPoint + patchPointIndex]], pointWeights[k], duWeights[k], dvWeights[k]);
            }
            else
            {
                uint32_t supportOffset = m_texcoordPatchPointsOffsets[iSurface];
                limit.AddWithWeight(m_texcoordPatchPoints[supportOffset + patchPointIndex - faceSize], pointWeights[k], duWeights[k], dvWeights[k]);
            }
        }
        
        return limit;
    }
};

#endif // SUBDIVISION_EVAL_HLSLI