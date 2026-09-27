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
//

#pragma once

#ifdef __cplusplus

#include <donut/core/math/math.h>

#include <assert.h>
#include <cstdint>
#include <stdio.h>

#include <cmath>

using namespace donut::math;

typedef vector<unsigned short, 2> uint16_t2;
typedef vector<unsigned short, 4> uint16_t4;

using std::round;
using std::max;
#else
#define assert(x)
#endif

enum class ClusterTessShape
{
    RECTANGULAR,
    SQUARE,
};

enum class ClusterTessPattern : uint32_t
{
    REGULAR,
    SLANTED
};

// Class for computing uv sample locations on a rectangular grid.
//
// The (edgeSegments + 1) defines the number of distinct
// vertex locations on the rectangular surface patche's edge.
// For this multiple vertices may by snapped into indentical
// locations on the edge.
//
struct GridSampler
{
    uint16_t4 edgeSegments;

#ifndef __cplusplus
    // only need the code for HLSL
    uint16_t GridSizeX() { return max(edgeSegments.x, edgeSegments.z); }
    uint16_t GridSizeY() { return max(edgeSegments.y, edgeSegments.w); }
    uint16_t2 GridSize() { return uint16_t2(GridSizeX(), GridSizeY()); }

    float UV0(uint16_t u_index)
    {
        return round(edgeSegments.x / (float)(GridSizeX()) * u_index) / edgeSegments.x;
    }

    float UV1(uint16_t u_index)
    {
        return round(edgeSegments.z / (float)(GridSizeX()) * u_index) / edgeSegments.z;
    }

    float U1V(uint16_t v_index)
    {
        return round(edgeSegments.y / (float)(GridSizeY()) * v_index) / edgeSegments.y;
    }

    float U0V(uint16_t v_index)
    {
        return  round(edgeSegments.w / (float)(GridSizeY()) * v_index) / edgeSegments.w;
    }

    float2 RegularInteriorUV(uint16_t i, uint16_t j)
    {
        assert(0 < i && i < GridSizeX());
        assert(0 < j && j < GridSizeY());
        float clusterU = i / (float)(GridSizeX());
        float clusterV = j / (float)(GridSizeY());
        return float2(clusterU, clusterV);
    }

    float2 SlantedInteriorUV(uint16_t i, uint16_t j)
    {
        assert(0 < i && i < GridSizeX());
        assert(0 < j && j < GridSizeY());
        float Du = UV1(i) - UV0(i);
        float Dv = U1V(j) - U0V(j);
        float clusterU = (UV0(i) + U0V(j) * Du) / (1.0f - Du * Dv);
        float clusterV = (U0V(j) + UV0(i) * Dv) / (1.0f - Du * Dv);
        return float2(clusterU, clusterV);
    }

    float2 BoundaryUV(uint16_t i, uint16_t j)
    {
        if (j == 0)
            return float2(UV0(i), 0.0f);
        if (j == GridSizeY())
            return float2(UV1(i), 1.0f);

        if (i == 0)
            return float2(0.0f, U0V(j));
        if (i == GridSizeX())
            return float2(1.0f, U1V(j));
        assert(false);

        return float2(0.0f, 0.0f);
    }

    float2 UV(uint16_t2 uvIndex, ClusterTessPattern pattern)
    {
        uint16_t i = uvIndex.x, j = uvIndex.y;
        // interior uv locations
        if (0 < i && i < GridSizeX() && 0 < j && j < GridSizeY())
        {
            if (pattern == ClusterTessPattern::SLANTED)
               return SlantedInteriorUV(i, j);
            else
               return RegularInteriorUV(i, j);
        }
        else
            return BoundaryUV(i, j);

        return float2(0.0f, 0.0f);
    }

    // The functions below estimate the parametric edge lengths du and dv around any point on
    // a surface, i.e., if you wanted to tessellate at this point, what should be the spacing.
    // They are needed for displacement texture filtering on the surface.
    //
    // TODO: define something that is C0 continuous across clusters.  The lerp is not quite C0.
    float DU(float2 uvVals)
    {
        float v = uvVals.y;
        return 1 / ((1.0f - v) * float(edgeSegments.x) + v * float(edgeSegments.z));
    }

    float DV(float2 uvVals)
    {
        float u = uvVals.x;
        return 1 / ((1.0f - u) * float(edgeSegments.w) + u * float(edgeSegments.y));
    }

    bool IsEmpty()
    {
        return GridSizeX() == 0 && GridSizeY() == 0;
    }
#endif
};


// Cluster class.
//
// Tessellator's representation of Clusters.
// This representation contains all data necessary for tessellating analytic
// surface patches (tmr surfaces) into RTX Cluster
// primitives.
//
// This base class stores the target edge resolutions (number of edge segments).
// It provides a number of helper methods for computing resolution of an NxM
// patch (max of the parallel edges) and (u, v)-locations of the cluster samples.
//

struct Cluster
{
    uint32_t iSurface;  // index of the surface (patch) generating this cluster
    uint32_t nVertexOffset;  // vertex array index of this cluster's [0, 0]-corner
    uint16_t2 offset;  // cluster's offset inside sample grid
    uint sizeX : 8;  // cluster's m_size
    uint sizeY : 8;  // cluster's m_size

#ifndef __cplusplus
    // HLSL code only

    inline uint32_t VerticesPerCluster()
    {
        return (sizeX + 1) * (sizeY + 1);
    }

    inline uint32_t QuadsPerCluster()
    {
        return sizeX * sizeY;
    }

    inline uint32_t TrianglesPerCluster() { return 2 * QuadsPerCluster(); }

    uint16_t2 Linear2Idx2D(uint16_t indexLinear)
    {
        const uint16_t vertices_u = (uint16_t)(sizeX + 1);
        return uint16_t2((uint16_t)(indexLinear % vertices_u),
            (uint16_t)(indexLinear / vertices_u));
    }

    inline bool Equals(Cluster other)
    {
        return (iSurface == other.iSurface) && (nVertexOffset == other.nVertexOffset)
            && (offset.x == other.offset.x) && (offset.y == other.offset.y)
            && (sizeX == other.sizeX) && (sizeY == other.sizeY);
    }
#endif
};
#ifndef __cplusplus
Cluster MakeCluster(uint32_t iSurface = 0u,
    uint32_t vertexOffset = 0u,
    uint16_t2  offset = uint16_t2(0u, 0u),
    uint sizeX = 0u,
    uint sizeY = 0u)
{
    Cluster c;
    c.iSurface = iSurface;
    c.nVertexOffset = vertexOffset;
    c.offset = offset;
    c.sizeX = sizeX;
    c.sizeY = sizeY;
    return c;
}
#endif

struct ClusterTessShadingData
{
    uint16_t4 m_edgeSegments;
    float2   m_texcoords[4];
    uint32_t m_surfaceId;
    uint32_t m_vertexOffset;

    uint16_t2  m_clusterOffset;
    uint   m_clusterSizeX : 8;
    uint   m_clusterSizeY : 8;
    uint   m_pad0 : 16;
};
