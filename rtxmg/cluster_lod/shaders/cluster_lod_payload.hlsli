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

// cluster_lod_payload.hlsli
// Cluster-payload accessors used by the cluster-LOD hit shader.
//
// The hit shader reads from the SAME `groupData` byte buffer that the
// CLAS-build hardware dereferences — i.e. the opaque packed nvclusterlod
// blob:  [Group hdr | Cluster hdr ×N | per-cluster vertices | per-cluster
// indices | …repeat per cluster…].  ClusterAddress gives the bindless SRV
// and the absolute byte offset of the Cluster header; the header stores
// vertex/index offsets relative to itself.
//
// CLAS-build wire format (set by preloaded.cpp::InitClas and
// stream_update_scene.hlsl):
//   indexFormat        = IndexFormat8bit
//   indexBufferStride  = 1     (consecutive uint8 indices, 3 bytes per tri)
//   vertexBufferStride = 12    (float3, 4-B aligned)
//
// The hit shader has to read the same uint8-stride-1 index format, so the
// 3 bytes for triangle T live at byte offsets [T*3, T*3+1, T*3+2] — only
// 4-B-aligned when T % 4 == 0.  ClusterLodLoadTriangleIndices() handles all
// alignments with two unconditional `Load<uint>` calls + a bit-extract.

#pragma once

#include "rtxmg/cluster_lod/shaderio.h"

static const uint kClusterLodBytesPerPosition = 12u;  // sizeof(float3)
static const uint kClusterLodIndicesPerTri    =  3u;  // uint8 indices, stride 1
static const uint kClusterLodBytesPerTri      = kClusterLodIndicesPerTri;  // 1 byte per index

// Single vertex position by per-cluster vertex index.  vertexByteBase is 4-B
// aligned (the cluster header before it is 16 B aligned, and vertices are
// float3-stride within their region).
float3 ClusterLodLoadPosition(ByteAddressBuffer groupData,
                        uint              vertexByteBase,
                        uint              vertexIdx)
{
    return groupData.Load<float3>(vertexByteBase
                                  + vertexIdx * kClusterLodBytesPerPosition);
}

// All three corner indices of a triangle, packed into uint3.
// The 3 consecutive uint8s at `indexByteBase + T*3` are 4-B aligned only for
// T%4 == 0; otherwise they sit in a misaligned uint32 window or straddle two.
// Two unconditional Load<uint>s + bit-shift handles every case branchlessly.
uint3 ClusterLodLoadTriangleIndices(ByteAddressBuffer groupData,
                              uint              indexByteBase,
                              uint              localTriIdx)
{
    const uint byteOffset    = indexByteBase + localTriIdx * kClusterLodBytesPerTri;
    const uint alignedOffset = byteOffset & ~3u;
    const uint shiftBits     = (byteOffset & 3u) * 8u;

    const uint lo = groupData.Load<uint>(alignedOffset);
    const uint hi = groupData.Load<uint>(alignedOffset + 4u);

    // Logical 64-bit window {hi,lo} shifted right by `shiftBits`, take
    // low 24 bits = (i2 << 16 | i1 << 8 | i0).  shiftBits ∈ {0,8,16,24}.
    // Only shiftBits > 8 actually spills into `hi`; suppress the hi
    // contribution otherwise (and avoid the `hi << 32` undefined-shift
    // when shiftBits == 0).
    const uint hiContrib = (shiftBits > 8u) ? (hi << (32u - shiftBits)) : 0u;
    const uint packed    = (lo >> shiftBits) | hiContrib;

    return uint3( packed         & 0xFFu,
                 (packed >>  8u) & 0xFFu,
                 (packed >> 16u) & 0xFFu);
}

void ClusterLodLoadTrianglePositions(ByteAddressBuffer            groupData,
                               shaderio::ClusterAddress     clusterAddress,
                               uint                         localTriIdx,
                               out float3                   v0,
                               out float3                   v1,
                               out float3                   v2)
{
    const shaderio::Cluster cluster =
        groupData.Load<shaderio::Cluster>(clusterAddress.byteOffset);
    const uint vertexByteBase = clusterAddress.byteOffset + cluster.vertices;
    const uint indexByteBase  = clusterAddress.byteOffset + cluster.triangles;

    const uint3 idx = ClusterLodLoadTriangleIndices(groupData, indexByteBase, localTriIdx);
    v0 = ClusterLodLoadPosition(groupData, vertexByteBase, idx.x);
    v1 = ClusterLodLoadPosition(groupData, vertexByteBase, idx.y);
    v2 = ClusterLodLoadPosition(groupData, vertexByteBase, idx.z);
}

// Byte offset (within groupData) of the cluster's TEXCOORD_0 array.
// Layout (matches baker.cpp's vertex packing):
//   positions  : vertexCount * 12 bytes
//   normals    : vertexCount * 4  bytes (if shaderio::ClusterAttribute::VertexNormal)
//   align to 8 bytes
//   tex0       : vertexCount * 8  bytes (if shaderio::ClusterAttribute::VertexTex0)
//   ...
// Returns 0 when the cluster lacks tex0 — callers must guard on
// (cluster.attributeBits & shaderio::ClusterAttribute::VertexTex0).
uint ClusterLodTex0ByteBase(shaderio::Cluster        cluster,
                      shaderio::ClusterAddress clusterAddress)
{
    const uint vertexCount = cluster.vertexCountMinusOne + 1u;
    uint off = clusterAddress.byteOffset + cluster.vertices;
    // Position array — absent when the upload stripped it from the resident
    // blob, in which case positions live only in the CLAS-build staging and the
    // hit shader fetches them from the AS (ray-tracing position fetch).
    if ((cluster.attributeBits & shaderio::ClusterAttribute::StrippedVertexPos) == 0u)
        off += vertexCount * 12u;                               // positions
    if (cluster.attributeBits & shaderio::ClusterAttribute::VertexNormal)
        off += vertexCount * 4u;                                // packed normal+tangent
    off = (off + 7u) & ~7u;                                     // align to float2 (8B)
    return off;
}

// All three TEXCOORD_0 UVs for a triangle. Caller must confirm the cluster
// has tex0; otherwise returns zero UVs (safe default for alpha-test).
void ClusterLodLoadTriangleTex0(ByteAddressBuffer            groupData,
                          shaderio::ClusterAddress     clusterAddress,
                          uint                         localTriIdx,
                          out float2                   uv0,
                          out float2                   uv1,
                          out float2                   uv2)
{
    const shaderio::Cluster cluster =
        groupData.Load<shaderio::Cluster>(clusterAddress.byteOffset);
    if ((cluster.attributeBits & shaderio::ClusterAttribute::VertexTex0) == 0u)
    {
        uv0 = uv1 = uv2 = float2(0.f, 0.f);
        return;
    }
    const uint indexByteBase = clusterAddress.byteOffset + cluster.triangles;
    const uint tex0Base      = ClusterLodTex0ByteBase(cluster, clusterAddress);

    const uint3 idx = ClusterLodLoadTriangleIndices(groupData, indexByteBase, localTriIdx);
    if (cluster.encodingBits & shaderio::ClusterEncoding::QuantizedTex0)
    {
        // Po2-grid quantized (see shaderio.h): 16B header (float2 base,
        // float2 step — step an exact power of two) then one uint per vertex
        // (u16 deltaU | u16 deltaV << 16).  float(delta) * step is exact, so
        // this decode reproduces the baker's values bit-for-bit.
        const float2 base      = groupData.Load<float2>(tex0Base);
        const float2 step      = groupData.Load<float2>(tex0Base + 8u);
        const uint   deltaBase = tex0Base + 16u;
        const uint   d0 = groupData.Load<uint>(deltaBase + idx.x * 4u);
        const uint   d1 = groupData.Load<uint>(deltaBase + idx.y * 4u);
        const uint   d2 = groupData.Load<uint>(deltaBase + idx.z * 4u);
        uv0 = base + float2(float(d0 & 0xFFFFu), float(d0 >> 16u)) * step;
        uv1 = base + float2(float(d1 & 0xFFFFu), float(d1 >> 16u)) * step;
        uv2 = base + float2(float(d2 & 0xFFFFu), float(d2 >> 16u)) * step;
    }
    else
    {
        uv0 = groupData.Load<float2>(tex0Base + idx.x * 8u);
        uv1 = groupData.Load<float2>(tex0Base + idx.y * 8u);
        uv2 = groupData.Load<float2>(tex0Base + idx.z * 8u);
    }
}

// ---------------------------------------------------------------------------
// Per-vertex normal decode.
//
// The baker packs one uint32 per vertex (normal in the low ATTRENC_NORMAL_BITS,
// tangent in the upper bits) via shaderio::NormalPack — octahedral encoding,
// see baking/attribute_encoding.h.  That header is host-only C++ so it can't be
// #included here; the decode below mirrors its NormalUnpack / OctToVec.
// ---------------------------------------------------------------------------

static const uint kClusterLodAttrEncNormalBits = 22u;  // must match ATTRENC_NORMAL_BITS

// Octahedral decode (jcgt.org/published/0003/02/01), mirrors OctToVec.
float3 ClusterLodOctToVec(float2 e)
{
    float3 v = float3(e.x, e.y, 1.0f - abs(e.x) - abs(e.y));
    if (v.z < 0.0f)
    {
        float2 os = float2(e.x >= 0.0f ? 1.0f : -1.0f,
                           e.y >= 0.0f ? 1.0f : -1.0f);
        v.x = (1.0f - abs(e.y)) * os.x;
        v.y = (1.0f - abs(e.x)) * os.y;
    }
    return normalize(v);
}

// Decode the normal stored in the low kClusterLodAttrEncNormalBits of a packed
// normal+tangent word (mirrors shaderio::NormalUnpack).
float3 ClusterLodNormalUnpack(uint packed)
{
    const uint   mask = (1u << (kClusterLodAttrEncNormalBits / 2u)) - 1u;  // 2047
    const uint2  pv   = uint2(packed, packed >> 11u) & uint2(mask, mask);
    const float2 v    = (float2(float(pv.x), float(pv.y)) / float(mask)) * 2.0f - 1.0f;
    return ClusterLodOctToVec(v);
}

// Byte offset (within groupData) of the cluster's packed normal+tangent array.
// Sits immediately after the position array; see the layout in ClusterLodTex0ByteBase.
// Caller must guard on (cluster.attributeBits & shaderio::ClusterAttribute::VertexNormal).
uint ClusterLodNormalByteBase(shaderio::Cluster        cluster,
                        shaderio::ClusterAddress clusterAddress)
{
    const uint vertexCount = cluster.vertexCountMinusOne + 1u;
    // Normals ARE the region start when the upload stripped positions.
    const uint posBytes =
        (cluster.attributeBits & shaderio::ClusterAttribute::StrippedVertexPos) ? 0u
                                                                               : vertexCount * 12u;
    return clusterAddress.byteOffset + cluster.vertices + posBytes;
}

// All three per-vertex (object-space) normals for a triangle. Caller must
// confirm the cluster has normals; otherwise returns zero (caller falls back
// to the geometric normal).
void ClusterLodLoadTriangleNormals(ByteAddressBuffer            groupData,
                             shaderio::ClusterAddress     clusterAddress,
                             uint                         localTriIdx,
                             out float3                   n0,
                             out float3                   n1,
                             out float3                   n2)
{
    const shaderio::Cluster cluster =
        groupData.Load<shaderio::Cluster>(clusterAddress.byteOffset);
    if ((cluster.attributeBits & shaderio::ClusterAttribute::VertexNormal) == 0u)
    {
        n0 = n1 = n2 = float3(0.f, 0.f, 0.f);
        return;
    }
    const uint indexByteBase = clusterAddress.byteOffset + cluster.triangles;
    const uint normalBase    = ClusterLodNormalByteBase(cluster, clusterAddress);

    const uint3 idx = ClusterLodLoadTriangleIndices(groupData, indexByteBase, localTriIdx);
    n0 = ClusterLodNormalUnpack(groupData.Load<uint>(normalBase + idx.x * 4u));
    n1 = ClusterLodNormalUnpack(groupData.Load<uint>(normalBase + idx.y * 4u));
    n2 = ClusterLodNormalUnpack(groupData.Load<uint>(normalBase + idx.z * 4u));
}
