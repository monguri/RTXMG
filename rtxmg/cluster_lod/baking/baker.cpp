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

#include <algorithm>
#include <cassert>
#include <cstring>
#include <vector>

#include <donut/core/log.h>
#include <donut/core/math/math.h>

// meshopt_clusterlod.h is a single-file header-only library.
// Exactly one .cpp must define CLUSTERLOD_IMPLEMENTATION before including it.
#include <meshoptimizer.h>
#define CLUSTERLOD_IMPLEMENTATION
#include "meshopt_clusterlod.h"

#include "rtxmg/cluster_lod/baked_geometry.h"
#include "rtxmg/cluster_lod/baker.h"
#include "rtxmg/cluster_lod/baking/attribute_encoding.h"
#include "rtxmg/cluster_lod/group_codec.h"

using namespace donut::math;
using namespace donut;

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

namespace {

// ---------------------------------------------------------------------------
// Optional vertex compression, gated on BakerConfig::useCompressedData.
// QuantizeGeometry() lossily drops low mantissa bits before the LOD build so
// the arithmetic packer in group_codec.h has common zero bits to strip.
// ---------------------------------------------------------------------------

// Drop the low `dropBits` mantissa bits of a float (round-to-nearest), leaving
// inf/nan untouched and flushing denormals to zero. Matches vk QuantizeFloat.
inline float QuantizeFloat(float value, uint32_t dropBits)
{
    if (dropBits == 0)
        return value;
    union { uint32_t u32; float f32; } un;
    un.f32          = value;
    uint32_t ui     = un.u32;
    const int32_t  mask  = (1 << dropBits) - 1;
    const int32_t  round = (1 << dropBits) >> 1;
    int32_t        e     = ui & 0x7f800000;
    uint32_t       rui   = (ui + round) & ~mask;
    ui = (e == 0x7f800000) ? ui : rui;   // leave inf/nan
    ui = (e == 0)          ? 0u : ui;    // flush denormals
    un.u32 = ui;
    return un.f32;
}

// Quantize a geometry's source positions + texcoords in place, before LOD build.
void QuantizeGeometry(GeometryStorage& geometry, uint32_t posDropBits, uint32_t texDropBits)
{
    for (float3& p : geometry.vertexPositions)
    {
        p.x = QuantizeFloat(p.x, posDropBits);
        p.y = QuantizeFloat(p.y, posDropBits);
        p.z = QuantizeFloat(p.z, posDropBits);
    }

    if (geometry.vertexPositions.empty() || geometry.vertexAttributes.empty())
        return;

    const size_t stride = geometry.vertexAttributes.size() / geometry.vertexPositions.size();
    const size_t count  = geometry.vertexPositions.size();
    for (uint32_t t = 0; t < 2; t++)
    {
        const uint32_t usedBit = (t == 0) ? shaderio::ClusterAttribute::VertexTex0 : shaderio::ClusterAttribute::VertexTex1;
        if (!(geometry.attributeBits & usedBit))
            continue;
        const uint32_t off = (t == 0) ? geometry.attributeTex0offset : geometry.attributeTex1offset;
        for (size_t v = 0; v < count; v++)
        {
            float* uv = &geometry.vertexAttributes[v * stride + off];
            uv[0] = QuantizeFloat(uv[0], texDropBits);
            uv[1] = QuantizeFloat(uv[1], texDropBits);
        }
    }
}

// Zero-fill the padding gap between two adjacent span-like regions.
template<typename T0, typename T1>
void PadZeroes(std::span<T0> previous, T1* next)
{
    size_t padSize = reinterpret_cast<size_t>(next)
                   - reinterpret_cast<size_t>(previous.data() + previous.size());
    if (padSize)
        std::memset(previous.data() + previous.size(), 0, padSize);
}

// ---------------------------------------------------------------------------
// Material state-bit helpers.  They read the importer-built
// geometry.localMaterialStateBits lookup rather than the Scene's material
// table, which keeps the baker free of any Scene dependency.
// ---------------------------------------------------------------------------

// cluster → group: aggregate one cluster's stateBits, setting the _MIXED variant
// when it disagrees with its siblings.
inline void ApplyMaterialStateBits(uint32_t& stateBits, uint32_t clusterBits, bool isFirst)
{
    if (!isFirst)
    {
        if ((stateBits & shaderio::ClusterState::AlphaMasked) != (clusterBits & shaderio::ClusterState::AlphaMasked))
            stateBits |= shaderio::ClusterState::AlphaMaskedMixed;
        if ((stateBits & shaderio::ClusterState::TwoSided) != (clusterBits & shaderio::ClusterState::TwoSided))
            stateBits |= shaderio::ClusterState::TwoSidedMixed;
    }
    stateBits |= clusterBits;
}

// material → cluster: same _MIXED detection, folding one local material's
// alpha-masked / two-sided bits into the running cluster stateBits.
inline void ApplyMaterialStateBits(uint32_t& stateBits, const GeometryStorage& geometry,
                                   uint32_t localMaterialID, bool isFirst)
{
    const uint8_t materialBits = (localMaterialID < geometry.localMaterialStateBits.size())
                                     ? geometry.localMaterialStateBits[localMaterialID]
                                     : uint8_t(0);
    const bool materialAlpha   = (materialBits & shaderio::ClusterState::AlphaMasked) != 0;
    const bool materialTwoSide = (materialBits & shaderio::ClusterState::TwoSided)    != 0;

    if (!isFirst)
    {
        if (materialAlpha   != ((stateBits & shaderio::ClusterState::AlphaMasked) != 0))
            stateBits |= shaderio::ClusterState::AlphaMaskedMixed;
        if (materialTwoSide != ((stateBits & shaderio::ClusterState::TwoSided)    != 0))
            stateBits |= shaderio::ClusterState::TwoSidedMixed;
    }
    if (materialAlpha)
        stateBits |= shaderio::ClusterState::AlphaMasked;
    if (materialTwoSide)
        stateBits |= shaderio::ClusterState::TwoSided;
}

// Sets the high 2 bits of a per-triangle material byte; the caller fills the
// low 6 with the material index itself.
inline void ApplyMaterialTriangleBits(uint8_t& triangleBits, const GeometryStorage& geometry,
                                      uint32_t localMaterialID)
{
    const uint8_t materialBits = (localMaterialID < geometry.localMaterialStateBits.size())
                                     ? geometry.localMaterialStateBits[localMaterialID]
                                     : uint8_t(0);
    if (materialBits & shaderio::ClusterState::AlphaMasked)
        triangleBits |= uint8_t(shaderio::kClusterTriangleAlphaMasked);
    if (materialBits & shaderio::ClusterState::TwoSided)
        triangleBits |= uint8_t(shaderio::kClusterTriangleTwoSided);
}

// Reads the per-vertex local material ID out of the float-encoded attribute
// stream.  Only valid on multi-material geometry; the caller must check.
inline uint8_t GetMaterialLocalIndex(const GeometryStorage& geometry, uint32_t vertexIndex,
                                     uint32_t attributeStride)
{
    return uint8_t(geometry.vertexAttributes[vertexIndex * attributeStride
                                             + geometry.attributeMaterialOffset]);
}

// ---------------------------------------------------------------------------
// StoreGroup pass-1 helpers — one cluster's payload, written into the temp
// GroupStorage.  `vertexDataOffset` is the running offset into that blob's
// vertex region and is advanced in place; `triangleDataOffset` is the byte
// offset of the cluster's own triangle payload.
// ---------------------------------------------------------------------------

// Builds the cluster's local vertex list and its 8-bit index bytes, escalating
// groupCluster.localMaterialID when a vertex disagrees with the cluster-uniform
// material.  Returns the deduplicated vertex count.
uint32_t DedupClusterVertices(const GeometryStorage& geometry, uint32_t attributeStride,
                              bool hasMultiMaterial, uint8_t localMaterialID,
                              const clodCluster& cluster, uint32_t triangleDataOffset,
                              GroupStorage& storage, shaderio::Cluster& groupCluster,
                              uint32_t* localVertices,
                              uint32_t* cacheEarlyValue, uint32_t* cacheEarlyPos)
{
    uint32_t vertexCount = 0;

    std::memset(cacheEarlyValue, ~0, sizeof(uint32_t) * 256);

    for (uint32_t i = 0; i < cluster.index_count; i++)
    {
        uint32_t vertexIndex = cluster.indices[i];
        uint32_t cacheIndex  = ~0u;

        if (cacheEarlyValue[vertexIndex & 0xFF] == vertexIndex)
        {
            cacheIndex = cacheEarlyPos[vertexIndex & 0xFF];
        }
        else
        {
            for (uint32_t v = 0; v < vertexCount; v++)
            {
                if (localVertices[v] == vertexIndex)
                {
                    cacheIndex = v;
                    break;
                }
            }
        }

        if (cacheIndex == ~0u)
        {
            cacheIndex                          = vertexCount++;
            localVertices[cacheIndex]           = vertexIndex;
            cacheEarlyValue[vertexIndex & 0xFF] = vertexIndex;
            cacheEarlyPos[vertexIndex & 0xFF]   = cacheIndex;

            // Any newly-seen vertex whose material disagrees with the
            // cluster-uniform one flips the cluster to per-triangle material
            // storage.
            if (hasMultiMaterial
                && localMaterialID != GetMaterialLocalIndex(geometry, vertexIndex, attributeStride))
            {
                groupCluster.localMaterialID = uint8_t(shaderio::kPerTriangleMaterials);
            }
        }

        storage.indices[i + triangleDataOffset] = uint8_t(cacheIndex);
    }

    return vertexCount;
}

// Appends one material byte per triangle after a mixed cluster's index bytes.
// Returns the cluster stateBits aggregated over those materials.
uint32_t EmitPerTriangleMaterials(const GeometryStorage& geometry, uint32_t attributeStride,
                                  const clodCluster& cluster, uint32_t triangleCount,
                                  uint32_t triangleDataOffset, GroupStorage& storage)
{
    uint32_t stateBits = 0;
    for (uint32_t t = 0; t < triangleCount; ++t)
    {
        uint8_t vertexMaterial[3];
        vertexMaterial[0] = GetMaterialLocalIndex(geometry, cluster.indices[t * 3 + 0], attributeStride);
        vertexMaterial[1] = GetMaterialLocalIndex(geometry, cluster.indices[t * 3 + 1], attributeStride);
        vertexMaterial[2] = GetMaterialLocalIndex(geometry, cluster.indices[t * 3 + 2], attributeStride);

        const bool diff01 = vertexMaterial[0] != vertexMaterial[1];
        const bool diff12 = vertexMaterial[1] != vertexMaterial[2];
        if (diff01 || diff12)
        {
            // Pick the material of the pair that agrees, falling back to V0 when
            // all three differ.  Picking the odd vertex out instead mis-materials
            // cluster-internal seams.
            if (diff01 && !diff12)
                vertexMaterial[0] = vertexMaterial[1];
        }
        ApplyMaterialStateBits(stateBits, geometry, vertexMaterial[0], /*isFirst=*/(t == 0));
        uint8_t triangleByte = vertexMaterial[0];
        ApplyMaterialTriangleBits(triangleByte, geometry, vertexMaterial[0]);
        storage.indices[t + triangleDataOffset] = triangleByte;
    }
    return stateBits;
}

// Writes the cluster's positions and folds them into its bbox.
void PackClusterPositions(const GeometryStorage& geometry, const uint32_t* localVertices,
                          uint32_t vertexCount, uint32_t& vertexDataOffset,
                          GroupStorage& storage, shaderio::BBox& bbox)
{
    for (uint32_t v = 0; v < vertexCount; v++)
    {
        float3 pos = geometry.vertexPositions[localVertices[v]];
        *reinterpret_cast<float3*>(&storage.vertices[vertexDataOffset + v * 3]) = pos;
        bbox.lo = min(bbox.lo, pos);
        bbox.hi = max(bbox.hi, pos);
    }
    vertexDataOffset += vertexCount * 3;
}

// Writes one packed word per vertex holding the normal, plus the tangent when
// the geometry carries one.  No-op for geometry without normals.
void PackClusterNormals(const GeometryStorage& geometry, uint32_t attributeStride,
                        const uint32_t* localVertices, uint32_t vertexCount,
                        uint32_t& vertexDataOffset, GroupStorage& storage)
{
    if (!(geometry.attributeBits & shaderio::ClusterAttribute::VertexNormal))
        return;

    if (geometry.attributeBits & shaderio::ClusterAttribute::VertexTangent)
    {
        for (uint32_t v = 0; v < vertexCount; v++)
        {
            const float* na = &geometry.vertexAttributes[localVertices[v] * attributeStride
                                                           + geometry.attributeNormalOffset];
            const float* ta = &geometry.vertexAttributes[localVertices[v] * attributeStride
                                                           + geometry.attributeTangentOffset];
            float3 normal  = *reinterpret_cast<const float3*>(na);
            float4 tangent = *reinterpret_cast<const float4*>(ta);

            uint32_t encoded = shaderio::NormalPack(normal);
            encoded |= shaderio::TangentPack(normal, tangent) << ATTRENC_NORMAL_BITS;

            *reinterpret_cast<uint32_t*>(&storage.vertices[vertexDataOffset + v]) = encoded;
        }
    }
    else
    {
        for (uint32_t v = 0; v < vertexCount; v++)
        {
            const float* na = &geometry.vertexAttributes[localVertices[v] * attributeStride
                                                           + geometry.attributeNormalOffset];
            float3 normal = *reinterpret_cast<const float3*>(na);
            *reinterpret_cast<uint32_t*>(&storage.vertices[vertexDataOffset + v])
                = shaderio::NormalPack(normal);
        }
    }
    vertexDataOffset += vertexCount;
}

// Writes TEXCOORD_0/1 as po2-grid quantized values where the cluster allows it
// (see QuantizeClusterTexCoords), raw float2s otherwise.  Returns the cluster's
// ClusterEncoding bits.
uint8_t PackClusterTexCoords(const GeometryStorage& geometry, uint32_t attributeStride,
                             const uint32_t* localVertices, uint32_t vertexCount,
                             bool quantizeTexCoords, uint32_t& vertexDataOffset,
                             GroupStorage& storage)
{
    uint8_t quantizedTexBits = 0;
    for (uint32_t t = 0; t < 2; t++)
    {
        uint32_t usedBit         = (t == 0) ? shaderio::ClusterAttribute::VertexTex0 : shaderio::ClusterAttribute::VertexTex1;
        uint32_t attributeTexOff = (t == 0) ? geometry.attributeTex0offset : geometry.attributeTex1offset;

        if (!(geometry.attributeBits & usedBit))
            continue;

        // align to float2
        vertexDataOffset = (vertexDataOffset + 1u) & ~1u;

        auto uvAt = [&](uint32_t v) {
            const float* ua = &geometry.vertexAttributes[localVertices[v] * attributeStride
                                                           + attributeTexOff];
            return *reinterpret_cast<const float2*>(ua);
        };

        float2   lo{}, step{};
        uint32_t texDeltas[kMaxClusterVertices * 2];
        const bool quantize = quantizeTexCoords
                           && QuantizeClusterTexCoords(vertexCount, uvAt, lo, step, texDeltas);
        if (quantize)
        {
            StoreQuantizedTexHeader(&storage.vertices[vertexDataOffset], lo, step);
            uint32_t* deltas =
                reinterpret_cast<uint32_t*>(&storage.vertices[vertexDataOffset + 4]);
            for (uint32_t v = 0; v < vertexCount; v++)
                deltas[v] = texDeltas[v * 2] | (texDeltas[v * 2 + 1] << 16);
            vertexDataOffset += 4 + vertexCount;
            quantizedTexBits |= uint8_t((t == 0) ? shaderio::ClusterEncoding::QuantizedTex0
                                                 : shaderio::ClusterEncoding::QuantizedTex1);
        }
        else
        {
            for (uint32_t v = 0; v < vertexCount; v++)
            {
                *reinterpret_cast<float2*>(&storage.vertices[vertexDataOffset + v * 2]) = uvAt(v);
            }
            vertexDataOffset += vertexCount * 2;
        }
    }
    return quantizedTexBits;
}

// Folds the cluster's shortest / longest triangle edge into its bbox.
void AccumulateClusterEdgeExtents(const GeometryStorage& geometry, const uint32_t* localVertices,
                                  GroupStorage& storage, uint32_t triangleDataOffset,
                                  uint32_t triangleCount, shaderio::BBox& bbox)
{
    for (uint32_t tri = 0; tri < triangleCount; tri++)
    {
        float3 triPos[3];
        for (uint32_t v = 0; v < 3; v++)
            triPos[v] = geometry.vertexPositions[localVertices[storage.indices[triangleDataOffset + tri * 3 + v]]];

        for (uint32_t e = 0; e < 3; e++)
        {
            float d       = length(triPos[e] - triPos[(e + 1) % 3]);
            bbox.shortestEdge = std::min(bbox.shortestEdge, d);
            bbox.longestEdge  = std::max(bbox.longestEdge,  d);
        }
    }
}

// ---------------------------------------------------------------------------
// StoreGroup pass-2 helpers.
// ---------------------------------------------------------------------------

// Folds one stored group's counts and bounds into the per-LOD-level and
// geometry-wide totals.
void AccumulateGeometryStats(GeometryStorage& geometry, const GroupInfo& groupInfo,
                             uint32_t level, const clodGroup& group,
                             const shaderio::BBox& groupBbox,
                             uint32_t clusterMaxTrianglesCount, uint32_t clusterMaxVerticesCount)
{
    if (level >= geometry.lodLevels.size())
    {
        const shaderio::LodLevel* prev = level ? &geometry.lodLevels[level - 1] : nullptr;
        shaderio::LodLevel init{};
        init.clusterOffset         = prev ? prev->clusterOffset + prev->clusterCount : 0;
        init.groupOffset           = prev ? prev->groupOffset   + prev->groupCount   : 0;
        init.minBoundingSphereRadius = FLT_MAX;
        init.minMaxQuadricError      = FLT_MAX;
        geometry.lodLevels.push_back(init);
    }

    geometry.lodLevels[level].clusterCount += groupInfo.clusterCount;
    geometry.lodLevels[level].groupCount++;
    geometry.lodLevels[level].minBoundingSphereRadius =
        std::min(geometry.lodLevels[level].minBoundingSphereRadius, group.simplified.radius);
    geometry.lodLevels[level].minMaxQuadricError =
        std::min(geometry.lodLevels[level].minMaxQuadricError, group.simplified.error);

    geometry.clusterMaxTrianglesCount = std::max(clusterMaxTrianglesCount, geometry.clusterMaxTrianglesCount);
    geometry.clusterMaxVerticesCount  = std::max(clusterMaxVerticesCount,  geometry.clusterMaxVerticesCount);

    if (level == 0)
    {
        geometry.hiClustersCount += groupInfo.clusterCount;
        geometry.hiTriangleCount += groupInfo.triangleCount;
        geometry.hiVerticesCount += groupInfo.vertexCount;
    }
    geometry.totalClustersCount += groupInfo.clusterCount;
    geometry.totalTriangleCount += groupInfo.triangleCount;
    geometry.totalVerticesCount += groupInfo.vertexCount;

    geometry.bbox.lo = min(geometry.bbox.lo, groupBbox.lo);
    geometry.bbox.hi = max(geometry.bbox.hi, groupBbox.hi);
}

// Copies the packed temp group into geometry.groupData at groupInfo.offsetBytes,
// rebasing each cluster's vertex/index offsets onto its own header.
void CopyGroupToStorage(GeometryStorage& geometry, const GroupInfo& groupInfo,
                        uint32_t level, uint32_t groupStateBits, const clodGroup& group,
                        const GroupStorage& src, uint32_t clusterCount)
{
    GroupStorage dst(&geometry.groupData[groupInfo.offsetBytes], groupInfo);

    dst.group->clusterResidentID      = 0;
    dst.group->groupResidentID        = 0;
    dst.group->lodLevel               = uint16_t(level);
    // Read by traversal_run_groups' alpha-queue selection.
    dst.group->stateBits              = uint8_t(groupStateBits);
    dst.group->clusterCount           = uint16_t(groupInfo.clusterCount);
    dst.group->traversalMetric.boundingSphereX      = group.simplified.center[0];
    dst.group->traversalMetric.boundingSphereY      = group.simplified.center[1];
    dst.group->traversalMetric.boundingSphereZ      = group.simplified.center[2];
    dst.group->traversalMetric.boundingSphereRadius = group.simplified.radius;
    dst.group->traversalMetric.maxQuadricError      = group.simplified.error;

    std::memcpy(dst.clusters.data(), src.clusters.data(), dst.clusters.size_bytes());

    // Patch cluster vertex/index offsets to be relative to their cluster header.
    // For a compressed blob cl.vertices still refers to the UNCOMPRESSED layout
    // (where DecompressGroup writes), so the assert bound must be that size.
    const bool   compressed   = groupInfo.uncompressedSizeBytes != 0u;
    const size_t vertBoundSize = compressed ? size_t(groupInfo.uncompressedSizeBytes) : groupInfo.sizeBytes;
    for (uint32_t c = 0; c < clusterCount; c++)
    {
        shaderio::Cluster& cl = dst.clusters[c];
        cl.vertices = dst.GetClusterLocalOffset(c, dst.vertices.data() + cl.vertices, vertBoundSize);
        if (compressed)
            // CompressGroup hijacked cl.triangles to the group offset of this
            // cluster's COMPRESSED vertex stream (in the vertices region).
            cl.triangles = dst.GetClusterLocalOffset(c, dst.vertices.data() + cl.triangles);
        else
            cl.triangles = dst.GetClusterLocalOffset(c, dst.indices.data() + cl.triangles);
    }

    std::memcpy(dst.clusterGeneratingGroups.data(),
                src.clusterGeneratingGroups.data(),
                dst.clusterGeneratingGroups.size_bytes());
    PadZeroes(dst.clusterGeneratingGroups, dst.clusterBboxes.data());

    std::memcpy(dst.clusterBboxes.data(), src.clusterBboxes.data(), dst.clusterBboxes.size_bytes());
    std::memcpy(dst.indices.data(),       src.indices.data(),       dst.indices.size_bytes());
    PadZeroes(dst.indices, dst.vertices.data());
    std::memcpy(dst.vertices.data(),      src.vertices.data(),      dst.vertices.size_bytes());

    // Zero tail padding to end of the aligned blob.
    const uint8_t* blobEnd = geometry.groupData.data() + groupInfo.offsetBytes + groupInfo.sizeBytes;
    PadZeroes(dst.vertices, blobEnd);
}

} // namespace

// ---------------------------------------------------------------------------
// BakeContext — per-Build() state, passed through clodBuild callback.
// ---------------------------------------------------------------------------

struct ClusterLodBaker::BakeContext
{
    GeometryStorage*    geometry = nullptr;
    ClusterLodBaker*    baker    = nullptr;

    GroupInfo worstCaseInfo;

    // Temp buffer for one group: [GroupStorage blob | vertex caches]
    //   vertexCacheEarlyValue[256]   — quick early-out check
    //   vertexCacheEarlyPos[256]     — position in local vertex list
    //   vertexCacheLocal[groupSize * clusterVertices] — per-cluster vertex remap
    std::vector<uint8_t> tempBuf;

    uint32_t worstCaseStorageSize = 0;
    uint32_t tempBufStride        = 0;   // total bytes per temp buffer slot
};

// ---------------------------------------------------------------------------
// ClusterLodBaker constructor
// ---------------------------------------------------------------------------

ClusterLodBaker::ClusterLodBaker(const BakerConfig& config)
    : m_config(config)
{
}

// ---------------------------------------------------------------------------
// StoreGroup — packs one clodGroup into geometry.groupData.
// ---------------------------------------------------------------------------

uint32_t ClusterLodBaker::StoreGroup(BakeContext*  ctx,
                                     uint32_t      groupIndex,
                                     const void*   clusterLodGroupPtr,
                                     uint32_t      clusterCount,
                                     const void*   clustersPtr)
{
    const clodGroup*   group    = static_cast<const clodGroup*>(clusterLodGroupPtr);
    const clodCluster* clusters = static_cast<const clodCluster*>(clustersPtr);

    GeometryStorage& geometry = *ctx->geometry;

    uint32_t level = uint32_t(group->depth);

    // Temp storage for this group (located at the start of tempBuf).
    uint8_t* groupTempData = ctx->tempBuf.data();

    GroupInfo    groupTempInfo    = ctx->worstCaseInfo;
    GroupStorage groupTempStorage(groupTempData, groupTempInfo);

    // Vertex caches live after the GroupStorage blob.
    uint32_t* vertexCacheEarlyValue = reinterpret_cast<uint32_t*>(groupTempData + ctx->worstCaseStorageSize);
    uint32_t* vertexCacheEarlyPos   = vertexCacheEarlyValue + 256;
    uint32_t* vertexCacheLocal      = vertexCacheEarlyPos   + 256;

    uint32_t clusterMaxVerticesCount  = 0;
    uint32_t clusterMaxTrianglesCount = 0;

    shaderio::BBox groupBbox = { {FLT_MAX, FLT_MAX, FLT_MAX}, {-FLT_MAX, -FLT_MAX, -FLT_MAX}, 0.0f, 0.0f };

    GroupInfo groupInfo = {};
    // Aggregated from each cluster in pass 1, written to the group header in pass 2.
    uint32_t  groupStateBits = 0;

    // ------------------------------------------------------------------
    // Pass 1: fill temp GroupStorage.
    // Performs vertex de-duplication so we know final vertexDataCount
    // before making the allocation in geometry.groupData.
    // ------------------------------------------------------------------
    {
        uint32_t attributeStride = geometry.vertexPositions.empty()
                                       ? 0u
                                       : uint32_t(geometry.vertexAttributes.size()
                                                  / geometry.vertexPositions.size());

        // Multi-material assets carry a per-vertex material ID in vertexAttributes
        // at attributeMaterialOffset.
        const bool hasMultiMaterial = geometry.localMaterialIDs.size() > 1;

        uint32_t triangleOffset     = 0;   // triangle count (index triplets)
        uint32_t vertexOffset       = 0;
        uint32_t vertexDataOffset   = 0;
        // triangleDataOffset is in BYTES — running offset into the group
        // blob's triangle payload. For uniform clusters it advances by
        // triangleCount * 3 (index bytes only); for mixed clusters it
        // additionally advances by triangleCount (one material byte per
        // triangle, appended after the index bytes).
        uint32_t triangleDataOffset = 0;

        for (uint32_t c = 0; c < clusterCount; c++)
        {
            uint32_t* localVertices = &vertexCacheLocal[vertexOffset];

            const clodCluster& tempCluster = clusters[c];

            shaderio::Cluster& groupCluster  = groupTempStorage.clusters[c];
            uint32_t           triangleCount = uint32_t(tempCluster.index_count / 3);
            uint32_t           stateBits     = 0;

            groupCluster.vertices  = vertexDataOffset;
            groupCluster.triangles = triangleDataOffset;

            // --- material discovery: assume uniform until a vertex
            // disagrees, then escalate to shaderio::kPerTriangleMaterials.
            uint8_t localMaterialID = 0;
            if (hasMultiMaterial)
            {
                localMaterialID              = GetMaterialLocalIndex(geometry, tempCluster.indices[0], attributeStride);
                groupCluster.localMaterialID = localMaterialID;
            }
            else
            {
                groupCluster.localMaterialID = 0;
            }
            ApplyMaterialStateBits(stateBits, geometry, localMaterialID, /*isFirst=*/true);

            const uint32_t vertexCount =
                DedupClusterVertices(geometry, attributeStride, hasMultiMaterial, localMaterialID,
                                     tempCluster, triangleDataOffset, groupTempStorage,
                                     groupCluster, localVertices,
                                     vertexCacheEarlyValue, vertexCacheEarlyPos);
            triangleDataOffset += triangleCount * 3;

            // Mixed clusters append one material byte per triangle after their
            // index bytes.
            if (hasMultiMaterial
                && groupCluster.localMaterialID == uint8_t(shaderio::kPerTriangleMaterials))
            {
                stateBits = EmitPerTriangleMaterials(geometry, attributeStride, tempCluster,
                                                     triangleCount, triangleDataOffset,
                                                     groupTempStorage);
                triangleDataOffset += triangleCount;
            }

            shaderio::BBox bbox = { {FLT_MAX, FLT_MAX, FLT_MAX}, {-FLT_MAX, -FLT_MAX, -FLT_MAX},
                                    FLT_MAX, -FLT_MAX };

            PackClusterPositions(geometry, localVertices, vertexCount, vertexDataOffset,
                                 groupTempStorage, bbox);
            PackClusterNormals(geometry, attributeStride, localVertices, vertexCount,
                               vertexDataOffset, groupTempStorage);
            const uint8_t quantizedTexBits =
                PackClusterTexCoords(geometry, attributeStride, localVertices, vertexCount,
                                     m_config.quantizeTexCoords, vertexDataOffset,
                                     groupTempStorage);
            AccumulateClusterEdgeExtents(geometry, localVertices, groupTempStorage,
                                         groupCluster.triangles, triangleCount, bbox);

            groupBbox.lo = min(groupBbox.lo, bbox.lo);
            groupBbox.hi = max(groupBbox.hi, bbox.hi);

            groupTempStorage.clusterBboxes[c]           = bbox;
            groupTempStorage.clusterGeneratingGroups[c] = uint32_t(tempCluster.refined);

            groupCluster.triangleCountMinusOne = uint8_t(triangleCount - 1);
            groupCluster.vertexCountMinusOne   = uint8_t(vertexCount - 1);
            groupCluster.lodLevel              = uint8_t(level);
            groupCluster.groupChildIndex       = uint8_t(c);
            groupCluster.attributeBits         = uint8_t(geometry.attributeBits);
            groupCluster.stateBits             = uint8_t(stateBits);
            groupCluster.encodingBits          = quantizedTexBits;

            ApplyMaterialStateBits(groupStateBits, stateBits, /*isFirst=*/(c == 0));

            clusterMaxTrianglesCount = std::max(clusterMaxTrianglesCount, triangleCount);
            clusterMaxVerticesCount  = std::max(clusterMaxVerticesCount,  vertexCount);

            vertexOffset   += vertexCount;
            triangleOffset += triangleCount;
        }

        groupInfo.offsetBytes                 = 0;
        groupInfo.reserved1                   = 0;
        groupInfo.clusterCount                = uint8_t(clusterCount);
        groupInfo.triangleCount               = uint16_t(triangleOffset);
        groupInfo.vertexCount                 = uint16_t(vertexOffset);
        groupInfo.lodLevel                    = uint8_t(level);
        groupInfo.attributeBits               = uint8_t(geometry.attributeBits);
        groupInfo.vertexDataCount             = vertexDataOffset;
        // Stored explicitly, since the payload also holds the mixed clusters'
        // per-triangle material bytes and so exceeds triangleCount * 3.
        groupInfo.triangleDataCount           = triangleDataOffset;
        groupInfo.uncompressedVertexDataCount = 0;
        groupInfo.uncompressedSizeBytes       = 0;
        groupInfo.sizeBytes                   = groupInfo.ComputeSize();
    }

    // Shrinks groupInfo.sizeBytes, which is what pass 2 copies.
    if (m_config.useCompressedData)
        CompressGroup(groupTempStorage, groupInfo, geometry, vertexCacheLocal);

    // ------------------------------------------------------------------
    // Pass 2: allocate in geometry.groupData and copy from temp storage.
    // (Single-threaded — no mutex needed.)
    // ------------------------------------------------------------------

    AccumulateGeometryStats(geometry, groupInfo, level, *group, groupBbox,
                            clusterMaxTrianglesCount, clusterMaxVerticesCount);

    // Allocate space in groupData.
    if (groupIndex == ~0u)
    {
        groupIndex = uint32_t(geometry.groupInfos.size());
        geometry.groupInfos.resize(groupIndex + 1);
    }

    groupInfo.offsetBytes = geometry.groupData.size();
    geometry.groupData.resize(groupInfo.offsetBytes + groupInfo.sizeBytes);
    geometry.groupInfos[groupIndex] = groupInfo;

    CopyGroupToStorage(geometry, groupInfo, level, groupStateBits, *group,
                       groupTempStorage, clusterCount);

    return groupIndex;
}

// ---------------------------------------------------------------------------
// Static callback forwarded into the instance.
// ---------------------------------------------------------------------------

int ClusterLodBaker::GroupCallback(void*              ctx,
                                     clodGroup          group,
                                     const clodCluster* clusters,
                                     size_t             cluster_count,
                                     size_t             /*task_index*/,
                                     unsigned int       /*thread_index*/)
{
    auto* bctx = static_cast<BakeContext*>(ctx);
    // The return value becomes `clodCluster::refined` on every cluster derived
    // from this group, i.e. the generating group traversal_run_groups cuts on.
    return int(bctx->baker->StoreGroup(bctx, ~0u, &group, uint32_t(cluster_count), clusters));
}

// ---------------------------------------------------------------------------
// Build — entry point.  Calls meshoptimizer's clodBuild, then hierarchy.
// ---------------------------------------------------------------------------

void ClusterLodBaker::Build(GeometryStorage& geometry)
{
    // Lossy, and must run before the LOD build so simplification sees the
    // quantized positions that CompressGroup() will later pack.
    if (m_config.useCompressedData)
        QuantizeGeometry(geometry, m_config.compressionPosDropBits, m_config.compressionTexDropBits);

    clodConfig clusterLodInfo = m_config.meshoptPreferRayTracing
        ? clodDefaultConfigRT(m_config.clusterTriangles)
        : clodDefaultConfig  (m_config.clusterTriangles);

    clusterLodInfo.cluster_fill_weight  = m_config.meshoptFillWeight;
    clusterLodInfo.cluster_split_factor = m_config.meshoptSplitFactor;
    clusterLodInfo.max_vertices         = m_config.clusterVertices;
    clusterLodInfo.partition_size       = m_config.clusterGroupSize;
    clusterLodInfo.partition_spatial    = true;
    clusterLodInfo.partition_sort       = true;
    clusterLodInfo.optimize_clusters    = true;

    // Account for meshopt_partitionClusters using a target with higher worst case.
    while ((clusterLodInfo.partition_size + clusterLodInfo.partition_size / 3) > m_config.clusterGroupSize)
        clusterLodInfo.partition_size--;

    clusterLodInfo.simplify_error_merge_previous = m_config.lodErrorMergePrevious;
    clusterLodInfo.simplify_error_merge_additive = m_config.lodErrorMergeAdditive;
    clusterLodInfo.simplify_error_edge_limit     = m_config.lodErrorEdgeLimit;

    // Build mesh description.
    clodMesh inputMesh                = {};
    inputMesh.vertex_positions        = reinterpret_cast<const float*>(geometry.vertexPositions.data());
    inputMesh.vertex_count            = geometry.vertexPositions.size();
    inputMesh.vertex_positions_stride = sizeof(float3);
    inputMesh.index_count             = geometry.triangles.size() * 3;
    inputMesh.indices                 = reinterpret_cast<const uint32_t*>(geometry.triangles.data());

    // 16 covers the worst-case attribute stride NORMAL(3) + TANGENT(4) +
    // TEX0(2) + TEX1(2) + MATERIAL(1), with headroom.
    float attributeWeights[16] = {};
    if (geometry.attributesWithWeights)
    {
        if (m_config.simplifyNormalWeight > 0.0f && (geometry.attributeBits & shaderio::ClusterAttribute::VertexNormal))
        {
            attributeWeights[geometry.attributeNormalOffset + 0] = m_config.simplifyNormalWeight;
            attributeWeights[geometry.attributeNormalOffset + 1] = m_config.simplifyNormalWeight;
            attributeWeights[geometry.attributeNormalOffset + 2] = m_config.simplifyNormalWeight;
        }
        if (m_config.simplifyTexCoordWeight > 0.0f && (geometry.attributeBits & shaderio::ClusterAttribute::VertexTex0))
        {
            attributeWeights[geometry.attributeTex0offset + 0] = m_config.simplifyTexCoordWeight;
            attributeWeights[geometry.attributeTex0offset + 1] = m_config.simplifyTexCoordWeight;
        }
        if (m_config.simplifyTexCoordWeight > 0.0f && (geometry.attributeBits & shaderio::ClusterAttribute::VertexTex1))
        {
            attributeWeights[geometry.attributeTex1offset + 0] = m_config.simplifyTexCoordWeight;
            attributeWeights[geometry.attributeTex1offset + 1] = m_config.simplifyTexCoordWeight;
        }
        if (m_config.simplifyTangentWeight > 0.0f
            && (geometry.attributeBits & shaderio::ClusterAttribute::VertexTangent))
        {
            attributeWeights[geometry.attributeTangentOffset + 0] = m_config.simplifyTangentWeight;
            attributeWeights[geometry.attributeTangentOffset + 1] = m_config.simplifyTangentWeight;
            attributeWeights[geometry.attributeTangentOffset + 2] = m_config.simplifyTangentWeight;
            if (m_config.simplifyTangentSignWeight > 0.0f)
                attributeWeights[geometry.attributeTangentOffset + 3] = m_config.simplifyTangentSignWeight;
        }
        // Heavy weight on the per-vertex local-material-ID float keeps the
        // simplifier from collapsing edges across material boundaries.
        if (m_config.simplifyMaterialWeight > 0.0f
            && geometry.localMaterialIDs.size() > 1
            && geometry.attributeMaterialOffset != ~0u)
        {
            attributeWeights[geometry.attributeMaterialOffset + 0] = m_config.simplifyMaterialWeight;
        }

        inputMesh.attribute_count          = geometry.attributesWithWeights;
        inputMesh.vertex_attributes        = geometry.vertexAttributes.data();
        inputMesh.vertex_attributes_stride = sizeof(float) * inputMesh.attribute_count;
        inputMesh.attribute_weights        = attributeWeights;
    }

    // Pre-size worst-case temp buffer.
    BakeContext ctx;
    ctx.geometry = &geometry;
    ctx.baker    = this;

    ctx.worstCaseInfo.clusterCount      = uint8_t(m_config.clusterGroupSize);
    ctx.worstCaseInfo.vertexCount       = uint16_t(m_config.clusterGroupSize * m_config.clusterVertices);
    ctx.worstCaseInfo.triangleCount     = uint16_t(m_config.clusterGroupSize * m_config.clusterTriangles);
    ctx.worstCaseInfo.attributeBits     = uint8_t(geometry.attributeBits);
    ctx.worstCaseInfo.vertexDataCount   = ctx.worstCaseInfo.EstimateVertexDataCount();
    // Multi-material geometry may append a material byte per triangle, which the
    // worst-case temp buffer has to cover.
    ctx.worstCaseInfo.triangleDataCount =
        ctx.worstCaseInfo.EstimateTriangleDataCount(/*hasTriangleMaterials=*/
            geometry.localMaterialIDs.size() > 1);
    ctx.worstCaseInfo.sizeBytes         = ctx.worstCaseInfo.ComputeSize();

    ctx.worstCaseStorageSize = uint32_t(align_up(uint32_t(ctx.worstCaseInfo.sizeBytes), 4));
    ctx.tempBufStride        = ctx.worstCaseStorageSize
                             + sizeof(uint32_t) * 256          // vertexCacheEarlyValue
                             + sizeof(uint32_t) * 256          // vertexCacheEarlyPos
                             + sizeof(uint32_t) * m_config.clusterGroupSize * m_config.clusterVertices;
    ctx.tempBuf.resize(ctx.tempBufStride);

    // Reserve geometry arrays.
    size_t reservedClusters  = (geometry.triangles.size() + m_config.clusterTriangles - 1) / m_config.clusterTriangles;
    size_t reservedGroups    = (reservedClusters  + m_config.clusterGroupSize - 1) / m_config.clusterGroupSize;
    size_t reservedTriangles = geometry.triangles.size();

    reservedClusters  = size_t(double(reservedClusters)  * 2.0);
    reservedGroups    = size_t(double(reservedGroups)     * 3.0);
    reservedTriangles = size_t(double(reservedTriangles)  * 2.0);

    size_t reservedData = 0;
    reservedData += sizeof(shaderio::Group)   * reservedGroups;
    reservedData += sizeof(shaderio::Cluster) * reservedClusters;
    reservedData += sizeof(shaderio::BBox)    * reservedClusters;
    reservedData += sizeof(uint32_t)          * reservedClusters;
    reservedData += sizeof(uint8_t)           * reservedTriangles;
    reservedData += sizeof(float3)            * reservedClusters * m_config.clusterVertices;

    geometry.groupData.reserve(reservedData);
    geometry.groupInfos.reserve(reservedGroups);
    geometry.lodLevels.reserve(32);

    // Run the LOD builder.
    clodBuild(clusterLodInfo, inputMesh, &ctx, GroupCallback, nullptr);

    // The baked group blobs are self-contained; free the raw mesh arrays.
    geometry.triangles        = {};
    geometry.vertexPositions  = {};
    geometry.vertexAttributes = {};

    geometry.lodLevelsCount = uint32_t(geometry.lodLevels.size());

    if (geometry.lodLevelsCount == 0)
    {
        log::error("ClusterLodBaker: clodBuild produced no LOD levels.");
        return;
    }

    {
        const shaderio::LodLevel& lastLevel = geometry.lodLevels.back();
        // The LOD DAG should collapse to a single root group; >1 cluster in that
        // group is fine (disconnected foliage cards can't decimate further).
        // Warning, never error: donut pops a modal box for Error/Fatal and this
        // runs on parallel bake workers.
        if (lastLevel.groupCount != 1)
        {
            log::warning("ClusterLodBaker: LOD DAG root has %u groups (expected 1); clusters=%u.",
                         lastLevel.groupCount, lastLevel.clusterCount);
        }

        // The renderer copies this into RenderInstance::lowDetailClusterStateBits
        // as the alpha / two-sided hint for the low-detail render path.
        geometry.lowDetailClusterStateBits = 0;
        if (lastLevel.groupCount == 1 && lastLevel.clusterCount == 1
            && lastLevel.groupOffset < geometry.groupInfos.size())
        {
            const GroupInfo& lastGroupInfo = geometry.groupInfos[lastLevel.groupOffset];
            GroupView        lastGroupView{
                std::span<const uint8_t>(geometry.groupData), lastGroupInfo};
            if (!lastGroupView.clusters.empty())
                geometry.lowDetailClusterStateBits = uint8_t(lastGroupView.clusters[0].stateBits);
        }
    }

    geometry.groupInfos.shrink_to_fit();
    geometry.groupData.shrink_to_fit();
    geometry.lodLevels.shrink_to_fit();

    // Per-LOD Inspector totals.  Summed here, while the freshly baked groups are
    // still ordinary heap memory — at runtime groupData is an mmap of the shard,
    // so the same sweep would fault in the whole geometry pool.
    geometry.lodStats.assign(geometry.lodLevels.size(), {});
    for (size_t level = 0; level < geometry.lodLevels.size(); ++level)
    {
        const shaderio::LodLevel& lv = geometry.lodLevels[level];
        LodStats&                 st = geometry.lodStats[level];
        for (uint32_t k = 0;
             k < lv.groupCount && (lv.groupOffset + k) < geometry.groupInfos.size(); ++k)
        {
            const GroupInfo& gi = geometry.groupInfos[lv.groupOffset + k];
            GroupView        gv{std::span<const uint8_t>(geometry.groupData), gi};
            const GroupAttributeBytes ab = ComputeGroupAttributeBytes(gi, gv.clusters.data());

            st.totGroups++;
            st.totClusters    += gi.clusterCount;
            st.totTris        += gi.triangleCount;
            st.totBytes       += gi.sizeBytes;
            st.totDeviceBytes += gi.GetDeviceSize();
            st.posBytes       += ab.positions;
            st.nrmBytes       += ab.normals;
            st.uvBytes        += ab.Uvs();
            st.quantUv    |= uint8_t(ab.quantTex0 || ab.quantTex1);
            st.compressed |= uint8_t(gi.uncompressedSizeBytes != 0);
        }
    }
    geometry.lodStats.shrink_to_fit();

    BuildHierarchy(geometry);

    geometry.lodNodeBboxes.resize(geometry.lodNodes.size());
    ComputeBboxesRecursive(geometry, 0);
}

// ---------------------------------------------------------------------------
// BuildHierarchy — builds a spatial LOD node tree over the stored groups.
// ---------------------------------------------------------------------------

void ClusterLodBaker::BuildHierarchy(GeometryStorage& geometry)
{
    uint32_t lodLevelCount = geometry.lodLevelsCount;

    struct Range { uint32_t offset; uint32_t count; };
    std::vector<Range> lodNodeRanges(lodLevelCount);

    // Allocate node array:
    //   [0]          = top root
    //   [1 .. L]     = one per-LOD-level root
    //   [L+1 ..]     = remaining interior nodes
    {
        uint32_t nodeOffset = 1 + lodLevelCount;

        for (uint32_t L = 0; L < lodLevelCount; L++)
        {
            const shaderio::LodLevel& lod = geometry.lodLevels[L];
            uint32_t nodeCount      = lod.groupCount;
            uint32_t iterationCount = nodeCount;

            while (iterationCount > 1)
            {
                iterationCount = (iterationCount + m_config.preferredNodeWidth - 1) / m_config.preferredNodeWidth;
                nodeCount += iterationCount;
            }
            nodeCount--;   // subtract the root already counted above (goes in slot 1+L)

            lodNodeRanges[L].offset = nodeOffset;
            lodNodeRanges[L].count  = nodeCount;
            nodeOffset += nodeCount;
        }
        geometry.lodNodes.resize(nodeOffset);
    }

    // Build per-level trees.
    for (uint32_t L = 0; L < lodLevelCount; L++)
    {
        const shaderio::LodLevel& lod         = geometry.lodLevels[L];
        const Range&              lodNodeRange = lodNodeRanges[L];

        uint32_t nodeCount      = lod.groupCount;
        uint32_t nodeOffset     = lodNodeRange.offset;
        uint32_t lastNodeOffset = nodeOffset;

        // Create leaf nodes (one per group in this LOD level).
        for (uint32_t g = 0; g < nodeCount; g++)
        {
            uint32_t  groupID   = g + lod.groupOffset;
            GroupView groupView(geometry.groupData, geometry.groupInfos[groupID]);

            shaderio::Node& node = (nodeCount == 1)
                ? geometry.lodNodes[1 + L]
                : geometry.lodNodes[nodeOffset++];

            node                                      = {};
            node.groupRange.isGroup                   = 1;
            node.groupRange.groupIndex                = groupID;
            node.groupRange.groupClusterCountMinusOne = geometry.groupInfos[groupID].clusterCount - 1;
            node.traversalMetric                      = groupView.group->traversalMetric;
        }
        if (nodeCount == 1)
            nodeOffset++;

        // Build interior node levels bottom-up.
        uint32_t iterationCount = nodeCount;

        std::vector<uint32_t>       partitionedIndices;
        std::vector<shaderio::Node> oldNodes;

        while (iterationCount > 1)
        {
            uint32_t        lastNodeCount = iterationCount;
            shaderio::Node* lastNodes     = &geometry.lodNodes[lastNodeOffset];

            // Spatially cluster the child nodes.
            partitionedIndices.resize(lastNodeCount);
            meshopt_spatialClusterPoints(
                partitionedIndices.data(),
                &lastNodes->traversalMetric.boundingSphereX,
                lastNodeCount, sizeof(shaderio::Node),
                m_config.preferredNodeWidth);

            // Re-order last nodes by partition.
            oldNodes.assign(lastNodes, lastNodes + lastNodeCount);
            for (uint32_t n = 0; n < lastNodeCount; n++)
                lastNodes[n] = oldNodes[partitionedIndices[n]];

            iterationCount = (lastNodeCount + m_config.preferredNodeWidth - 1) / m_config.preferredNodeWidth;

            shaderio::Node* newNodes = (iterationCount == 1)
                ? &geometry.lodNodes[1 + L]
                : &geometry.lodNodes[nodeOffset];

            for (uint32_t n = 0; n < iterationCount; n++)
            {
                shaderio::Node* childrenNodes = &lastNodes[n * m_config.preferredNodeWidth];
                uint32_t childCount = std::min((n + 1) * m_config.preferredNodeWidth, lastNodeCount)
                                    - n * m_config.preferredNodeWidth;

                shaderio::Node& node = newNodes[n];
                node                              = {};
                node.nodeRange.isGroup            = 0;
                node.nodeRange.childCountMinusOne = childCount - 1;
                node.nodeRange.childOffset        = lastNodeOffset + n * m_config.preferredNodeWidth;
                node.traversalMetric.maxQuadricError = 0.0f;

                for (uint32_t c = 0; c < childCount; c++)
                {
                    node.traversalMetric.maxQuadricError = std::max(
                        node.traversalMetric.maxQuadricError,
                        childrenNodes[c].traversalMetric.maxQuadricError);
                }

                meshopt_Bounds merged = meshopt_computeSphereBounds(
                    &childrenNodes[0].traversalMetric.boundingSphereX,
                    childCount, sizeof(shaderio::Node),
                    &childrenNodes[0].traversalMetric.boundingSphereRadius,
                    sizeof(shaderio::Node));

                node.traversalMetric.boundingSphereX      = merged.center[0];
                node.traversalMetric.boundingSphereY      = merged.center[1];
                node.traversalMetric.boundingSphereZ      = merged.center[2];
                node.traversalMetric.boundingSphereRadius = merged.radius;
            }

            lastNodeOffset = nodeOffset;
            nodeOffset += iterationCount;
        }
        nodeOffset--;   // undo last advance — per-LOD root goes to slot 1+L, not nodeOffset
        assert(lodNodeRange.offset + lodNodeRange.count == nodeOffset);
    }

    // Build top-level root over all LOD-level roots.
    {
        meshopt_Bounds merged = meshopt_computeSphereBounds(
            &geometry.lodNodes[1].traversalMetric.boundingSphereX,
            lodLevelCount, sizeof(shaderio::Node),
            &geometry.lodNodes[1].traversalMetric.boundingSphereRadius,
            sizeof(shaderio::Node));

        shaderio::Node& root = geometry.lodNodes[0];
        root                                      = {};
        root.nodeRange.isGroup                    = 0;
        root.nodeRange.childCountMinusOne         = lodLevelCount - 1;
        root.nodeRange.childOffset                = 1;
        root.traversalMetric.boundingSphereX      = merged.center[0];
        root.traversalMetric.boundingSphereY      = merged.center[1];
        root.traversalMetric.boundingSphereZ      = merged.center[2];
        root.traversalMetric.boundingSphereRadius = merged.radius;
        root.traversalMetric.maxQuadricError      = 0.0f;

        for (uint32_t c = 0; c < lodLevelCount; c++)
        {
            root.traversalMetric.maxQuadricError = std::max(
                root.traversalMetric.maxQuadricError,
                geometry.lodNodes[1 + c].traversalMetric.maxQuadricError);
        }
    }
}

// ---------------------------------------------------------------------------
// ComputeBboxesRecursive — recursively compute lodNodeBboxes.
// ---------------------------------------------------------------------------

void ClusterLodBaker::ComputeBboxesRecursive(GeometryStorage& geometry, size_t i)
{
    const shaderio::Node& node = geometry.lodNodes[i];
    shaderio::BBox&       bbox = geometry.lodNodeBboxes[i];

    bbox = { {FLT_MAX, FLT_MAX, FLT_MAX}, {-FLT_MAX, -FLT_MAX, -FLT_MAX}, 0.0f, 0.0f };

    if (node.groupRange.isGroup)
    {
        const GroupInfo& groupInfo = geometry.groupInfos[node.groupRange.groupIndex];
        GroupView        groupView(geometry.groupData, groupInfo);

        for (uint32_t c = 0; c < groupInfo.clusterCount; c++)
        {
            bbox.lo = min(bbox.lo, groupView.clusterBboxes[c].lo);
            bbox.hi = max(bbox.hi, groupView.clusterBboxes[c].hi);
        }
    }
    else
    {
        for (uint32_t n = 0; n <= node.nodeRange.childCountMinusOne; n++)
            ComputeBboxesRecursive(geometry, node.nodeRange.childOffset + n);

        for (uint32_t n = 0; n <= node.nodeRange.childCountMinusOne; n++)
        {
            const shaderio::BBox& child = geometry.lodNodeBboxes[node.nodeRange.childOffset + n];
            bbox.lo = min(bbox.lo, child.lo);
            bbox.hi = max(bbox.hi, child.hi);
        }
    }
}
