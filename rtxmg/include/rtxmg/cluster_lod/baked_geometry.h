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

// baked_geometry.h — the baked cluster-LoD asset format.  Owns the group blob
// layout (GroupInfo, GroupView / GroupStorage, and the byte accounting the
// streaming allocator and the Inspector both size from), the per-geometry
// wrapper around it (GeometryBase / GeometryView / GeometryStorage, LodStats),
// and the CLAS geometry-index/flags encoding the host-side CLAS argument fill
// shares with its HLSL twins.  ClusterLodBaker writes it, .nvsngeo shards store
// it verbatim, and both residency modes read it back.

#pragma once

#include <cassert>
#include <cstdint>
#include <span>
#include <vector>

#include <donut/core/math/math.h>
#include <nvrhi/nvrhiHLSL.h>  // for nvrhi::rt::cluster::GeometryIndexAndFlags / ClusterGeometryFlags

#include "serialization.h"
#include "shaderio.h"

using namespace donut::math;

// ---------------------------------------------------------------------------
// align_up — portable power-of-two alignment helper.
// ---------------------------------------------------------------------------

inline size_t align_up(size_t v, size_t a)
{
    return (v + a - 1u) & ~(a - 1u);
}

// shaderio::Cluster::vertexCountMinusOne is 8 bits, so this bounds any
// per-cluster vertex-indexed scratch array.
inline constexpr uint32_t kMaxClusterVertices = 256;

// ---------------------------------------------------------------------------
// CLAS geometry-index/flags encoding for the CPU-side CLAS argument fill.  The
// geometryIndex doubles as the alpha-mask channel (0 opaque / 1 alpha-masked)
// so the hardware can skip any-hit for opaque hits.  Must stay in sync with the
// HLSL twins in cluster_lod_material_resolve.hlsli, which the GPU-side
// stream_update_scene.hlsl uses.
// ---------------------------------------------------------------------------

inline nvrhi::rt::cluster::GeometryIndexAndFlags
ClasEncodeBaseGeometryIndexAndFlagsFromState(uint32_t stateBits)
{
    const bool alphaMasked = (stateBits & shaderio::ClusterState::AlphaMasked) != 0u;
    const bool twoSided    = (stateBits & shaderio::ClusterState::TwoSided)    != 0u;

    uint32_t flags = alphaMasked
                         ? 0u
                         : uint32_t(nvrhi::rt::cluster::ClusterGeometryFlags::Opaque);
    if (twoSided)
        flags |= uint32_t(nvrhi::rt::cluster::ClusterGeometryFlags::CullDisable);

    nvrhi::rt::cluster::GeometryIndexAndFlags out{};
    out.geometryIndex = alphaMasked ? 1u : 0u;
    out.reserved      = 0u;
    out.geometryFlags = flags;
    return out;
}

inline nvrhi::rt::cluster::GeometryIndexAndFlags
ClasEncodePerTriangleGeometryIndexAndFlags(uint8_t triangleMaterialByte)
{
    const bool alphaMasked = (triangleMaterialByte & shaderio::kClusterTriangleAlphaMasked) != 0u;
    const bool twoSided    = (triangleMaterialByte & shaderio::kClusterTriangleTwoSided)    != 0u;

    uint32_t flags = alphaMasked
                         ? 0u
                         : uint32_t(nvrhi::rt::cluster::ClusterGeometryFlags::Opaque);
    if (twoSided)
        flags |= uint32_t(nvrhi::rt::cluster::ClusterGeometryFlags::CullDisable);

    nvrhi::rt::cluster::GeometryIndexAndFlags out{};
    out.geometryIndex = alphaMasked ? 1u : 0u;
    out.reserved      = 0u;
    out.geometryFlags = flags;
    return out;
}

// ---------------------------------------------------------------------------
// GroupInfo — metadata describing one cluster group stored in groupData.
//
// Binary layout of the blob at groupData[offsetBytes .. offsetBytes+sizeBytes):
//   Group(32B) → [16B align] → Clusters(16B*n) → [4B align] → GeneratingGroups(4B*n)
//   → [16B align] → BBoxes(32B*n) → TriangleData(TriangleDataBytes()) → [8B align]
//   → Vertices(4B * vertexDataCount) → [16B align]
// ---------------------------------------------------------------------------

struct GroupInfo
{
    uint64_t offsetBytes : 42;
    uint64_t sizeBytes   : 22;

    uint16_t vertexCount;
    uint16_t triangleCount;

    uint8_t  lodLevel;
    uint8_t  clusterCount;
    uint8_t  attributeBits;
    uint8_t  reserved1 = 0;

    // Total byte count of the triangle payload across the group's clusters:
    // index bytes (3 per triangle) plus, when any cluster is multi-material,
    // one material byte per triangle.
    uint32_t triangleDataCount           = 0;
    uint32_t _reserved2                  = 0;  // 8-byte alignment for trailing uint64 bitfield

    uint64_t vertexDataCount             : 21;
    // Non-zero only for compressed groups; zero means uncompressed.
    uint64_t uncompressedVertexDataCount : 21;
    uint64_t uncompressedSizeBytes       : 22;

    uint32_t GetDeviceSize() const
    {
        return uint32_t(uncompressedSizeBytes ? uncompressedSizeBytes : sizeBytes);
    }

    // Conservative upper bound on vertexDataCount (used to pre-size temp buffers).
    uint32_t EstimateVertexDataCount() const
    {
        uint32_t dataCount = vertexCount * 3u;  // positions (3 floats each)
        if (attributeBits & shaderio::ClusterAttribute::VertexNormal)
            dataCount += vertexCount * 1u;       // packed normal (1 uint32 stored as float)
        if (attributeBits & shaderio::ClusterAttribute::VertexTex0)
        {
            dataCount += vertexCount * 2u;
            dataCount += clusterCount;           // potential float2 alignment padding per cluster
        }
        if (attributeBits & shaderio::ClusterAttribute::VertexTex1)
        {
            dataCount += vertexCount * 2u;
            dataCount += clusterCount;
        }
        return dataCount;
    }

    // Conservative upper bound on triangleDataCount, given whether any
    // cluster in the group emits per-triangle material bytes.
    uint32_t EstimateTriangleDataCount(bool hasTriangleMaterials) const
    {
        uint32_t dataCount = uint32_t(triangleCount) * 3u;
        if (hasTriangleMaterials)
            dataCount += uint32_t(triangleCount);  // +1 byte per triangle
        return dataCount;
    }

    // Resolved triangle payload size in bytes.  A zero triangleDataCount means
    // the GroupInfo was filled by a path that doesn't set it, so fall back to
    // index-only sizing.
    size_t TriangleDataBytes() const
    {
        return triangleDataCount != 0u
                   ? size_t(triangleDataCount)
                   : size_t(triangleCount) * 3u;
    }

    size_t ComputeSize() const
    {
        size_t s = sizeof(shaderio::Group);                                         // 32
        s = align_up(s, 16) + sizeof(shaderio::Cluster) * clusterCount;
        s = align_up(s,  4) + sizeof(uint32_t)          * clusterCount;
        s = align_up(s, 16) + sizeof(shaderio::BBox)    * clusterCount;
        s = s               + TriangleDataBytes();
        s = align_up(s,  8) + sizeof(float)              * vertexDataCount;
        return align_up(s, 16);
    }

    size_t ComputeUncompressedSectionSize() const
    {
        size_t s = sizeof(shaderio::Group);
        s = align_up(s, 16) + sizeof(shaderio::Cluster) * clusterCount;
        s = align_up(s,  4) + sizeof(uint32_t)          * clusterCount;
        s = align_up(s, 16) + sizeof(shaderio::BBox)    * clusterCount;
        s = s               + TriangleDataBytes();
        return align_up(s,  8);
    }
};

// ---------------------------------------------------------------------------
// Group vertex-attribute byte accounting — the single source of truth shared by
// the streaming allocator (resident-pool sizing) and the Inspector's
// residency-vs-disk breakdown.  Both walk cluster HEADERS only, which live in
// the uncompressed section, so they work on a compressed group without
// decompressing its vertex payload.
// ---------------------------------------------------------------------------

// Per-attribute byte sizes of a group's vertex data in the uncompressed / baked
// layout, plus the attribute/encoding presence used for labeling.
struct GroupAttributeBytes
{
    uint64_t positions = 0;   // 12 * vertexCount, summed over clusters
    uint64_t normals   = 0;   // 4 * vertexCount when present, else 0
    uint64_t uv0       = 0;   // uv0 block bytes, as baked (quantized or raw)
    uint64_t uv1       = 0;   // uv1 block bytes, as baked
    bool     hasNormals = false;
    bool     hasTex0 = false, hasTex1 = false;
    bool     quantTex0 = false, quantTex1 = false;  // po2-grid quantized UV
    uint64_t Uvs() const { return uv0 + uv1; }
};

inline GroupAttributeBytes ComputeGroupAttributeBytes(const GroupInfo&         info,
                                               const shaderio::Cluster* clusters)
{
    GroupAttributeBytes a;
    for (uint32_t c = 0; c < info.clusterCount; ++c)
    {
        const shaderio::Cluster& cl = clusters[c];
        const uint64_t vc = uint64_t(cl.vertexCountMinusOne) + 1u;
        a.positions += vc * 12u;
        if (cl.attributeBits & shaderio::ClusterAttribute::VertexNormal)
        {
            a.hasNormals = true;
            a.normals += vc * 4u;
        }
        if (cl.attributeBits & shaderio::ClusterAttribute::VertexTex0)
        {
            a.hasTex0 = true;
            const bool q = (cl.encodingBits & shaderio::ClusterEncoding::QuantizedTex0) != 0;
            a.quantTex0 = a.quantTex0 || q;
            a.uv0 += q ? (16u + vc * 4u) : vc * 8u;
        }
        if (cl.attributeBits & shaderio::ClusterAttribute::VertexTex1)
        {
            a.hasTex1 = true;
            const bool q = (cl.encodingBits & shaderio::ClusterEncoding::QuantizedTex1) != 0;
            a.quantTex1 = a.quantTex1 || q;
            a.uv1 += q ? (16u + vc * 4u) : vc * 8u;
        }
    }
    return a;
}

// Device (resident-pool) blob size a group occupies under the given strip
// config.  Must match the streaming allocator's per-group allocation size
// EXACTLY (streaming.cpp ComputeStrippedGroupSizes).
inline uint64_t ResidentGroupDeviceBytes(const GroupInfo&         info,
                                         const shaderio::Cluster* clusters,
                                         bool stripPositions, bool stripNormals)
{
    // Nothing stripped => the blob is uploaded verbatim in its baked layout.
    // Stripping either channel makes the upload rewrite authoritative for the
    // vertex region, so the size follows its derivation below.
    if (!stripPositions && !stripNormals)
        return info.GetDeviceSize();

    size_t off = info.ComputeUncompressedSectionSize();  // vertex region start
    for (uint32_t c = 0; c < info.clusterCount; ++c)
    {
        const shaderio::Cluster& cl = clusters[c];
        const size_t vc = size_t(cl.vertexCountMinusOne) + 1u;
        off = align_up(off, 8);  // slice start (deterministic pads)
        if (!stripPositions)
            off += vc * 12u;
        if (!stripNormals && (cl.attributeBits & shaderio::ClusterAttribute::VertexNormal))
            off += vc * 4u;
        if (cl.attributeBits & shaderio::ClusterAttribute::VertexTex0)
        {
            off = align_up(off, 8);
            off += (cl.encodingBits & shaderio::ClusterEncoding::QuantizedTex0) ? (16u + vc * 4u) : vc * 8u;
        }
        if (cl.attributeBits & shaderio::ClusterAttribute::VertexTex1)
        {
            off = align_up(off, 8);
            off += (cl.encodingBits & shaderio::ClusterEncoding::QuantizedTex1) ? (16u + vc * 4u) : vc * 8u;
        }
    }
    return off;
}

// ---------------------------------------------------------------------------
// GroupView — read-only accessor into a group blob within a groupData buffer.
// ---------------------------------------------------------------------------

struct GroupView
{
    const uint8_t*                     raw       = nullptr;
    size_t                             rawSize   = 0;
    const shaderio::Group*             group     = nullptr;
    std::span<const shaderio::Cluster> clusters;
    std::span<const uint32_t>          clusterGeneratingGroups;
    std::span<const shaderio::BBox>    clusterBboxes;
    std::span<const uint8_t>           indices;
    std::span<const float>             vertices;

    GroupView() = default;

    // groupDatas is the full flat groupData array; info.offsetBytes locates this group.
    GroupView(std::span<const uint8_t> groupDatas, const GroupInfo& info)
        : rawSize(info.sizeBytes)
    {
        assert(info.offsetBytes + info.sizeBytes <= groupDatas.size());
        raw = &groupDatas[info.offsetBytes];

        auto addr = [](const void* p) { return reinterpret_cast<size_t>(p); };

        group    = reinterpret_cast<const shaderio::Group*>(raw);
        clusters = { reinterpret_cast<const shaderio::Cluster*>(align_up(addr(raw) + sizeof(shaderio::Group), 16)),
                     info.clusterCount };
        clusterGeneratingGroups = { reinterpret_cast<const uint32_t*>(align_up(addr(clusters.data() + info.clusterCount), 4)),
                                    info.clusterCount };
        clusterBboxes = { reinterpret_cast<const shaderio::BBox*>(align_up(addr(clusterGeneratingGroups.data() + info.clusterCount), 16)),
                          info.clusterCount };
        indices  = { reinterpret_cast<const uint8_t*>(clusterBboxes.data() + info.clusterCount),
                     info.TriangleDataBytes() };
        vertices = { reinterpret_cast<const float*>(align_up(addr(indices.data() + indices.size()), 8)),
                     info.vertexDataCount };

        assert(addr(vertices.data() + info.vertexDataCount) - addr(raw) <= size_t(info.sizeBytes));
    }

    const uint8_t* GetClusterIndices(size_t clusterIndex) const
    {
        return reinterpret_cast<const uint8_t*>(
            reinterpret_cast<size_t>(&clusters[clusterIndex]) + clusters[clusterIndex].triangles);
    }
    const float3* GetClusterVertices(size_t clusterIndex) const
    {
        return reinterpret_cast<const float3*>(
            reinterpret_cast<size_t>(&clusters[clusterIndex]) + clusters[clusterIndex].vertices);
    }
};

// ---------------------------------------------------------------------------
// GroupStorage — read-write accessor used when building/filling a group blob.
// ---------------------------------------------------------------------------

struct GroupStorage
{
    uint8_t*                     raw     = nullptr;
    size_t                       rawSize = 0;
    shaderio::Group*             group   = nullptr;
    std::span<shaderio::Cluster> clusters;
    std::span<uint32_t>          clusterGeneratingGroups;
    std::span<shaderio::BBox>    clusterBboxes;
    std::span<uint8_t>           indices;
    std::span<float>             vertices;

    GroupStorage() = default;

    // groupData points to the start of this group's blob; info.offsetBytes is NOT applied.
    explicit GroupStorage(void* groupData, const GroupInfo& info)
        : rawSize(info.sizeBytes)
    {
        auto addr = [](const void* p) { return reinterpret_cast<size_t>(p); };

        raw      = static_cast<uint8_t*>(groupData);
        group    = static_cast<shaderio::Group*>(groupData);
        clusters = { reinterpret_cast<shaderio::Cluster*>(align_up(addr(raw) + sizeof(shaderio::Group), 16)),
                     info.clusterCount };
        clusterGeneratingGroups = { reinterpret_cast<uint32_t*>(align_up(addr(clusters.data() + info.clusterCount), 4)),
                                    info.clusterCount };
        clusterBboxes = { reinterpret_cast<shaderio::BBox*>(align_up(addr(clusterGeneratingGroups.data() + info.clusterCount), 16)),
                          info.clusterCount };
        indices  = { reinterpret_cast<uint8_t*>(clusterBboxes.data() + info.clusterCount),
                     info.TriangleDataBytes() };
        vertices = { reinterpret_cast<float*>(align_up(addr(indices.data() + indices.size()), 8)),
                     info.vertexDataCount };

        assert(addr(vertices.data() + info.vertexDataCount) - addr(raw) <= rawSize);
    }

    // Returns byte offset of `input` relative to the header of cluster[clusterIndex].
    // `overrideSize` (when non-zero) replaces rawSize for the bounds assert: a
    // compressed blob's cluster.vertices offsets address the larger UNCOMPRESSED
    // layout, so they legitimately point past the compressed blob's end.
    uint32_t GetClusterLocalOffset(uint32_t clusterIndex, const void* input, size_t overrideSize = 0) const
    {
        assert(reinterpret_cast<size_t>(input) >= reinterpret_cast<size_t>(&clusters[clusterIndex]));
        assert(reinterpret_cast<size_t>(input) < reinterpret_cast<size_t>(raw) + (overrideSize ? overrideSize : rawSize));
        return uint32_t(reinterpret_cast<size_t>(input) - reinterpret_cast<size_t>(&clusters[clusterIndex]));
    }

    // Returns a writable pointer at byte offset `localOffset` from the header of
    // cluster[clusterIndex] (inverse of GetClusterLocalOffset). Used as the
    // per-cluster vertex-data destination during decompression.
    uint32_t* GetClusterLocalData(uint32_t clusterIndex, uint32_t localOffset)
    {
        return reinterpret_cast<uint32_t*>(reinterpret_cast<size_t>(&clusters[clusterIndex]) + localOffset);
    }
};

// ---------------------------------------------------------------------------
// LodStats — per-LOD-level totals for the Inspector, summed over that level's
// groups at bake time.  Baked rather than computed at runtime because the sum
// needs each group's cluster headers, and groupData is a zero-copy span into
// the mmap'd shard: walking it scene-wide faults in the entire geometry pool
// (one page per group) for a few bytes each.
//
// Everything here describes the BAKED layout.  The resident size under channel
// stripping is approximated by subtracting posBytes / nrmBytes, which ignores
// the per-cluster 8-byte realignment and so drifts slightly from
// ResidentGroupDeviceBytes().
// ---------------------------------------------------------------------------

struct LodStats
{
    uint64_t totTris        = 0;
    uint64_t totBytes       = 0;  // sum of GroupInfo::sizeBytes (on-disk stored)
    uint64_t totDeviceBytes = 0;  // sum of GroupInfo::GetDeviceSize(), unstripped
    uint64_t posBytes       = 0;
    uint64_t nrmBytes       = 0;
    uint64_t uvBytes        = 0;
    uint32_t totGroups      = 0;
    uint32_t totClusters    = 0;
    uint8_t  quantUv        = 0;  // any cluster uses po2-grid quantized UVs
    uint8_t  compressed     = 0;  // any group is arithmetic-compressed on disk
    uint16_t _pad0          = 0;
    uint32_t _pad1          = 0;
};

// ---------------------------------------------------------------------------
// GeometryBase — metadata shared between GeometryStorage and GeometryView.
// Serialised verbatim as the first block in the .nvsngeo cache file entry.
// ---------------------------------------------------------------------------

struct GeometryBase
{
    uint32_t attributeBits = 0;

    uint32_t clusterMaxVerticesCount  = 0;
    uint32_t clusterMaxTrianglesCount = 0;

    uint32_t lodLevelsCount = 0;

    // Highest-detail LOD (level 0) totals.
    uint32_t hiTriangleCount = 0;
    uint32_t hiVerticesCount = 0;
    uint32_t hiClustersCount = 0;

    // Sum across all LOD levels.
    uint32_t totalTriangleCount = 0;
    uint32_t totalVerticesCount = 0;
    uint32_t totalClustersCount = 0;

    // Object-space bounding box (set during baking).
    shaderio::BBox bbox = { {FLT_MAX, FLT_MAX, FLT_MAX}, {-FLT_MAX, -FLT_MAX, -FLT_MAX}, 0.0f, 0.0f };

    // Aggregate Cluster::stateBits of the lowest-detail LOD level, copied into
    // RenderInstance::lowDetailClusterStateBits so traversal_init knows whether
    // the fallback cluster needs alpha-test / two-sided handling.  Populated by
    // Build() once the LOD hierarchy is finalised.
    uint8_t  lowDetailClusterStateBits = 0;
};

// ---------------------------------------------------------------------------
// GeometryView — non-owning spans into GeometryStorage vectors or a
// memory-mapped cache file.  Set by MakeGeometryView() or CacheFileView.
//
// The lifetime of the underlying data must exceed this view.
// ---------------------------------------------------------------------------

struct GeometryView : GeometryBase
{
    std::span<const uint8_t>             groupData;
    std::span<const GroupInfo>           groupInfos;
    std::span<const shaderio::LodLevel>  lodLevels;
    std::span<const LodStats>            lodStats;  // parallel to lodLevels
    std::span<const shaderio::Node>      lodNodes;
    std::span<const shaderio::BBox>      lodNodeBboxes;
    std::span<const uint32_t>            localMaterialIDs;
    std::span<const uint8_t>             localMaterialStateBits;

    // Returns the number of bytes needed to serialise this geometry into a
    // .nvsngeo cache file (GeometryBase block + all span payloads, each
    // preceded by a 16-byte count, rounded up to 16-byte alignment).
    uint64_t GetCachedSize() const
    {
        uint64_t size = 0;
        size += (sizeof(GeometryBase) + serialization::ALIGN_MASK) & ~serialization::ALIGN_MASK;
        size += serialization::GetCachedSize(groupData);
        size += serialization::GetCachedSize(groupInfos);
        size += serialization::GetCachedSize(lodLevels);
        size += serialization::GetCachedSize(lodStats);
        size += serialization::GetCachedSize(lodNodes);
        size += serialization::GetCachedSize(lodNodeBboxes);
        size += serialization::GetCachedSize(localMaterialIDs);
        size += serialization::GetCachedSize(localMaterialStateBits);
        return size;
    }
};

// ---------------------------------------------------------------------------
// GeometryStorage — CPU-side owned geometry data.  ClusterLodGltfImporter fills
// the raw input mesh; ClusterLodBaker::Build() consumes it and fills the baked
// output plus the GeometryBase stats.  MakeGeometryView() then spans both.
// ---------------------------------------------------------------------------

struct GeometryStorage : GeometryBase
{
    // Offsets (in floats) of each attribute within the flat vertexAttributes
    // array.  Stride = sum of component counts.
    uint32_t attributeNormalOffset  = 0;
    uint32_t attributeTex0offset    = 0;
    uint32_t attributeTex1offset    = 0;
    uint32_t attributeTangentOffset = 0;
    // Float-stride offset of the per-vertex local-material-ID attribute used
    // by the simplifier to avoid merging vertices across material boundaries.
    // Only populated for multi-material geometries (localMaterialIDs.size()
    // > 1); a sentinel of UINT32_MAX means "not present".
    uint32_t attributeMaterialOffset = ~0u;
    // Number of floats per vertex that participate in simplification weighting.
    uint32_t attributesWithWeights  = 0;

    // ----- Raw input mesh (transient — cleared after baking) -----
    std::vector<float3>  vertexPositions;
    std::vector<float>   vertexAttributes;   // flat: [stride*v0 | stride*v1 | ...]
    std::vector<uint3>   triangles;

    // ----- Baked output -----
    std::vector<uint8_t>             groupData;
    std::vector<GroupInfo>           groupInfos;
    std::vector<shaderio::LodLevel>  lodLevels;
    std::vector<LodStats>            lodStats;   // parallel to lodLevels
    std::vector<shaderio::Node>      lodNodes;
    std::vector<shaderio::BBox>      lodNodeBboxes;
    // Per-geometry local→global material ID table. Single-material geometry:
    // size 1, contents = {global material ID}. Multi-material geometry: size N,
    // contents = the N unique glTF-file-global material IDs referenced by this
    // geometry's primitives, in first-seen order. The local index is the
    // position in this vector and is what gets written into Cluster::
    // localMaterialID / per-triangle material bytes.
    std::vector<uint32_t>            localMaterialIDs;
    // Parallel to localMaterialIDs — ClusterState AlphaMasked / TwoSided bits
    // for the n'th local material.  Filled by the importer so the baker can
    // compute per-cluster stateBits without depending on Scene.
    std::vector<uint8_t>             localMaterialStateBits;
};

// ---------------------------------------------------------------------------
// MakeGeometryView implementation (after GeometryStorage is fully defined).
// ---------------------------------------------------------------------------

inline GeometryView MakeGeometryView(const GeometryStorage& storage)
{
    GeometryView view;
    static_cast<GeometryBase&>(view) = storage;

    view.groupData        = storage.groupData;
    view.groupInfos       = storage.groupInfos;
    view.lodLevels        = storage.lodLevels;
    view.lodStats         = storage.lodStats;
    view.lodNodes         = storage.lodNodes;
    view.lodNodeBboxes    = storage.lodNodeBboxes;
    view.localMaterialIDs       = storage.localMaterialIDs;
    view.localMaterialStateBits = storage.localMaterialStateBits;
    return view;
}
