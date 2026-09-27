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

#pragma once

#include <cstdint>

#include "baked_geometry.h"
#include "rtxmg/utils/hash.h"

// ---------------------------------------------------------------------------
// BakerConfig — parameters controlling cluster LOD hierarchy generation.
//
// SemanticHash() is stored in every cache header, so changing any hashed field
// re-bakes the cache in place, overwriting the shards baked at the old settings.
// That is the accepted policy: LoadManifest warns and names what changed.
// ---------------------------------------------------------------------------

struct BakerConfig
{
    // Cluster geometry limits (meshoptimizer cluster constraints).  Smaller
    // clusters multiply the group count and blow the streaming working set past
    // the CLAS pool, wedging streaming.
    // Override with --clustersize <tris> <verts>.
    uint32_t clusterVertices   = 128;
    uint32_t clusterTriangles  = 128;
    // Maximum number of clusters per group (partition_size).  Capped by the
    // 32-bit cluster mask in traversal_blas_merging (static_assert below).
    uint32_t clusterGroupSize  = 32;

    // LOD hierarchy node branching factor.
    uint32_t preferredNodeWidth = 8;

    // Error propagation across LOD levels: a group's error becomes
    // max(childError * Previous, ownError) + Additive * ownError.  Previous
    // below 1 breaks the DAG's error monotonicity (a parent can then advertise
    // less error than its children).  Override with --loderrormerge.
    float lodErrorMergePrevious = 1.0f;
    float lodErrorMergeAdditive = 0.0f;
    float lodErrorEdgeLimit     = 0.0f;

    // meshoptimizer cluster LOD tuning.
    bool  meshoptPreferRayTracing = true;
    // Named, zero-initialized padding so the bytes the compiler inserts after
    // this bool are deterministic on disk instead of stack garbage.  Not
    // compared in operator==.
    uint8_t _pad0[3]              = {};
    float meshoptFillWeight       = 0.5f;
    float meshoptSplitFactor      = 1.5f;

    // Arithmetic-packed vertex data (--compress), decompressed per group at
    // upload time.
    bool     useCompressedData      = false;
    uint8_t  _pad1[3]               = {};  // see _pad0
    uint32_t compressionPosDropBits = 7;
    uint32_t compressionTexDropBits = 7;

    // Simplification attribute weights (0 = not used), overridable with
    // --simplifyweights <normal> <texcoord>.  Attribute error folds into each
    // group's baked maxQuadricError — the value traversal tests against
    // lodPixelError — so these weights directly scale the streaming working
    // set.  Both must be non-zero: at 0, flat/fan geometry reports near-zero
    // quadric error for coplanar collapses and a UV-shredded coarse LOD passes
    // the lodPixelError test from a few meters away, and shading reads the
    // baked normals so the simplifier has to preserve those too.
    float simplifyNormalWeight    = 0.5f;
    float simplifyTexCoordWeight  = 0.5f;
    float simplifyTangentWeight   = 0.0f;
    float simplifyTangentSignWeight = 0.0f;
    // Deliberately large: keeps the simplifier from collapsing edges that span
    // different materials.  Used only when a geometry is multi-material
    // (GeometryStorage::attributeMaterialOffset != ~0u).
    float simplifyMaterialWeight  = 32.0f;

    // Store per-cluster TEXCOORD arrays po2-grid-quantized instead of raw
    // float2s, halving their bytes.  Per-cluster adaptive: falls back to raw
    // float2 when the cluster's UV range would need a step coarser than 2^-14
    // (>~0.13 texel of error at 4K) or is too small to amortize the header.
    // Under --compress the delta words are arithmetic-packed on top, which
    // beats packing raw float2s because they carry no exponent field.
    bool    quantizeTexCoords = true;
    uint8_t _pad2[3]          = {};  // see _pad0

    // Field-wise equality.  Do NOT memcmp this struct: default-init leaves the
    // padding after each bool uninitialized, and Debug /RTC1 fills it with 0xCC
    // where optimized builds leave 0x00, so a memcmp cache check would reject
    // any cache baked by a different build config.
    bool operator==(const BakerConfig& o) const
    {
        return clusterVertices          == o.clusterVertices
            && clusterTriangles         == o.clusterTriangles
            && clusterGroupSize         == o.clusterGroupSize
            && preferredNodeWidth       == o.preferredNodeWidth
            && lodErrorMergePrevious    == o.lodErrorMergePrevious
            && lodErrorMergeAdditive    == o.lodErrorMergeAdditive
            && lodErrorEdgeLimit        == o.lodErrorEdgeLimit
            && meshoptPreferRayTracing  == o.meshoptPreferRayTracing
            && meshoptFillWeight        == o.meshoptFillWeight
            && meshoptSplitFactor       == o.meshoptSplitFactor
            && useCompressedData        == o.useCompressedData
            && compressionPosDropBits   == o.compressionPosDropBits
            && compressionTexDropBits   == o.compressionTexDropBits
            && simplifyNormalWeight     == o.simplifyNormalWeight
            && simplifyTexCoordWeight   == o.simplifyTexCoordWeight
            && simplifyTangentWeight    == o.simplifyTangentWeight
            && simplifyTangentSignWeight== o.simplifyTangentSignWeight
            && simplifyMaterialWeight   == o.simplifyMaterialWeight
            && quantizeTexCoords        == o.quantizeTexCoords;
    }
    bool operator!=(const BakerConfig& o) const { return !(*this == o); }

    // Revision of the compressed group-blob layout.  2 = UV quantization folded
    // into the compress path.
    static constexpr uint32_t kCompressedLayoutVersion = 2;

    // Digest of the fields that change baked output.  The cache headers store
    // this instead of the struct, so a newly added field invalidates nothing
    // until it is listed here — the struct itself cannot say that, because both
    // its size and its operator== change the moment a field appears.
    uint64_t SemanticHash() const
    {
        uint64_t h = rtxmg::kFnv1aOffsetBasis;
        h = rtxmg::Fnv1aValue(clusterVertices,          h);
        h = rtxmg::Fnv1aValue(clusterTriangles,         h);
        h = rtxmg::Fnv1aValue(clusterGroupSize,         h);
        h = rtxmg::Fnv1aValue(preferredNodeWidth,       h);
        h = rtxmg::Fnv1aValue(lodErrorMergePrevious,    h);
        h = rtxmg::Fnv1aValue(lodErrorMergeAdditive,    h);
        h = rtxmg::Fnv1aValue(lodErrorEdgeLimit,        h);
        h = rtxmg::Fnv1aValue(uint8_t(meshoptPreferRayTracing), h);
        h = rtxmg::Fnv1aValue(meshoptFillWeight,        h);
        h = rtxmg::Fnv1aValue(meshoptSplitFactor,       h);
        h = rtxmg::Fnv1aValue(uint8_t(useCompressedData), h);
        // Compressed blob layout revision, folded in only under compression.
        // A change to that layout leaves an uncompressed bake byte-identical,
        // so invalidating uncompressed caches for it would cost a re-bake of
        // every warm cache to no effect.  Bump on any compressed-layout change.
        if (useCompressedData)
            h = rtxmg::Fnv1aValue(kCompressedLayoutVersion, h);
        h = rtxmg::Fnv1aValue(compressionPosDropBits,   h);
        h = rtxmg::Fnv1aValue(compressionTexDropBits,   h);
        h = rtxmg::Fnv1aValue(simplifyNormalWeight,     h);
        h = rtxmg::Fnv1aValue(simplifyTexCoordWeight,   h);
        h = rtxmg::Fnv1aValue(simplifyTangentWeight,    h);
        h = rtxmg::Fnv1aValue(simplifyTangentSignWeight, h);
        h = rtxmg::Fnv1aValue(simplifyMaterialWeight,   h);
        h = rtxmg::Fnv1aValue(uint8_t(quantizeTexCoords), h);
        return h;
    }
};

static_assert(BakerConfig{}.clusterGroupSize <= shaderio::kTraversalBlasMergingMaxGroupClusters,
              "clusterGroupSize exceeds the 32-bit cluster mask in "
              "traversal_blas_merging.hlsl — clusters past bit 31 would be "
              "dropped from the merged BLAS and clusters 0..31 re-marked");

// ---------------------------------------------------------------------------
// ClusterLodBaker — generates a cluster LOD hierarchy from raw mesh data.
//
// Usage:
//   ClusterLodBaker baker;
//   baker.Build(geometryStorage);  // populates groupData, lodNodes, etc.
//
// After Build(), the raw input vectors (vertexPositions, vertexAttributes,
// triangles) are cleared to free memory.
// ---------------------------------------------------------------------------

class ClusterLodBaker
{
public:
    explicit ClusterLodBaker(const BakerConfig& config = {});

    // Build the cluster LOD hierarchy for one geometry.
    // Populates the baked-output fields of geometry and clears the raw input.
    void Build(GeometryStorage& geometry);

private:
    BakerConfig m_config;

    // Internal context for a single Build() call.
    struct BakeContext;

    // Store one meshoptimizer-produced group into geometry.groupData.
    // Returns the groupIndex that was stored.
    uint32_t StoreGroup(BakeContext* ctx,
                        uint32_t     groupIndex,
                        const void*  clusterLodGroupPtr,   // typed as clodGroup* in .cpp
                        uint32_t     clusterCount,
                        const void*  clustersPtr);   // typed as clodCluster* in .cpp

    // Build the spatial LOD node hierarchy over the stored groups.
    void BuildHierarchy(GeometryStorage& geometry);

    // Recursively compute lodNodeBboxes from cluster bboxes.
    void ComputeBboxesRecursive(GeometryStorage& geometry, size_t nodeIdx);

    // Static C-style callbacks forwarded to instance methods.
    static int  GroupCallback(void* ctx, struct clodGroup group,
                                const struct clodCluster* clusters,
                                size_t cluster_count, size_t task_index,
                                unsigned int thread_index);
};
