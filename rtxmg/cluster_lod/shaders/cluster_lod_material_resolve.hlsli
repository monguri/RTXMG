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

// cluster_lod_material_resolve.hlsli
//
// Material / alpha-mask resolve helpers for the cluster-LOD path.
//
// All readers go through `shaderio::GroupAddress { srvIndex, byteOffset }` —
// never raw VAs — so they share the bindless ByteAddressBuffer path with the
// rest of the cluster-LOD shaders (see cluster_lod_payload.hlsli).

#ifndef RTXMG_CLUSTER_LOD_MATERIAL_RESOLVE_HLSLI
#define RTXMG_CLUSTER_LOD_MATERIAL_RESOLVE_HLSLI

#include "rtxmg/cluster_lod/shaderio.h"
#include <nvrhi/nvrhiHLSL.h>  // for nvrhi::rt::cluster::GeometryIndexAndFlags / ClusterGeometryFlags

// Load one byte from a ByteAddressBuffer at an arbitrary byte offset.
//
// ByteAddressBuffer.Load requires a 4-byte-aligned address and silently rounds
// an unaligned one DOWN, which scrambles per-byte arrays; this helper does the
// aligned-load-then-shift dance so callers can ignore alignment.  Used for the
// per-triangle material bytes after each cluster's triangle payload — the only
// sub-uint32 array in the group blob.
uint ClusterLoadGroupDataByte(ByteAddressBuffer groupData, uint byteOffset)
{
    const uint aligned = byteOffset & ~3u;
    const uint shift   = (byteOffset & 3u) * 8u;
    return (groupData.Load(aligned) >> shift) & 0xFFu;
}

// Byte offset (relative to the group blob start) at which the per-triangle
// material byte array begins for the given cluster header: immediately after
// the index bytes (triangleCount * 3) in the cluster's triangle payload.
//
// Valid only when the cluster is multi-material
// (cluster.localMaterialID == shaderio::kPerTriangleMaterials); callers must
// check that before reading the returned offset.
uint ClusterGetTriangleMaterialsByteOffset(shaderio::Cluster cluster,
                                            uint              clusterByteOffset)
{
    const uint triangleCount = cluster.triangleCountMinusOne + 1u;
    return clusterByteOffset + cluster.triangles + triangleCount * 3u;
}

// ---------------------------------------------------------------------------
// High-level material resolve helpers used by the cluster-LoD any-hit /
// closest-hit shaders.
//
// The full t_MaterialConstants table holds subd materials first, cluster-LoD
// materials after. shaderio::Geometry::materialBaseID is the slot where the
// cluster-LoD section starts. The local→scene-global mapping for each
// geometry lives in a flat StructuredBuffer (bound as
// t_ClusterLodLocalMaterialIDs at register t15) sliced by
// Geometry::localMaterialsOffset / localMaterialsCount.
// ---------------------------------------------------------------------------

// Returns the cluster's effective local material ID for the given triangle.
// For uniform-material clusters this is just cluster.localMaterialID. For
// mixed clusters (cluster.localMaterialID == shaderio::kPerTriangleMaterials)
// it reads the per-triangle material byte (low 6 bits = local index).
//
// triangleByteBase is the byte offset (in groupData) of the per-triangle
// material array — ClusterGetTriangleMaterialsByteOffset(cluster,
// clusterByteOffset). Callers may pass 0 when the cluster is known to be
// uniform.
uint ResolveClusterLocalMaterialID(shaderio::Cluster   cluster,
                                   ByteAddressBuffer   groupData,
                                   uint                triangleByteBase,
                                   uint                triangleID)
{
    if (cluster.localMaterialID == shaderio::kPerTriangleMaterials)
    {
        const uint b = ClusterLoadGroupDataByte(groupData, triangleByteBase + triangleID);
        return b & shaderio::kLocalMaterialMask;
    }
    return cluster.localMaterialID;
}

// Maps a per-cluster local material ID to a slot in t_MaterialConstants for
// the given cluster-LoD geometry. The local→global indirection lives in
// t_ClusterLodLocalMaterialIDs; geom.materialBaseID is added on top to translate
// the cluster-LoD-global index into the unified material table slot.
//
// localMaterialIDsBuffer should be the scene-wide flat buffer bound at
// register t15 (StructuredBuffer<uint>).
uint ResolveMaterialIDFromLocal(shaderio::Geometry      geom,
                                StructuredBuffer<uint>  localMaterialIDsBuffer,
                                uint                    localMaterialID)
{
    if (geom.localMaterialsCount == 0)
        return geom.materialBaseID;  // no indirection — fall back to base
    const uint clamped = min(localMaterialID, geom.localMaterialsCount - 1u);
    const uint global  = localMaterialIDsBuffer[geom.localMaterialsOffset + clamped];
    return geom.materialBaseID + global;
}

// ---------------------------------------------------------------------------
// CLAS base-geometry-index/flag encoding helpers.
// Convert Cluster::stateBits (uniform cluster) or a per-triangle material
// byte (mixed cluster) into a nvrhi::rt::cluster::GeometryIndexAndFlags
// value — written to IndirectTriangleClasArgs::baseGeometryIndexAndFlags
// or, in packed-uint32 form, each entry of geometryIndexAndFlagsBuffer.
//
// Layout (nvrhi::rt::cluster::GeometryIndexAndFlags):
//   geometryIndex : 24   (0 = opaque slot, 1 = alpha-mask slot)
//   reserved      : 5
//   geometryFlags : 3    (bit 0 = CullDisable, bit 1 = NoDupAH, bit 2 = Opaque)
//
// Convention:
//   * Opaque (no ALPHAMASKED bit): geometryIndex = 0, Opaque flag set so the
//     hardware skips any-hit invocation — perf optimisation.
//   * Alpha-masked: geometryIndex = 1, no Opaque flag so any-hit fires.
//   * Two-sided: CullDisable flag set regardless of opacity.
// ---------------------------------------------------------------------------
nvrhi::rt::cluster::GeometryIndexAndFlags ClasEncodeBaseGeometryIndexAndFlagsFromState(uint stateBits)
{
    const bool alphaMasked = (stateBits & shaderio::ClusterState::AlphaMasked) != 0u;
    const bool twoSided    = (stateBits & shaderio::ClusterState::TwoSided)    != 0u;

    uint flags = alphaMasked ? 0u : (uint)nvrhi::rt::cluster::ClusterGeometryFlags::Opaque;
    if (twoSided)
        flags |= (uint)nvrhi::rt::cluster::ClusterGeometryFlags::CullDisable;

    nvrhi::rt::cluster::GeometryIndexAndFlags result;
    result.geometryIndex = alphaMasked ? 1u : 0u;
    result.reserved      = 0u;
    result.geometryFlags = flags;
    return result;
}

nvrhi::rt::cluster::GeometryIndexAndFlags ClasEncodePerTriangleGeometryIndexAndFlags(uint triangleMaterialByte)
{
    const bool alphaMasked = (triangleMaterialByte & shaderio::kClusterTriangleAlphaMasked) != 0u;
    const bool twoSided    = (triangleMaterialByte & shaderio::kClusterTriangleTwoSided)    != 0u;

    uint flags = alphaMasked ? 0u : (uint)nvrhi::rt::cluster::ClusterGeometryFlags::Opaque;
    if (twoSided)
        flags |= (uint)nvrhi::rt::cluster::ClusterGeometryFlags::CullDisable;

    nvrhi::rt::cluster::GeometryIndexAndFlags result;
    result.geometryIndex = alphaMasked ? 1u : 0u;
    result.reserved      = 0u;
    result.geometryFlags = flags;
    return result;
}

// Bit-reinterpret between uint32 and GeometryIndexAndFlags.
// u_NewClasGeometryIndices is a RWStructuredBuffer<uint> (DXC's bit-field-struct
// stores misbehaved), so its writers pack through _toUint.
nvrhi::rt::cluster::GeometryIndexAndFlags GeometryIndexAndFlagsFromUint(uint v)
{
    nvrhi::rt::cluster::GeometryIndexAndFlags r;
    r.geometryIndex = v & 0x00FFFFFFu;
    r.reserved      = (v >> 24) & 0x1Fu;
    r.geometryFlags = (v >> 29) & 0x7u;
    return r;
}

uint GeometryIndexAndFlagsToUint(nvrhi::rt::cluster::GeometryIndexAndFlags g)
{
    return (g.geometryIndex & 0x00FFFFFFu)
         | ((g.reserved & 0x1Fu) << 24)
         | ((g.geometryFlags & 0x7u) << 29);
}

#endif  // RTXMG_CLUSTER_LOD_MATERIAL_RESOLVE_HLSLI
