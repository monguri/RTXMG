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

// ClusterLodPreloaded — the non-streaming residency backend: every LoD level of
// every cluster-LoD geometry is uploaded at scene-load time and stays resident.
//
// This is a supported reference implementation.  It is not dead code, not a
// legacy fallback, and not scheduled for removal.  It exists so that the
// cluster-LoD CLAS build and the traversal it feeds can be read without the
// demand-paging machinery layered on top, and so a run can be compared against
// streaming with everything else held equal.  Select it with --preload; it
// implements the same ClusterLodResources interface ClusterLodStreaming does,
// minus the streaming hooks.

#pragma once

#include <vector>

#include <nvrhi/nvrhi.h>
#include <donut/engine/DescriptorTableManager.h>

#include "rtxmg/cluster_lod/baked_geometry.h"
#include "rtxmg/cluster_lod/gltf_model.h"
#include "rtxmg/cluster_lod/resources.h"
#include "rtxmg/cluster_lod/resources_base.h"
#include "rtxmg/cluster_lod/shaderio.h"
#include "rtxmg/utils/buffer.h"

// ---------------------------------------------------------------------------
// PreloadGeometry — GPU buffers for one geometry entry: BaseGeometry's LoD-tree
// and metadata buffers plus the preload-only bulk `groupData` / `clasData`.
// ---------------------------------------------------------------------------

struct PreloadGeometry : BaseGeometry
{
    // Every group blob of the geometry, back to back.  Two consumers, one
    // buffer: the CLAS-build hardware dereferences it by raw VA per cluster,
    // and the hit shader reads it bindlessly via groupDataSRVHandle.
    RTXMGBuffer<uint8_t>                      groupData;

    // Ray-tracing CLAS data (populated by InitClas).
    RTXMGBuffer<uint8_t>                      clasData;

    donut::engine::DescriptorHandle           groupDataSRVHandle;  // ByteAddressBuffer view of groupData

    // One GeometryIndexAndFlags per triangle of each mixed
    // (ClusterState *Mixed) cluster, concatenated; each such cluster's
    // IndirectTriangleClasArgs::geometryIndexAndFlagsBuffer points at its slice.
    // Empty when the geometry has no mixed clusters.
    RTXMGBuffer<nvrhi::rt::cluster::GeometryIndexAndFlags> clasGeometryIndices;
};

// ---------------------------------------------------------------------------
// ClusterLodPreloaded — orchestrates preload of all geometries.
// ---------------------------------------------------------------------------

class ClusterLodPreloaded : public ClusterLodResourcesBase
{
public:
    // Upload geometry buffers and build CLASes.  Must be called once before
    // any render frames that reference cluster LOD geometry.
    // bakerConfig + clasPositionTruncateBits resolve the CLAS position
    // truncation, so a compressed bake gets the same smaller CLAS as streaming.
    // clusterLodMaterialBaseID = slot in RTXMGScene::m_materialBuffer where
    // cluster-LoD materials start.  hasAlphaMask drives the CLAS-build
    // maxUniqueGeometryCount / maxGeometryIndex decision (2/1 vs 1/0) and gates
    // the per-geometry mixed-cluster geometryIndexAndFlagsBuffer allocation.
    void Init(const std::vector<GeometryView>&       geometries,
              const std::vector<ClusterLodInstance>& instances,
              const BakerConfig&                     bakerConfig,
              uint32_t                               clasPositionTruncateBits,
              uint32_t                               clusterLodMaterialBaseID,
              bool                                   hasAlphaMask,
              donut::engine::DescriptorTableManager* descriptorTable,
              nvrhi::IDevice*                        device,
              nvrhi::ICommandList*                   commandList);

    // Log key fields of the CPU-side shader geometry table (debug only).
    void LogGeometryData() const;

protected:
    // Build CLASes for every cluster and the low-detail BLAS for every
    // geometry.  Writes scene-global m_residentClasAddresses (and
    // populates each PreloadGeometry::clasData backing store) and updates
    // lowDetailBlasAddress in m_shaderGeometries.
    void InitClas(const std::vector<GeometryView>&       geometries,
                  donut::engine::DescriptorTableManager* descriptorTable,
                  nvrhi::IDevice*                        device,
                  nvrhi::ICommandList*                   commandList);

    // Preload-specific per-geometry resources.
    std::vector<PreloadGeometry>             m_geometries;

    // Prefix sums assigning each geometry its scene-global ID range: geom g's
    // clusters occupy [m_geomFirstClusterResidentID[g], …[g+1]), and likewise
    // for groups.  InitClas uses them to write the scene-global resident tables
    // and to patch each group blob with its IDs.
    std::vector<uint32_t>                    m_geomFirstClusterResidentID;
    std::vector<uint32_t>                    m_geomFirstGroupResidentID;
    uint32_t                                 m_sceneTotalClusters = 0;
    uint32_t                                 m_sceneTotalGroups   = 0;
};
