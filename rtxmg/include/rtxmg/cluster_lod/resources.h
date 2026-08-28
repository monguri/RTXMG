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

// ClusterLodResources — abstract interface that ClusterLodPass and
// ClusterLodBlasPass consume, so the same passes can run against either
// ClusterLodPreloaded (everything resident at startup, no streaming) or
// ClusterLodStreaming (per-frame request emission + readback + on-demand
// load loop).
//
// This is the whole contract a residency backend must implement.  The
// streaming-only half — the per-frame command hooks and the buffers behind
// them — is IClusterLodStreamingHooks, reached through GetStreamingHooks().

#pragma once

#include <cstdint>

#include <nvrhi/nvrhi.h>

#include "rtxmg/cluster_lod/shaderio.h"
#include "rtxmg/cluster_lod/streaming_hooks.h"
#include "rtxmg/utils/buffer.h"

class ClusterLodResources
{
public:
    virtual ~ClusterLodResources() = default;

    // Buffer of one shaderio::Geometry per geometry; consumed by traversal
    // and BLAS-build shaders for per-geometry SRV/UAV bindless indices.
    virtual const RTXMGBuffer<shaderio::Geometry>& GetShaderGeometriesBuffer() const = 0;

    // Buffer of one shaderio::RenderInstance per instance.
    virtual const RTXMGBuffer<shaderio::RenderInstance>& GetShaderRenderInstancesBuffer() const = 0;

    virtual uint32_t        GetRenderInstanceCount()         const = 0;

    // Maximum depth of the LOD node tree across all loaded geometries —
    // drives the multipass traversal_run dispatch count.
    virtual uint32_t        GetMaxNodeTreeDepth()            const = 0;

    // Maximum per-geometry group count across all loaded geometries (the size
    // of the largest geometry's group-address table). Bounds the per-geom
    // `groupIndex` used by traversal_run_groups; sizes the diagnostic
    // rendered-group bitmap.
    virtual uint32_t        GetMaxPerGeometryGroups()        const = 0;

    // Clusters per group the bake was produced with — selects
    // traversal_blas_merging's GROUP_CLUSTER_COUNT permutation.
    virtual uint32_t        GetMaxClustersPerGroup()         const = 0;

    // Scene-global per-resident-cluster tables.
    //   - GetResidentClasAddressesBuffer: RWStructuredBuffer<uint64_t>
    //     [count], indexed by clusterResidentID.  Bound as SRV(t4) into
    //     blas_insert_clusters.
    //   - GetResidentClustersBuffer: RWStructuredBuffer<ClusterAddress>
    //     [count], indexed by clusterResidentID.  Bound as SRV into the hit
    //     shader.
    // `count` is sceneTotalClusters in preload mode, maxResidentClusters in
    // streaming mode — opaque to consumers, which just index by the
    // clusterResidentID value emitted by traversal_run_groups.
    virtual const RTXMGBuffer<uint64_t>& GetResidentClasAddressesBuffer() const = 0;
    virtual const RTXMGBuffer<shaderio::ClusterAddress>& GetResidentClustersBuffer() const = 0;

    // Scene-global StreamingGroup table indexed by groupResidentID, bound at u7
    // in the traversal pass.  traversal_run_groups clears the age field on
    // visit; stream_age_groups increments it and emits eviction signals.  Preload
    // returns a placeholder — it never age-filters, so the age write is inert.
    virtual const RTXMGBuffer<shaderio::StreamingGroup>& GetResidentGroupsBuffer() const = 0;

    // The streaming half of the contract, or null on a backend that has none
    // (preload).  Consumers read null as "streaming off".
    virtual const IClusterLodStreamingHooks* GetStreamingHooks() const { return nullptr; }
    virtual       IClusterLodStreamingHooks* GetStreamingHooks()       { return nullptr; }
};
