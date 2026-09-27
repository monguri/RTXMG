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

// ClusterLodPass — the LoD traversal.  One compute pass per frame walks each
// instance's cluster DAG and emits the flat list of clusters that should be
// rendered at the current view.  ClusterLodSystem owns where this sits in the
// frame; ClusterLodBlasPass consumes what it writes.
//
// Base path (no BLAS sharing).  Every dispatch shares one binding set, and the
// counters buffer doubles as the indirect-args source:
//
//   traversal_init                  One thread per instance.  Frustum / screen-
//                                   size / HiZ-culls the instance, then tests
//                                   the root's second-coarsest child: if even
//                                   that is fine enough the instance never
//                                   enters the queue and renders from its
//                                   persistent low-detail BLAS, which is also
//                                   the default this kernel pre-fills into
//                                   instanceBlasAddrs.
//   traversal_setup[TraversalRun]   1x1x1.  Clamps the queue counters and
//                                   writes the first indirect grid.
//   loop maxNodeTreeDepth times:
//     traversal_run                 Indirect.  One thread per CHILD of a queued
//                                   node, wave-packed so nodes with differing
//                                   child counts still fill consecutive lanes.
//                                   A child either re-enters the node queue or,
//                                   if it is a leaf group, moves to the group
//                                   queue — unless it is not resident, in which
//                                   case it is dropped and a streaming load
//                                   request is emitted instead.
//     traversal_setup[PassNodesOnly|PassCombined]  1x1x1; advances the window.
//     traversal_run_groups          Indirect, on the last pass only.  One
//                                   thread per group.  Continuous LoD resolves
//                                   here: a cluster is emitted once the finer
//                                   group it was simplified from is itself fine
//                                   enough to stop at, or does not exist.
//   traversal_setup[BlasInsertion]  1x1x1.  Clamps the cluster count and writes
//                                   the BLAS-insert indirect grid.
//
// With BLAS sharing on, three kernels replace traversal_init and decide per
// geometry which instances have to be traversed at all: instance_classify_lod
// (cull + per-geometry LoD histogram) -> blas_elect_sharing_provider (elect one
// provider instance per geometry) -> traversal_init_blas_sharing (enqueue only
// providers and view-dependent instances; consumers reuse the provider's BLAS).
//
// With merging additionally on, traversal_blas_merging appends each merged
// per-geometry proxy's clusters to the same render list.  It also performs the
// streaming age update, which is why the standalone stream_age_groups dispatch
// is skipped on those frames.
//
// After Execute() the caller can read renderClusterInfos, instanceBuildInfos,
// instanceBlasAddrs and the counters buffer.

#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <string>

#include <nvrhi/nvrhi.h>
#include <donut/engine/ShaderFactory.h>

#include "rtxmg/cluster_lod/resources.h"
#include "rtxmg/cluster_lod/shaderio.h"
#include "rtxmg/utils/buffer.h"

// HiZ occlusion: the traversal cull samples the previous-frame depth pyramid the
// ZBuffer produces.  Forward-declared — only inline accessors are used (host),
// no link dependency on the hiz library.
class ZBuffer;

struct ClusterLodPassParams
{
    // Per-frame view/LOD constants — filled by the caller each frame.
    shaderio::SceneBuildingConstants constants;

    // Runtime-togglable per frame.
    bool useBlasSharing = false;
    // Caching is a runtime sub-mode of sharing: the sharing shaders are
    // compiled as a USE_BLAS_CACHING={0,1} permutation and this selects which.
    bool useBlasCaching = false;
    // Total clusters across this frame's cached-BLAS builds.  When caching is
    // on, the traversal cluster budget is reduced by this so the per-instance
    // and cached builds together fit the shared CLAS-address pool.
    uint32_t patchCachedClustersCount = 0u;
    bool useBlasMerging = false;
    // The renderer's streaming-mode flag; only used by the useBlasMerging
    // branch's assert (merging is streaming-only).
    bool useStreaming   = true;
    // Dispatch size for traversal_blas_merging (one thread per resident active
    // group); the shader also guards on the GPU-side count.
    uint32_t activeGroupsCount = 0u;

    // Previous-frame depth pyramid backing the culling PSOs' space-1 binding.
    // Always pass the renderer's ZBuffer when it has one; the occlusion test
    // itself is gated by constants.hizNumLODs, which Execute() derives from
    // this.  Null only when there is no ZBuffer, in which case the culling PSOs
    // go unused.
    const ZBuffer* zbuffer = nullptr;
    // Runtime toggle for the HiZ occlusion test alone (frustum culling keeps
    // running).  When false Execute() forces constants.hizNumLODs = 0, making
    // the shader-side IntersectHiz a no-op while the HiZ set stays bound.
    bool useHizOcclusion = true;
};

class ClusterLodPass
{
public:
    // Maximum number of simultaneous traversal tasks in the ring buffer.  Sizes
    // BOTH the node and the group queue: 2 × 1M slots × 8B = 16 MB.
    static constexpr uint32_t kMaxTraversalInfos  = 1u << 20;

    // Output clusters per frame is an init-time budget (1u << renderClusterBits)
    // set from the UI / --render-cluster-bits; committing a new value
    // re-initializes the cluster-LoD resources and this pass.  Each
    // render cluster costs ~16 B of index/VA arrays (renderClusterInfos +
    // CLAS-VA pool), so the 20-bit default is 1M ≈ 16 MB; the CLAS *data* pool
    // is budgeted separately.
    static constexpr uint32_t kDefaultRenderClusterBits = 20;  // 1M
    static constexpr uint32_t kMinRenderClusterBits     = 16;  // 64K
    static constexpr uint32_t kMaxRenderClusterBits     = 25;  // 33M

    // Widest GROUP_CLUSTER_COUNT permutation precompiled in shaders.cfg, and the
    // limit of traversal_blas_merging's uint4 cluster bitmask.
    static constexpr uint32_t kMaxGroupClusterCount = 128;

    // Create PSOs, allocate GPU buffers.  Must be called once before any frame.
    // descriptorTable is the global bindless heap (needed for ResourceDescriptorHeap
    // accesses inside the traversal shaders).
    void Init(const ClusterLodResources&    resources,
              nvrhi::DescriptorTableHandle  descriptorTable,
              donut::engine::ShaderFactory* shaderFactory,
              nvrhi::IDevice*              device,
              bool                         debugClusterLod = false,
              uint32_t                     renderClusterBits = kDefaultRenderClusterBits);

    // Execute the LOD traversal for one frame.
    // commandList must be open.  After the call it remains open.
    void Execute(nvrhi::IDevice*              device,
                 nvrhi::ICommandList*         commandList,
                 const ClusterLodPassParams&  params);

    // ------------------------------------------------------------------
    // Output accessors (valid after Execute)
    // ------------------------------------------------------------------

    // uint2 array of size kMaxRenderClusters; entries [0..numRendered) are valid.
    nvrhi::IBuffer* GetRenderClusterInfosBuffer() const
    {
        return m_renderClusterInfos.GetBuffer();
    }

    // Per-frame render-cluster budget (1u << renderClusterBits), set at Init.
    uint32_t GetMaxRenderClusters() const { return m_maxRenderClusters; }

    // Everything this pass allocates: the traversal queues and the render-cluster
    // list the budget above sizes, plus the per-instance and per-geometry tables.
    // Feeds the VRAM window's Cluster LOD Metadata bucket.
    uint64_t GetMetadataBytes() const;

    RTXMGBuffer<uint2>& GetRenderClusterInfosTyped()
    {
        return m_renderClusterInfos;
    }

    // SceneBuildingCounters[1] — contains numRenderedClusters / blasBuildCounter etc. after Execute.
    nvrhi::IBuffer* GetCountersBuffer() const
    {
        return m_counters.GetBuffer();
    }
    // Typed accessor for diagnostic Download() — non-const because Download
    // allocates a staging readback buffer.
    RTXMGBuffer<shaderio::SceneBuildingCounters>& GetCountersTyped()
    {
        return m_counters;
    }

    // InstanceBuildInfo[numInstances] — clusterReferencesCount valid after Execute.
    nvrhi::IBuffer* GetInstanceBuildInfosBuffer() const
    {
        return m_instanceBuildInfos.GetBuffer();
    }

    // GeometryBuildInfo[numGeometries] — per-geometry sharing/caching decisions
    // (u10).  blas_cache_gather_clusters writes cachedBuildIndex here and
    // blas_cache_stage_move reads it.  Empty buffer when BLAS sharing is off.
    nvrhi::IBuffer* GetGeometryBuildInfosBuffer() const
    {
        return m_geometryBuildInfos.GetBuffer();
    }
    // Typed accessor for diagnostic Log() calls — sub-passes that need to dump
    // the per-instance build state across the BLAS build use this.  Returns a
    // non-const ref because RTXMGBuffer::Log() is non-const (allocates a
    // staging readback buffer on first call).
    RTXMGBuffer<shaderio::InstanceBuildInfo>& GetInstanceBuildInfosTyped()
    {
        return m_instanceBuildInfos;
    }

    // Per-instance BLAS addresses (GpuVirtualAddress[numInstances]).
    // Seeded with lowDetailBlasAddress by traversal_init; overwritten for slow-path
    // instances by ClusterLodBlasPass after BLAS build.
    nvrhi::IBuffer* GetInstanceBlasAddrsBuffer() const
    {
        return m_instanceBlasAddrs.GetBuffer();
    }

    // Render-stats: per-instance rendered-triangle counts (valid after Execute).
    // ClusterLodBlasPass::instance_assign_blas reads this to build the instanced
    // total, resolving each instance's sharing/merging owner.
    nvrhi::IBuffer* GetPerInstanceTrianglesBuffer() const
    {
        return m_perInstanceTriangles.GetBuffer();
    }

    RTXMGBuffer<nvrhi::GpuVirtualAddress>& GetInstanceBlasAddrsTyped()
    {
        return m_instanceBlasAddrs;
    }

private:
    void CreatePipelines(donut::engine::ShaderFactory* shaderFactory,
                         nvrhi::IDevice* device);
    void CreateBindingLayout(nvrhi::IDevice* device);

    // ---- PSOs ---------------------------------------------------------------
    struct Pipelines
    {
        nvrhi::ComputePipelineHandle computeTraversalInit;
        nvrhi::ComputePipelineHandle computeTraversalRun;      // nodes
        nvrhi::ComputePipelineHandle computeTraversalGroups;   // groups → clusters
        nvrhi::ComputePipelineHandle computeBuildSetup;

        // BLAS-sharing shaders, compiled as a USE_BLAS_CACHING={0,1}
        // permutation; index [0] = caching off, [1] = on, selected per frame
        // by params.useBlasCaching.
        std::array<nvrhi::ComputePipelineHandle, 2> computeInstanceClassifyLod;
        std::array<nvrhi::ComputePipelineHandle, 2> computeGeometryBlasSharing;
        std::array<nvrhi::ComputePipelineHandle, 2> computeTraversalInitBlasSharing;  // replaces computeTraversalInit when sharing
        nvrhi::ComputePipelineHandle computeTraversalMerge;
    };
    Pipelines m_pipelines;

    // Persistent volatile CBV for SceneBuildingConstants (b0), allocated in
    // Init() and writeBuffer'd each frame.
    nvrhi::BufferHandle m_constantsCB;

    nvrhi::BindingLayoutHandle   m_bindingLayout;
    nvrhi::BindingLayoutHandle   m_bindlessLayout;
    nvrhi::DescriptorTableHandle m_descriptorTable;
    // HiZ occlusion (register space 1): texture-array[HIZ_MAX_LODS] + sampler on
    // the culling PSOs (compiled with -D CLUSTER_LOD_HIZ_OCCLUSION=1).  The set is
    // rebuilt per frame from the ZBuffer's HiZ textures; the sampler persists.
    nvrhi::BindingLayoutHandle   m_hizBindingLayout;
    nvrhi::SamplerHandle         m_hizSampler;

    // ---- Persistent data from ClusterLodResources ----
    const RTXMGBuffer<shaderio::Geometry>*       m_geometriesBuffer       = nullptr;
    const RTXMGBuffer<shaderio::RenderInstance>* m_renderInstancesBuffer  = nullptr;
    const RTXMGBuffer<shaderio::StreamingGroup>* m_residentGroupsBuffer = nullptr;  // u7

    // Bindings consumed by traversal_run.hlsl's load-emit branch.  Streaming
    // points these at the real buffers; preload falls back to the dummies so
    // the BindingSet stays valid (its load-emit branch never fires — every
    // group is resident).
    nvrhi::IBuffer* m_streamingShaderBuffer     = nullptr;  // u8
    nvrhi::IBuffer* m_streamingLoadGroupsBuffer = nullptr;  // u9 full request ring
    nvrhi::BufferHandle  m_streamingDummyAggregate;   // preload u8 fallback (stride = sizeof(SceneStreaming))
    nvrhi::BufferHandle  m_streamingDummyLoadGroups;  // preload u9 fallback (stride = 8, uint2)

    // Streaming age-filter buffers, read by traversal_blas_merging only but part
    // of the shared layout, so preload falls back to dummies the same way.
    nvrhi::IBuffer* m_activeGroupsBuffer  = nullptr;  // u15
    nvrhi::IBuffer* m_groupIDsBuffer      = nullptr;  // u16
    nvrhi::IBuffer* m_unloadRequestBuffer = nullptr;  // u17 full request ring
    uint64_t        m_unloadRingBytes     = 0u;       // 0 => bind the dummy whole
    nvrhi::BufferHandle m_dummyActiveGroups;
    nvrhi::BufferHandle m_dummyGroupIDs;
    nvrhi::BufferHandle m_dummyUnloadRequest;

    uint32_t        m_numInstances        = 0;
    uint32_t        m_maxRenderClusters   = 1u << kDefaultRenderClusterBits;  // set at Init
    uint32_t        m_numGeometries       = 0;   // BLAS sharing: blas_elect_sharing_provider dispatch size
    uint32_t        m_maxNodeTreeDepth    = 1;   // how many traversal_run dispatches per frame
    uint32_t        m_maxClustersPerGroup = 32;  // picks the merge kernel's GROUP_CLUSTER_COUNT
    bool            m_debugClusterLod     = false;

    // ---- Per-frame GPU buffers ----
    RTXMGBuffer<shaderio::SceneBuildingCounters> m_counters;
    RTXMGBuffer<shaderio::InstanceBuildInfo>     m_instanceBuildInfos;
    RTXMGBuffer<nvrhi::GpuVirtualAddress>        m_instanceBlasAddrs;
    RTXMGBuffer<uint2>                           m_traversalNodeQ;   // interior node queue
    RTXMGBuffer<uint2>                           m_traversalGroupQ;  // leaf-group queue
    RTXMGBuffer<uint2>                           m_renderClusterInfos;

    // BLAS sharing — per-geometry tables (u10/u11), one entry per unique
    // geometry; histograms cleared each frame.  Always created so the
    // BindingSet shape is stable, but only used when params.useBlasSharing.
    RTXMGBuffer<shaderio::GeometryBuildInfo>      m_geometryBuildInfos;
    RTXMGBuffer<shaderio::GeometryBuildHistogram> m_geometryHistograms;

    // Per-instance visibility flags (u12).  Always allocated + cleared each
    // frame; tagged with shaderio::InstanceVisibility::UsesMerged by
    // traversal_init_blas_sharing and read by traversal_run.
    RTXMGBuffer<uint32_t>                        m_instanceVisibility;

    // Render-stats (TRACK_RENDER_STATS): per-instance rendered-triangle tally,
    // plus a per-resident-cluster "seen this frame" flag for the unique-triangle
    // CLAS dedup.  Always allocated so the BindingSet shape is stable.
    RTXMGBuffer<uint32_t>                        m_perInstanceTriangles;
    RTXMGBuffer<uint32_t>                        m_uniqueSeenClusters;
};
