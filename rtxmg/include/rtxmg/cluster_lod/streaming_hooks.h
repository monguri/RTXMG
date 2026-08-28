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

// IClusterLodStreamingHooks — the streaming half of the cluster-LoD resource
// contract: the four per-frame command hooks the renderer drives around
// traversal, plus the buffers, counts and residency reporting only a streaming
// residency backend can supply.  Separate from ClusterLodResources so that
// interface stays the short list every residency backend must implement; a
// backend without streaming (ClusterLodPreloaded) returns null from
// GetStreamingHooks() and every consumer reads that as "streaming off".

#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <vector>

#include <nvrhi/nvrhi.h>

#include "rtxmg/cluster_lod/geometry_group.h"
#include "rtxmg/cluster_lod/shaderio.h"

namespace rtxmg
{
struct StreamingConfig;
struct StreamingStats;
}

class IClusterLodStreamingHooks
{
public:
    virtual ~IClusterLodStreamingHooks() = default;

    // Per-frame policy, passed to StageResidencyUpdate.
    struct FrameSettings
    {
        bool                              useBlasCaching        = false;
        uint32_t                          blasCacheMaxClusters  = 0;
        uint32_t                          blasCacheMaxBuilds    = 0;
        uint32_t                          blasCacheMinLevel     = 0;
        uint32_t                          ageThreshold          = 16;
    };

    // The renderer drives these four in this exact order, each with a recording
    // command list.  The order is a data dependency, not a convention, so the
    // constraint that pins each one is spelled out:
    //   1. StageResidencyUpdate  reads the prior frame's readback and stages the
    //      group blobs into the upload ring.  Host + transfer work only.
    //   2. ApplyResidencyUpdate  applies the staged loads via
    //      stream_update_scene.hlsl and kicks the persistent CLAS allocator.
    //      MUST precede traversal, so this frame sees the new residency.
    //   3. FinalizeResidency     age filter + CLAS Move-op + status publish.
    //      MUST follow traversal (which resets the ages of the groups it used and
    //      emits the load requests) and precede dynamic BLAS setup, so BLAS sees
    //      this frame's resident CLAS addresses.
    //   4. CaptureFrameRequests  copies the request bitset toward the readback
    //      that step 1 pops next frame.
    //
    // SignalTasksSubmitted is the odd one out: no command list, and it MUST be
    // called after the frame's command list has been submitted because it fences
    // the streaming tasks against that submission.
    virtual void StageResidencyUpdate(nvrhi::ICommandList* commandList,
                                      const FrameSettings& settings) = 0;
    virtual void ApplyResidencyUpdate(nvrhi::ICommandList* commandList) = 0;
    virtual void FinalizeResidency   (nvrhi::ICommandList* commandList, bool runAgeFilter) = 0;
    virtual void CaptureFrameRequests(nvrhi::ICommandList* commandList) = 0;
    virtual void SignalTasksSubmitted() = 0;

    // Fraction [0,1] of the streaming budgets in use; drives rtxmg::
    // AdaptiveLodError.  Readback-based, so ~1 frame stale.
    virtual float GetLoadFactor() const = 0;

    // Residency / pool snapshot for the profiler.
    virtual void GetStats(rtxmg::StreamingStats& out) const = 0;

    // The effective (post-init, defaults-resolved) configuration this backend
    // is running with.
    virtual const rtxmg::StreamingConfig& GetStreamingConfig() const = 0;

    // What is resident right now, for a UI that shows residency per geometry
    // and LoD level.  Filled by GetResidencyReport.
    struct ResidencyReport
    {
        // Resident-blob policy: a stripped channel can never be resident.
        bool stripResidentPositions = false;
        bool stripResidentNormals   = false;

        // Monotonic change counter of the resident set.
        uint64_t residencyEpoch = 0;

        // Every resident group, in active-list order: the always-resident
        // low-detail prefix leads, so the pinned-vs-streamable split is an index
        // compare against pinnedGroupsCount.
        std::vector<rtxmg::GeometryGroup> residentGroups;
        uint32_t                          pinnedGroupsCount = 0;

        // Per geometry, the LoD level whose BLAS is cached — that level stays
        // available even when its groups would otherwise stream out.
        // shaderio::kTraversalInvalidLodLevel when nothing is cached.
        std::vector<uint32_t> cachedBlasLevels;

        // Resident CLAS bytes per (geometry, LoD level): a view of the backend's
        // readback, valid until the next frame.  Empty until the first readback
        // lands, and on a backend with no persistent CLAS allocator.
        std::span<const std::array<uint64_t, shaderio::kMaxLodLevels>> residentClasBytes;
    };

    // Fills `out` and arms the CLAS-sizes readback that populates
    // residentClasBytes a frame or more later.  Returns true when residentGroups
    // was rebuilt — that sweep is O(resident), so it only reruns when the
    // residency epoch moves.  Pass the same report back every frame.
    virtual bool GetResidencyReport(ResidencyReport& out) = 0;

    // Bindings consumed by the traversal_run.hlsl load-emit branch (u8/u9).
    virtual nvrhi::IBuffer* GetStreamingShaderBuffer()         const = 0;
    virtual nvrhi::IBuffer* GetStreamingLoadGroupsBuffer()     const = 0;

    // traversal_blas_merging folds the streaming age filter into its dispatch,
    // so it needs the age-filter buffers: the active-groups buffer (bound whole;
    // the shader adds persistentGroupsCount), the group-IDs table, and the full
    // unload request ring.
    virtual nvrhi::IBuffer* GetActiveGroupsBuffer()            const = 0;
    virtual nvrhi::IBuffer* GetGroupIDsBuffer()                const = 0;
    virtual nvrhi::IBuffer* GetUnloadRequestBuffer()           const = 0;
    virtual uint64_t        GetUnloadRequestRingBytes()        const = 0;
    // Resident active-group suffix count (excludes the persistent low-detail
    // prefix) — the merge dispatch size, same count stream_age_groups uses.
    virtual uint32_t        GetActiveGroupsCount()             const = 0;

    // Per-frame inputs the BLAS pass consumes when useBlasCaching is set.
    //   * patchCachedBlasCount      — cached BLASes (re)built this frame.
    //   * geometryPatchesBuffer/Off — this task's StreamingGeometryPatch slot.
    //   * cachedBlasPoolBlocks      — AS-storage blocks of the cached-BLAS pool
    //                                 (MOVE destinations; transitioned by the
    //                                 BLAS pass to AccelStructBuildBlas).
    virtual uint32_t        GetMaxCachedBlasBuilds()       const = 0;
    virtual uint32_t        GetPatchCachedBlasCount()      const = 0;
    virtual uint32_t        GetPatchCachedClustersCount()  const = 0;
    virtual nvrhi::IBuffer* GetGeometryPatchesBuffer()     const = 0;
    virtual uint64_t        GetGeometryPatchesByteOffset() const = 0;
    virtual const std::vector<nvrhi::IBuffer*>* GetCachedBlasPoolBlocks() = 0;

    // Frees all cached BLASes and zeroes their shader-visible addresses. The
    // renderer calls this when the BLAS-caching toggle flips so no instance can
    // dangle on a cached BLAS whose pool reservation is no longer held.
    virtual void ResetCachedBlas(nvrhi::ICommandList* commandList) = 0;
};
