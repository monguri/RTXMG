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

#include "rtxmg/cluster_lod/system.h"

#include "rtxmg/cluster_lod/preloaded.h"
#include "rtxmg/profiler/statistics.h"

namespace rtxmg
{

void ClusterLodSystem::Update(nvrhi::IDevice*      device,
                              nvrhi::ICommandList* commandList,
                              const FrameParams&   params)
{
    if (!m_resources)
        return;

    // Preloaded path only: the log API lives on the concrete subclass.
    if (params.debugClusterLod)
    {
        if (!m_loggedPreloadGeometry)
        {
            if (auto* preloaded = dynamic_cast<ClusterLodPreloaded*>(m_resources))
                preloaded->LogGeometryData();
            m_loggedPreloadGeometry = true;
        }
    }

    // Streaming runs as four hooks around traversal; the preloaded path has no
    // hooks at all.  FinalizeResidency must precede the BLAS build so the BLAS
    // consumes the CLAS addresses this frame's streaming update published.
    IClusterLodStreamingHooks* streaming = m_resources->GetStreamingHooks();

    // Per-frame cache policy: MaxClusters/MaxBuilds bound what HandleBlasCaching
    // (re)builds this frame; blasCacheMinLevel is a coarse-tail level count.
    m_frameSettings.useBlasCaching       = params.useBlasCaching;
    m_frameSettings.blasCacheMinLevel    = params.blasCacheMinLevel;
    m_frameSettings.blasCacheMaxClusters = params.blasCacheMaxClusters;
    m_frameSettings.blasCacheMaxBuilds   = streaming ? streaming->GetMaxCachedBlasBuilds() : 0u;

    {
        ScopedGPUTimer timer(stats::clusterAccelSamplers.clusterLodUploadTime, commandList);
        stats::clusterAccelSamplers.clusterLodHostTime.Start();
        if (streaming)
            streaming->StageResidencyUpdate(commandList, m_frameSettings);
        stats::clusterAccelSamplers.clusterLodHostTime.Stop();
    }
    if (streaming)
        streaming->ApplyResidencyUpdate(commandList);

    {
        ClusterLodPassParams passParams;
        passParams.constants      = params.constants;
        // Always pass the depth pyramid when present; Execute() gates the
        // occlusion test on useCulling + useHizOcclusion + GetNumHiZLODs.
        passParams.zbuffer        = params.zbuffer;
        passParams.useHizOcclusion = params.useHizOcclusion;
        passParams.useBlasSharing = params.useBlasSharing;
        passParams.useBlasCaching = params.useBlasCaching;
        passParams.useBlasMerging = params.useBlasMerging;
        passParams.useStreaming   = params.useStreaming;
        // Reserve shared-CLAS-pool room for this frame's cached-BLAS builds by
        // shrinking the traversal cluster budget (see ClusterLodPass::Execute).
        if (params.useBlasCaching)
            passParams.patchCachedClustersCount = streaming ? streaming->GetPatchCachedClustersCount() : 0u;
        // Sizes the merge dispatch: one thread per resident active group.
        if (params.useBlasMerging)
        {
            passParams.activeGroupsCount = streaming ? streaming->GetActiveGroupsCount() : 0u;
        }
        ScopedGPUTimer timer(stats::clusterAccelSamplers.clusterLodTraversalTime, commandList);
        m_pass.Execute(device, commandList, passParams);
    }

    // traversal_blas_merging already runs the age filter inline.
    if (streaming)
        streaming->FinalizeResidency(commandList, /*runAgeFilter=*/!params.useBlasMerging);

    {
        ClusterLodBlasPassParams blasParams;
        blasParams.useBlasSharing = params.useBlasSharing;
        blasParams.useBlasCaching = params.useBlasCaching;
        if (params.useBlasCaching)
        {
            // Streaming-owned; null/0 on the preloaded path, which has no caching.
            blasParams.patchCachedBlasCount      = streaming ? streaming->GetPatchCachedBlasCount()      : 0u;
            blasParams.geometryPatchesBuffer     = streaming ? streaming->GetGeometryPatchesBuffer()     : nullptr;
            blasParams.geometryPatchesByteOffset = streaming ? streaming->GetGeometryPatchesByteOffset() : 0ull;
            blasParams.cachedBlasPoolBlocks      = streaming ? streaming->GetCachedBlasPoolBlocks()      : nullptr;
        }
        ScopedGPUTimer timer(stats::clusterAccelSamplers.clusterLodBlasBuildTime, commandList);
        m_blasPass.Execute(device, commandList, m_pass, blasParams);
    }

    // BLAS stats for the Profiler "Streaming" tab — async readback, 1-frame lag.
    // Summing the per-build sizes host-side avoids needing a 64-bit GPU atomic.
    {
        const auto counters  = m_pass.GetCountersTyped().Download(commandList, /*async=*/true);
        const auto blasSizes = m_blasPass.GetBlasSizesTyped().Download(commandList, /*async=*/true);
        if (!counters.empty())
        {
            const shaderio::SceneBuildingCounters c = counters[0];
            // The first async readback maps a not-yet-written staging buffer, so
            // reject the impossible counts it returns and keep the last good value.
            if (c.blasBuildCounter <= uint32_t(blasSizes.size()))
            {
                m_counters = c;
                uint64_t total = 0;
                for (uint32_t i = 0; i < c.blasBuildCounter; ++i)
                    total += blasSizes[i];
                m_blasActualBytes = total;
            }
        }
    }

    if (streaming)
        streaming->CaptureFrameRequests(commandList);

    if (params.useStreaming)
    {
        // Submit the streaming + accel half early so the GPU runs it while the
        // host records the render half, and so SignalTasksSubmitted has a
        // submission to fence the completed tasks against (it requires one).
        stats::clusterAccelSamplers.clusterLodSubmitIdleTime.Start(commandList);
        commandList->close();
        device->executeCommandList(commandList);
        if (streaming)
            streaming->SignalTasksSubmitted();
        commandList->open();
        stats::clusterAccelSamplers.clusterLodSubmitIdleTime.Stop();
    }
}

}  // namespace rtxmg
